import os
import argparse
import json
import logging
import glob
import time
import datetime
import warnings
import traceback
import gc
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Import custom modules
from models.domain_adapter import UnifiedDomainAdapter
from models.gat import DeviceGAT
from utils.spot import SPOTDetector
from utils.rcl import root_cause_localization
from utils.pattern_matcher import PatternMatcher

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure hardware
num_cores = max(1, os.cpu_count() // 2 - 1)
torch.set_num_threads(num_cores)
torch.set_num_interop_threads(num_cores)

# --- Worker Function for Multiprocessing ---
def process_device_spot(args):
    """
    Parallel processing function for SPOT detection on a single device.
    """
    device_idx, device_mse, detector, warmup_windows, init_points, num_windows = args

    device_anomaly_labels = np.zeros(num_windows, dtype=np.int32)
    device_thresholds = np.full(num_windows, np.nan, dtype=np.float32)

    if detector is None:
        return device_idx, device_anomaly_labels, device_thresholds

    for window_idx in range(num_windows):
        if window_idx < warmup_windows:
            device_anomaly_labels[window_idx] = 0
            device_thresholds[window_idx] = np.nan
        elif window_idx < init_points:
            device_anomaly_labels[window_idx] = 0
            device_thresholds[window_idx] = np.nan
        else:
            mse_value = device_mse[window_idx]
            _, threshold, label = detector.detect_point(np.array([mse_value]))
            device_anomaly_labels[window_idx] = int(label)
            device_thresholds[window_idx] = threshold

    return device_idx, device_anomaly_labels, device_thresholds

# --- Switch Tester Class ---
class SwitchTester:
    """
    Tester class for a single switch using Pretrained ADA and Trained GAT models.
    """

    def __init__(self, switch_name, config, logger=None):
        self.switch_name = switch_name
        self.config = config
        self.logger = logger
        self.device = torch.device('cpu') # Inference usually on CPU for stability in multiprocessing

        # Paths
        self.model_save_dir = os.path.join(config['model_dir'], switch_name, "model")
        if not os.path.exists(self.model_save_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_save_dir}")

        # Load GAT Config
        with open(os.path.join(self.model_save_dir, 'config.json'), 'r') as f:
            self.gat_config = json.load(f)

        # 1. Load Pretrained ADA (UnifiedDomainAdapter)
        if self.logger: self.logger.info("Loading pretrained Domain Adapter...")
        
        # Use config path provided via args
        pretrained_ada_path = config['ada_path']
        if os.path.exists(pretrained_ada_path):
            checkpoint = torch.load(pretrained_ada_path, map_location=self.device)
            model_config = checkpoint['model_config']

            self.domain_adapter = UnifiedDomainAdapter(
                input_feature_dim=model_config['input_feature_dim'],
                transformer_d_model=config['transformer_d_model'],
                transformer_nhead=config['transformer_nhead'],
                transformer_num_layers=config['transformer_num_layers'],
                transformer_out_dim=config['transformer_out_dim'],
                num_domains=model_config['num_domains'],
                mapped_dim=model_config['mapped_dim'],
                reconstruction_dim=model_config['input_feature_dim']
            ).to(self.device)
            self.domain_adapter.load_state_dict(checkpoint['ada_state_dict'])
        else:
            raise FileNotFoundError(f"ADA model not found at {pretrained_ada_path}")

        # Freeze ADA
        for param in self.domain_adapter.parameters():
            param.requires_grad = False
        self.domain_adapter.eval()

        # 2. Load Trained GAT (DeviceGAT)
        if self.logger: self.logger.info("Loading trained GAT model...")
        
        self.gat_model = DeviceGAT(
            input_feature_dim=config['input_feature_dim'],
            mapped_dim=config['transformer_out_dim'],
            hidden_dim=config['gat_hidden_dim'],
            gat_heads=config['gat_heads'],
            dropout=config['dropout'],
            decoder_hidden_dims=config['decoder_hidden_dims']
        ).to(self.device)

        gat_path = os.path.join(self.model_save_dir, 'gat_model.pt')
        gat_checkpoint = torch.load(gat_path, map_location=self.device)
        
        # Handle state_dict loading
        if isinstance(gat_checkpoint, dict) and 'gat_state_dict' in gat_checkpoint:
            self.gat_model.load_state_dict(gat_checkpoint['gat_state_dict'])
        else:
            self.gat_model.load_state_dict(gat_checkpoint)
        
        self.gat_model.eval()

    def extract_features_from_csv(self, csv_file_path):
        """Extracts features from CSV for inference."""
        df = pd.read_csv(csv_file_path).sort_values("timestamp")
        
        feature_cols = [
            'voltage', 'currentMultiBias1', 'currentMultiBias2', 'currentMultiBias3', 'currentMultiBias4',
            'currentMultiRXPower1', 'currentMultiRXPower2', 'currentMultiRXPower3', 'currentMultiRXPower4',
            'currentMultiTXPower1', 'currentMultiTXPower2', 'currentMultiTXPower3', 'currentMultiTXPower4',
            'temperature'
        ]

        local_feat, remote_feat = [], []
        for col in feature_cols:
            l_col, r_col = f'local_{col}', f'remote_{col}'
            if l_col in df.columns and r_col in df.columns:
                local_feat.append(df[l_col].values.astype(np.float32))
                remote_feat.append(df[r_col].values.astype(np.float32))

        if not local_feat or not remote_feat:
            return None, None, df, None

        return np.stack(local_feat, axis=1), np.stack(remote_feat, axis=1), df, feature_cols

    def process_single_switch(self, switch_path, pattern_matcher=None):
        """
        Process all CSVs in a switch folder: Feature Ext -> GAT -> SPOT -> RCA -> Pattern Match.
        """
        if self.logger: self.logger.info(f"Processing switch: {switch_path}")

        csv_files = glob.glob(os.path.join(switch_path, "*.csv"))
        if not csv_files:
            return {}

        # 1. Prepare Data
        csv_data, csv_names = [], []
        feature_names = None
        min_length = float('inf')

        for csv_file in csv_files:
            try:
                l_feat, r_feat, df_orig, f_names = self.extract_features_from_csv(csv_file)
                if l_feat is None: continue
                
                if feature_names is None: feature_names = f_names

                T = len(l_feat)
                if T < self.config['window_len']: continue

                csv_data.append({
                    'local_features': l_feat,
                    'remote_features': r_feat,
                    'original_df': df_orig,
                    'local_domain': str(df_orig.get('local_vendor_cls', pd.Series(['unknown'])).iloc[0]),
                    'remote_domain': str(df_orig.get('remote_vendor_cls', pd.Series(['unknown'])).iloc[0])
                })
                csv_names.append(os.path.basename(csv_file))
                min_length = min(min_length, T)
            except Exception as e:
                if self.logger: self.logger.error(f"Error reading {csv_file}: {e}")

        if not csv_data: return {}

        # 2. Initialize Results Containers
        results_dict = {}
        for i, (data, csv_name) in enumerate(zip(csv_data, csv_names)):
            res_df = data['original_df'].copy()
            # Init columns
            for feat in feature_names:
                res_df[f'local_reconstructed_{feat}'] = np.nan
                res_df[f'remote_reconstructed_{feat}'] = np.nan
            
            for prefix in ['local', 'remote']:
                res_df[f'{prefix}_anomaly_label'] = 0
                res_df[f'{prefix}_threshold'] = np.nan
                res_df[f'{prefix}_root_cause_score'] = 0.0
                res_df[f'{prefix}_pattern_score'] = np.nan
                res_df[f'{prefix}_pattern_category'] = -1
                res_df[f'{prefix}_pattern_description'] = ""

            results_dict[csv_name] = {'result_df': res_df, 'original_length': len(res_df)}

        # 3. Sliding Window Inference
        window_len = self.config['window_len']
        num_windows = min_length - window_len + 1
        num_devices = 2 * len(csv_data)
        
        mse_matrix = np.zeros((num_windows, num_devices))
        root_cause_matrix = np.zeros((num_windows, num_devices))
        
        # Pattern matching state
        interval = 144
        last_pattern_window_idx = -interval
        last_pattern_res = (None, None, None) # scores, cats, descs
        pattern_results_list = []

        # -- Main Loop --
        for window_idx in tqdm(range(num_windows), desc=f"Inference {self.switch_name}", disable=not self.config['show_progress']):
            start_idx = window_idx
            end_idx = start_idx + window_len

            # Construct Batch [2*M, T, F]
            all_feats, all_types = [], []
            for data in csv_data:
                all_feats.append(data['local_features'][start_idx:end_idx])
                all_types.append(1) # Local
                all_feats.append(data['remote_features'][start_idx:end_idx])
                all_types.append(0) # Remote
            
            window_tensor = torch.tensor(np.stack(all_feats), dtype=torch.float32).to(self.device)
            type_tensor = torch.tensor(all_types, dtype=torch.long).to(self.device)

            with torch.no_grad():
                # ADA Feature Extraction
                node_features = self.domain_adapter.feature_extractor(window_tensor)

                # Pattern Matching (Periodic)
                if pattern_matcher and (window_idx - last_pattern_window_idx >= interval or window_idx == 0):
                    p_scores, p_cats, p_descs = pattern_matcher.match_patterns_from_features(node_features, return_description=True)
                    last_pattern_res = (p_scores, p_cats, p_descs)
                    last_pattern_window_idx = window_idx
                
                # Use current or last pattern result
                p_scores, p_cats, p_descs = last_pattern_res

                # GAT Inference
                reconstructed, edge_index, edge_weights = self.gat_model(node_features, type_tensor)

                # MSE Calculation
                target = window_tensor[:, -1, :]
                mse_per_node = F.mse_loss(reconstructed, target, reduction='none').mean(dim=1)
                mse_matrix[window_idx] = mse_per_node.cpu().numpy()

                # Root Cause Localization
                # Use dummy labels as we are in unsupervised testing
                dummy_labels = torch.ones_like(mse_per_node, dtype=torch.long)
                # !!! Fix tuple unpacking here !!!
                rc_scores, _ = root_cause_localization(
                    mse_per_node, dummy_labels, edge_index, edge_weights, device=self.device
                )
                root_cause_matrix[window_idx] = rc_scores.cpu().numpy()

                # Store Pattern Results
                if p_scores is not None:
                    pattern_results_list.append({
                        'window_idx': window_idx,
                        'time_idx': end_idx - 1,
                        'scores': p_scores.cpu().numpy(),
                        'cats': p_cats.cpu().numpy(),
                        'descs': p_descs
                    })

                # Store Reconstruction
                time_idx = end_idx - 1
                for dev_idx, csv_name in enumerate(csv_names):
                    res_info = results_dict[csv_name]
                    if time_idx >= res_info['original_length']: continue
                    
                    df = res_info['result_df']
                    l_idx, r_idx = dev_idx * 2, dev_idx * 2 + 1
                    
                    # Save Reconstructed Features
                    for c_idx, fname in enumerate(feature_names):
                        df.at[time_idx, f'local_reconstructed_{fname}'] = reconstructed[l_idx, c_idx].item()
                        df.at[time_idx, f'remote_reconstructed_{fname}'] = reconstructed[r_idx, c_idx].item()
                    
                    # Save RC Scores
                    df.at[time_idx, 'local_root_cause_score'] = rc_scores[l_idx].item()
                    df.at[time_idx, 'remote_root_cause_score'] = rc_scores[r_idx].item()

        # 4. Fill Pattern Matching Results (Batch update)
        for res in pattern_results_list:
            t_idx = res['time_idx']
            for dev_idx, csv_name in enumerate(csv_names):
                df = results_dict[csv_name]['result_df']
                if t_idx >= len(df): continue
                
                l_idx, r_idx = dev_idx * 2, dev_idx * 2 + 1
                
                df.at[t_idx, 'local_pattern_score'] = res['scores'][l_idx]
                df.at[t_idx, 'remote_pattern_score'] = res['scores'][r_idx]
                df.at[t_idx, 'local_pattern_category'] = res['cats'][l_idx]
                df.at[t_idx, 'remote_pattern_category'] = res['cats'][r_idx]
                df.at[t_idx, 'local_pattern_description'] = res['descs'][l_idx]
                df.at[t_idx, 'remote_pattern_description'] = res['descs'][r_idx]

        del pattern_results_list
        gc.collect()

        # 5. SPOT Anomaly Detection (Parallelized)
        if self.logger: self.logger.info("Running SPOT detection...")
        
        init_points = 24 * 12 * 3
        warmup = 287
        
        spot_detectors = []
        for i in range(num_devices):
            detector = SPOTDetector(
                q=self.config.get('spot_q', 1e-4),
                dynamic=self.config.get('spot_dynamic', True)
            )
            dev_mse = mse_matrix[:, i]
            
            # Init Data
            if len(dev_mse) > init_points:
                init_data = dev_mse[warmup:init_points].reshape(-1, 1)
            else:
                init_data = dev_mse.reshape(-1, 1)
            
            if len(init_data) > 0:
                detector.fit(init_data)
                spot_detectors.append(detector)
            else:
                spot_detectors.append(None)

        # Parallel Execution
        tasks = []
        for i in range(num_devices):
            tasks.append((i, mse_matrix[:, i], spot_detectors[i], warmup, init_points, num_windows))

        anomaly_labels = np.zeros((num_windows, num_devices), dtype=np.int32)
        thresholds = np.full((num_windows, num_devices), np.nan, dtype=np.float32)

        with ProcessPoolExecutor(max_workers=max(1, mp.cpu_count()-2)) as executor:
            for res in executor.map(process_device_spot, tasks):
                m_idx, m_labels, m_thresh = res
                anomaly_labels[:, m_idx] = m_labels
                thresholds[:, m_idx] = m_thresh

        # 6. Save SPOT Results to DataFrame
        for window_idx in range(num_windows):
            time_idx = window_idx + window_len - 1
            for dev_idx, csv_name in enumerate(csv_names):
                df = results_dict[csv_name]['result_df']
                if time_idx >= len(df): continue
                
                l_idx, r_idx = dev_idx * 2, dev_idx * 2 + 1
                
                df.at[time_idx, 'local_anomaly_label'] = anomaly_labels[window_idx, l_idx]
                df.at[time_idx, 'local_threshold'] = thresholds[window_idx, l_idx]
                df.at[time_idx, 'remote_anomaly_label'] = anomaly_labels[window_idx, r_idx]
                df.at[time_idx, 'remote_threshold'] = thresholds[window_idx, r_idx]

        return {k: v['result_df'] for k, v in results_dict.items()}

    def save_results(self, results, output_dir):
        switch_out = os.path.join(output_dir, self.switch_name)
        os.makedirs(switch_out, exist_ok=True)
        
        saved = []
        for name, df in results.items():
            fname = name.replace('.csv', '_results.csv')
            path = os.path.join(switch_out, fname)
            df.to_csv(path, index=False)
            saved.append(path)
        
        return saved

# --- Utilities ---
def setup_logger(name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), mode='w')
        ch = logging.StreamHandler()
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt); ch.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(ch)
    return logger

def load_pattern_templates(template_dir):
    templates = []
    # Map filenames to categories (Simplified for brevity, use your full mapping)
    mapping = {
        'template1_down.csv': (1, 'connector'), 'template1_up.csv': (1, 'connector'),
        'template4_down.csv': (4, 'device'),    'template4_up.csv': (4, 'device'),
        # Add all others here...
    }
    
    if not os.path.exists(template_dir):
        return []

    for fname, (cat, desc) in mapping.items():
        fpath = os.path.join(template_dir, fname)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                # Stack all 14 value columns
                cols = [df[f'value{i}'].values if f'value{i}' in df else np.zeros(len(df)) for i in range(1,15)]
                data = np.column_stack(cols).astype(np.float32)
                templates.append({'data': data, 'category': cat, 'description': desc})
            except Exception as e:
                print(f"Error loading template {fname}: {e}")
    return templates

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="GAT Testing & Anomaly Detection")
    parser.add_argument('--data_root', type=str, default='./pod17_normalized', help="Dataset root")
    parser.add_argument('--model_dir', type=str, default='./model_results/GAT_models', help="Where GAT models are saved")
    parser.add_argument('--ada_path', type=str, required=True, help="Path to pretrained ADA .pt file")
    parser.add_argument('--output_dir', type=str, default='./experiment_results/final_results')
    parser.add_argument('--template_dir', type=str, default='./failure_templates')
    parser.add_argument('--spot_q', type=float, default=1e-5)
    
    args = parser.parse_args()

    # Config merging
    config = {
        'input_feature_dim': 14,
        'transformer_d_model': 64,
        'transformer_nhead': 8,
        'transformer_num_layers': 3,
        'transformer_out_dim': 64,
        'mapped_dim': 32,
        'gat_hidden_dim': 64,
        'gat_heads': 4,
        'decoder_hidden_dims': [128, 64],
        'dropout': 0.1,
        'window_len': 288,
        'model_dir': args.model_dir,
        'ada_path': args.ada_path,
        'spot_q': args.spot_q,
        'spot_dynamic': True,
        'show_progress': True
    }

    # Load Templates
    templates = load_pattern_templates(args.template_dir)
    print(f"Loaded {len(templates)} failure templates.")

    # Find Switches
    switch_dirs = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
    
    # Run
    for switch in switch_dirs:
        try:
            logger = setup_logger(f"test_{switch}", "./logs")
            tester = SwitchTester(switch, config, logger)
            
            # Setup Pattern Matcher
            matcher = None
            if templates:
                matcher = PatternMatcher(tester.domain_adapter, config['input_feature_dim'], tester.device)
                matcher.build_template_library(templates)

            # Process
            results = tester.process_single_switch(os.path.join(args.data_root, switch), matcher)
            
            if results:
                tester.save_results(results, args.output_dir)
                logger.info("Results saved.")
            else:
                logger.warning("No results generated.")
                
        except Exception as e:
            print(f"Failed to test {switch}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    mp.freeze_support() # For Windows support
    main()