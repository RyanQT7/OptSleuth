import os
import json
import argparse
import logging
import sys
import traceback
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.domain_adapter import UnifiedDomainAdapter
from models.gat import DeviceGAT
from data.switch_dataset import SwitchDataset
from utils.rcl import root_cause_localization
from utils.pattern_matcher import PatternMatcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SwitchTrainer:
    """
    Trainer for a single switch.
    Integrates:
    1. Frozen Pretrained ADA Model (Feature Extraction)
    2. Trainable GAT Model
    3. Learnable Parameter Alpha (Balances Root Cause Weight and Pattern Matching Weight)
    """
    def __init__(self, switch_name, config, pretrained_ada_path, device):
        self.switch_name = switch_name
        self.config = config
        self.device = device

        if not os.path.exists(pretrained_ada_path):
            raise FileNotFoundError(f"Pretrained model file does not exist: {pretrained_ada_path}")
            
        logger.info(f"[{switch_name}] Loading ADA model: {pretrained_ada_path}")
        checkpoint = torch.load(pretrained_ada_path, map_location=device)
        ada_config = checkpoint['model_config']
        
        self.domain_adapter = UnifiedDomainAdapter(
            input_feature_dim=ada_config['input_feature_dim'],
            transformer_d_model=config['transformer_d_model'],
            transformer_nhead=config['transformer_nhead'],
            transformer_num_layers=config['transformer_num_layers'],
            transformer_out_dim=config['transformer_out_dim'],
            num_domains=ada_config['num_domains'],
            mapped_dim=ada_config['mapped_dim'],
            reconstruction_dim=ada_config['input_feature_dim']
        ).to(device)
        
        self.domain_adapter.load_state_dict(checkpoint['ada_state_dict'])
        
        for param in self.domain_adapter.parameters():
            param.requires_grad = False
        self.domain_adapter.eval()

        self.gat_model = DeviceGAT(
            input_feature_dim=config['input_feature_dim'],
            mapped_dim=config['transformer_out_dim'],
            hidden_dim=config['gat_hidden_dim'],
            gat_heads=config['gat_heads'],
            dropout=config['dropout'],
            decoder_hidden_dims=config['decoder_hidden_dims']
        ).to(device)

        self.alpha = nn.Parameter(torch.tensor(0.0, device=device))

        trainable_parameters = list(self.gat_model.parameters()) + [self.alpha]
        self.optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config['lr_gat'],
            weight_decay=config['weight_decay']
        )

    def train_epoch(self, dataloader, pattern_matcher=None):
        """Train one Epoch"""
        self.gat_model.train()
        self.domain_adapter.eval()
        
        total_loss = 0
        total_mse = 0
        total_alpha = 0
        samples = 0

        for batch in dataloader:
            window_data, domain_labels, node_types = batch
            
            window_data = window_data.to(self.device).squeeze(0)
            node_types = node_types.to(self.device).squeeze(0)
            
            self.optimizer.zero_grad()
            samples += 1

            with torch.no_grad():
                node_features = self.domain_adapter.feature_extractor(window_data)

            pattern_scores = None
            if pattern_matcher is not None:
                with torch.no_grad():
                    pattern_scores, _ = pattern_matcher.match_patterns_from_features(node_features)
                    pattern_scores = pattern_scores.to(self.device)

            reconstructed, edge_index, edge_weights = self.gat_model(node_features, node_types)

            target = window_data[:, -1, :]
            
            mse_per_node = F.mse_loss(reconstructed, target, reduction='none').mean(dim=1)

            with torch.no_grad():
                temp_labels = torch.ones_like(mse_per_node, dtype=torch.long)
                
                _, score_pr = root_cause_localization(
                    mse_per_node.detach(),
                    temp_labels,
                    edge_index,
                    edge_weights,
                    device=self.device
                )
                w1 = score_pr.detach()

            alpha_val = torch.sigmoid(self.alpha)
            
            if pattern_scores is not None:
                w2 = pattern_scores.detach()
                w = alpha_val * w1 + (1 - alpha_val) * w2
            else:
                w = score_pr.detach()

            loss = (w * mse_per_node).mean()

            loss.backward()
            
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer.param_groups[0]['params'], 
                    self.config['grad_clip']
                )
            
            self.optimizer.step()

            total_loss += loss.item()
            total_mse += mse_per_node.mean().item()
            total_alpha += alpha_val.item()

        avg_loss = total_loss / samples if samples > 0 else 0
        avg_mse = total_mse / samples if samples > 0 else 0
        avg_alpha = total_alpha / samples if samples > 0 else 0
        
        return avg_loss, avg_mse, avg_alpha

    def save_models(self, save_dir):
        """Save model state and configuration"""
        switch_save_dir = os.path.join(save_dir, self.switch_name, "model")
        os.makedirs(switch_save_dir, exist_ok=True)
        
        save_dict = {
            'gat_state_dict': self.gat_model.state_dict(),
            'alpha': self.alpha.detach().cpu().item(),
            'alpha_sigmoid': torch.sigmoid(self.alpha).detach().cpu().item()
        }
        
        torch.save(save_dict, os.path.join(switch_save_dir, 'gat_model.pt'))
        
        with open(os.path.join(switch_save_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
            
        logger.info(f"Model saved to: {switch_save_dir} (Alpha: {save_dict['alpha_sigmoid']:.4f})")

def load_templates(template_path):
    """Load failure pattern templates"""
    pattern_templates = []
    
    if not os.path.exists(template_path):
        logger.warning(f"Template path does not exist: {template_path}")
        return []

    template_categories = {
        'template1_down.csv': {'category': 1, 'description': 'optical_connector'},
        'template1_up.csv': {'category': 1, 'description': 'optical_connector'},
        'template2_down.csv': {'category': 2, 'description': 'optical_connector'},
        'template2_up.csv': {'category': 2, 'description': 'optical_connector'},
    }
    
    for filename, info in template_categories.items():
        file_path = os.path.join(template_path, filename)
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                
                value_columns = []
                for i in range(1, 15):
                    col_name = f'value{i}'
                    if col_name in df.columns:
                        value_columns.append(df[col_name].values)
                    else:
                        value_columns.append(np.zeros(len(df)))
                
                data_values = np.column_stack(value_columns).astype(np.float32)
                
                pattern_templates.append({
                    'data': data_values,
                    'category': info['category'],
                    'description': info['description']
                })
            except Exception as e:
                logger.error(f"Failed to load template {filename}: {e}")
                
    if pattern_templates:
        logger.info(f"Successfully loaded {len(pattern_templates)} pattern templates")
    else:
        logger.warning("No pattern templates loaded")
        
    return pattern_templates

def main():
    parser = argparse.ArgumentParser(description="GAT Training for Optical Networks")
    parser.add_argument('--data_root', type=str, default='./pod17_normalized', help='Dataset root directory')
    parser.add_argument('--ada_path', type=str, required=True, help='Path to pretrained ADA model (.pt)')
    parser.add_argument('--template_dir', type=str, default='../failure_templates', help='Failure templates directory')
    parser.add_argument('--save_root', type=str, default='./model_results/GAT_models', help='Results save directory')
    
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")

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
        'batch_size': args.batch_size,
        'lr_gat': args.lr,
        'gat_epochs': args.epochs,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'save_dir': args.save_root
    }

    pattern_templates = load_templates(args.template_dir)
    
    if not os.path.exists(args.data_root):
        logger.error(f"Data directory does not exist: {args.data_root}")
        sys.exit(1)

    switch_dirs = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
    
    global_vendor_map = {}
    
    results_summary = {}

    print("=" * 60)
    print(f"Starting training for {len(switch_dirs)} switches")
    print("=" * 60)

    for switch_dir in tqdm(switch_dirs, desc="Total Progress"):
        switch_path = os.path.join(args.data_root, switch_dir)
        switch_name = switch_dir
        
        try:
            logger.info(f"Processing switch: {switch_name}")
            
            dataset = SwitchDataset(
                switch_path, 
                global_vendor_map, 
                window_len=config['window_len'], 
                feature_dim=config['input_feature_dim']
            )
            
            if len(dataset) == 0:
                logger.warning(f"Dataset for switch {switch_name} is empty, skipping")
                continue

            dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

            trainer = SwitchTrainer(switch_name, config, args.ada_path, device)
            
            matcher = None
            if pattern_templates:
                matcher = PatternMatcher(
                    trainer.domain_adapter, 
                    input_feature_dim=config['input_feature_dim'], 
                    device=device
                )
                matcher.build_template_library(pattern_templates)

            epoch_pbar = tqdm(range(args.epochs), desc=f"Training {switch_name}", leave=False)
            final_alpha = 0.5
            
            for epoch in epoch_pbar:
                loss, mse, alpha = trainer.train_epoch(dataloader, matcher)
                final_alpha = alpha
                epoch_pbar.set_postfix({'Loss': f"{loss:.4f}", 'MSE': f"{mse:.4f}", 'Alpha': f"{alpha:.4f}"})

            trainer.save_models(args.save_root)
            results_summary[switch_name] = {'status': 'success', 'final_alpha': final_alpha}

        except Exception as e:
            logger.error(f"Failed to train switch {switch_name}: {e}")
            traceback.print_exc()
            results_summary[switch_name] = {'status': 'failed', 'error': str(e)}

    os.makedirs(args.save_root, exist_ok=True)
    with open(os.path.join(args.save_root, 'global_info.json'), 'w') as f:
        json.dump({
            'vendor_map': global_vendor_map, 
            'config': config,
            'summary': results_summary
        }, f, indent=2)
        
    print("\nTraining completed!")

if __name__ == "__main__":
    main()