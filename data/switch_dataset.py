import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class SwitchDataset(Dataset):
    """
    Dataset class representing a single network switch.
    It aggregates data from all optical device (CSV files) located in a switch's
    directory to form a comprehensive graph snapshot for each time window.
    """

    def __init__(self, switch_path, vendor_id_map, window_len=288, step=12, feature_dim=14, max_rows=12 * 24 * 3):
        self.switch_path = switch_path
        self.window_len = window_len
        self.step = step
        self.feature_dim = feature_dim
        self.max_rows = max_rows
        self.samples = []

        csv_files = glob.glob(os.path.join(switch_path, "*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {switch_path}")

        logger.info(f"Processing switch {os.path.basename(switch_path)}: {len(csv_files)} files found.")

        csv_data = []
        min_timestamps = []

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file).sort_values("timestamp")
                if df.empty:
                    continue

                if len(df) > self.max_rows:
                    df = df.head(self.max_rows)

                local_features, remote_features = self._extract_features(df)
                if local_features is None:
                    continue

                csv_data.append({
                    'local_features': local_features,
                    'remote_features': remote_features,
                    'local_domain': str(df['local_vendor_cls'].iloc[0]),
                    'remote_domain': str(df['remote_vendor_cls'].iloc[0])
                })
                min_timestamps.append(len(local_features))

            except Exception as e:
                logger.warning(f"Error reading {csv_file}: {e}")
                continue

        if not csv_data:
            raise ValueError(f"No valid data for switch {switch_path}")

        min_length = min(min_timestamps)
        if min_length < window_len:
            raise ValueError(f"Data length {min_length} < Window length {window_len}")

        self._create_samples(csv_data, min_length, vendor_id_map)

    def _extract_features(self, df):
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
            return None, None

        return np.stack(local_feat, axis=1), np.stack(remote_feat, axis=1)

    def _create_samples(self, csv_data, data_length, vendor_id_map):
        num_devices = len(csv_data)

        for data in csv_data:
            for domain in [data['local_domain'], data['remote_domain']]:
                if domain not in vendor_id_map:
                    vendor_id_map[domain] = len(vendor_id_map)

        for start_idx in range(0, data_length - self.window_len + 1, self.step):
            end_idx = start_idx + self.window_len
            
            all_features, all_domains, all_types = [], [], []

            for data in csv_data:
                all_features.append(data['local_features'][start_idx:end_idx])
                all_domains.append(vendor_id_map[data['local_domain']])
                all_types.append(1) 

                all_features.append(data['remote_features'][start_idx:end_idx])
                all_domains.append(vendor_id_map[data['remote_domain']])
                all_types.append(0) 

            window_data = np.stack(all_features, axis=0)

            self.samples.append({
                'window_data': window_data,
                'domain_labels': np.array(all_domains),
                'node_types': np.array(all_types),
                'num_devices': num_devices
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample['window_data'], dtype=torch.float32),
            torch.tensor(sample['domain_labels'], dtype=torch.long),
            torch.tensor(sample['node_types'], dtype=torch.long)
        )