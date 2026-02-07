import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class OpticalDataset(Dataset):
    """
    Dataset class for training the Domain Adaptation model.
    It reads optical performance monitoring data from CSV files, extracts features,
    and generates sliding window samples for both local and remote nodes.
    """
    def __init__(self, dataset_root, vendor_id_map, window_len=288, step=12 * 3):
        self.samples = []
        self.window_len = window_len
        self.step = step
        self.feature_cols = [
            'voltage', 'currentMultiBias1', 'currentMultiBias2', 'currentMultiBias3', 'currentMultiBias4',
            'currentMultiRXPower1', 'currentMultiRXPower2', 'currentMultiRXPower3', 'currentMultiRXPower4',
            'currentMultiTXPower1', 'currentMultiTXPower2', 'currentMultiTXPower3', 'currentMultiTXPower4',
            'temperature'
        ]

        csv_files = []
        switch_dirs = [d for d in os.listdir(dataset_root)
                       if os.path.isdir(os.path.join(dataset_root, d))]
        for switch in switch_dirs:
            switch_path = os.path.join(dataset_root, switch)
            if os.path.exists(switch_path):
                csv_files.extend(glob.glob(os.path.join(switch_path, "*.csv")))

        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in {dataset_root}")

        print(f"Found {len(csv_files)} CSV files")

        for csv_file in tqdm(csv_files, desc="Processing CSV files"):
            try:
                df = pd.read_csv(csv_file).sort_values("timestamp")

                if len(df) == 0:
                    print(f"Warning: File {csv_file} is empty, skipping.")
                    continue

                required_cols = ['local_vendor_cls', 'remote_vendor_cls', 'timestamp']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"Warning: File {csv_file} is missing columns {missing_cols}, skipping.")
                    continue

                max_points = 12 * 24 * 3
                if len(df) > max_points:
                    df = df.iloc[:max_points]

                try:
                    local_domain = str(df['local_vendor_cls'].iloc[0])
                    remote_domain = str(df['remote_vendor_cls'].iloc[0])

                    if pd.isna(local_domain) or local_domain == 'nan' or local_domain == '':
                        print(f"Warning: File {csv_file} has invalid local_vendor_cls, skipping.")
                        continue
                    if pd.isna(remote_domain) or remote_domain == 'nan' or remote_domain == '':
                        print(f"Warning: File {csv_file} has invalid remote_vendor_cls, skipping.")
                        continue

                except Exception as e:
                    print(f"Warning: Failed to read vendor_cls from file {csv_file}: {e}, skipping.")
                    continue

                if local_domain not in vendor_id_map:
                    vendor_id_map[local_domain] = len(vendor_id_map)
                if remote_domain not in vendor_id_map:
                    vendor_id_map[remote_domain] = len(vendor_id_map)

                local_features = []
                remote_features = []

                for col in self.feature_cols:
                    local_col = 'local_' + col
                    remote_col = 'remote_' + col

                    if local_col in df.columns and remote_col in df.columns:
                        local_features.append(df[local_col].values.astype(np.float32))
                        remote_features.append(df[remote_col].values.astype(np.float32))

                if not local_features or not remote_features:
                    print(f"Warning: File {csv_file} is missing feature columns, skipping.")
                    continue

                local_features = np.stack(local_features, axis=1)
                remote_features = np.stack(remote_features, axis=1)

                num_time_points = len(local_features)

                if num_time_points < window_len:
                    print(f"Warning: File {csv_file} has fewer data points ({num_time_points}) than window length ({window_len}), skipping.")
                    continue

                for start_idx in range(0, num_time_points - window_len + 1, step):
                    end_idx = start_idx + window_len

                    local_seq = local_features[start_idx:end_idx]
                    self.samples.append({
                        'sequence': local_seq,
                        'domain_label': vendor_id_map[local_domain],
                        'node_type': 1
                    })

                    remote_seq = remote_features[start_idx:end_idx]
                    self.samples.append({
                        'sequence': remote_seq,
                        'domain_label': vendor_id_map[remote_domain],
                        'node_type': 0
                    })

            except Exception as e:
                print(f"Warning: Error processing file {csv_file}: {e}, skipping.")
                continue

        if len(self.samples) == 0:
            raise ValueError(f"No valid training samples created.")

        print(f"Successfully created {len(self.samples)} samples")
        print(f"Global vendor map: {vendor_id_map}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sequence = sample['sequence']
        domain_label = sample['domain_label']
        node_type = sample['node_type']

        return (
            torch.tensor(sequence, dtype=torch.float32),
            torch.tensor(domain_label, dtype=torch.long),
            torch.tensor(node_type, dtype=torch.long)
        )


class SwitchOpticalTestDataset(Dataset):
    """
    Dataset class for testing/inference on a specific switch directory.
    It preserves the original dataframe for result reconstruction and generates
    sequential sliding windows (step=1) for granular anomaly detection.
    """
    def __init__(self, switch_path, vendor_id_map, window_len=288):
        self.samples = []
        self.window_len = window_len
        self.feature_cols = [
            'voltage', 'currentMultiBias1', 'currentMultiBias2', 'currentMultiBias3', 'currentMultiBias4',
            'currentMultiRXPower1', 'currentMultiRXPower2', 'currentMultiRXPower3', 'currentMultiRXPower4',
            'currentMultiTXPower1', 'currentMultiTXPower2', 'currentMultiTXPower3', 'currentMultiTXPower4',
            'temperature'
        ]

        csv_files = sorted(glob.glob(os.path.join(switch_path, "*.csv")))
        if len(csv_files) == 0:
            raise ValueError(f"No CSV found in {switch_path}")

        all_data = []
        self.original_dfs = {}
        self.file_indices = {}

        for file_idx, csv_file in enumerate(csv_files):
            try:
                df = pd.read_csv(csv_file).sort_values("timestamp")

                if len(df) == 0:
                    logger.warning(f"File {csv_file} is empty, skipping.")
                    continue

                required_cols = ['local_vendor_cls', 'remote_vendor_cls', 'timestamp']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logger.warning(f"File {csv_file} is missing columns {missing_cols}, skipping.")
                    continue

                local_domain = str(df['local_vendor_cls'].iloc[0])
                remote_domain = str(df['remote_vendor_cls'].iloc[0])

                if pd.isna(local_domain) or local_domain == 'nan' or local_domain == '':
                    logger.warning(f"File {csv_file} has invalid local_vendor_cls, skipping.")
                    continue
                if pd.isna(remote_domain) or remote_domain == 'nan' or remote_domain == '':
                    logger.warning(f"File {csv_file} has invalid remote_vendor_cls, skipping.")
                    continue

                if local_domain not in vendor_id_map:
                    logger.warning(f"Unknown vendor {local_domain} found in test set, skipping file {csv_file}")
                    continue
                if remote_domain not in vendor_id_map:
                    logger.warning(f"Unknown vendor {remote_domain} found in test set, skipping file {csv_file}")
                    continue

                local_features = []
                remote_features = []

                for col in self.feature_cols:
                    local_col = 'local_' + col
                    remote_col = 'remote_' + col

                    if local_col in df.columns and remote_col in df.columns:
                        local_features.append(df[local_col].values.astype(np.float32))
                        remote_features.append(df[remote_col].values.astype(np.float32))

                if not local_features or not remote_features:
                    logger.warning(f"File {csv_file} is missing feature columns, skipping.")
                    continue

                local_features = np.stack(local_features, axis=1)
                remote_features = np.stack(remote_features, axis=1)

                data_dict = {
                    'local_seq': local_features,
                    'remote_seq': remote_features,
                    'local_domain': local_domain,
                    'remote_domain': remote_domain,
                    'file_path': csv_file,
                    'original_df': df.copy()
                }

                all_data.append(data_dict)
                self.original_dfs[file_idx] = df.copy()
                self.file_indices[file_idx] = csv_file

            except Exception as e:
                logger.error(f"Error processing file {csv_file}: {e}, skipping.", exc_info=True)
                continue

        if len(all_data) == 0:
            raise ValueError(f"No valid data found in switch directory {switch_path}")

        for file_idx, device_data in enumerate(all_data):
            local_seq = device_data['local_seq']
            remote_seq = device_data['remote_seq']
            num_time_points = len(local_seq)

            if num_time_points < window_len:
                logger.warning(
                    f"File {device_data['file_path']} has fewer data points ({num_time_points}) than window length ({window_len}), skipping.")
                continue

            for start_idx in range(0, num_time_points - window_len + 1, 1):
                end_idx = start_idx + window_len

                self.samples.append({
                    'local_seqs': local_seq[start_idx:end_idx][np.newaxis, :, :],
                    'remote_seqs': remote_seq[start_idx:end_idx][np.newaxis, :, :],
                    'local_domains': np.array([vendor_id_map[device_data['local_domain']]]),
                    'remote_domains': np.array([vendor_id_map[device_data['remote_domain']]]),
                    'file_idx': file_idx,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'num_devices': 1
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        M = sample['num_devices']
        all_seqs = np.concatenate([sample['local_seqs'], sample['remote_seqs']], axis=0)
        domain_labels = np.concatenate([sample['local_domains'], sample['remote_domains']])
        node_types = np.concatenate([
            np.ones(M, dtype=np.int64),
            np.zeros(M, dtype=np.int64)
        ])
        return (
            torch.tensor(all_seqs, dtype=torch.float32),
            torch.tensor(domain_labels, dtype=torch.long),
            torch.tensor(node_types, dtype=torch.long),
            M,
            sample['file_idx'],
            sample['start_idx'],
            sample['end_idx']
        )