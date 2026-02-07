import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import multiprocessing

# === Configuration ===

LOCAL_FEATURES = [
    'local_voltage', 'local_current', 'local_currentTXPower', 'local_currentRXPower',
    'local_currentMultiBias1', 'local_currentMultiBias2', 'local_currentMultiBias3',
    'local_currentMultiBias4', 'local_currentMultiRXPower1', 'local_currentMultiRXPower2',
    'local_currentMultiRXPower3', 'local_currentMultiRXPower4', 'local_currentMultiTXPower1',
    'local_currentMultiTXPower2', 'local_currentMultiTXPower3', 'local_currentMultiTXPower4',
    'local_temperature'
]

REMOTE_FEATURES = [
    'remote_voltage', 'remote_current', 'remote_currentTXPower', 'remote_currentRXPower',
    'remote_currentMultiBias1', 'remote_currentMultiBias2', 'remote_currentMultiBias3',
    'remote_currentMultiBias4', 'remote_currentMultiRXPower1', 'remote_currentMultiRXPower2',
    'remote_currentMultiRXPower3', 'remote_currentMultiRXPower4', 'remote_currentMultiTXPower1',
    'remote_currentMultiTXPower2', 'remote_currentMultiTXPower3', 'remote_currentMultiTXPower4',
    'remote_temperature'
]

INVALID_VALUES = {-1, -255, -40}

# === Helper Functions ===

def get_all_csv_files(root_dir):
    """
    Recursively retrieve all CSV files within the specified directory structure.
    """
    return list(Path(root_dir).rglob("*.csv"))

def check_invalid_values(df, prefix):
    """
    Detect if specific columns contain known invalid sensor readings.
    Returns True if any invalid values are found in the critical columns.
    """
    check_cols = [
        f'{prefix}current', f'{prefix}temperature',
        f'{prefix}currentMultiTXPower1', f'{prefix}currentMultiTXPower2',
        f'{prefix}currentMultiTXPower3', f'{prefix}currentMultiTXPower4',
        f'{prefix}currentMultiRXPower1', f'{prefix}currentMultiRXPower2',
        f'{prefix}currentMultiRXPower3', f'{prefix}currentMultiRXPower4'
    ]
    existing_cols = [c for c in check_cols if c in df.columns]
    
    if not existing_cols:
        return False
        
    mask = df[existing_cols].isin(INVALID_VALUES).any(axis=1)
    return mask.any()

# === Phase 1: Statistics Calculation ===

def extract_valid_data_for_stats(file_path, timestamp_limit):
    """
    Reads a single CSV file, filters out invalid rows or files based on
    data integrity checks, and returns valid data for statistical computation.
    """
    try:
        df = pd.read_csv(file_path)
        
        if 'timestamp' in df.columns:
            pass 
        
        if 'local_vendorSn' not in df.columns or 'remote_vendorSn' not in df.columns:
            return None

        if df['local_vendorSn'].nunique() != 1 or df['remote_vendorSn'].nunique() != 1:
            return None

        if check_invalid_values(df, 'local_') or check_invalid_values(df, 'remote_'):
            return None

        valid_cols = [c for c in LOCAL_FEATURES + REMOTE_FEATURES if c in df.columns]
        return df[valid_cols]

    except Exception:
        return None

def compute_global_stats(data_dir, timestamp_limit, max_workers):
    """
    Computes global Mean and Standard Deviation for normalization by aggregating
    valid data from all files using parallel processing.
    """
    csv_files = get_all_csv_files(data_dir)
    valid_dfs = []
    
    print(f"[INFO] Phase 1: Computing global statistics from {len(csv_files)} files...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_valid_data_for_stats, f, timestamp_limit): f 
            for f in csv_files
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Aggregating Data"):
            res = future.result()
            if res is not None and not res.empty:
                valid_dfs.append(res)
    
    if not valid_dfs:
        raise ValueError("No valid data found to compute statistics.")

    full_df = pd.concat(valid_dfs, ignore_index=True)
    
    print("[INFO] Filtering outliers for robust statistics...")
    cleaned_df = full_df.copy()
    for col in full_df.columns:
        lower = full_df[col].quantile(0.05)
        upper = full_df[col].quantile(0.95)
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower) & (cleaned_df[col] <= upper)]

    stats = {}
    for col in full_df.columns:
        mean_val = cleaned_df[col].mean()
        std_val = cleaned_df[col].std()
        if std_val == 0 or np.isnan(std_val):
            std_val = 1e-6
        stats[col] = {'mean': float(mean_val), 'std': float(std_val)}
        
    return stats

# === Phase 2: Transformation & Saving ===

def transform_and_save(file_path, output_root, stats, clip_range=(-5, 5)):
    """
    Normalizes a single file using the computed global statistics, appends
    vendor classification columns, and saves the result maintaining the directory structure.
    """
    try:
        df = pd.read_csv(file_path)
        
        if 'local_vendorSn' not in df.columns or 'remote_vendorSn' not in df.columns:
            return 

        df['local_vendor_cls'] = 'local_' + df['local_vendorSn'].astype(str)
        df['remote_vendor_cls'] = 'remote_' + df['remote_vendorSn'].astype(str)

        for col, params in stats.items():
            if col in df.columns:
                mean = params['mean']
                std = params['std']
                df[col] = (df[col] - mean) / std
                df[col] = df[col].clip(clip_range[0], clip_range[1])

        file_path_obj = Path(file_path)
        switch_id = file_path_obj.parent.name 
        file_name = file_path_obj.name
        
        save_dir = Path(output_root) / switch_id
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / file_name
        df.to_csv(save_path, index=False)
        
        return True

    except Exception as e:
        return False

def apply_transformations(data_dir, output_dir, stats, max_workers):
    """
    Executes the normalization and saving process across all files in parallel.
    """
    csv_files = get_all_csv_files(data_dir)
    print(f"[INFO] Phase 2: Transforming and saving {len(csv_files)} files to {output_dir}...")
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(transform_and_save, f, output_dir, stats): f 
            for f in csv_files
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Normalizing"):
            if future.result():
                success_count += 1
                
    print(f"[SUCCESS] Processed {success_count}/{len(csv_files)} files.")

# === Main Entry Point ===

if __name__ == "__main__":
    RAW_DATA_DIR = "./mock_dataset" 
    PROCESSED_DATA_DIR = "./processed_dataset"
    TIMESTAMP_LIMIT = "2026-01-01 00:00:00"
    
    MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)

    if os.path.exists(PROCESSED_DATA_DIR):
        shutil.rmtree(PROCESSED_DATA_DIR)

    try:
        norm_stats = compute_global_stats(
            data_dir=RAW_DATA_DIR, 
            timestamp_limit=TIMESTAMP_LIMIT, 
            max_workers=MAX_WORKERS
        )
        print("[INFO] Statistics computation complete.")
        
        apply_transformations(
            data_dir=RAW_DATA_DIR, 
            output_dir=PROCESSED_DATA_DIR, 
            stats=norm_stats, 
            max_workers=MAX_WORKERS
        )
        
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")