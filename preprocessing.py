```python
"""
Data preprocessing and feature engineering for CNC Mill Tool Wear Dataset.
Author: 985211-MAX / DT-CNC-Benchmark
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_mill_data(path='./data/cnc_mill'):
    """
    Load raw CSV files from the CNC Mill dataset.
    Expected structure: each experiment is a separate CSV file.
    """
    import os
    from glob import glob
    
    all_dfs = []
    file_list = glob(os.path.join(path, '*.csv'))
    for f in file_list:
        df = pd.read_csv(f)
        df['experiment_id'] = os.path.basename(f).split('.')[0]
        all_dfs.append(df)
    
    if not all_dfs:
        raise FileNotFoundError(f"No CSV files found in {path}. Please download the dataset from Kaggle.")
    
    data = pd.concat(all_dfs, ignore_index=True)
    return data

def extract_features(df, window_size=50, step=10):
    """
    Sliding window feature extraction.
    For each window, compute statistical features from raw signals.
    """
    from scipy.stats import kurtosis, skew
    
    feature_list = []
    label_list = []
    exp_ids = []
    
    for exp_id, group in df.groupby('experiment_id'):
        group = group.sort_values('time').reset_index(drop=True)
        vib_cols = [c for c in group.columns if 'vib' in c.lower() or 'vibration' in c.lower()]
        motor_cols = [c for c in group.columns if 'load' in c.lower() or 'current' in c.lower()]
        wear_col = 'wear' if 'wear' in group.columns else group.columns[-1]  # heuristic
        
        for start in range(0, len(group) - window_size, step):
            window = group.iloc[start:start+window_size]
            
            # Vibration features
            vib_feat = {}
            for col in vib_cols:
                sig = window[col].values
                vib_feat[f'{col}_rms'] = np.sqrt(np.mean(sig**2))
                vib_feat[f'{col}_kurtosis'] = kurtosis(sig)
                vib_feat[f'{col}_peak'] = np.max(np.abs(sig))
                vib_feat[f'{col}_crest'] = vib_feat[f'{col}_peak'] / (vib_feat[f'{col}_rms'] + 1e-8)
            
            # Motor features
            motor_feat = {}
            for col in motor_cols:
                sig = window[col].values
                motor_feat[f'{col}_mean'] = np.mean(sig)
                motor_feat[f'{col}_std'] = np.std(sig)
                motor_feat[f'{col}_max'] = np.max(sig)
            
            # Wear label (value at window end)
            wear_value = window[wear_col].iloc[-1]
            label_list.append(wear_value)
            
            # Combine all features
            feat = {**vib_feat, **motor_feat}
            feature_list.append(feat)
            exp_ids.append(exp_id)
    
    X = pd.DataFrame(feature_list)
    y = np.array(label_list)
    return X, y, exp_ids

def normalize_features(X_train, X_test):
    """Z-score normalization using training statistics."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

if __name__ == '__main__':
    # Example usage
    print("Loading data...")
    df = load_mill_data()  # adjust path as needed
    print(f"Data loaded, shape: {df.shape}")
    
    print("Extracting features...")
    X, y, exp_ids = extract_features(df)
    print(f"Extracted {X.shape[0]} windows, {X.shape[1]} features.")
    
    # Save intermediate features for faster reloading
    X.to_csv('mill_features.csv', index=False)
    np.save('mill_labels.npy', y)
    pd.Series(exp_ids).to_csv('mill_exp_ids.csv', index=False)
    print("Features saved.")
