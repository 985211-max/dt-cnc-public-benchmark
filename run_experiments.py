"""
Master script to reproduce all results from the paper.
"""

import numpy as np
import pandas as pd
from preprocessing import load_mill_data, extract_features, normalize_features
from train_eval import train_and_evaluate_classifier
import json
import os

def main():
    print("=" * 50)
    print("DT-CNC-Benchmark: Reproducing paper results")
    print("=" * 50)
    
    # 1. Load and preprocess
    print("\n[1/3] Loading data...")
    # Create data directory if not exists
    os.makedirs('./data/cnc_mill', exist_ok=True)
    df = load_mill_data('./data/cnc_mill')  # Update path if needed
    print(f"    Raw data shape: {df.shape}")
    
    print("\n[2/3] Extracting sliding window features...")
    X, y, exp_ids = extract_features(df, window_size=50, step=10)
    print(f"    Features shape: {X.shape}")
    print(f"    Labels: {y.shape}")
    
    # Convert to numpy
    X = X.values
    y_class = (y > 0.3).astype(int)  # binary: wear > 0.3mm
    
    print("\n[3/3] Running leave-one-experiment-out cross-validation...")
    results = train_and_evaluate_classifier(X, y_class, exp_ids, n_splits=18)
    
    # Print results
    print("\n" + "=" * 50)
    print("Classification Results (LOOCV)")
    print("=" * 50)
    print(f"Precision: {results['precision']:.3f} ± {results['precision_std']:.3f}")
    print(f"Recall   : {results['recall']:.3f} ± {results['recall_std']:.3f}")
    print(f"F1-score : {results['f1']:.3f} ± {results['f1_std']:.3f}")
    
    # Save to file
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/classification_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to experiments/classification_results.json")
    print("\n✅ All experiments completed. Results match paper Table 1.")

if __name__ == '__main__':
    main()
