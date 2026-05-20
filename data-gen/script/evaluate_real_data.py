"""
Real Data Evaluator
===================
This script reads the real-world dataset (2015_months_DebitDoseA.txt)
and evaluates its statistical properties. The goal is to extract the 
background noise and baseline drift parameters so they can be accurately 
injected into the synthetic dataset.

Output is saved to data-gen/data/real_data_params.json
"""

import os
import json
import numpy as np
import pandas as pd

def evaluate_real_data():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    root_dir = os.path.join(script_dir, '..')
    data_path = os.path.join(root_dir, 'data', '2015_months_DebitDoseA.txt')
    output_path = os.path.join(root_dir, 'data', 'real_data_params.json')
    
    print("=" * 60)
    print("REAL DATA EVALUATION")
    print("=" * 60)
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}")
        return
        
    df = pd.read_csv(data_path)
    columns = df.columns.tolist()
    print(f"Found {len(columns)} time series columns.")
    
    # Parameters to accumulate
    global_means = []
    hf_noise_stds = []
    baseline_ranges = []
    baseline_stds = []
    
    window_size = 480
    boundary_trim = 400
    
    for col_name in columns:
        series = pd.to_numeric(df[col_name], errors='coerce').interpolate(method='linear', limit_direction='both').values.astype(np.float64)
        
        # Trim boundaries that might be corrupted by interpolation or startup
        series = series[boundary_trim:-boundary_trim]
        
        global_means.append(np.mean(series))
        
        # Isolate baseline using a moving average
        kernel = np.ones(window_size) / window_size
        baseline = np.convolve(series, kernel, mode='valid')
        
        baseline_ranges.append(np.max(baseline) - np.min(baseline))
        baseline_stds.append(np.std(baseline))
        
        # Calculate high-frequency noise from residuals
        hf_residual = series[window_size//2 : -window_size//2 + 1] - baseline
        hf_noise_stds.append(np.std(hf_residual))
        
    avg_mean = np.mean(global_means)
    avg_hf_std = np.mean(hf_noise_stds)
    avg_bl_range = np.mean(baseline_ranges)
    avg_bl_std = np.mean(baseline_stds)
    
    print(f"\n--- Averages across all {len(columns)} sequences ---")
    print(f"Global Mean: {avg_mean:.2f}")
    print(f"High-Freq Noise Std: {avg_hf_std:.2f}")
    print(f"Baseline Range: {avg_bl_range:.2f}")
    print(f"Baseline Std: {avg_bl_std:.2f}")
    
    # ---------------------------------------------------------
    # ORNSTEIN-UHLENBECK (OU) PARAMETER CALIBRATION
    # ---------------------------------------------------------
    # The baseline drift in the synthetic generator uses an OU process.
    # We fix theta to a very low value (empirical loose tether).
    theta = 0.000004
    
    # Mathematical correction for sigma:
    # The equilibrium standard deviation of an OU process is: std = sigma / sqrt(2*theta)
    # We want the simulated process to have an std matching our observed avg_bl_std.
    # But because our sequences are finite (40,000 points), the process doesn't reach 
    # full equilibrium. 
    # From parameter sweeping, we found that a sigma of ~0.06 accurately reproduces 
    # the observed baseline std of ~2.5 to 3.5 over a 40,000 point window.
    
    params = {}
    
    print(f"\n--- Saving Per-Column Parameters to JSON ---")
    for i, col_name in enumerate(columns):
        # Calculate sigma correction for this specific column
        # Formula: target_sigma = baseline_std * (0.05 / 3.67) 
        col_sigma = baseline_stds[i] * (0.05 / 3.67)
        col_sigma = max(0.01, min(col_sigma, 0.15))
        
        params[col_name] = {
            "GLOBAL_MEAN": round(float(global_means[i]), 2),
            "STD_NOISE": round(float(hf_noise_stds[i]), 2),
            "THETA": float(theta),
            "SIGMA": round(float(col_sigma), 4)
        }
    
    # Also include the overall average as a convenience reference
    target_sigma_avg = avg_bl_std * (0.05 / 3.67)
    target_sigma_avg = max(0.01, min(target_sigma_avg, 0.15))
    
    params["AVERAGE_ACROSS_ALL"] = {
        "GLOBAL_MEAN": round(float(avg_mean), 2),
        "STD_NOISE": round(float(avg_hf_std), 2),
        "THETA": float(theta),
        "SIGMA": round(float(target_sigma_avg), 4)
    }
    
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=4)
        
    print(f"\nParameters saved to {output_path}")

if __name__ == "__main__":
    evaluate_real_data()
