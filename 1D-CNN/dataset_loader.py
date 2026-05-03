import numpy as np
import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(window_size=256, stride=32, test_size=0.2, random_state=42):
    """
    Loads the synthetic dataset and prepares sliding windows for 1D-CNN.
    Uses a temporal split (first 80% train, last 20% test) to prevent
    data leakage from overlapping windows.
    Saves normalization stats for consistent inference.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    data_path = os.path.join(script_dir, '..', '1D-CNN', 'data', 'training_dataset.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_custom_dataset.py first.")
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    gamma_dose = df['gamma_dose'].values
    is_anomaly = df['is_anomaly'].values.astype(int)
    
    # --- Temporal split BEFORE windowing to prevent data leakage ---
    split_point = int(len(gamma_dose) * (1 - test_size))
    
    train_signal = gamma_dose[:split_point]
    train_labels = is_anomaly[:split_point]
    test_signal = gamma_dose[split_point:]
    test_labels = is_anomaly[split_point:]
    
    print(f"Temporal split: Train signal = {len(train_signal)} points, Test signal = {len(test_signal)} points")
    
    # Apply sliding window to each split independently
    X_train_windows, y_train = _create_windows(train_signal, train_labels, window_size, stride)
    X_test_windows, y_test = _create_windows(test_signal, test_labels, window_size, stride)
    
    print(f"Generated {len(X_train_windows)} train windows and {len(X_test_windows)} test windows of size {window_size}.")
    print(f"Train anomaly windows: {np.sum(y_train > 0)} | Train normal windows: {np.sum(y_train == 0)}")
    print(f"Test anomaly windows: {np.sum(y_test > 0)} | Test normal windows: {np.sum(y_test == 0)}")
    
    # Normalize using global mean and std from the training set to preserve time-series shapes
    train_mean = float(np.mean(X_train_windows))
    train_std = float(np.std(X_train_windows))
    
    # Save normalization stats for inference
    stats_path = os.path.join(script_dir, 'saved_models', 'normalization_stats.json')
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump({'mean': train_mean, 'std': train_std}, f)
    print(f"Normalization stats saved: mean={train_mean:.4f}, std={train_std:.4f}")
    
    X_train_scaled = (X_train_windows - train_mean) / train_std
    X_test_scaled = (X_test_windows - train_mean) / train_std
    
    # Reshape to (batch, channels, sequence) for PyTorch 1D-CNN
    X_train_final = X_train_scaled.reshape(-1, 1, window_size)
    X_test_final = X_test_scaled.reshape(-1, 1, window_size)
    
    print(f"X_train shape: {X_train_final.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    num_classes = len(np.unique(is_anomaly))
    print(f"Detected {num_classes} total classes.")
    
    return X_train_final, X_test_final, y_train, y_test, num_classes


def _create_windows(signal, labels, window_size, stride, anomaly_threshold=0.10):
    """
    Create sliding windows from a signal segment.
    A window is labeled as anomalous if anomaly points exceed anomaly_threshold
    fraction of the window.
    """
    X_windows = []
    y_windows = []
    
    num_windows = (len(signal) - window_size) // stride + 1
    
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        
        window_data = signal[start:end]
        window_labels = labels[start:end]
        
        # Label as anomaly if it occupies at least anomaly_threshold of the window
        anomaly_points = np.sum(window_labels > 0)
        if anomaly_points > (window_size * anomaly_threshold):
            # Get the most frequent anomaly class in this window
            non_zero_labels = window_labels[window_labels > 0]
            values, counts = np.unique(non_zero_labels, return_counts=True)
            window_label = values[np.argmax(counts)]
        else:
            window_label = 0
        
        X_windows.append(window_data)
        y_windows.append(window_label)
        
    return np.array(X_windows), np.array(y_windows)
