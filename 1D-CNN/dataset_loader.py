import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(window_size=256, stride=32, test_size=0.2, random_state=42):
    """
    Loads the synthetic dataset and prepares sliding windows for 1D-CNN.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    data_path = os.path.join(script_dir, '..', 'data-gen', 'data', 'synthetic_custom_dataset.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_custom_dataset.py first.")
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    gamma_dose = df['gamma_dose'].values
    is_anomaly = df['is_anomaly'].values.astype(int)
    
    # Apply sliding window
    X_windows = []
    y_windows = []
    
    num_windows = (len(gamma_dose) - window_size) // stride + 1
    
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        
        window_data = gamma_dose[start:end]
        window_labels = is_anomaly[start:end]
        
        # The label of the window is the maximum class ID present
        window_label = np.max(window_labels)
        
        X_windows.append(window_data)
        y_windows.append(window_label)
        
    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)
    
    print(f"Generated {len(X_windows)} windows of size {window_size}.")
    print(f"Total anomaly windows: {np.sum(y_windows > 0)}")
    print(f"Total normal windows: {np.sum(y_windows == 0)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_windows, y_windows, test_size=test_size, random_state=random_state, stratify=y_windows
    )
    
    # Normalize: Fit scaler on training set only
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, window_size)
    X_test_flat = X_test.reshape(-1, window_size)
    
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # PyTorch Conv1d expects input shape: (batch_size, channels, sequence_length)
    X_train_final = X_train_scaled.reshape(-1, 1, window_size)
    X_test_final = X_test_scaled.reshape(-1, 1, window_size)
    
    print(f"X_train shape: {X_train_final.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    num_classes = len(np.unique(is_anomaly))
    print(f"Detected {num_classes} total classes.")
    
    return X_train_final, X_test_final, y_train, y_test, num_classes
