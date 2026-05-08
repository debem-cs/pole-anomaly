import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import pandas as pd
import json
from model import Anomaly1DCNN

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
    
    # Normalize by centering EACH window to 0, but using global std to preserve relative amplitudes
    train_window_means = np.mean(X_train_windows, axis=1, keepdims=True)
    X_train_centered = X_train_windows - train_window_means
    
    test_window_means = np.mean(X_test_windows, axis=1, keepdims=True)
    X_test_centered = X_test_windows - test_window_means
    
    # Calculate global std on the centered training windows
    train_std = float(np.std(X_train_centered))
    
    # Save normalization stats for inference
    stats_path = os.path.join(script_dir, 'saved_models', 'normalization_stats.json')
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump({'std': train_std}, f)
    print(f"Normalization stats saved: per-window centering, global std={train_std:.4f}")
    
    X_train_scaled = X_train_centered / train_std
    X_test_scaled = X_test_centered / train_std
    
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
def main():
    window_size = 512
    batch_size = 64
    epochs = 150
    learning_rate = 0.001
    patience = 10
    
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    logs_dir = os.path.join(script_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f'training_log.txt')
    
    open(log_path, 'w').close()  # Clear log file for a fresh run
    
    def log(msg):
        print(msg)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    
    log("=" * 60)
    log("1D-CNN MULTI-CLASS TRAINING LOG")
    log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)
    
    # Hyperparameters
    log("\n--- Hyperparameters ---")
    log(f"Window Size: {window_size}")
    log(f"Batch Size: {batch_size}")
    log(f"Max Epochs: {epochs}")
    log(f"Learning Rate: {learning_rate}")
    log(f"Early Stopping Patience: {patience}")
    log(f"Sliding Window Stride: 32")
    log(f"LR Scheduler: ReduceLROnPlateau (patience=4, factor=0.5)")
    log(f"Data Augmentation: None (randomness handled by data generator)")
    log(f"Train/Test Split: Temporal (no data leakage)")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"\n--- Environment ---")
    log(f"PyTorch Version: {torch.__version__}")
    log(f"Device: {device}")
    if device.type == 'cuda':
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"CUDA Version: {torch.version.cuda}")
    
    log("\nLoading and preprocessing data...")
    X_train, X_test, y_train, y_test, num_classes = load_data(window_size=window_size, stride=32)
    
    log(f"\n--- Dataset ---")
    log(f"Training samples: {len(X_train)}")
    log(f"Test samples: {len(X_test)}")
    log(f"Number of classes: {num_classes}")
    log(f"Input shape: {X_train.shape}")
    
    # Class distribution
    class_counts = np.bincount(y_train.astype(int), minlength=num_classes)
    log(f"\n--- Class Distribution (Training Set) ---")
    
    class_mapping_path = os.path.join(script_dir, '..', 'data-gen', 'data', 'class_mapping.txt')
    class_names = {}
    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, 'r') as f:
            for line in f:
                if ':' in line:
                    c_id, name = line.split(':')
                    class_names[int(c_id.strip())] = name.strip()
    
    for i, count in enumerate(class_counts):
        name = class_names.get(i, f"Class {i}")
        pct = 100 * count / len(y_train)
        log(f"  {name} (ID {i}): {count} samples ({pct:.1f}%)")
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    # Split training into train + validation (80/20 of the training portion)
    val_size = int(0.2 * len(X_train_t))
    train_size = len(X_train_t) - val_size
    
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(X_train_t), generator=generator)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = TensorDataset(X_train_t[train_indices], y_train_t[train_indices])
    val_dataset = TensorDataset(X_train_t[val_indices], y_train_t[val_indices])
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    log(f"\n--- Split ---")
    log(f"Train: {train_size} | Validation: {val_size} | Test: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = Anomaly1DCNN(in_channels=1, num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"\n--- Model Architecture ---")
    log(f"Total Parameters: {total_params:,}")
    log(f"Trainable Parameters: {trainable_params:,}")
    
    # Handle class imbalance using class weights
    total = len(y_train)
    class_weights = total / (num_classes * np.maximum(class_counts, 1))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    log(f"\n--- Class Weights ---")
    log(f"Weights: {np.round(class_weights, 2)}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)
    
    save_dir = os.path.join(script_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'best_1d_cnn_pytorch.pth')
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    log(f"\n{'=' * 60}")
    log("TRAINING")
    log(f"{'=' * 60}")
    log(f"{'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} {'LR':<12} {'Status'}")
    log("-" * 62)
    
    train_start = datetime.now()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        # Step the LR scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
            status = "* saved"
        else:
            patience_counter += 1
            status = f"patience {patience_counter}/{patience}"
            if patience_counter >= patience:
                status = "EARLY STOP"
        
        log(f"{epoch+1:02d}/{epochs:<5} {train_loss:<14.4f} {val_loss:<14.4f} {current_lr:<12.6f} {status}")
        
        if patience_counter >= patience:
            break
    
    train_duration = datetime.now() - train_start
    log(f"\nTraining Duration: {train_duration}")
    log(f"Best Validation Loss: {best_val_loss:.4f}")
                
    log(f"\n{'=' * 60}")
    log("TEST EVALUATION")
    log(f"{'=' * 60}")
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            total += y_batch.size(0)
            correct += (preds == y_batch).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_acc = correct / total
    
    log(f"Test Loss: {test_loss:.4f}")
    log(f"Test Accuracy: {test_acc:.4f} ({correct}/{total})")
    
    # Per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    log(f"\n--- Per-Class Accuracy ---")
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == c).sum() / mask.sum()
            name = class_names.get(c, f"Class {c}")
            log(f"  {name} (ID {c}): {class_acc:.4f} ({(all_preds[mask] == c).sum()}/{mask.sum()})")
    
    log(f"\nModel saved to: {model_path}")
    log(f"Log saved to: {log_path}")
    log(f"\n{'=' * 60}")

if __name__ == "__main__":
    main()
