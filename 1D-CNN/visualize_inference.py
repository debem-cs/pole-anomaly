import os
import json
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Import the model architecture
from model import Anomaly1DCNN

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    data_path = os.path.join(script_dir, '..', 'data-gen', 'data', 'synthetic_custom_dataset.csv')
    model_path = os.path.join(script_dir, 'saved_models', 'best_1d_cnn_pytorch.pth')
    stats_path = os.path.join(script_dir, 'saved_models', 'normalization_stats.json')
    logs_dir = os.path.join(script_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f'inference_log.txt')
    
    def log(msg):
        print(msg)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        log("Missing dataset or trained model!")
        return

    log("=" * 60)
    log("1D-CNN MULTI-CLASS INFERENCE LOG")
    log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    # 1. Load Data
    log("\nLoading data...")
    df = pd.read_csv(data_path)
    
    segment_size = min(50000, len(df))
    df_segment = df.iloc[:segment_size]
    
    gamma_dose = df_segment['gamma_dose'].values
    is_anomaly = df_segment['is_anomaly'].values
    time_steps = df_segment['time_step'].values
    
    log(f"\n--- Dataset ---")
    log(f"Total dataset size: {len(df):,} points")
    log(f"Inference segment: {segment_size:,} points")
    log(f"True anomaly points in segment: {int(np.sum(is_anomaly > 0)):,}")
    
    # Read num_classes from class_mapping.txt (authoritative source)
    # This prevents mismatch when a regenerated dataset doesn't contain all anomaly types
    class_mapping_path = os.path.join(script_dir, '..', 'data-gen', 'data', 'class_mapping.txt')
    class_names = {}
    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, 'r') as f:
            for line in f:
                if ':' in line:
                    c_id, name = line.split(':')
                    class_names[int(c_id.strip())] = name.strip()
        num_classes = len(class_names)
    else:
        num_classes = len(np.unique(df['is_anomaly'].values))
        class_names = {i: f"Class {i}" for i in range(num_classes)}
    
    log(f"Number of classes: {num_classes}")
    log(f"Class mapping: {class_names}")
    
    # 2. Load normalization stats from training (CRITICAL: must match training normalization)
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        train_mean = stats['mean']
        train_std = stats['std']
        log(f"\nLoaded training normalization stats: mean={train_mean:.4f}, std={train_std:.4f}")
    else:
        log("\n[WARNING] normalization_stats.json not found! Falling back to segment-level stats.")
        log("          This WILL cause a performance mismatch. Please retrain the model first.")
        train_mean = float(np.mean(gamma_dose))
        train_std = float(np.std(gamma_dose))
    
    # 3. Setup Model
    window_size = 512
    stride = 10
    smoothing_window = 30
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"\n--- Environment ---")
    log(f"PyTorch Version: {torch.__version__}")
    log(f"Device: {device}")
    if device.type == 'cuda':
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    
    log(f"\n--- Inference Parameters ---")
    log(f"Window Size: {window_size}")
    log(f"Stride: {stride}")
    log(f"Smoothing Window: {smoothing_window}")
    log(f"Model: {model_path}")
    
    model = Anomaly1DCNN(in_channels=1, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    log(f"Model Parameters: {total_params:,}")
    
    # 4. Continuous Inference via Sliding Window with ACCUMULATION
    log(f"\nRunning inference...")
    inference_start = datetime.now()
    
    # Accumulate probabilities across all overlapping windows
    probabilities = np.zeros((segment_size, num_classes))
    counts = np.zeros(segment_size)
    
    num_windows = (segment_size - window_size) // stride + 1
    
    for start in range(0, segment_size - window_size + 1, stride):
        end = start + window_size
        window_data = gamma_dose[start:end]
        
        # Use the same normalization as training
        window_scaled = (window_data - train_mean) / train_std
        x_tensor = torch.tensor(window_scaled, dtype=torch.float32).view(1, 1, window_size).to(device)
        
        with torch.no_grad():
            output = model(x_tensor)
            prob = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        # Accumulate predictions across ALL points in the window (not just center)
        probabilities[start:end] += prob
        counts[start:end] += 1

    # Average the accumulated probabilities
    for i in range(len(counts)):
        if counts[i] > 0:
            probabilities[i] /= counts[i]
        else:
            probabilities[i] = np.nan
    
    inference_duration = datetime.now() - inference_start
    log(f"Inference completed in {inference_duration}")
    log(f"Total windows processed: {num_windows:,}")
    log(f"Windows per second: {num_windows / max(inference_duration.total_seconds(), 0.001):.0f}")
    
    # Compute detection statistics against ground truth
    log(f"\n--- Detection Summary ---")
    valid_indices = np.where(~np.isnan(probabilities[:, 0]))[0]
    
    if len(valid_indices) > 0:
        predicted_classes = np.argmax(probabilities[valid_indices], axis=1)
        max_confidences = np.max(probabilities[valid_indices], axis=1)
        true_classes = is_anomaly[valid_indices].astype(int)
        
        for c in range(num_classes):
            name = class_names.get(c, f"Class {c}")
            detected = np.sum(predicted_classes == c)
            if detected > 0:
                avg_conf = np.mean(max_confidences[predicted_classes == c])
                log(f"  {name}: {detected} windows detected (avg confidence: {avg_conf:.2%})")
            else:
                log(f"  {name}: 0 windows detected")
        
        # Per-class confusion matrix metrics
        log(f"\n{'=' * 80}")
        log("PER-CLASS CONFUSION MATRIX ANALYSIS")
        log(f"{'=' * 80}")
        log(f"Total evaluated windows: {len(valid_indices)}")
        
        overall_correct = np.sum(predicted_classes == true_classes)
        overall_acc = overall_correct / len(valid_indices)
        log(f"Overall Accuracy: {overall_acc:.4f} ({overall_correct}/{len(valid_indices)})")
        
        # Header
        log(f"\n{'Class':<22} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6}  {'Prec':>7} {'Recall':>7} {'F1':>7} {'Spec':>7}")
        log("-" * 80)
        
        # Macro averages accumulator
        precisions = []
        recalls = []
        f1s = []
        specificities = []
        
        for c in range(num_classes):
            name = class_names.get(c, f"Class {c}")
            
            # Binary: is this class or not?
            tp = int(np.sum((predicted_classes == c) & (true_classes == c)))
            tn = int(np.sum((predicted_classes != c) & (true_classes != c)))
            fp = int(np.sum((predicted_classes == c) & (true_classes != c)))
            fn = int(np.sum((predicted_classes != c) & (true_classes == c)))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            specificities.append(specificity)
            
            log(f"{name:<22} {tp:>6} {tn:>6} {fp:>6} {fn:>6}  {precision:>7.4f} {recall:>7.4f} {f1:>7.4f} {specificity:>7.4f}")
        
        log("-" * 80)
        
        macro_prec = np.mean(precisions)
        macro_rec = np.mean(recalls)
        macro_f1 = np.mean(f1s)
        macro_spec = np.mean(specificities)
        
        log(f"{'Macro Average':<22} {'':>6} {'':>6} {'':>6} {'':>6}  {macro_prec:>7.4f} {macro_rec:>7.4f} {macro_f1:>7.4f} {macro_spec:>7.4f}")
        
        # Weighted averages (weighted by true class support)
        supports = np.array([np.sum(true_classes == c) for c in range(num_classes)])
        total_support = supports.sum()
        if total_support > 0:
            w_prec = np.sum(np.array(precisions) * supports) / total_support
            w_rec = np.sum(np.array(recalls) * supports) / total_support
            w_f1 = np.sum(np.array(f1s) * supports) / total_support
            log(f"{'Weighted Average':<22} {'':>6} {'':>6} {'':>6} {'':>6}  {w_prec:>7.4f} {w_rec:>7.4f} {w_f1:>7.4f}")
        
        log(f"\nLegend: TP=True Positive, TN=True Negative, FP=False Positive, FN=False Negative")
        log(f"        Prec=Precision, Spec=Specificity")
    
    # 5. Visualization
    log(f"\nGenerating Plotly graph...")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Background Gamma
    fig.add_trace(
        go.Scatter(x=time_steps, y=gamma_dose, mode='lines', name='Simulated Gamma Noise',
                   line=dict(color='Teal', width=1), opacity=0.7),
        secondary_y=False,
    )
    
    # Ground Truth Anomalies
    anomalous_idx = np.where(is_anomaly > 0)[0]
    hover_texts = [f"True Anomaly: {class_names.get(is_anomaly[idx], is_anomaly[idx])}" for idx in anomalous_idx]
    
    fig.add_trace(
        go.Scatter(x=time_steps[anomalous_idx], y=gamma_dose[anomalous_idx], mode='markers',
                   name='True Anomalies', marker=dict(color='Red', size=5), text=hover_texts, hoverinfo="x+y+text"),
        secondary_y=False,
    )
    
    # CNN Probability Curves
    colors = ['magenta', 'orange', 'yellow', 'lime', 'cyan', 'dodgerblue', 'white', 'pink', 'gold', 'lightgreen']
    for c in range(1, num_classes):
        df_probs = pd.DataFrame({'prob': probabilities[:, c]})
        df_probs['prob'] = df_probs['prob'].interpolate(method='linear', limit_direction='both')
        # Apply a rolling average to smooth out the high-frequency sliding window fluctuations
        df_probs['prob'] = df_probs['prob'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
        smooth_probs = df_probs['prob'].values
        
        color = colors[(c - 1) % len(colors)]
        class_label = class_names.get(c, f"Class {c}")
        
        fig.add_trace(
            go.Scatter(x=time_steps, y=smooth_probs * 100, mode='lines', name=f'{class_label} Confidence (%)',
                       line=dict(color=color, width=3)),
            secondary_y=True,
        )
    
    fig.update_layout(
        title='1D-CNN Multi-Class Inference: Streaming Anomaly Classification',
        xaxis_title='Time Steps',
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    fig.update_yaxes(title_text="Gamma Dose Level", secondary_y=False)
    fig.update_yaxes(title_text="CNN Confidence Probability (%)", range=[0, 105], secondary_y=True, showgrid=False)
    
    output_html = os.path.join(logs_dir, 'cnn_inference_visualization.html')
    output_png = os.path.join(logs_dir, 'cnn_inference_visualization.png')
    
    fig.write_html(output_html)
    try:
        fig.write_image(output_png)
    except Exception as e:
        pass
        
    log(f"\nVisualization saved to {output_html}")
    log(f"Log saved to {log_path}")
    log(f"\n{'=' * 60}")

if __name__ == "__main__":
    main()
