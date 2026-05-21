import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model_selection import MODEL_DISPLAY_NAMES, get_model, select_model


def main():
    model_name = select_model()
    display_name = MODEL_DISPLAY_NAMES[model_name]

    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    data_path = os.path.join(script_dir, 'data', 'validation_dataset.csv')
    model_path = os.path.join(script_dir, 'saved_models', model_name, f'best_{model_name}.pth')
    stats_path = os.path.join(script_dir, 'saved_models', model_name, 'normalization_stats.json')
    logs_dir = os.path.join(script_dir, 'logs', model_name)
    os.makedirs(logs_dir, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f'inference_log.txt')

    open(log_path, 'w').close()  # Clear log file for a fresh run

    def log(msg):
        print(msg)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    if not os.path.exists(data_path) or not os.path.exists(model_path):
        log("Missing dataset or trained model!")
        log(f"  Data path: {data_path} (exists: {os.path.exists(data_path)})")
        log(f"  Model path: {model_path} (exists: {os.path.exists(model_path)})")
        return

    log("=" * 60)
    log(f"{display_name} MULTI-CLASS INFERENCE LOG")
    log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    # 1. Load Data
    log("\nLoading data...")
    df = pd.read_csv(data_path)

    segment_size = min(100000, len(df))
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
        train_std = stats['std']
        log(f"\nLoaded training normalization stats: per-window centering, std={train_std:.4f}")
    else:
        log("\n[WARNING] normalization_stats.json not found! Falling back to segment-level stats.")
        log("          This WILL cause a performance mismatch. Please retrain the model first.")
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
    log(f"Model: {display_name}")
    log(f"Window Size: {window_size}")
    log(f"Stride: {stride}")
    log(f"Smoothing Window: {smoothing_window}")
    log(f"Model path: {model_path}")

    model = get_model(model_name, num_classes=num_classes, in_channels=1).to(device)
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

        # Use per-window centering + global std normalization (same as training)
        window_mean = np.mean(window_data)
        window_scaled = (window_data - window_mean) / train_std
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

    # ============ EVENT-LEVEL EVALUATION ============
    # Group contiguous anomaly points into discrete events and evaluate per-event.
    # This avoids the misleading point-level metrics caused by sliding window boundary spillover.
    log(f"\n{'=' * 80}")
    log("EVENT-LEVEL EVALUATION")
    log(f"{'=' * 80}")
    log("(Each contiguous anomaly region counts as one event)")

    events = []
    idx = 0
    while idx < segment_size:
        label = int(is_anomaly[idx])
        if label > 0:
            evt_start = idx
            while idx < segment_size and int(is_anomaly[idx]) == label:
                idx += 1
            events.append((evt_start, idx, label))
        else:
            idx += 1

    log(f"Total anomaly events in segment: {len(events)}")

    # Classify each event using the model's averaged accumulated probabilities
    event_true = []
    event_pred = []
    event_conf = []
    event_details = []
    event_probs_all = []

    for evt_start, evt_end, true_class in events:
        event_probs = probabilities[evt_start:evt_end]
        valid_mask = ~np.isnan(event_probs[:, 0])

        if valid_mask.sum() == 0:
            continue

        avg_probs = np.mean(event_probs[valid_mask], axis=0)
        pred_class = int(np.argmax(avg_probs))
        confidence = float(avg_probs[pred_class])

        event_true.append(true_class)
        event_pred.append(pred_class)
        event_conf.append(confidence)
        event_probs_all.append(avg_probs)
        event_details.append({
            'start': evt_start, 'end': evt_end,
            'length': evt_end - evt_start,
            'true': true_class, 'pred': pred_class, 'conf': confidence
        })

    event_true = np.array(event_true)
    event_pred = np.array(event_pred)
    event_conf = np.array(event_conf)
    event_probs_all = np.array(event_probs_all)

    log(f"Events with valid predictions: {len(event_true)}")

    # Per-event results table
    log(f"\n{'#':<4} {'True Class':<20} {'Predicted':<20} {'Conf':>7} {'Pts':>7} {''}")
    log("-" * 66)

    for i, det in enumerate(event_details):
        true_name = class_names.get(det['true'], f"Class {det['true']}")
        pred_name = class_names.get(det['pred'], f"Class {det['pred']}")
        result = "OK" if det['true'] == det['pred'] else "MISS"
        log(f"{i+1:<4} {true_name:<20} {pred_name:<20} {det['conf']:>6.1%} {det['length']:>7} {result}")

    # Event-level confusion matrix
    all_evt_classes = sorted(set(event_true.tolist() + event_pred.tolist()))

    log(f"\n--- Event-Level Confusion Matrix (rows=true, cols=predicted) ---")
    header = f"{'':>16}"
    for c in all_evt_classes:
        name = class_names.get(c, f"Cls{c}")
        header += f" {name[:10]:>10}"
    log(header)
    log("-" * (16 + 11 * len(all_evt_classes)))

    for true_c in sorted(set(event_true)):
        true_name = class_names.get(true_c, f"Class {true_c}")
        row = f"{true_name[:16]:>16}"
        for pred_c in all_evt_classes:
            count = int(np.sum((event_true == true_c) & (event_pred == pred_c)))
            row += f" {count:>10}"
        log(row)

    # Summary
    correct = int(np.sum(event_true == event_pred))
    detected_any = int(np.sum(event_pred > 0))
    total_events = len(event_true)

    log(f"\n--- Event-Level Summary ---")
    log(f"Exact Classification: {correct}/{total_events} ({correct/total_events:.1%})")
    log(f"Detection Rate (any anomaly): {detected_any}/{total_events} ({detected_any/total_events:.1%})")
    log(f"Mean Confidence: {np.mean(event_conf):.1%}")

    # Per-class metrics at event level
    log(f"\n{'Class':<20} {'Support':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Spec':>8} {'FAR':>8}")
    log("-" * 73)

    e_precs, e_recs, e_f1s, e_specs, e_fars = [], [], [], [], []
    for c in sorted(set(event_true.tolist() + event_pred.tolist())):
        if c == 0:
            continue
        name = class_names.get(c, f"Class {c}")
        tp = int(np.sum((event_pred == c) & (event_true == c)))
        fp = int(np.sum((event_pred == c) & (event_true != c)))
        fn = int(np.sum((event_pred != c) & (event_true == c)))
        tn = int(np.sum((event_pred != c) & (event_true != c)))
        support = int(np.sum(event_true == c))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        far = 1.0 - spec

        e_precs.append(prec)
        e_recs.append(rec)
        e_f1s.append(f1)
        e_specs.append(spec)
        e_fars.append(far)

        log(f"{name:<20} {support:>8} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {spec:>8.4f} {far:>8.4f}")

    log("-" * 73)
    if e_precs:
        log(f"{'Macro Average':<20} {'':>8} {np.mean(e_precs):>8.4f} {np.mean(e_recs):>8.4f} {np.mean(e_f1s):>8.4f} {np.mean(e_specs):>8.4f} {np.mean(e_fars):>8.4f}")

    # Generate Confusion Matrix Plot
    plt.style.use('default')
    cm = confusion_matrix(event_true, event_pred, labels=all_evt_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[class_names.get(c, f"C{c}") for c in all_evt_classes],
                yticklabels=[class_names.get(c, f"C{c}") for c in all_evt_classes])
    plt.title(f'{display_name} - Event-Level Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    cm_path = os.path.join(logs_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    log(f"\nConfusion matrix saved to {cm_path}")

    # Generate ROC Curves
    try:
        plt.style.use('default')
        plt.figure(figsize=(10, 8))
        y_true_bin = label_binarize(event_true, classes=range(num_classes))
        colors = ['magenta', 'orange', 'yellow', 'lime', 'cyan', 'dodgerblue', 'white', 'pink']
        for c in range(1, num_classes):
            if np.sum(y_true_bin[:, c]) > 0:
                fpr, tpr, _ = roc_curve(y_true_bin[:, c], event_probs_all[:, c])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=colors[(c-1)%len(colors)], lw=2,
                         label=f'{class_names.get(c, f"Class {c}")} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='white', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{display_name} - ROC Curves')
        plt.legend(loc="lower right")
        roc_path = os.path.join(logs_dir, 'roc_curves.png')
        plt.savefig(roc_path, bbox_inches='tight')
        plt.close()
        log(f"ROC curves saved to {roc_path}")
    except Exception as e:
        log(f"Could not generate ROC curves: {e}")

    # 5. Visualization (Matplotlib)
    log(f"\nGenerating Inference Visualization plot...")
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(15, 6))

    # Background Gamma
    ax1.plot(time_steps, gamma_dose, color='teal', linewidth=1, alpha=0.7, label='Simulated Gamma Noise')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Gamma Dose Level')

    # Ground Truth Anomalies
    anomalous_idx = np.where(is_anomaly > 0)[0]
    ax1.scatter(time_steps[anomalous_idx], gamma_dose[anomalous_idx], color='red', s=5, label='True Anomalies', zorder=5)

    # CNN Probability Curves
    ax2 = ax1.twinx()
    colors = ['magenta', 'orange', 'green', 'lime', 'cyan', 'dodgerblue', 'purple', 'pink', 'gold', 'olive']
    for c in range(1, num_classes):
        df_probs = pd.DataFrame({'prob': probabilities[:, c]})
        df_probs['prob'] = df_probs['prob'].interpolate(method='linear', limit_direction='both')
        df_probs['prob'] = df_probs['prob'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
        smooth_probs = df_probs['prob'].values

        color = colors[(c - 1) % len(colors)]
        class_label = class_names.get(c, f"Class {c}")
        ax2.plot(time_steps, smooth_probs * 100, color=color, linewidth=2, label=f'{class_label} Confidence (%)')

    ax2.set_ylabel('CNN Confidence Probability (%)')
    ax2.set_ylim(0, 105)

    plt.title(f'{display_name} Multi-Class Inference: Streaming Anomaly Classification', fontsize=14)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.01, 0.99))

    # Limit x-axis to first 3 anomalies for the static PNG snapshot
    is_anom_bool = is_anomaly > 0
    diffs = np.diff(is_anom_bool.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    if len(is_anom_bool) > 0 and is_anom_bool[0]: starts = np.insert(starts, 0, 0)
    if len(is_anom_bool) > 0 and is_anom_bool[-1]: ends = np.append(ends, len(is_anom_bool))
    
    if len(starts) >= 3:
        padding = 2000
        xlim_start = time_steps[max(0, starts[0] - padding)]
        xlim_end = time_steps[min(len(time_steps)-1, ends[2] + padding)]
        ax1.set_xlim(xlim_start, xlim_end)
    elif len(starts) > 0:
        padding = 2000
        xlim_start = time_steps[max(0, starts[0] - padding)]
        xlim_end = time_steps[min(len(time_steps)-1, ends[-1] + padding)]
        ax1.set_xlim(xlim_start, xlim_end)

    output_png = os.path.join(logs_dir, 'cnn_inference_visualization.png')
    plt.savefig(output_png, bbox_inches='tight', dpi=200)
    plt.close()
    
    log(f"\nVisualization saved to {output_png}")

    # 6. Visualization (Plotly)
    log(f"\nGenerating Plotly graph...")

    fig_plotly = make_subplots(specs=[[{"secondary_y": True}]])

    fig_plotly.add_trace(
        go.Scatter(x=time_steps, y=gamma_dose, mode='lines', name='Simulated Gamma Noise',
                   line=dict(color='teal', width=1), opacity=0.7),
        secondary_y=False,
    )

    hover_texts = [f"True Anomaly: {class_names.get(is_anomaly[idx], is_anomaly[idx])}" for idx in anomalous_idx]

    fig_plotly.add_trace(
        go.Scatter(x=time_steps[anomalous_idx], y=gamma_dose[anomalous_idx], mode='markers',
                   name='True Anomalies', marker=dict(color='red', size=5), text=hover_texts, hoverinfo="x+y+text"),
        secondary_y=False,
    )

    for c in range(1, num_classes):
        df_probs = pd.DataFrame({'prob': probabilities[:, c]})
        df_probs['prob'] = df_probs['prob'].interpolate(method='linear', limit_direction='both')
        df_probs['prob'] = df_probs['prob'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
        smooth_probs = df_probs['prob'].values

        color = colors[(c - 1) % len(colors)]
        class_label = class_names.get(c, f"Class {c}")

        fig_plotly.add_trace(
            go.Scatter(x=time_steps, y=smooth_probs * 100, mode='lines', name=f'{class_label} Confidence (%)',
                       line=dict(color=color, width=3)),
            secondary_y=True,
        )

    fig_plotly.update_layout(
        title=f'{display_name} Multi-Class Inference: Streaming Anomaly Classification',
        xaxis_title='Time Steps',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    fig_plotly.update_yaxes(title_text="Gamma Dose Level", secondary_y=False)
    fig_plotly.update_yaxes(title_text="CNN Confidence Probability (%)", range=[0, 105], secondary_y=True, showgrid=False)

    output_html = os.path.join(logs_dir, 'cnn_inference_visualization.html')
    fig_plotly.write_html(output_html)

    log(f"Visualization saved to {output_html}")
    log(f"Log saved to {log_path}")
    log(f"\n{'=' * 60}")

if __name__ == "__main__":
    main()
