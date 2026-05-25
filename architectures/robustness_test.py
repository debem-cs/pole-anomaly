"""
Robustness Test for Anomaly Classifier
==============================================
Generates small test datasets with progressively increasing:
  1) Background noise intensity (std_noise multiplier)
  2) Anomaly shape deformation (variance parameter of generate_anomaly)

Each axis is swept independently while the other is held at its baseline value.
The trained model is evaluated on each dataset at event-level granularity.
Produces a summary log and an interactive Plotly chart.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import importlib.util

from model_selection import MODEL_DISPLAY_NAMES, get_model, select_model

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
root_dir = os.path.join(script_dir, '..')

# Import the data generation script directly
import sys
sys.path.append(os.path.join(root_dir, 'data-gen'))
import generate_custom_dataset as gcd


# ─────────────────────────────────────────────
# Dataset Generation
# ─────────────────────────────────────────────
def generate_test_dataset(noise_multiplier=1.0, variance_level=0.04,
                          num_anomalies=50, seed=42):
    """
    Generate a small synthetic dataset with controllable noise and
    deformation, by calling the shared generator in
    data-gen/generate_custom_dataset.py.

    The generator's `noise_multiplier` scales the AR(1) innovation std
    (and therefore the marginal HF std). Since anomaly amplitudes in
    the new generator are *absolute* counts (no longer multiples of
    sigma_HF), the SNR drops automatically when noise grows --- no
    need for the amplitude rescaling hack the v1 robustness script
    used.
    """
    # Keep the same anomaly density as the training dataset.
    mean_train_anomalies = (gcd.CONFIG["NUM_ANOMALIES_MIN"]
                             + gcd.CONFIG["NUM_ANOMALIES_MAX"]) // 2
    spacing = gcd.CONFIG["TOTAL_POINTS"] // mean_train_anomalies
    N = spacing * (num_anomalies + 1)

    return gcd.create_synthetic_dataset(
        total_points=N,
        num_anomalies=num_anomalies,
        noise_multiplier=noise_multiplier,
        variance_override=variance_level,
        return_df=True,
        seed=seed,
    )


# ─────────────────────────────────────────────
# Event-level Evaluation (reuses inference logic)
# ─────────────────────────────────────────────
from inference_engine import run_sliding_window_inference, evaluate_detection_events

def evaluate_on_dataset(df, model, device, train_std, num_classes, window_size=512, stride=10):
    """
    Run sliding-window inference and compute event-level accuracy.
    Returns (event_accuracy, detection_rate, mean_confidence, num_events, macro_f1)
    """
    gamma_dose = df['gamma_dose'].values
    is_anomaly = df['is_anomaly'].values
    
    probabilities, counts, num_windows, _ = run_sliding_window_inference(
        gamma_dose, model, device, train_std, num_classes, window_size, stride
    )
    
    results = evaluate_detection_events(
        probabilities, is_anomaly, num_classes, bg_threshold=0.3, tolerance=50
    )
    
    total = results['total_true_events']
    correct = results['tp']
    
    event_details = results['event_details']
    detected = 0
    confidences = []
    
    for det in event_details:
        # A true event was "detected" if it was a TP or a Misclassification (i.e. not entirely missed)
        if det['type'] in ['TP (Hit)', 'FN (Misclassified)']:
            detected += 1
            confidences.append(det['conf'])
            
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    event_acc = correct / total if total > 0 else 0.0
    det_rate = detected / total if total > 0 else 0.0
    
    return event_acc, det_rate, mean_conf, total, results['macro_f1']


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    model_name = select_model()
    display_name = MODEL_DISPLAY_NAMES[model_name]

    logs_dir = os.path.join(script_dir, 'logs', model_name)
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, 'robustness_test_log.txt')

    open(log_path, 'w').close()

    def log(msg):
        print(msg)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log("=" * 70)
    log(f"{display_name} ROBUSTNESS TEST")
    log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # ── Load background characteristics from generator CONFIG ──
    global_mean = gcd.CONFIG["BASELINE_MEAN"]
    ar1_phi = gcd.CONFIG["AR1_PHI"]
    ar1_sigma_eps = gcd.CONFIG["AR1_SIGMA_EPS"]
    # Marginal HF std implied by the AR(1) model.
    marginal_hf_std = ar1_sigma_eps / np.sqrt(1.0 - ar1_phi ** 2)

    log(f"\nAR(1) phi: {ar1_phi}")
    log(f"AR(1) innovation sigma: {ar1_sigma_eps:.4f}")
    log(f"Marginal HF std: {marginal_hf_std:.4f}")
    log(f"Global mean: {global_mean:.2f}")

    # ── Load templates ──
    anomalies_dir = os.path.join(root_dir, 'data-gen', 'anomalies')
    import glob
    template_files = glob.glob(os.path.join(anomalies_dir, '*.csv'))
    templates = {}
    for fp in template_files:
        name = os.path.basename(fp).replace('.csv', '')
        templates[name] = pd.read_csv(fp)

    class_map = {name: i + 1 for i, name in enumerate(sorted(templates.keys()))}

    # ── Load class mapping from training ──
    class_mapping_path = os.path.join(root_dir, 'data-gen', 'data', 'class_mapping.txt')
    class_names = {}
    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, 'r') as f:
            for line in f:
                if ':' in line:
                    c_id, name = line.split(':')
                    class_names[int(c_id.strip())] = name.strip()
        num_classes = len(class_names)
        # Rebuild class_map from training mapping
        class_map = {}
        for c_id, name in class_names.items():
            if c_id > 0:
                class_map[name] = c_id
    else:
        num_classes = len(templates) + 1

    log(f"Templates loaded: {list(templates.keys())}")
    log(f"Class mapping: {class_map}")
    log(f"Number of classes: {num_classes}")

    # ── Load model ──
    model_path = os.path.join(script_dir, 'saved_models', model_name, f'best_{model_name}.pth')
    stats_path = os.path.join(script_dir, 'saved_models', model_name, 'normalization_stats.json')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")

    model = get_model(model_name, num_classes=num_classes, in_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    with open(stats_path, 'r') as f:
        train_std = json.load(f)['std']
    log(f"Training std: {train_std:.4f}")

    # ── Configuration ──
    NUM_ANOMALIES = 50
    BASELINE_VARIANCE = 0.04  # default variance used in training data
    BASELINE_NOISE_MULT = 1.0

    # Sweep ranges (20 evenly spaced values each)
    noise_multipliers = np.linspace(1, 3, 10).tolist()
    deform_multipliers = np.linspace(1, 8, 10).tolist()

    # ── Sweep 1: Noise ──
    log(f"\n{'=' * 70}")
    log("SWEEP 1: BACKGROUND NOISE INTENSITY")
    log(f"(deformation held at baseline variance = {BASELINE_VARIANCE})")
    log(f"{'=' * 70}")
    log(f"\n{'Noise Mult.':<14} {'Events':<9} {'Macro F1':<12}")
    log("-" * 37)

    noise_results = []
    for mult in noise_multipliers:
        df_test = generate_test_dataset(
            noise_multiplier=mult, variance_level=BASELINE_VARIANCE,
            num_anomalies=NUM_ANOMALIES, seed=12345,
        )

        acc, det, conf, n_events, f1 = evaluate_on_dataset(
            df_test, model, device, train_std, num_classes
        )

        noise_results.append((mult, n_events, f1))
        log(f"{mult:<14.2f} {n_events:<9} {f1:<12.1%}")

    # ── Sweep 2: Deformation ──
    log(f"\n{'=' * 70}")
    log("SWEEP 2: ANOMALY SHAPE DEFORMATION")
    log(f"(noise held at baseline multiplier = {BASELINE_NOISE_MULT})")
    log(f"{'=' * 70}")
    log(f"\n{'Deform Mult.':<14} {'Events':<9} {'Macro F1':<12}")
    log("-" * 37)

    deform_results = []
    for mult in deform_multipliers:
        actual_variance = mult * BASELINE_VARIANCE
        df_test = generate_test_dataset(
            noise_multiplier=BASELINE_NOISE_MULT,
            variance_level=actual_variance,
            num_anomalies=NUM_ANOMALIES, seed=12345,
        )

        acc, det, conf, n_events, f1 = evaluate_on_dataset(
            df_test, model, device, train_std, num_classes
        )

        deform_results.append((mult, n_events, f1))
        log(f"{mult:<14.2f} {n_events:<9} {f1:<12.1%}")

    # ── Summary ──
    log(f"\n{'=' * 70}")
    log("SUMMARY")
    log(f"{'=' * 70}")

    # Find thresholds where F1 drops below a certain level (e.g. 50%)
    noise_threshold = None
    for mult, _, f1 in noise_results:
        if f1 < 0.5 and noise_threshold is None:
            noise_threshold = mult

    deform_threshold = None
    for mult, _, f1 in deform_results:
        if f1 < 0.5 and deform_threshold is None:
            deform_threshold = mult

    if noise_threshold:
        log(f"Noise: F1 Score first drops below 50% at multiplier = {noise_threshold:.1f}x")
    else:
        log("Noise: model maintained >50% F1 Score across all tested multipliers")

    if deform_threshold:
        log(f"Deformation: F1 Score first drops below 50% at multiplier = {deform_threshold:.1f}x")
    else:
        log("Deformation: model maintained >50% F1 Score across all tested variance levels")

    # ── Plot (Matplotlib) ──
    log("\nGenerating robustness chart...")
    plt.style.use('default')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    noise_x = [r[0] for r in noise_results]
    noise_y = [r[2] * 100 for r in noise_results]
    ax1.plot(noise_x, noise_y, marker='o', linestyle='-', color='red', linewidth=3, markersize=8, label='Macro F1 Score')
    ax1.set_title('Background Noise Intensity')
    ax1.set_xlabel(f'Noise Multiplier (×1 = AR(1) innov σ {ar1_sigma_eps:.2f} → marginal HF σ {marginal_hf_std:.2f})')
    ax1.set_ylabel('F1 Score (%)')
    ax1.set_ylim(0, 110)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend()
    
    deform_x = [r[0] for r in deform_results]
    deform_y = [r[2] * 100 for r in deform_results]
    ax2.plot(deform_x, deform_y, marker='o', linestyle='-', color='blue', linewidth=3, markersize=8, label='Macro F1 Score')
    ax2.set_title('Anomaly Shape Deformation')
    ax2.set_xlabel(f'Deformation Multiplier (×1 = {BASELINE_VARIANCE} variance)')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend()
    
    plt.suptitle(f'{display_name} Robustness: F1 Score vs. Noise & Deformation', fontsize=16)
    plt.tight_layout()
    
    output_png = os.path.join(logs_dir, 'robustness_test.png')
    plt.savefig(output_png, bbox_inches='tight', dpi=200)
    plt.close()
    
    log(f"PNG saved to: {output_png}")

    # ── Plot (Plotly HTML) ──
    fig_plotly = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Background Noise Intensity',
            'Anomaly Shape Deformation'
        ),
        horizontal_spacing=0.12
    )

    # ── Left panel: Noise sweep ──
    fig_plotly.add_trace(go.Scatter(
        x=[r[0] for r in noise_results],
        y=[r[2] * 100 for r in noise_results],
        mode='lines+markers',
        name='Macro F1 Score',
        line=dict(color='red', width=3),
        marker=dict(size=8),
        legendgroup='noise', legendgrouptitle_text='Noise Sweep',
        hovertemplate='Noise ×%{x:.1f}<br>F1 Score: %{y:.1f}%<extra></extra>'
    ), row=1, col=1)

    # ── Right panel: Deformation sweep ──
    fig_plotly.add_trace(go.Scatter(
        x=[r[0] for r in deform_results],
        y=[r[2] * 100 for r in deform_results],
        mode='lines+markers',
        name='Macro F1 Score',
        line=dict(color='blue', width=3),
        marker=dict(size=8),
        legendgroup='deform', legendgrouptitle_text='Deformation Sweep',
        hovertemplate='Deformation ×%{x:.1f}<br>F1 Score: %{y:.1f}%<extra></extra>'
    ), row=1, col=2)

    # ── Layout ──
    fig_plotly.update_layout(
        title=dict(text=f'{display_name} Robustness: F1 Score vs. Noise & Deformation', x=0.5),
        template='plotly_white',
        height=500, width=1100,
        legend=dict(
            orientation='v',
            bgcolor='rgba(255,255,255,0.4)',
            font=dict(size=11)
        )
    )

    fig_plotly.update_xaxes(title_text=f'Noise Multiplier (×1 = AR(1) innov σ {ar1_sigma_eps:.2f} → marginal HF σ {marginal_hf_std:.2f})', row=1, col=1)
    fig_plotly.update_xaxes(title_text=f'Deformation Multiplier (×1 = {BASELINE_VARIANCE} variance)', row=1, col=2)
    fig_plotly.update_yaxes(title_text='F1 Score (%)', range=[0, 110], row=1, col=1)
    fig_plotly.update_yaxes(range=[0, 110], row=1, col=2)

    output_html = os.path.join(logs_dir, 'robustness_test.html')
    fig_plotly.write_html(output_html)
    log(f"Chart saved to: {output_html}")

    log(f"\nLog saved to: {log_path}")
    log("=" * 70)


if __name__ == "__main__":
    main()
