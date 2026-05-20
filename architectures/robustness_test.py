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
import plotly.graph_objects as go
from datetime import datetime
import importlib.util

from model_selection import MODEL_DISPLAY_NAMES, get_model, select_model

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
root_dir = os.path.join(script_dir, '..')

# Load anomaly_generator directly from its file path
_gen_path = os.path.join(root_dir, 'data-gen', 'src', 'anomaly_generator.py')
_spec = importlib.util.spec_from_file_location("anomaly_generator", _gen_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
generate_anomaly = _mod.generate_anomaly


# ─────────────────────────────────────────────
# Dataset Generation
# ─────────────────────────────────────────────
def generate_test_dataset(templates, class_map, global_mean, base_std_noise, theta, sigma,
                          noise_multiplier=1.0, variance_level=0.04, num_anomalies=50,
                          seed=42):
    """
    Generate a small synthetic dataset with controllable noise and deformation.
    Returns a DataFrame with columns: time_step, gamma_dose, is_anomaly
    """
    rng = np.random.RandomState(seed)
    template_names = list(templates.keys())

    # Compute signal length: ~1200 spacing between anomalies
    spacing = 1200
    N = spacing * (num_anomalies + 2)

    # Pre-generate anomaly shapes
    inject_points = [spacing * (i + 1) + rng.randint(-100, 100) for i in range(num_anomalies)]

    anomaly_injections = []
    for idx in inject_points:
        chosen_name = rng.choice(template_names)
        df_template = templates[chosen_name]

        target_amplitude = base_std_noise * rng.uniform(3.0, 7.6)
        target_period = rng.randint(80, 400)

        t_discrete, v_discrete = generate_anomaly(
            df_template,
            amplitude=target_amplitude,
            period=target_period,
            variance=variance_level
        )

        anomaly_injections.append({
            'start': idx,
            'values': v_discrete,
            'template_name': chosen_name,
            'class_id': class_map[chosen_name]
        })

    # Generate baseline (OU process)
    baseline = np.zeros(N)
    noise_steps = rng.normal(scale=sigma, size=N)
    val = global_mean
    for i in range(N):
        val = val + theta * (global_mean - val) + noise_steps[i]
        baseline[i] = val

    # Apply scaled high-frequency noise
    effective_std = base_std_noise * noise_multiplier
    background = rng.normal(loc=baseline, scale=effective_std)
    background = np.clip(background, 0, None)

    labels = np.zeros(N, dtype=int)

    # Inject anomalies
    for anom in anomaly_injections:
        start = anom['start']
        for j, v in enumerate(anom['values']):
            pos = start + j
            if 0 <= pos < N:
                background[pos] += v
                if v > 1e-2:
                    labels[pos] = anom['class_id']

    df = pd.DataFrame({
        'time_step': np.arange(N),
        'gamma_dose': background,
        'is_anomaly': labels
    })

    return df


# ─────────────────────────────────────────────
# Event-level Evaluation (reuses inference logic)
# ─────────────────────────────────────────────
def evaluate_on_dataset(df, model, device, train_std, num_classes, window_size=512, stride=10):
    """
    Run sliding-window inference and compute event-level accuracy.
    Returns (event_accuracy, detection_rate, mean_confidence, num_events)
    """
    gamma_dose = df['gamma_dose'].values
    is_anomaly = df['is_anomaly'].values
    segment_size = len(df)

    # Accumulate probabilities
    probabilities = np.zeros((segment_size, num_classes))
    counts = np.zeros(segment_size)

    for start in range(0, segment_size - window_size + 1, stride):
        end = start + window_size
        window_data = gamma_dose[start:end]
        window_mean = np.mean(window_data)
        window_scaled = (window_data - window_mean) / train_std

        x_tensor = torch.tensor(window_scaled, dtype=torch.float32).view(1, 1, window_size).to(device)

        with torch.no_grad():
            output = model(x_tensor)
            prob = torch.softmax(output, dim=1).cpu().numpy()[0]

        probabilities[start:end] += prob
        counts[start:end] += 1

    # Average
    for i in range(len(counts)):
        if counts[i] > 0:
            probabilities[i] /= counts[i]

    # Extract events
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

    if len(events) == 0:
        return 0.0, 0.0, 0.0, 0

    # Classify each event
    correct = 0
    detected = 0
    confidences = []

    event_true = []
    event_pred = []

    for evt_start, evt_end, true_class in events:
        event_probs = probabilities[evt_start:evt_end]
        valid_mask = counts[evt_start:evt_end] > 0

        if valid_mask.sum() == 0:
            continue

        avg_probs = np.mean(event_probs[valid_mask], axis=0)
        pred_class = int(np.argmax(avg_probs))
        confidence = float(avg_probs[pred_class])

        event_true.append(true_class)
        event_pred.append(pred_class)

        if pred_class == true_class:
            correct += 1
        if pred_class > 0:
            detected += 1
        confidences.append(confidence)

    total = len(events)
    event_acc = correct / total if total > 0 else 0.0
    det_rate = detected / total if total > 0 else 0.0
    mean_conf = np.mean(confidences) if confidences else 0.0

    # Calculate Event-Level Macro F1
    event_true = np.array(event_true)
    event_pred = np.array(event_pred)

    e_f1s = []
    # Only calculate F1 over classes that actually appeared in the true labels
    for c in sorted(set(event_true.tolist())):
        if c == 0:
            continue
        tp = int(np.sum((event_pred == c) & (event_true == c)))
        fp = int(np.sum((event_pred == c) & (event_true != c)))
        fn = int(np.sum((event_pred != c) & (event_true == c)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        e_f1s.append(f1)

    macro_f1 = np.mean(e_f1s) if e_f1s else 0.0

    return event_acc, det_rate, mean_conf, total, macro_f1


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

    # ── Load real background characteristics ──
    filename = os.path.join(root_dir, 'data-gen', 'data', '2015_months_DebitDoseA.txt')
    try:
        data_gamma = np.genfromtxt(filename, delimiter=',', skip_header=1)
        mois = 3
        real_background = data_gamma[:, mois]
        Nh = 400
        real_background = real_background[Nh:-Nh]
        real_background = real_background[~np.isnan(real_background)]
        global_mean = np.mean(real_background)
        window_size_bg = 480
        kernel = np.ones(window_size_bg) / window_size_bg
        baseline = np.convolve(real_background, kernel, mode='valid')
        hf_residual = real_background[window_size_bg//2 : -window_size_bg//2 + 1] - baseline
        base_std_noise = np.std(hf_residual)
        baseline_diffs = np.diff(baseline)
        sigma = np.std(baseline_diffs)
        theta = 0.000004
    except FileNotFoundError:
        global_mean = 100.0
        base_std_noise = 2.55
        theta = 0.000004
        sigma = 0.010

    log(f"\nBaseline noise std: {base_std_noise:.4f}")
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
            templates, class_map, global_mean, base_std_noise, theta, sigma,
            noise_multiplier=mult, variance_level=BASELINE_VARIANCE,
            num_anomalies=NUM_ANOMALIES, seed=12345
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
            templates, class_map, global_mean, base_std_noise, theta, sigma,
            noise_multiplier=BASELINE_NOISE_MULT, variance_level=actual_variance,
            num_anomalies=NUM_ANOMALIES, seed=12345
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

    # ── Plot ──
    log("\nGenerating robustness chart...")

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Background Noise Intensity',
            'Anomaly Shape Deformation'
        ),
        horizontal_spacing=0.12
    )

    # ── Left panel: Noise sweep ──
    fig.add_trace(go.Scatter(
        x=[r[0] for r in noise_results],
        y=[r[2] * 100 for r in noise_results],
        mode='lines+markers',
        name='Macro F1 Score',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=8),
        legendgroup='noise', legendgrouptitle_text='Noise Sweep',
        hovertemplate='Noise ×%{x:.1f}<br>F1 Score: %{y:.1f}%<extra></extra>'
    ), row=1, col=1)

    # ── Right panel: Deformation sweep ──
    fig.add_trace(go.Scatter(
        x=[r[0] for r in deform_results],
        y=[r[2] * 100 for r in deform_results],
        mode='lines+markers',
        name='Macro F1 Score',
        line=dict(color='#4ECDC4', width=3),
        marker=dict(size=8),
        legendgroup='deform', legendgrouptitle_text='Deformation Sweep',
        hovertemplate='Deformation ×%{x:.1f}<br>F1 Score: %{y:.1f}%<extra></extra>'
    ), row=1, col=2)

    # ── Layout ──
    fig.update_layout(
        title=dict(text=f'{display_name} Robustness: F1 Score vs. Noise & Deformation', x=0.5),
        template='plotly_dark',
        height=500, width=1100,
        legend=dict(
            orientation='v',
            bgcolor='rgba(0,0,0,0.4)',
            font=dict(size=11)
        )
    )

    fig.update_xaxes(title_text=f'Noise Multiplier (×1 = {base_std_noise:.2f} std dev)', row=1, col=1)
    fig.update_xaxes(title_text=f'Deformation Multiplier (×1 = {BASELINE_VARIANCE} variance)', row=1, col=2)
    fig.update_yaxes(title_text='F1 Score (%)', range=[0, 110], row=1, col=1)
    fig.update_yaxes(range=[0, 110], row=1, col=2)

    output_html = os.path.join(logs_dir, 'robustness_test.html')
    fig.write_html(output_html)
    log(f"Chart saved to: {output_html}")

    try:
        output_png = os.path.join(logs_dir, 'robustness_test.png')
        fig.write_image(output_png, width=1100, height=500, scale=2)
        log(f"PNG saved to: {output_png}")
    except Exception:
        pass

    log(f"\nLog saved to: {log_path}")
    log("=" * 70)


if __name__ == "__main__":
    main()
