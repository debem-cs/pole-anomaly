"""
Build all figures for the LaTeX report from training/inference/robustness logs
and the raw data files. Produces consistently styled PNGs in documentation/figures/.

Run from anywhere:
    py documentation/figures/build_figures.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
DOC = HERE.parent
ROOT = DOC.parent
DATA_GEN = ROOT / "data-gen"
ARCH = ROOT / "architectures"
OUT = HERE
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["1d_cnn", "resnet", "cnn_attention"]
MODEL_PRETTY = {"1d_cnn": "1D-CNN", "resnet": "ResNet 1D", "cnn_attention": "CNN+Attention"}
MODEL_COLOR = {"1d_cnn": "#1f77b4", "resnet": "#d62728", "cnn_attention": "#2ca02c"}
CLASS_COLORS = {
    0: "#888888",
    1: "#d62728",  # bell
    2: "#ff7f0e",  # bell_sq
    3: "#1f77b4",  # fast_ascend
    4: "#9467bd",  # fast_descend
    5: "#2ca02c",  # M
    6: "#8c564b",  # square
}
CLASS_NAMES = {
    0: "Normal",
    1: "bell",
    2: "bell\\_sq",
    3: "fast\\_ascend",
    4: "fast\\_descend",
    5: "M",
    6: "square",
}
# Plain ASCII labels for matplotlib (no LaTeX escaping inside python)
CLASS_NAMES_PLAIN = {
    0: "Normal",
    1: "bell",
    2: "bell_sq",
    3: "fast_ascend",
    4: "fast_descend",
    5: "M",
    6: "square",
}

# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib global style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def _save(fig, name: str):
    path = OUT / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Real 2015 gamma dose data — overview, histogram, baseline decomposition
# ─────────────────────────────────────────────────────────────────────────────
def fig_real_data_overview():
    print("[1/13] real_data_overview")
    df = pd.read_csv(DATA_GEN / "data" / "2015_months_DebitDoseA.txt")
    columns = df.columns.tolist()
    month_names = {col: col.split(" ")[0] for col in columns}

    fig, axes = plt.subplots(len(columns), 1, figsize=(8.5, 9), sharey=True)
    for ax, col in zip(axes, columns):
        series = pd.to_numeric(df[col], errors="coerce").interpolate(method="linear", limit_direction="both")
        ax.plot(series.values, color="#2a6e9a", linewidth=0.4)
        ax.set_title(f"Time period starting {month_names[col]}", loc="left")
        ax.set_ylabel("Gamma dose")
    axes[-1].set_xlabel("Sample index (1 sample / minute)")
    fig.suptitle("Real gamma dose measurements, four months of 2015", y=1.00)
    fig.tight_layout()
    _save(fig, "real_data_overview")


def fig_real_data_stats():
    print("[2/13] real_data_stats")
    df = pd.read_csv(DATA_GEN / "data" / "2015_months_DebitDoseA.txt")
    cols = df.columns.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    # Histogram of values, one curve per month
    for i, col in enumerate(cols):
        series = pd.to_numeric(df[col], errors="coerce").interpolate("linear", limit_direction="both").values
        axes[0].hist(series, bins=80, alpha=0.45, density=True, label=col.split(" ")[0])
    axes[0].set_xlabel("Gamma dose value")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution of values per month")
    axes[0].legend(fontsize=8)

    # Autocorrelation of residuals (after removing the moving-average baseline)
    window = 480
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce").interpolate("linear", limit_direction="both").values
        kernel = np.ones(window) / window
        baseline = np.convolve(series, kernel, mode="valid")
        pad_l = window // 2
        pad_r = len(series) - len(baseline) - pad_l
        baseline = np.concatenate([
            np.full(pad_l, baseline[0]),
            baseline,
            np.full(pad_r, baseline[-1]),
        ])
        residual = series - baseline
        residual = residual[window:-window]
        # autocorr up to lag 200
        r = residual - residual.mean()
        denom = np.dot(r, r)
        lags = np.arange(0, 60)
        acf = np.array([np.dot(r[:len(r) - lag], r[lag:]) / denom for lag in lags])
        axes[1].plot(lags, acf, label=col.split(" ")[0], linewidth=1.0)
    axes[1].axhline(0.0, color="k", linewidth=0.6)
    axes[1].set_xlabel("Lag (minutes)")
    axes[1].set_ylabel("Autocorrelation")
    axes[1].set_title("Residual autocorrelation after baseline removal")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "real_data_stats")


def fig_real_data_decomposition():
    print("[3/13] real_data_decomposition")
    df = pd.read_csv(DATA_GEN / "data" / "2015_months_DebitDoseA.txt")
    col = df.columns[0]
    series = pd.to_numeric(df[col], errors="coerce").interpolate("linear", limit_direction="both").values
    window = 480
    kernel = np.ones(window) / window
    baseline_valid = np.convolve(series, kernel, mode="valid")
    pad_l = window // 2
    pad_r = len(series) - len(baseline_valid) - pad_l
    baseline = np.concatenate([
        np.full(pad_l, baseline_valid[0]),
        baseline_valid,
        np.full(pad_r, baseline_valid[-1]),
    ])
    residual = series - baseline

    # Trim a 10k snapshot to keep it readable
    n_show = 10_000
    t = np.arange(n_show)
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t, series[:n_show], color="#2a6e9a", linewidth=0.5, label="raw signal")
    axes[0].plot(t, baseline[:n_show], color="#d62728", linewidth=1.5, label="moving-average baseline (480-pt window)")
    axes[0].set_ylabel("Gamma dose")
    axes[0].set_title("Decomposition of one month of real gamma data (10 000-point snapshot)", loc="left")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(t, baseline[:n_show], color="#d62728", linewidth=1.2)
    axes[1].set_ylabel("Baseline only")
    axes[1].set_title("Slow drift — modelled as an Ornstein–Uhlenbeck process", loc="left")

    axes[2].plot(t, residual[:n_show], color="#444", linewidth=0.4)
    axes[2].set_ylabel("Residuals")
    axes[2].set_xlabel("Sample index")
    axes[2].set_title("Residuals — modelled as zero-mean Gaussian noise", loc="left")

    fig.tight_layout()
    _save(fig, "real_data_decomposition")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Anomaly templates
# ─────────────────────────────────────────────────────────────────────────────
def _draw_template(ax, df, title):
    t_dense, v_dense = [], []
    for i in range(len(df) - 1):
        r0, r1 = df.iloc[i], df.iloc[i + 1]
        t0, v0 = r0["time"], r0["value"]
        t1, v1 = r1["time"], r1["value"]
        n = max(10, int(100 * (t1 - t0))) if t1 > t0 else 2
        ts = np.linspace(t0, t1, n)
        mode = r0["interp"] if "interp" in df.columns else "linear"
        if mode in ("exp", "exp_up") and t1 > t0:
            x = (ts - t0) / (t1 - t0)
            y = (np.exp(3.0 * x) - 1) / (np.exp(3.0) - 1)
            vs = v0 + (v1 - v0) * y
        elif mode == "exp_down" and t1 > t0:
            x = (ts - t0) / (t1 - t0)
            y = 1.0 - (np.exp(3.0 * (1.0 - x)) - 1) / (np.exp(3.0) - 1)
            vs = v0 + (v1 - v0) * y
        elif mode == "bell" and t1 > t0:
            x = (ts - t0) / (t1 - t0)
            y = 0.5 - 0.5 * np.cos(np.pi * x)
            vs = v0 + (v1 - v0) * y
        else:
            vs = np.linspace(v0, v1, n)
        t_dense.extend(ts[:-1] if i < len(df) - 2 else ts)
        v_dense.extend(vs[:-1] if i < len(df) - 2 else vs)
    ax.plot(t_dense, v_dense, color="#1f77b4", linewidth=2)
    ax.scatter(df["time"], df["value"], color="#d62728", s=18, zorder=5, label="keypoints")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 1.20)
    ax.set_title(title)
    ax.set_xlabel("normalised time")
    ax.set_ylabel("amplitude")


def fig_anomaly_templates():
    print("[4/13] anomaly_templates")
    files = sorted((DATA_GEN / "anomalies").glob("*.csv"))
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.5))
    for ax, f in zip(axes.ravel(), files):
        df = pd.read_csv(f)
        name = f.stem.replace("_", " ")
        _draw_template(ax, df, name)
    fig.suptitle("Six hand-designed anomaly templates (normalised time and amplitude)", y=1.0)
    fig.tight_layout()
    _save(fig, "anomaly_templates")


def fig_template_deformation():
    print("[5/13] template_deformation")
    sys.path.insert(0, str(DATA_GEN))
    from anomaly_generator import generate_anomaly  # type: ignore

    files = sorted((DATA_GEN / "anomalies").glob("*.csv"))
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.5))
    rng = np.random.default_rng(42)
    for ax, f in zip(axes.ravel(), files):
        df = pd.read_csv(f)
        for j in range(5):
            np.random.seed(int(rng.integers(0, 1_000_000)))
            amplitude = 2.53 * rng.uniform(2.5, 7.6)
            period = int(rng.integers(200, 800))
            variance = float(rng.uniform(0.02, 0.10))
            t, v = generate_anomaly(df, amplitude=amplitude, period=period, variance=variance)
            ax.plot(t, v, alpha=0.7, linewidth=1.0)
        ax.set_title(f.stem.replace("_", " "))
        ax.set_xlabel("time step")
        ax.set_ylabel("dose offset")
    fig.suptitle("Five stochastic realisations per template (variance ∈ [0.02, 0.10])", y=1.0)
    fig.tight_layout()
    _save(fig, "template_deformation")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Synthetic dataset preview — sample the first chunk
# ─────────────────────────────────────────────────────────────────────────────
def fig_synthetic_dataset_preview():
    print("[6/13] synthetic_dataset_preview")
    path = ARCH / "data" / "training_dataset.csv"
    # Stream-read first ~30k rows
    df = pd.read_csv(path, nrows=30_000)
    t = df["time_step"].values
    g = df["gamma_dose"].values
    lbl = df["is_anomaly"].values.astype(int)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, g, color="#2a6e9a", linewidth=0.5, label="simulated gamma signal")

    # Coloured spans for each anomaly
    in_anom, start, current_label = False, 0, 0
    for i, val in enumerate(lbl):
        if val > 0 and not in_anom:
            in_anom, start, current_label = True, i, val
        elif val == 0 and in_anom:
            ax.axvspan(t[start], t[i - 1], color=CLASS_COLORS[current_label], alpha=0.25)
            in_anom = False
    if in_anom:
        ax.axvspan(t[start], t[-1], color=CLASS_COLORS[current_label], alpha=0.25)

    handles = [plt.matplotlib.patches.Patch(color=CLASS_COLORS[c], alpha=0.35, label=CLASS_NAMES_PLAIN[c])
               for c in range(1, 7)]
    ax.legend(handles=handles, loc="upper right", ncol=3, fontsize=8)
    ax.set_xlabel("time step")
    ax.set_ylabel("simulated dose")
    ax.set_title("Synthetic dataset preview (first 30 000 of 10 M points; anomaly spans coloured by class)")
    fig.tight_layout()
    _save(fig, "synthetic_dataset_preview")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Training curves — parse text logs
# ─────────────────────────────────────────────────────────────────────────────
def parse_training_log(model: str):
    path = ARCH / "logs" / model / "training_log.txt"
    epochs, train_loss, val_loss, val_f1 = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        in_training = False
        for line in f:
            if line.startswith("Epoch    Train"):
                in_training = True
                continue
            if in_training:
                m = re.match(r"\s*(\d+)/\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.e-]+)\s+", line)
                if m:
                    epochs.append(int(m.group(1)))
                    train_loss.append(float(m.group(2)))
                    val_loss.append(float(m.group(3)))
                    val_f1.append(float(m.group(4)))
                elif line.strip() == "" or line.startswith("Training"):
                    break
    return np.array(epochs), np.array(train_loss), np.array(val_loss), np.array(val_f1)


def fig_training_curves():
    print("[7/13] training_curves_comparison")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in MODELS:
        e, tl, vl, f1 = parse_training_log(m)
        c = MODEL_COLOR[m]
        ax1.plot(e, tl, color=c, linestyle="--", linewidth=1.0, alpha=0.7)
        ax1.plot(e, vl, color=c, linewidth=1.6, label=MODEL_PRETTY[m])
        ax2.plot(e, f1, color=c, linewidth=1.6, label=MODEL_PRETTY[m])
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax1.set_title("Train (dashed) and validation (solid) loss")
    ax1.legend(loc="upper right")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("macro F1 (window level)")
    ax2.set_title("Validation F1 score")
    ax2.set_ylim(0.5, 0.85)
    ax2.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "training_curves_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Confusion matrices — parse inference logs
# ─────────────────────────────────────────────────────────────────────────────
def parse_confusion_matrix(model: str):
    """Read the Event-Level Confusion Matrix block from inference_log.txt.

    The 6x6 matrix is the bottom 6 rows; only rows where the last 6 tokens
    parse as integers are kept.
    """
    path = ARCH / "logs" / model / "inference_log.txt"
    rows = []
    in_cm = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "Event-Level Confusion Matrix" in line:
                in_cm = True
                continue
            if not in_cm:
                continue
            if "Event-Level Summary" in line or line.startswith("---") and rows:
                break
            stripped = line.strip()
            if not stripped:
                if rows:
                    break
                continue
            tokens = stripped.split()
            if len(tokens) < 7:
                continue
            try:
                counts = [int(x) for x in tokens[-6:]]
            except ValueError:
                continue
            name = " ".join(tokens[:-6])
            rows.append((name, counts))
    if not rows:
        return None, None
    return [r[0] for r in rows], np.array([r[1] for r in rows])


def fig_confusion_matrices():
    print("[8/13] confusion_matrices")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    class_order = ["bell", "bell_sq", "fast_ascend", "fast_descend", "M", "square"]

    for ax, m in zip(axes, MODELS):
        names, matrix = parse_confusion_matrix(m)
        if matrix is None:
            ax.set_visible(False)
            continue
        # Normalize by row
        row_sums = matrix.sum(axis=1, keepdims=True)
        norm = np.divide(matrix, np.maximum(row_sums, 1), where=row_sums > 0)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(class_order)))
        ax.set_yticks(range(len(class_order)))
        ax.set_xticklabels(class_order, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_order, fontsize=8)
        for i in range(len(class_order)):
            for j in range(len(class_order)):
                v = matrix[i, j]
                ax.text(j, i, f"{v}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
        ax.set_title(MODEL_PRETTY[m])
        ax.set_xlabel("predicted")
        if m == "1d_cnn":
            ax.set_ylabel("true")
    fig.suptitle("Event-level confusion matrices on the synthetic test segment (100 k points, 59 events)", y=1.02)
    fig.tight_layout()
    _save(fig, "confusion_matrices")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Robustness — parse logs
# ─────────────────────────────────────────────────────────────────────────────
def parse_robustness(model: str):
    path = ARCH / "logs" / model / "robustness_test_log.txt"
    noise_pts, deform_pts = [], []
    section = None  # "noise" | "deform" | None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "SWEEP 1: BACKGROUND NOISE" in line:
                section = "noise"; continue
            if "SWEEP 2: ANOMALY SHAPE DEFORMATION" in line:
                section = "deform"; continue
            if "SUMMARY" in line:
                section = None; continue
            m = re.match(r"\s*([\d.]+)\s+\d+\s+([\d.]+)%", line)
            if m and section is not None:
                pt = (float(m.group(1)), float(m.group(2)))
                (noise_pts if section == "noise" else deform_pts).append(pt)
    return noise_pts, deform_pts


def fig_robustness():
    print("[9/13] robustness")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in MODELS:
        noise, deform = parse_robustness(m)
        c = MODEL_COLOR[m]
        x1, y1 = zip(*noise)
        x2, y2 = zip(*deform)
        ax1.plot(x1, y1, color=c, marker="o", linewidth=1.6, markersize=4, label=MODEL_PRETTY[m])
        ax2.plot(x2, y2, color=c, marker="o", linewidth=1.6, markersize=4, label=MODEL_PRETTY[m])
    ax1.axhline(50, color="k", linestyle=":", linewidth=0.8)
    ax2.axhline(50, color="k", linestyle=":", linewidth=0.8)
    ax1.set_xlabel("noise multiplier  ($\\times 2.53$ std)"); ax1.set_ylabel("event-level macro F1 (%)")
    ax1.set_title("Robustness to background noise"); ax1.set_ylim(0, 105); ax1.legend(loc="lower left")
    ax2.set_xlabel("deformation multiplier  ($\\times 0.04$ variance)"); ax2.set_ylabel("event-level macro F1 (%)")
    ax2.set_title("Robustness to anomaly-shape deformation"); ax2.set_ylim(0, 105); ax2.legend(loc="lower left")
    fig.tight_layout()
    _save(fig, "robustness")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Real-data inference — re-run for clean visuals
# ─────────────────────────────────────────────────────────────────────────────
def fig_real_data_inference():
    print("[10/13] real_data_inference")
    import torch
    sys.path.insert(0, str(ARCH))
    from model_selection import get_model  # type: ignore
    from inference_engine import run_sliding_window_inference  # type: ignore

    # Class mapping
    cm = {}
    with open(DATA_GEN / "data" / "class_mapping.txt") as f:
        for line in f:
            if ":" in line:
                c, n = line.split(":")
                cm[int(c.strip())] = n.strip()
    num_classes = len(cm)

    # Real data, first month only, 10k window
    df = pd.read_csv(DATA_GEN / "data" / "2015_months_DebitDoseA.txt")
    col = df.columns[0]
    series = pd.to_numeric(df[col], errors="coerce").interpolate("linear", limit_direction="both").values.astype(np.float32)
    n_show = 10_000
    series_show = series[:n_show]
    t = np.arange(n_show)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for ax, model_name in zip(axes, MODELS):
        # Load model
        model = get_model(model_name, num_classes=num_classes, in_channels=1).to(device)
        model.load_state_dict(torch.load(
            ARCH / "saved_models" / model_name / f"best_{model_name}.pth",
            map_location=device, weights_only=True))
        model.eval()
        with open(ARCH / "saved_models" / model_name / "normalization_stats.json") as f:
            train_std = json.load(f)["std"]

        probs, counts, _, _ = run_sliding_window_inference(
            series_show, model, device, train_std, num_classes,
            window_size=512, stride=10, batch_size=256
        )
        # Smooth
        smoothing = 30
        smooth = np.zeros_like(probs)
        for c in range(num_classes):
            s = pd.Series(probs[:, c]).interpolate(method="linear", limit_direction="both")
            smooth[:, c] = s.rolling(window=smoothing, center=True, min_periods=1).mean().values

        # Plot dose on left axis
        ax.plot(t, series_show, color="#2a6e9a", linewidth=0.4, alpha=0.7, label="gamma dose")
        ax.set_ylabel("dose"); ax.set_ylim(np.percentile(series_show, 1) - 2, np.percentile(series_show, 99.5) + 2)

        # Right axis for probabilities — only anomaly classes (1..)
        axR = ax.twinx()
        for c in range(1, num_classes):
            axR.plot(t, smooth[:, c] * 100, color=CLASS_COLORS[c], linewidth=1.2, label=CLASS_NAMES_PLAIN[c])
        # Faint normal-class probability
        axR.plot(t, smooth[:, 0] * 100, color="#888888", linewidth=0.8, alpha=0.4, label="Normal")
        axR.set_ylim(0, 105); axR.set_ylabel("class confidence (%)")
        axR.grid(False)
        ax.set_title(f"{MODEL_PRETTY[model_name]} — inference on February 2015 (10 000-point snapshot)", loc="left")

    # One legend for all — built from class colours (no extra twinx call!)
    legend_handles = [plt.Line2D([0], [0], color=CLASS_COLORS[c], label=CLASS_NAMES_PLAIN[c]) for c in range(0, 7)]
    fig.legend(handles=legend_handles, loc="upper center", ncol=7, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    axes[-1].set_xlabel("sample index")
    fig.tight_layout()
    _save(fig, "real_data_inference")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Failure analysis — class distribution shift
# ─────────────────────────────────────────────────────────────────────────────
def fig_failure_analysis():
    print("[11/13] failure_analysis")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    train_counts = [77674, 30218, 28106, 27004, 28769, 30963, 27251]
    train_pct = np.array(train_counts) / sum(train_counts) * 100
    labels = ["Normal", "bell", "bell_sq", "fast_ascend", "fast_descend", "M", "square"]
    real_pct = np.array([99.0] + [1.0 / 6] * 6)  # very rough — real data is almost all background

    x = np.arange(len(labels))
    w = 0.4
    ax1.bar(x - w/2, train_pct, w, label="Training set (window labels)", color="#2a6e9a")
    ax1.bar(x + w/2, real_pct, w, label="Real 2015 data (estimate)", color="#d62728")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_ylabel("share of windows (%)")
    ax1.set_title("Class prior shift: training vs deployment")
    ax1.legend(fontsize=8)

    # Class weights vs deployment prevalence — illustrate the bias
    weights = [0.46, 1.18, 1.27, 1.32, 1.24, 1.15, 1.31]
    ax2.bar(x, weights, color="#888888")
    ax2.axhline(1.0, color="k", linewidth=0.6, linestyle="--")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylabel("cross-entropy weight")
    ax2.set_title("Class weights used at training — penalise normal under-prediction")
    fig.tight_layout()
    _save(fig, "failure_analysis")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Parameter count + speed comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_model_cost():
    print("[12/13] model_cost")
    params = {"1d_cnn": 22_727, "resnet": 74_631, "cnn_attention": 109_655}
    train_minutes = {"1d_cnn": 4.30, "resnet": 10.03, "cnn_attention": 10.24}
    f1_event = {"1d_cnn": 93.42, "resnet": 100.00, "cnn_attention": 95.53}
    wps = {"1d_cnn": 10_664, "resnet": 14_232, "cnn_attention": 15_184}

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    xs = np.arange(3)
    names = [MODEL_PRETTY[m] for m in MODELS]

    axes[0].bar(xs, [params[m] / 1000 for m in MODELS],
                color=[MODEL_COLOR[m] for m in MODELS])
    axes[0].set_xticks(xs); axes[0].set_xticklabels(names, rotation=15)
    axes[0].set_ylabel("trainable parameters (k)")
    axes[0].set_title("Model size")

    axes[1].bar(xs, [train_minutes[m] for m in MODELS],
                color=[MODEL_COLOR[m] for m in MODELS])
    axes[1].set_xticks(xs); axes[1].set_xticklabels(names, rotation=15)
    axes[1].set_ylabel("training time (min)")
    axes[1].set_title("Training cost (35 epochs, RTX 5060)")

    axes[2].bar(xs, [f1_event[m] for m in MODELS],
                color=[MODEL_COLOR[m] for m in MODELS])
    axes[2].set_xticks(xs); axes[2].set_xticklabels(names, rotation=15)
    axes[2].set_ylabel("event-level macro F1 (%)")
    axes[2].set_title("Synthetic test performance")
    axes[2].set_ylim(80, 105)
    fig.tight_layout()
    _save(fig, "model_cost")


# ─────────────────────────────────────────────────────────────────────────────
# 13. Sliding-window illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_sliding_window():
    print("[13/13] sliding_window")
    # Use 30k synthetic preview slice
    path = ARCH / "data" / "training_dataset.csv"
    df = pd.read_csv(path, nrows=5000)
    g = df["gamma_dose"].values
    lbl = df["is_anomaly"].values.astype(int)
    t = np.arange(len(g))

    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.plot(t, g, color="#2a6e9a", linewidth=0.5)
    # Plot windows
    win = 512
    stride = 256  # show wider stride so windows don't overlap too densely
    y_base = g.min() - 1
    for i, start in enumerate(range(0, len(g) - win + 1, stride)):
        col = "#d62728" if lbl[start:start + win].sum() > 0.1 * win else "#888888"
        ax.add_patch(plt.matplotlib.patches.Rectangle(
            (start, y_base + i * 0.4), win, 0.3,
            facecolor=col, alpha=0.55, edgecolor="none"))
    ax.set_xlim(0, len(g))
    ax.set_ylim(y_base, g.max() + 1)
    ax.set_xlabel("sample index")
    ax.set_ylabel("dose")
    ax.set_title("Sliding-window construction (red = window contains $\\geq 10\\%$ anomaly points)")
    fig.tight_layout()
    _save(fig, "sliding_window")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    fig_real_data_overview()
    fig_real_data_stats()
    fig_real_data_decomposition()
    fig_anomaly_templates()
    fig_template_deformation()
    fig_synthetic_dataset_preview()
    fig_training_curves()
    fig_confusion_matrices()
    fig_robustness()
    fig_failure_analysis()
    fig_model_cost()
    fig_sliding_window()
    fig_real_data_inference()
    print("\nDone. Figures in:", OUT)


if __name__ == "__main__":
    main()
