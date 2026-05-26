"""
Build all figures for the LaTeX report from training/inference/robustness logs
and the raw data files. Produces consistently styled PNGs in documentation/figures/.

The figures cover two iterations of the data generator and three model
architectures, plus the noise re-analysis that motivated the v2 generator:

    v1 = white-Gaussian noise, OU baseline, six anomaly templates, dense
         (~7 anomalies / 10k samples) injection grid. Trained models live
         under  pole-anomaly-old/architectures/.

    v2 = AR(1) noise (phi ~ 0.86), tuned OU, three anomaly templates,
         sparse (~1.5 / 10k) injection grid, log-uniform absolute
         amplitudes/durations calibrated against 17 hand-labelled real
         events. Trained models live under architectures/.

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

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DOC = HERE.parent
ROOT = DOC.parent
DATA_GEN = ROOT / "data-gen"
ARCH = ROOT / "architectures"
OLD_ROOT = ROOT / "pole-anomaly-old"
OLD_DATA_GEN = OLD_ROOT / "data-gen"
OLD_ARCH = OLD_ROOT / "architectures"
OUT = HERE
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["1d_cnn", "resnet", "cnn_attention"]
MODEL_PRETTY = {
    "1d_cnn": "1D-CNN",
    "resnet": "ResNet 1D",
    "cnn_attention": "CNN+Attention",
}
MODEL_COLOR = {
    "1d_cnn": "#1f77b4",
    "resnet": "#d62728",
    "cnn_attention": "#2ca02c",
}

# v1 class palette (7 classes)
V1_CLASS_COLORS = {
    0: "#888888",
    1: "#d62728",  # bell
    2: "#ff7f0e",  # bell_sq
    3: "#1f77b4",  # fast_ascend
    4: "#9467bd",  # fast_descend
    5: "#2ca02c",  # M
    6: "#8c564b",  # square
}
V1_CLASS_NAMES = {
    0: "Normal",
    1: "bell",
    2: "bell_sq",
    3: "fast_ascend",
    4: "fast_descend",
    5: "M",
    6: "square",
}

# v2 class palette (4 classes)
V2_CLASS_COLORS = {
    0: "#888888",
    1: "#d62728",  # bell
    2: "#1f77b4",  # fast_ascend
    3: "#9467bd",  # fast_descend
}
V2_CLASS_NAMES = {
    0: "Normal",
    1: "bell",
    2: "fast_ascend",
    3: "fast_descend",
}

# -----------------------------------------------------------------------------
# Matplotlib global style
# -----------------------------------------------------------------------------
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


# =============================================================================
# REAL DATA (shared between v1 and v2)
# =============================================================================
def fig_real_data_overview():
    print("[01] real_data_overview")
    df = pd.read_csv(DATA_GEN / "data" / "2015_months_DebitDoseA.txt")
    columns = df.columns.tolist()
    month_names = {col: col.split(" ")[0] for col in columns}

    fig, axes = plt.subplots(len(columns), 1, figsize=(9.5, 8), sharey=True)
    for ax, col in zip(axes, columns):
        series = pd.to_numeric(df[col], errors="coerce").interpolate(
            method="linear", limit_direction="both")
        ax.plot(series.values, color="#2a6e9a", linewidth=0.4)
        ax.set_title(f"Recording starting {month_names[col]}", loc="left")
        ax.set_ylabel("dose")
    axes[-1].set_xlabel("sample index (1 min cadence)")
    fig.suptitle("Real gamma dose: four months of 2015 surveillance data", y=1.00)
    fig.tight_layout()
    _save(fig, "real_data_overview")


def fig_real_data_stats():
    print("[02] real_data_stats")
    df = pd.read_csv(DATA_GEN / "data" / "2015_months_DebitDoseA.txt")
    cols = df.columns.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce").interpolate(
            "linear", limit_direction="both").values
        axes[0].hist(series, bins=80, alpha=0.45, density=True,
                     label=col.split(" ")[0])
    axes[0].set_xlabel("gamma dose value")
    axes[0].set_ylabel("density")
    axes[0].set_title("Marginal distribution of raw values per month")
    axes[0].legend(fontsize=8)

    window = 480
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce").interpolate(
            "linear", limit_direction="both").values
        kernel = np.ones(window) / window
        baseline = np.convolve(series, kernel, mode="valid")
        pad_l = window // 2
        pad_r = len(series) - len(baseline) - pad_l
        baseline = np.concatenate([
            np.full(pad_l, baseline[0]),
            baseline,
            np.full(pad_r, baseline[-1]),
        ])
        residual = (series - baseline)[window:-window]
        r = residual - residual.mean()
        denom = np.dot(r, r)
        lags = np.arange(0, 50)
        acf = np.array([np.dot(r[:len(r) - lag], r[lag:]) / denom for lag in lags])
        axes[1].plot(lags, acf, label=col.split(" ")[0], linewidth=1.0)
    axes[1].axhline(0.0, color="k", linewidth=0.6)
    axes[1].set_xlabel("lag (minutes)")
    axes[1].set_ylabel("autocorrelation")
    axes[1].set_title("Residual autocorrelation (after MA-480 baseline removal)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "real_data_stats")


def fig_real_data_decomposition():
    print("[03] real_data_decomposition")
    df = pd.read_csv(DATA_GEN / "data" / "2015_months_DebitDoseA.txt")
    col = df.columns[0]
    series = pd.to_numeric(df[col], errors="coerce").interpolate(
        "linear", limit_direction="both").values
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

    n_show = 10_000
    t = np.arange(n_show)
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 6.5), sharex=True)
    axes[0].plot(t, series[:n_show], color="#2a6e9a", linewidth=0.5,
                 label="raw signal")
    axes[0].plot(t, baseline[:n_show], color="#d62728", linewidth=1.5,
                 label="baseline (MA-480)")
    axes[0].set_ylabel("dose")
    axes[0].set_title("Raw signal and slow baseline", loc="left")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(t, baseline[:n_show], color="#d62728", linewidth=1.2)
    axes[1].set_ylabel("baseline")
    axes[1].set_title("Slow drift only", loc="left")

    axes[2].plot(t, residual[:n_show], color="#444", linewidth=0.4)
    axes[2].set_ylabel("residual")
    axes[2].set_xlabel("sample index")
    axes[2].set_title("High-frequency residual (raw $-$ baseline)", loc="left")

    fig.suptitle("Decomposition of one month of real gamma data "
                 "(10k-point snapshot)", y=1.00)
    fig.tight_layout()
    _save(fig, "real_data_decomposition")


# =============================================================================
# ANOMALY TEMPLATES (v1 = 6, v2 = 3)
# =============================================================================
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
    ax.scatter(df["time"], df["value"], color="#d62728", s=18, zorder=5,
               label="keypoints")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 1.20)
    ax.set_title(title)
    ax.set_xlabel("normalised time")
    ax.set_ylabel("amplitude")


def fig_v1_anomaly_templates():
    print("[04] v1_anomaly_templates")
    files = sorted((OLD_DATA_GEN / "anomalies").glob("*.csv"))
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.8))
    for ax, f in zip(axes.ravel(), files):
        df = pd.read_csv(f)
        _draw_template(ax, df, f.stem.replace("_", " "))
    fig.suptitle("v1 generator: six hand-designed anomaly templates",
                 y=1.00)
    fig.tight_layout()
    _save(fig, "v1_anomaly_templates")


def fig_v2_anomaly_templates():
    print("[05] v2_anomaly_templates")
    files = sorted((DATA_GEN / "anomalies").glob("*.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, f in zip(axes, files):
        df = pd.read_csv(f)
        _draw_template(ax, df, f.stem.replace("_", " "))
    fig.suptitle("v2 generator: three templates retained "
                 "(bell, fast\\_ascend, fast\\_descend)", y=1.02)
    fig.tight_layout()
    _save(fig, "v2_anomaly_templates")


def _draw_deformations(template_files, axes_array, amp_lo, amp_hi,
                       period_lo, period_hi, var_lo, var_hi,
                       seed=42, log_amp=False):
    sys.path.insert(0, str(DATA_GEN))
    from anomaly_generator import generate_anomaly  # type: ignore

    rng = np.random.default_rng(seed)
    for ax, f in zip(axes_array.ravel(), template_files):
        df = pd.read_csv(f)
        for _ in range(5):
            np.random.seed(int(rng.integers(0, 1_000_000)))
            if log_amp:
                amp = float(np.exp(rng.uniform(np.log(amp_lo), np.log(amp_hi))))
                period = int(np.exp(rng.uniform(np.log(period_lo),
                                                np.log(period_hi))))
            else:
                amp = float(rng.uniform(amp_lo, amp_hi))
                period = int(rng.integers(period_lo, period_hi))
            variance = float(rng.uniform(var_lo, var_hi))
            t, v = generate_anomaly(df, amplitude=amp, period=period,
                                    variance=variance)
            ax.plot(t, v, alpha=0.7, linewidth=1.0)
        ax.set_title(f.stem.replace("_", " "))
        ax.set_xlabel("sample")
        ax.set_ylabel("dose offset")


def fig_v1_template_deformation():
    print("[06] v1_template_deformation")
    files = sorted((OLD_DATA_GEN / "anomalies").glob("*.csv"))
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.8))
    _draw_deformations(files, axes,
                       amp_lo=2.53 * 2.5, amp_hi=2.53 * 7.6,
                       period_lo=200, period_hi=800,
                       var_lo=0.02, var_hi=0.10, seed=42, log_amp=False)
    fig.suptitle("v1 generator: five stochastic realisations per template "
                 "(uniform amplitude/period/deformation)", y=1.00)
    fig.tight_layout()
    _save(fig, "v1_template_deformation")


def fig_v2_template_deformation():
    print("[07] v2_template_deformation")
    files = sorted((DATA_GEN / "anomalies").glob("*.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    _draw_deformations(files, axes,
                       amp_lo=15.0, amp_hi=40.0,
                       period_lo=30, period_hi=1000,
                       var_lo=0.02, var_hi=0.10, seed=7, log_amp=True)
    fig.suptitle("v2 generator: five realisations per template "
                 "(log-uniform absolute amplitude $\\in [15,40]$, "
                 "duration $\\in [30, 1000]$)", y=1.02)
    fig.tight_layout()
    _save(fig, "v2_template_deformation")


# =============================================================================
# SYNTHETIC PREVIEWS (v1 dense, v2 sparse)
# =============================================================================
def _read_synth_preview(csv_path, nrows):
    df = pd.read_csv(csv_path, nrows=nrows)
    return (df["time_step"].values,
            df["gamma_dose"].values,
            df["is_anomaly"].values.astype(int))


def _plot_synth_preview(ax, t, g, lbl, class_colors, class_names,
                        title, max_legend_classes=6):
    ax.plot(t, g, color="#2a6e9a", linewidth=0.5, label="synthetic signal")
    in_anom, start, current_label = False, 0, 0
    for i, val in enumerate(lbl):
        if val > 0 and not in_anom:
            in_anom, start, current_label = True, i, val
        elif val == 0 and in_anom:
            ax.axvspan(t[start], t[i - 1],
                       color=class_colors.get(current_label, "#888"),
                       alpha=0.30)
            in_anom = False
    if in_anom:
        ax.axvspan(t[start], t[-1],
                   color=class_colors.get(current_label, "#888"), alpha=0.30)

    handles = [plt.matplotlib.patches.Patch(
        color=class_colors[c], alpha=0.40, label=class_names[c])
        for c in list(class_names.keys())[1:max_legend_classes + 1]]
    ax.legend(handles=handles, loc="upper right",
              ncol=min(3, len(handles)), fontsize=8)
    ax.set_xlabel("sample")
    ax.set_ylabel("dose")
    ax.set_title(title)


def fig_v1_synthetic_preview():
    print("[08] v1_synthetic_preview")
    t, g, lbl = _read_synth_preview(
        OLD_DATA_GEN / "data" / "synthetic_custom_dataset.csv",
        nrows=30_000)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    _plot_synth_preview(ax, t, g, lbl, V1_CLASS_COLORS, V1_CLASS_NAMES,
                        title="v1 synthetic dataset preview "
                              "(first 30k of 10M samples; "
                              "$\\sim 7$ anomalies / 10k)",
                        max_legend_classes=6)
    fig.tight_layout()
    _save(fig, "v1_synthetic_preview")


def fig_v2_synthetic_preview():
    print("[09] v2_synthetic_preview")
    t, g, lbl = _read_synth_preview(
        ARCH / "data" / "training_dataset.csv",
        nrows=100_000)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    _plot_synth_preview(ax, t, g, lbl, V2_CLASS_COLORS, V2_CLASS_NAMES,
                        title="v2 synthetic dataset preview "
                              "(first 100k of 10M samples; "
                              "$\\sim 1.5$ anomalies / 10k)",
                        max_legend_classes=3)
    fig.tight_layout()
    _save(fig, "v2_synthetic_preview")


def fig_v1_vs_v2_synthetic():
    print("[10] v1_vs_v2_synthetic")
    t1, g1, lbl1 = _read_synth_preview(
        OLD_DATA_GEN / "data" / "synthetic_custom_dataset.csv",
        nrows=30_000)
    t2, g2, lbl2 = _read_synth_preview(
        ARCH / "data" / "training_dataset.csv",
        nrows=30_000)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    _plot_synth_preview(axes[0], t1, g1, lbl1, V1_CLASS_COLORS, V1_CLASS_NAMES,
                        title="v1: white noise + OU, six templates, dense",
                        max_legend_classes=6)
    _plot_synth_preview(axes[1], t2, g2, lbl2, V2_CLASS_COLORS, V2_CLASS_NAMES,
                        title="v2: AR(1) + tuned OU, three templates, sparse",
                        max_legend_classes=3)
    fig.suptitle("v1 vs v2 synthetic data (first 30k samples)", y=1.00)
    fig.tight_layout()
    _save(fig, "v1_vs_v2_synthetic")


# =============================================================================
# NOISE TEXTURE COMPARISON (IID vs AR(1))
# =============================================================================
def fig_noise_texture_comparison():
    print("[11] noise_texture_comparison")
    n = 600
    rng = np.random.default_rng(11)
    # IID Gaussian (v1)
    iid = rng.normal(scale=2.30, size=n)
    # AR(1) (v2)
    phi, sigma_eps = 0.86, 1.12
    ar = np.empty(n)
    state = 0.0
    eps = rng.normal(scale=sigma_eps, size=n)
    for i in range(n):
        state = phi * state + eps[i]
        ar[i] = state

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(iid, color="#1f77b4", linewidth=0.7,
                 label="v1: IID $\\mathcal{N}(0, 2.30^2)$")
    axes[0].plot(ar, color="#d62728", linewidth=0.7,
                 label="v2: AR(1) $\\varphi=0.86$, $\\sigma_\\varepsilon=1.12$")
    axes[0].set_xlabel("sample")
    axes[0].set_ylabel("residual")
    axes[0].set_title("Two noise textures with the same marginal $\\sigma=2.30$")
    axes[0].legend(loc="upper right", fontsize=8)

    # ACF comparison
    def acf_of(x, lmax=30):
        r = x - x.mean()
        denom = np.dot(r, r)
        return np.array([np.dot(r[:len(r) - k], r[k:]) / denom
                         for k in range(lmax)])

    lags = np.arange(30)
    axes[1].stem(lags - 0.15, acf_of(iid), basefmt=" ",
                 linefmt="#1f77b4", markerfmt="o",
                 label="v1 (IID)")
    axes[1].stem(lags + 0.15, acf_of(ar), basefmt=" ",
                 linefmt="#d62728", markerfmt="o",
                 label="v2 (AR(1))")
    # Empirical real-data ACF for reference
    df = pd.read_csv(DATA_GEN / "data" / "2015_months_DebitDoseA.txt")
    col = df.columns[0]
    series = pd.to_numeric(df[col], errors="coerce").interpolate(
        "linear", limit_direction="both").values
    w = 480
    kernel = np.ones(w) / w
    baseline = np.convolve(series, kernel, mode="valid")
    pad_l = w // 2
    pad_r = len(series) - len(baseline) - pad_l
    baseline = np.concatenate([np.full(pad_l, baseline[0]), baseline,
                               np.full(pad_r, baseline[-1])])
    residual = (series - baseline)[w:-w]
    axes[1].plot(lags, acf_of(residual, lmax=30), color="black",
                 marker="x", linewidth=1.0,
                 label="real 2015 (Feb)", linestyle="-")
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].set_xlabel("lag (samples)")
    axes[1].set_ylabel("autocorrelation")
    axes[1].set_title("ACF of v1, v2 noise vs. real-data residual")
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    _save(fig, "noise_texture_comparison")


# =============================================================================
# NOISE REANALYSIS (AR diagnostics)
# =============================================================================
def fig_noise_clean_residual():
    """Marginal + ACF + rolling-sigma diagnostics on the clean residual."""
    print("[12] noise_clean_residual")
    from scipy import stats as scstats

    df = pd.read_csv(DATA_GEN / "data" / "2015_months_DebitDoseA.txt")
    col = df.columns[0]  # February 2015
    series = pd.to_numeric(df[col], errors="coerce").interpolate(
        "linear", limit_direction="both").values

    w = 480
    kernel = np.ones(w) / w
    baseline = np.convolve(series, kernel, mode="valid")
    pad_l = w // 2
    pad_r = len(series) - len(baseline) - pad_l
    baseline = np.concatenate([np.full(pad_l, baseline[0]), baseline,
                               np.full(pad_r, baseline[-1])])
    residual = series - baseline

    # Robust outlier mask
    s = pd.Series(residual)
    mad = s.rolling(w, center=True, min_periods=1).apply(
        lambda v: np.median(np.abs(v - np.median(v))), raw=True).values
    sigma_loc = 1.4826 * np.maximum(mad, 1e-6)
    z = residual / sigma_loc
    mask = np.abs(z) <= 4.0
    clean = residual[mask][w:-w]

    sigma = float(np.std(clean, ddof=1))

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))

    # Histogram
    bins = np.linspace(np.quantile(clean, 0.001),
                       np.quantile(clean, 0.999), 80)
    axes[0, 0].hist(clean, bins=bins, density=True, color="#2a6e9a",
                    alpha=0.65, label="clean residual")
    xs = np.linspace(bins[0], bins[-1], 400)
    axes[0, 0].plot(xs, scstats.norm.pdf(xs, loc=0, scale=sigma),
                    color="#d62728", linewidth=1.5,
                    label=f"$\\mathcal{{N}}(0, {sigma:.2f})$")
    axes[0, 0].set_title("Marginal histogram + Gaussian fit")
    axes[0, 0].set_xlabel("residual")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].legend(fontsize=8)

    # Q-Q vs Normal
    scstats.probplot(clean[:20000], dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Q-Q vs.\\ Normal")
    axes[0, 1].get_lines()[0].set_color("#2a6e9a")
    axes[0, 1].get_lines()[1].set_color("#d62728")

    # ACF
    r = clean - clean.mean()
    denom = np.dot(r, r)
    lags = np.arange(0, 60)
    acf = np.array([np.dot(r[:len(r) - k], r[k:]) / denom for k in lags])
    axes[1, 0].stem(lags, acf, basefmt=" ",
                    linefmt="#2a6e9a", markerfmt="o")
    band = 2 / np.sqrt(len(clean))
    axes[1, 0].axhline(band, color="#d62728", linestyle="--", linewidth=0.8,
                       label=f"$\\pm 2/\\sqrt{{N}}$")
    axes[1, 0].axhline(-band, color="#d62728", linestyle="--", linewidth=0.8)
    axes[1, 0].axhline(0, color="k", linewidth=0.4)
    axes[1, 0].set_title(f"ACF (lag-1 = {acf[1]:.3f})")
    axes[1, 0].set_xlabel("lag")
    axes[1, 0].set_ylabel("autocorrelation")
    axes[1, 0].legend(fontsize=8)

    # Rolling sigma
    roll = pd.Series(clean).rolling(window=2000, min_periods=200).std().values
    axes[1, 1].plot(np.arange(len(roll)), roll, color="#2a6e9a", linewidth=0.8)
    axes[1, 1].axhline(sigma, color="#d62728", linestyle="--", linewidth=1.0,
                       label=f"global $\\sigma$ = {sigma:.2f}")
    axes[1, 1].set_title("Rolling-window standard deviation (W = 2000)")
    axes[1, 1].set_xlabel("sample")
    axes[1, 1].set_ylabel("$\\sigma$")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Clean residual diagnostics (February 2015, "
                 f"clean fraction = {mask.sum() / len(mask):.2f})", y=1.00)
    fig.tight_layout()
    _save(fig, "noise_clean_residual")


def fig_noise_ar1_innov():
    """AR(1) innovation diagnostics."""
    print("[13] noise_ar1_innov")
    from scipy import stats as scstats
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.stats.diagnostic import acorr_ljungbox

    df = pd.read_csv(DATA_GEN / "data" / "2015_months_DebitDoseA.txt")
    col = df.columns[0]
    series = pd.to_numeric(df[col], errors="coerce").interpolate(
        "linear", limit_direction="both").values
    w = 480
    kernel = np.ones(w) / w
    baseline = np.convolve(series, kernel, mode="valid")
    pad_l = w // 2
    pad_r = len(series) - len(baseline) - pad_l
    baseline = np.concatenate([np.full(pad_l, baseline[0]), baseline,
                               np.full(pad_r, baseline[-1])])
    residual = series - baseline

    s = pd.Series(residual)
    mad = s.rolling(w, center=True, min_periods=1).apply(
        lambda v: np.median(np.abs(v - np.median(v))), raw=True).values
    sigma_loc = 1.4826 * np.maximum(mad, 1e-6)
    z = residual / sigma_loc
    mask = np.abs(z) <= 4.0
    clean = residual[mask][w:-w]

    model = AutoReg(clean, lags=1, old_names=False).fit()
    innov = model.resid
    phi = float(model.params[1])
    sigma_eps = float(np.std(innov, ddof=1))

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))

    bins = np.linspace(np.quantile(innov, 0.001),
                       np.quantile(innov, 0.999), 80)
    axes[0, 0].hist(innov, bins=bins, density=True, color="#2a6e9a",
                    alpha=0.65, label="AR(1) innovations $\\varepsilon_t$")
    xs = np.linspace(bins[0], bins[-1], 400)
    axes[0, 0].plot(xs, scstats.norm.pdf(xs, loc=0, scale=sigma_eps),
                    color="#d62728", linewidth=1.5,
                    label=f"$\\mathcal{{N}}(0, {sigma_eps:.2f})$")
    axes[0, 0].set_title("AR(1) innovation histogram + Gaussian fit")
    axes[0, 0].set_xlabel("$\\varepsilon_t$")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].legend(fontsize=8)

    scstats.probplot(innov[:20000], dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Q-Q vs.\\ Normal")
    axes[0, 1].get_lines()[0].set_color("#2a6e9a")
    axes[0, 1].get_lines()[1].set_color("#d62728")

    r = innov - innov.mean()
    denom = np.dot(r, r)
    lags = np.arange(0, 50)
    acf = np.array([np.dot(r[:len(r) - k], r[k:]) / denom for k in lags])
    axes[1, 0].stem(lags, acf, basefmt=" ",
                    linefmt="#2a6e9a", markerfmt="o")
    band = 2 / np.sqrt(len(innov))
    axes[1, 0].axhline(band, color="#d62728", linestyle="--", linewidth=0.8,
                       label="$\\pm 2/\\sqrt{N}$")
    axes[1, 0].axhline(-band, color="#d62728", linestyle="--", linewidth=0.8)
    axes[1, 0].axhline(0, color="k", linewidth=0.4)
    axes[1, 0].set_title("ACF of innovations (should be $\\sim 0$)")
    axes[1, 0].set_xlabel("lag")
    axes[1, 0].set_ylabel("autocorrelation")
    axes[1, 0].legend(fontsize=8)

    lb = acorr_ljungbox(innov, lags=[10, 20, 50], return_df=True)
    txt = [f"AR(1) fit: $\\varphi = {phi:.3f}$",
           f"           $\\sigma_\\varepsilon = {sigma_eps:.3f}$",
           "",
           "Ljung--Box (innovations white if $p > 0.05$):"]
    for lag in [10, 20, 50]:
        p = float(lb.loc[lag, "lb_pvalue"])
        verdict = "white" if p > 0.05 else "not white"
        txt.append(f"  lag {lag}: $p = {p:.3g}$ $\\Rightarrow$ {verdict}")
    axes[1, 1].axis("off")
    axes[1, 1].text(0.04, 0.95, "\n".join(txt),
                    transform=axes[1, 1].transAxes, fontsize=10,
                    verticalalignment="top", family="serif")
    axes[1, 1].set_title("Whitening test summary")

    fig.suptitle("February 2015: AR(1) innovation diagnostics", y=1.00)
    fig.tight_layout()
    _save(fig, "noise_ar1_innov")


def fig_noise_psd_cmp():
    """Empirical PSD vs. AR(p) theoretical."""
    print("[14] noise_psd_cmp")
    from scipy import signal as scsignal
    from statsmodels.tsa.ar_model import AutoReg

    df = pd.read_csv(DATA_GEN / "data" / "2015_months_DebitDoseA.txt")
    col = df.columns[0]
    series = pd.to_numeric(df[col], errors="coerce").interpolate(
        "linear", limit_direction="both").values
    w = 480
    kernel = np.ones(w) / w
    baseline = np.convolve(series, kernel, mode="valid")
    pad_l = w // 2
    pad_r = len(series) - len(baseline) - pad_l
    baseline = np.concatenate([np.full(pad_l, baseline[0]), baseline,
                               np.full(pad_r, baseline[-1])])
    residual = series - baseline
    s = pd.Series(residual)
    mad = s.rolling(w, center=True, min_periods=1).apply(
        lambda v: np.median(np.abs(v - np.median(v))), raw=True).values
    sigma_loc = 1.4826 * np.maximum(mad, 1e-6)
    mask = np.abs(residual / sigma_loc) <= 4.0
    clean = residual[mask][w:-w]

    freqs, psd_emp = scsignal.periodogram(clean, fs=1.0)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.loglog(freqs[1:], psd_emp[1:], color="grey", alpha=0.5, linewidth=0.4,
              label="empirical periodogram")
    smoothed = pd.Series(psd_emp[1:]).rolling(50, min_periods=1).mean()
    ax.loglog(freqs[1:], smoothed, color="#2a6e9a", linewidth=1.4,
              label="empirical (smoothed)")

    colours = ["#d62728", "#2ca02c", "#9467bd"]
    for p, c in zip([1, 2, 5], colours):
        model = AutoReg(clean, lags=p, old_names=False).fit()
        phi = np.array(model.params[1:p + 1])
        sigma_eps = float(np.std(model.resid, ddof=1))
        omega = 2 * np.pi * freqs[1:]
        denom = np.ones_like(omega, dtype=complex)
        for k, phi_k in enumerate(phi, start=1):
            denom = denom - phi_k * np.exp(-1j * k * omega)
        psd_theory = sigma_eps ** 2 / np.abs(denom) ** 2
        ax.loglog(freqs[1:], psd_theory, color=c, linestyle="--",
                  linewidth=1.4, label=f"AR({p}) theoretical")

    ax.set_xlabel("frequency [1/sample]")
    ax.set_ylabel("PSD")
    ax.set_title("February 2015: empirical PSD vs. AR(p) theoretical fits")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    _save(fig, "noise_psd_cmp")


def fig_noise_stability():
    """phi and sigma_eps stability across the 4 months."""
    print("[15] noise_stability")
    from statsmodels.tsa.ar_model import AutoReg

    df = pd.read_csv(DATA_GEN / "data" / "2015_months_DebitDoseA.txt")
    months = df.columns.tolist()
    phis_per_month, sigmas_per_month = [], []
    for col in months:
        series = pd.to_numeric(df[col], errors="coerce").interpolate(
            "linear", limit_direction="both").values
        w = 480
        kernel = np.ones(w) / w
        baseline = np.convolve(series, kernel, mode="valid")
        pad_l = w // 2
        pad_r = len(series) - len(baseline) - pad_l
        baseline = np.concatenate([np.full(pad_l, baseline[0]), baseline,
                                   np.full(pad_r, baseline[-1])])
        residual = series - baseline
        s = pd.Series(residual)
        mad = s.rolling(w, center=True, min_periods=1).apply(
            lambda v: np.median(np.abs(v - np.median(v))), raw=True).values
        mask = np.abs(residual / (1.4826 * np.maximum(mad, 1e-6))) <= 4.0
        clean = residual[mask][w:-w]

        # Per-quarter
        phis_q, sigs_q = [], []
        n = len(clean)
        for q in range(4):
            chunk = clean[q * (n // 4):(q + 1) * (n // 4) if q < 3 else n]
            if len(chunk) < 500:
                continue
            try:
                m = AutoReg(chunk, lags=1, old_names=False).fit()
                phis_q.append(float(m.params[1]))
                sigs_q.append(float(np.std(m.resid, ddof=1)))
            except Exception:
                pass
        phis_per_month.append(phis_q)
        sigmas_per_month.append(sigs_q)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    month_short = [m.split(" ")[0] for m in months]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for k, (m, phis, sigs) in enumerate(
            zip(month_short, phis_per_month, sigmas_per_month)):
        x = np.arange(len(phis))
        axes[0].plot(x, phis, "o-", color=colors[k], label=m, markersize=6)
        axes[1].plot(x, sigs, "o-", color=colors[k], label=m, markersize=6)
    axes[0].axhline(0.86, color="k", linestyle="--", linewidth=0.7,
                    label="$\\varphi = 0.86$ (generator)")
    axes[1].axhline(1.12, color="k", linestyle="--", linewidth=0.7,
                    label="$\\sigma_\\varepsilon = 1.12$ (generator)")
    axes[0].set_xlabel("quarter index (1 = first $\\sim 10\\text{k}$ samples)")
    axes[0].set_ylabel("AR(1) $\\varphi$")
    axes[0].set_title("Per-quarter AR(1) coefficient stability")
    axes[0].legend(fontsize=8, loc="lower right", ncol=2)
    axes[0].set_ylim(0.80, 0.96)
    axes[1].set_xlabel("quarter index")
    axes[1].set_ylabel("AR(1) $\\sigma_\\varepsilon$")
    axes[1].set_title("Per-quarter innovation std")
    axes[1].set_ylim(1.05, 1.20)
    axes[1].legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    _save(fig, "noise_stability")


# =============================================================================
# HAND LABELS
# =============================================================================
def _baseline_ma(x, w):
    if w <= 1:
        return x.copy()
    s = pd.Series(x).rolling(window=w, center=True, min_periods=1).mean()
    return s.values


def fig_hand_labels_overview():
    print("[16] hand_labels_overview")
    df = pd.read_csv(DATA_GEN / "data" / "2015_months_DebitDoseA.txt")
    with open(DATA_GEN / "data" / "real_event_labels.json") as f:
        labels = json.load(f)
    labels = {k: v for k, v in labels.items() if not k.startswith("_")}

    fig, axes = plt.subplots(len(df.columns), 1, figsize=(10, 8.5),
                             sharey=False)
    for ax, col in zip(axes, df.columns):
        series = pd.to_numeric(df[col], errors="coerce").interpolate(
            "linear", limit_direction="both").values
        ax.plot(series, color="#2a6e9a", linewidth=0.4)
        col_labels = labels.get(col, [])
        for s, e in col_labels:
            ax.axvspan(s, e, color="orange", alpha=0.35)
        ax.set_title(f"{col.split(' ')[0]} -- {len(col_labels)} labels",
                     loc="left")
        ax.set_ylabel("dose")
    axes[-1].set_xlabel("sample index")
    fig.suptitle("Hand-labelled events across the four months of 2015 "
                 "(17 events total)", y=1.00)
    fig.tight_layout()
    _save(fig, "hand_labels_overview")


def fig_hand_labels_zoom():
    print("[17] hand_labels_zoom")
    df = pd.read_csv(DATA_GEN / "data" / "2015_months_DebitDoseA.txt")
    with open(DATA_GEN / "data" / "real_event_labels.json") as f:
        labels_all = json.load(f)
    labels_all = {k: v for k, v in labels_all.items() if not k.startswith("_")}

    col = "22/06/2015 07:31"
    labels = labels_all[col]
    series = pd.to_numeric(df[col], errors="coerce").interpolate(
        "linear", limit_direction="both").values

    pad = 600
    n = len(labels)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.5 * rows),
                             squeeze=False)
    for k, (s, e) in enumerate(labels):
        ax = axes[k // cols][k % cols]
        lo = max(0, s - pad)
        hi = min(len(series), e + pad)
        ax.plot(np.arange(lo, hi), series[lo:hi],
                color="#2a6e9a", linewidth=0.5, label="raw")
        for W, color in [(30, "#d62728"), (120, "#2ca02c")]:
            ax.plot(np.arange(lo, hi), _baseline_ma(series, W)[lo:hi],
                    linewidth=1.3, color=color, label=f"MA-{W}")
        ax.axvspan(s, e, color="orange", alpha=0.30)
        ax.set_title(f"samples {s}-{e} (n={e - s})", fontsize=9)
        if k == 0:
            ax.legend(fontsize=8, loc="upper right")
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"Hand-labelled events in {col.split(' ')[0]} 2015 "
                 f"($\\pm {pad}$-sample context)", y=1.00)
    fig.tight_layout()
    _save(fig, "hand_labels_zoom")


def fig_hand_label_stats():
    print("[18] hand_label_stats")
    df = pd.read_csv(DATA_GEN / "data" / "2015_months_DebitDoseA.txt")
    with open(DATA_GEN / "data" / "real_event_labels.json") as f:
        labels_all = json.load(f)
    labels_all = {k: v for k, v in labels_all.items() if not k.startswith("_")}

    durations, amplitudes = [], []
    for col, lst in labels_all.items():
        series = pd.to_numeric(df[col], errors="coerce").interpolate(
            "linear", limit_direction="both").values
        baseline = _baseline_ma(series, 4320)
        for s, e in lst:
            durations.append(e - s)
            peak = float(np.max(series[s:e] - baseline[s:e]))
            amplitudes.append(peak)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].hist(durations, bins=8, color="#2a6e9a", alpha=0.85, edgecolor="k")
    axes[0].set_title(f"Hand-labelled event durations (n = {len(durations)})")
    axes[0].set_xlabel("duration (samples)")
    axes[0].set_ylabel("count")
    axes[0].axvline(30, color="#d62728", linestyle="--", linewidth=0.9,
                    label="v2 lower bound (30)")
    axes[0].axvline(1000, color="#d62728", linestyle="--", linewidth=0.9,
                    label="v2 upper bound (1000)")
    axes[0].legend(fontsize=8)

    axes[1].hist(amplitudes, bins=8, color="#2a6e9a", alpha=0.85, edgecolor="k")
    axes[1].set_title("Hand-labelled event amplitudes "
                      "(peak above local baseline)")
    axes[1].set_xlabel("amplitude (counts)")
    axes[1].set_ylabel("count")
    axes[1].axvline(15, color="#d62728", linestyle="--", linewidth=0.9,
                    label="v2 lower bound (15)")
    axes[1].axvline(40, color="#d62728", linestyle="--", linewidth=0.9,
                    label="v2 upper bound (40)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Empirical event statistics used to calibrate v2", y=1.02)
    fig.tight_layout()
    _save(fig, "hand_label_stats")


# =============================================================================
# TRAINING CURVES (v1 and v2)
# =============================================================================
def _parse_training_log(path: Path):
    epochs, train_loss, val_loss, val_f1 = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        in_training = False
        for line in f:
            if line.startswith("Epoch    Train"):
                in_training = True
                continue
            if in_training:
                m = re.match(r"\s*(\d+)/\d+\s+([\d.]+)\s+([\d.]+)\s+"
                             r"([\d.]+)\s+([\d.e-]+)\s+", line)
                if m:
                    epochs.append(int(m.group(1)))
                    train_loss.append(float(m.group(2)))
                    val_loss.append(float(m.group(3)))
                    val_f1.append(float(m.group(4)))
                elif line.strip() == "" or line.startswith("Training"):
                    break
    return (np.array(epochs), np.array(train_loss),
            np.array(val_loss), np.array(val_f1))


def _plot_training_block(axes, root_dir: Path, title_prefix: str,
                         f1_ylim=(0.5, 1.0)):
    ax_loss, ax_f1 = axes
    for m in MODELS:
        path = root_dir / "architectures" / "logs" / m / "training_log.txt"
        if not path.exists():
            continue
        e, tl, vl, f1 = _parse_training_log(path)
        c = MODEL_COLOR[m]
        ax_loss.plot(e, tl, color=c, linestyle="--", linewidth=0.9,
                     alpha=0.65)
        ax_loss.plot(e, vl, color=c, linewidth=1.6, label=MODEL_PRETTY[m])
        ax_f1.plot(e, f1, color=c, linewidth=1.6, label=MODEL_PRETTY[m])
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title(f"{title_prefix} -- train (dashed) and val (solid) loss")
    ax_loss.legend(loc="upper right", fontsize=8)
    ax_f1.set_xlabel("epoch")
    ax_f1.set_ylabel("macro F1 (window level)")
    ax_f1.set_title(f"{title_prefix} -- validation F1 score")
    ax_f1.set_ylim(*f1_ylim)
    ax_f1.legend(loc="lower right", fontsize=8)


def fig_v1_training_curves():
    print("[19] v1_training_curves")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    _plot_training_block(axes, OLD_ROOT, "v1 (7 classes)",
                         f1_ylim=(0.55, 0.85))
    fig.tight_layout()
    _save(fig, "v1_training_curves")


def fig_v2_training_curves():
    print("[20] v2_training_curves")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    _plot_training_block(axes, ROOT, "v2 (4 classes)",
                         f1_ylim=(0.6, 0.95))
    fig.tight_layout()
    _save(fig, "v2_training_curves")


def fig_v1_vs_v2_training_curves():
    print("[21] v1_vs_v2_training_curves")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    _plot_training_block(axes[0], OLD_ROOT, "v1 (7 classes)",
                         f1_ylim=(0.55, 0.85))
    _plot_training_block(axes[1], ROOT, "v2 (4 classes)",
                         f1_ylim=(0.6, 0.9))
    fig.suptitle("Training dynamics: v1 (top) vs v2 (bottom)", y=1.00)
    fig.tight_layout()
    _save(fig, "v1_vs_v2_training_curves")


# =============================================================================
# CONFUSION MATRICES (v1 and v2)
# =============================================================================
def _parse_cm(path: Path):
    rows = []
    in_cm = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "Event-Level Confusion Matrix" in line:
                in_cm = True
                continue
            if not in_cm:
                continue
            if "Event-Level Summary" in line:
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("-"):
                if rows:
                    continue
                continue
            tokens = stripped.split()
            if len(tokens) < 2:
                continue
            try:
                counts = [int(x) for x in tokens
                          if re.fullmatch(r"-?\d+", x)]
            except ValueError:
                continue
            if not counts:
                continue
            name = " ".join(t for t in tokens
                            if not re.fullmatch(r"-?\d+", t))
            if not name:
                continue
            rows.append((name, counts))
    if not rows:
        return None, None
    return [r[0] for r in rows], np.array([r[1] for r in rows])


def _plot_cm_row(axes, model_paths, model_titles, row_labels, col_labels,
                  suptitle):
    for ax, m, title in zip(axes, model_paths, model_titles):
        names, matrix = _parse_cm(m)
        if matrix is None:
            ax.set_visible(False)
            continue
        # Truncate / pad to match expected shape
        nr, nc = len(row_labels), len(col_labels)
        matrix = matrix[:nr, :nc]
        row_sums = matrix.sum(axis=1, keepdims=True).astype(float)
        norm = np.where(row_sums > 0, matrix / np.maximum(row_sums, 1), 0.0)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(nc))
        ax.set_yticks(range(nr))
        ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(row_labels, fontsize=8)
        for i in range(min(matrix.shape[0], nr)):
            for j in range(min(matrix.shape[1], nc)):
                v = matrix[i, j]
                ax.text(j, i, f"{v}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black",
                        fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("predicted")
        if ax is axes[0]:
            ax.set_ylabel("true")
    return suptitle


def fig_v1_confusion_matrices():
    print("[22] v1_confusion_matrices")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    rows = ["bell", "bell_sq", "fast_ascend", "fast_descend", "M", "square"]
    cols = rows
    paths = [OLD_ARCH / "logs" / m / "inference_log.txt" for m in MODELS]
    _plot_cm_row(axes, paths, [MODEL_PRETTY[m] for m in MODELS],
                 rows, cols, "v1 event-level confusion matrices")
    fig.suptitle("v1 event-level confusion matrices "
                 "(synthetic test segment, 59 events)", y=1.02)
    fig.tight_layout()
    _save(fig, "v1_confusion_matrices")


def fig_v2_confusion_matrices():
    print("[23] v2_confusion_matrices")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    rows = ["bell", "fast_ascend", "fast_descend"]
    cols = ["Normal", "bell", "fast_ascend", "fast_descend"]
    paths = [ARCH / "logs" / m / "inference_log.txt" for m in MODELS]
    _plot_cm_row(axes, paths, [MODEL_PRETTY[m] for m in MODELS],
                 rows, cols, "v2 event-level confusion matrices")
    fig.suptitle("v2 event-level confusion matrices "
                 "(synthetic test segment, 15 events; "
                 "the ``Normal'' column = missed events)", y=1.02)
    fig.tight_layout()
    _save(fig, "v2_confusion_matrices")


# =============================================================================
# ROBUSTNESS (v1 and v2)
# =============================================================================
def _parse_robustness(path: Path):
    noise_pts, deform_pts = [], []
    section = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "SWEEP 1: BACKGROUND NOISE" in line:
                section = "noise"
                continue
            if "SWEEP 2: ANOMALY SHAPE DEFORMATION" in line:
                section = "deform"
                continue
            if "SUMMARY" in line:
                section = None
                continue
            m = re.match(r"\s*([\d.]+)\s+\d+\s+([\d.]+)%", line)
            if m and section is not None:
                pt = (float(m.group(1)), float(m.group(2)))
                (noise_pts if section == "noise"
                 else deform_pts).append(pt)
    return noise_pts, deform_pts


def _plot_robustness_block(axes, root_dir: Path):
    ax_noise, ax_deform = axes
    for m in MODELS:
        path = root_dir / "architectures" / "logs" / m / "robustness_test_log.txt"
        if not path.exists():
            continue
        noise, deform = _parse_robustness(path)
        if not noise:
            continue
        c = MODEL_COLOR[m]
        x1, y1 = zip(*noise)
        x2, y2 = zip(*deform)
        ax_noise.plot(x1, y1, color=c, marker="o", linewidth=1.6,
                      markersize=4, label=MODEL_PRETTY[m])
        ax_deform.plot(x2, y2, color=c, marker="o", linewidth=1.6,
                       markersize=4, label=MODEL_PRETTY[m])
    ax_noise.axhline(50, color="k", linestyle=":", linewidth=0.8)
    ax_deform.axhline(50, color="k", linestyle=":", linewidth=0.8)
    ax_noise.set_xlabel("noise multiplier")
    ax_noise.set_ylabel("event-level macro F1 (\\%)")
    ax_noise.set_title("Background noise sweep")
    ax_noise.set_ylim(0, 105)
    ax_noise.legend(loc="lower left", fontsize=8)
    ax_deform.set_xlabel("deformation multiplier")
    ax_deform.set_ylabel("event-level macro F1 (\\%)")
    ax_deform.set_title("Shape-deformation sweep")
    ax_deform.set_ylim(0, 105)
    ax_deform.legend(loc="lower left", fontsize=8)


def fig_v1_robustness():
    print("[24] v1_robustness")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    _plot_robustness_block(axes, OLD_ROOT)
    fig.suptitle("v1 robustness study (six classes, white-noise generator)",
                 y=1.02)
    fig.tight_layout()
    _save(fig, "v1_robustness")


def fig_v2_robustness():
    print("[25] v2_robustness")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    _plot_robustness_block(axes, ROOT)
    fig.suptitle("v2 robustness study (three classes, AR(1) generator)",
                 y=1.02)
    fig.tight_layout()
    _save(fig, "v2_robustness")


def fig_v1_vs_v2_robustness():
    print("[26] v1_vs_v2_robustness")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    _plot_robustness_block(axes[0], OLD_ROOT)
    _plot_robustness_block(axes[1], ROOT)
    axes[0, 0].set_title("v1 -- background noise sweep")
    axes[0, 1].set_title("v1 -- shape-deformation sweep")
    axes[1, 0].set_title("v2 -- background noise sweep")
    axes[1, 1].set_title("v2 -- shape-deformation sweep")
    fig.tight_layout()
    _save(fig, "v1_vs_v2_robustness")


# =============================================================================
# REAL DATA INFERENCE (v1 catastrophe, v2 recovery)
# =============================================================================
def _run_inference(arch_root: Path, data_gen_root: Path, models, n_show=10_000):
    """Returns dict: model -> (series, smooth probs (T x C), class_names)"""
    import torch
    sys.path.insert(0, str(arch_root))
    if "model_selection" in sys.modules:
        del sys.modules["model_selection"]
    if "inference_engine" in sys.modules:
        del sys.modules["inference_engine"]
    from model_selection import get_model  # type: ignore
    from inference_engine import run_sliding_window_inference  # type: ignore

    cm_path = data_gen_root / "data" / "class_mapping.txt"
    class_names = {}
    with open(cm_path) as f:
        for line in f:
            if ":" in line:
                c, n = line.split(":")
                class_names[int(c.strip())] = n.strip()
    num_classes = len(class_names)

    df = pd.read_csv(DATA_GEN / "data" / "2015_months_DebitDoseA.txt")
    col = df.columns[0]
    series = pd.to_numeric(df[col], errors="coerce").interpolate(
        "linear", limit_direction="both").values.astype(np.float32)
    series_show = series[:n_show]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for model_name in models:
        model_path = (arch_root / "saved_models" / model_name
                      / f"best_{model_name}.pth")
        stats_path = (arch_root / "saved_models" / model_name
                      / "normalization_stats.json")
        if not model_path.exists():
            print(f"  missing {model_path}, skipping")
            continue
        model = get_model(model_name, num_classes=num_classes,
                          in_channels=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device,
                                          weights_only=True))
        model.eval()
        with open(stats_path) as f:
            train_std = json.load(f)["std"]
        probs, _, _, _ = run_sliding_window_inference(
            series_show, model, device, train_std, num_classes,
            window_size=512, stride=10, batch_size=256)
        smoothing = 30
        smooth = np.zeros_like(probs)
        for c in range(num_classes):
            s = pd.Series(probs[:, c]).interpolate(
                method="linear", limit_direction="both")
            smooth[:, c] = s.rolling(window=smoothing, center=True,
                                     min_periods=1).mean().values
        out[model_name] = (series_show, smooth, class_names)
    sys.path.pop(0)
    return out


def _plot_real_inference_block(axes, results, class_colors, n_show, version):
    t = np.arange(n_show)
    for ax, (model_name, (series, smooth, class_names)) in zip(axes,
                                                                 results.items()):
        ax.plot(t, series, color="#2a6e9a", linewidth=0.5, alpha=0.75)
        ax.set_ylabel("dose")
        ax.set_ylim(np.percentile(series, 1) - 2,
                    np.percentile(series, 99.5) + 2)
        axR = ax.twinx()
        for c in sorted(class_names.keys()):
            color = class_colors.get(c, "#888")
            lw = 1.4 if c > 0 else 0.9
            alpha = 0.95 if c > 0 else 0.45
            axR.plot(t, smooth[:, c] * 100, color=color, linewidth=lw,
                     alpha=alpha, label=class_names[c])
        axR.set_ylim(0, 105)
        axR.set_ylabel("class confidence (\\%)")
        axR.grid(False)
        ax.set_title(f"{MODEL_PRETTY[model_name]} -- {version} inference",
                     loc="left")


def fig_v1_real_inference():
    print("[27] v1_real_inference")
    n_show = 10_000
    res = _run_inference(OLD_ARCH, OLD_DATA_GEN, MODELS, n_show=n_show)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.0), sharex=True)
    _plot_real_inference_block(axes, res, V1_CLASS_COLORS, n_show, "v1")
    handles = [plt.Line2D([0], [0], color=V1_CLASS_COLORS[c],
                          label=V1_CLASS_NAMES[c])
               for c in range(7)]
    fig.legend(handles=handles, loc="upper center", ncol=7, fontsize=8,
               bbox_to_anchor=(0.5, 1.02))
    axes[-1].set_xlabel("sample (February 2015)")
    fig.suptitle("v1 inference on real 2015 data "
                 "(10k snapshot of February): all three models fail",
                 y=1.04)
    fig.tight_layout()
    _save(fig, "v1_real_inference")


def fig_v2_real_inference():
    print("[28] v2_real_inference")
    n_show = 10_000
    res = _run_inference(ARCH, DATA_GEN, MODELS, n_show=n_show)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.0), sharex=True)
    _plot_real_inference_block(axes, res, V2_CLASS_COLORS, n_show, "v2")
    handles = [plt.Line2D([0], [0], color=V2_CLASS_COLORS[c],
                          label=V2_CLASS_NAMES[c])
               for c in range(4)]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 1.02))
    axes[-1].set_xlabel("sample (February 2015)")
    fig.suptitle("v2 inference on real 2015 data "
                 "(10k snapshot of February): models sit on background "
                 "and fire only on labelled event",
                 y=1.04)
    fig.tight_layout()
    _save(fig, "v2_real_inference")


def fig_v1_vs_v2_real_inference_resnet():
    print("[29] v1_vs_v2_real_inference_resnet")
    n_show = 10_000
    v1 = _run_inference(OLD_ARCH, OLD_DATA_GEN, ["resnet"], n_show=n_show)
    v2 = _run_inference(ARCH, DATA_GEN, ["resnet"], n_show=n_show)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    _plot_real_inference_block([axes[0]], v1, V1_CLASS_COLORS, n_show, "v1")
    _plot_real_inference_block([axes[1]], v2, V2_CLASS_COLORS, n_show, "v2")
    axes[-1].set_xlabel("sample (February 2015)")
    fig.suptitle("ResNet 1D on real 2015 data: v1 (false-positive plateaus) "
                 "vs v2 (silent on background, fires on event)", y=1.00)
    fig.tight_layout()
    _save(fig, "v1_vs_v2_real_inference_resnet")


# =============================================================================
# FAILURE ANALYSIS (class prior shift)
# =============================================================================
def fig_failure_analysis():
    print("[30] failure_analysis")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    # v1 class distribution from old training log
    v1_counts = [77674, 30218, 28106, 27004, 28769, 30963, 27251]
    v1_pct = np.array(v1_counts) / sum(v1_counts) * 100
    v1_labels = ["Normal", "bell", "bell_sq", "fast_ascend",
                 "fast_descend", "M", "square"]
    # v2 class distribution from new training log
    v2_counts = [229481, 7348, 6668, 6488]
    v2_pct = np.array(v2_counts) / sum(v2_counts) * 100
    v2_labels = ["Normal", "bell", "fast_ascend", "fast_descend"]
    # real-data estimate: 17 labelled events over ~160k samples
    # window-level normal proportion ~ 98-99%
    real_norm = 99.0
    real_anom = 1.0

    x = np.arange(len(v1_labels))
    w = 0.4
    ax1.bar(x - w/2, v1_pct, w, label="v1 training (7 classes)",
            color="#1f77b4")
    # Overlay v2 (different number of classes) -> pad zeros for missing
    v2_pad = [v2_pct[0], v2_pct[1], 0.0, v2_pct[2], v2_pct[3], 0.0, 0.0]
    ax1.bar(x + w/2, v2_pad, w, label="v2 training (4 classes)",
            color="#2ca02c")
    ax1.axhline(real_norm, color="#d62728", linestyle="--", linewidth=1.0,
                label="real Normal $\\sim 99\\%$")
    ax1.set_xticks(x)
    ax1.set_xticklabels(v1_labels, rotation=35, ha="right")
    ax1.set_ylabel("share of windows (\\%)")
    ax1.set_title("Training class prior: v1 vs v2 vs real deployment")
    ax1.legend(fontsize=8)

    # Class weights bar
    v1_weights = [0.46, 1.18, 1.27, 1.32, 1.24, 1.15, 1.31]
    v2_weights_padded = [0.27, 8.51, 0.0, 9.37, 9.63, 0.0, 0.0]
    ax2.bar(x - w/2, v1_weights, w, label="v1", color="#1f77b4")
    ax2.bar(x + w/2, v2_weights_padded, w, label="v2", color="#2ca02c")
    ax2.axhline(1.0, color="k", linewidth=0.6, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(v1_labels, rotation=35, ha="right")
    ax2.set_ylabel("cross-entropy weight")
    ax2.set_title("Class weights used at training")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "failure_analysis")


# =============================================================================
# MODEL COST (v2)
# =============================================================================
def fig_model_cost():
    print("[31] model_cost")
    # v2 numbers from training logs
    params = {"1d_cnn": 22_628, "resnet": 74_532, "cnn_attention": 109_460}
    train_minutes = {"1d_cnn": 3.81, "resnet": 7.36, "cnn_attention": 7.91}
    # event-level macro F1 from inference logs (15-event segment)
    f1_event = {"1d_cnn": 86.15, "resnet": 86.15, "cnn_attention": 86.15}

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    xs = np.arange(3)
    names = [MODEL_PRETTY[m] for m in MODELS]
    colors = [MODEL_COLOR[m] for m in MODELS]

    axes[0].bar(xs, [params[m] / 1000 for m in MODELS], color=colors)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(names, rotation=15)
    axes[0].set_ylabel("trainable parameters (k)")
    axes[0].set_title("Model size")

    axes[1].bar(xs, [train_minutes[m] for m in MODELS], color=colors)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(names, rotation=15)
    axes[1].set_ylabel("training time (min)")
    axes[1].set_title("Training cost (35 epochs max, RTX 5060)")

    axes[2].bar(xs, [f1_event[m] for m in MODELS], color=colors)
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(names, rotation=15)
    axes[2].set_ylabel("event-level macro F1 (\\%)")
    axes[2].set_title("v2 synthetic test performance")
    axes[2].set_ylim(70, 100)
    fig.tight_layout()
    _save(fig, "model_cost")


# =============================================================================
# SLIDING-WINDOW ILLUSTRATION (kept; works for both v1 and v2)
# =============================================================================
def fig_sliding_window():
    print("[32] sliding_window")
    path = ARCH / "data" / "training_dataset.csv"
    df = pd.read_csv(path, nrows=5000)
    g = df["gamma_dose"].values
    lbl = df["is_anomaly"].values.astype(int)
    t = np.arange(len(g))

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(t, g, color="#2a6e9a", linewidth=0.5)
    win = 512
    stride = 256
    y_base = g.min() - 1
    for i, start in enumerate(range(0, len(g) - win + 1, stride)):
        col = ("#d62728" if lbl[start:start + win].sum() > 0.1 * win
               else "#888888")
        ax.add_patch(plt.matplotlib.patches.Rectangle(
            (start, y_base + i * 0.4), win, 0.3,
            facecolor=col, alpha=0.55, edgecolor="none"))
    ax.set_xlim(0, len(g))
    ax.set_ylim(y_base, g.max() + 1)
    ax.set_xlabel("sample index")
    ax.set_ylabel("dose")
    ax.set_title("Sliding-window construction "
                 "(red = window contains $\\geq 10\\%$ anomaly samples)")
    fig.tight_layout()
    _save(fig, "sliding_window")


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    print("=== Real-data figures ===")
    fig_real_data_overview()
    fig_real_data_stats()
    fig_real_data_decomposition()

    print("\n=== Anomaly templates ===")
    fig_v1_anomaly_templates()
    fig_v2_anomaly_templates()
    fig_v1_template_deformation()
    fig_v2_template_deformation()

    print("\n=== Synthetic previews ===")
    fig_v1_synthetic_preview()
    fig_v2_synthetic_preview()
    fig_v1_vs_v2_synthetic()
    fig_noise_texture_comparison()

    print("\n=== Noise reanalysis ===")
    fig_noise_clean_residual()
    fig_noise_ar1_innov()
    fig_noise_psd_cmp()
    fig_noise_stability()

    print("\n=== Hand labels ===")
    fig_hand_labels_overview()
    fig_hand_labels_zoom()
    fig_hand_label_stats()

    print("\n=== Training curves ===")
    fig_v1_training_curves()
    fig_v2_training_curves()

    print("\n=== Confusion matrices ===")
    fig_v1_confusion_matrices()
    fig_v2_confusion_matrices()

    print("\n=== Robustness ===")
    fig_v1_robustness()
    fig_v2_robustness()
    fig_v1_vs_v2_robustness()

    print("\n=== Real-data inference ===")
    fig_v1_real_inference()
    fig_v2_real_inference()
    fig_v1_vs_v2_real_inference_resnet()

    print("\n=== Misc ===")
    fig_failure_analysis()
    fig_model_cost()
    fig_sliding_window()

    print("\nDone. Figures in:", OUT)


if __name__ == "__main__":
    main()
