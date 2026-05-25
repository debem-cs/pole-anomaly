"""
Visualization helper for hand-labelling real anomalies.

Produces two outputs per month under
``data-gen/logs/labeling/``:

1. A static PNG with several stacked panels:
     - raw signal + user labels
     - five low-pass filtered views (W = 15, 30, 60, 120, 240)
   Smoothing concentrates the signal of sustained events and washes
   out AR(1) noise, making subtle bumps easier to spot.

2. An interactive HTML (plotly) with all curves overlaid on the same
   axis, supporting box-zoom, pan, and hover, so you can drill into
   any region and refine your labels.

Sample indices are in the **original** (un-trimmed) sample index of
``2015_months_DebitDoseA.txt`` --- the same index used in
``data/real_event_labels.json``.
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "2015_months_DebitDoseA.txt")
LABELS_PATH = os.path.join(SCRIPT_DIR, "data", "real_event_labels.json")
OUT_DIR = os.path.join(SCRIPT_DIR, "logs", "labeling")
os.makedirs(OUT_DIR, exist_ok=True)

SMOOTHING_WINDOWS = [15, 30, 60, 120, 240]
DETREND_WINDOW = 4320  # ~3 days, slow baseline subtracted before smoothing


def centred_moving_average(x, w):
    if w <= 1:
        return x.copy()
    s = pd.Series(x).rolling(window=w, center=True, min_periods=1).mean()
    return s.values


def shade_labels(ax, labels):
    for s, e in labels:
        ax.axvspan(s, e, color="orange", alpha=0.35, zorder=0)


def robust_std(x):
    """MAD-based standard deviation, robust to outliers/events."""
    x = np.asarray(x)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def plot_month_static(name, series, labels, out_path):
    n_panels = 1 + len(SMOOTHING_WINDOWS)
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(16, 1.8 * n_panels),
                             sharex=True)
    t = np.arange(len(series))

    ax0 = axes[0]
    ax0.plot(t, series, color="#1f77b4", linewidth=0.4)
    shade_labels(ax0, labels)
    ax0.set_ylabel("raw")
    ax0.set_title(f"{name} - raw signal + {len(labels)} hand labels (orange)")

    for ax, W in zip(axes[1:], SMOOTHING_WINDOWS):
        sm = centred_moving_average(series, W)
        ax.plot(t, sm, color="#1f77b4", linewidth=0.7)
        shade_labels(ax, labels)
        ax.set_ylabel(f"MA-{W}")
        ax.set_title(f"low-pass (window = {W} samples = "
                     f"{W/60:.1f} hours)", fontsize=9)

    axes[-1].set_xlabel("sample index (original, un-trimmed)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def plot_month_interactive(name, series, labels, out_path):
    """Same content as static, but plotly so you can zoom in."""
    n_panels = 1 + len(SMOOTHING_WINDOWS)
    fig = make_subplots(
        rows=n_panels, cols=1, shared_xaxes=True,
        vertical_spacing=0.015,
        subplot_titles=["raw"] + [f"MA-{W} ({W/60:.1f} h)"
                                   for W in SMOOTHING_WINDOWS],
    )
    t = np.arange(len(series))

    fig.add_trace(
        go.Scatter(x=t, y=series, mode="lines",
                   line=dict(color="#1f77b4", width=0.6),
                   name="raw", hoverinfo="x+y"),
        row=1, col=1,
    )
    for k, W in enumerate(SMOOTHING_WINDOWS, start=2):
        sm = centred_moving_average(series, W)
        fig.add_trace(
            go.Scatter(x=t, y=sm, mode="lines",
                       line=dict(color="#1f77b4", width=0.9),
                       name=f"MA-{W}", hoverinfo="x+y",
                       showlegend=False),
            row=k, col=1,
        )

    for row in range(1, n_panels + 1):
        for s, e in labels:
            fig.add_vrect(
                x0=s, x1=e,
                fillcolor="orange", opacity=0.25,
                line_width=0, layer="below",
                row=row, col=1,
            )

    fig.update_layout(
        title=(f"{name} - hand-labelling view "
               f"({len(labels)} current labels)"),
        height=200 + 180 * n_panels,
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(title_text="sample index (original)",
                     row=n_panels, col=1)
    fig.write_html(out_path, include_plotlyjs="cdn")


def plot_month_static_detrended(name, series, labels, out_path):
    """Same as plot_month_static, but the slow baseline is removed first
    so all panels are centred on zero. Each panel also shows the
    +/-2*sigma noise band (robust MAD-based estimate)."""
    baseline = centred_moving_average(series, DETREND_WINDOW)
    detrended = series - baseline

    n_panels = 1 + len(SMOOTHING_WINDOWS)
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(16, 1.8 * n_panels),
                             sharex=True)
    t = np.arange(len(series))

    def add_sigma_band(ax, series_panel):
        sigma = robust_std(series_panel)
        for sign in (+1, -1):
            ax.axhline(sign * 2 * sigma, color="#d62728",
                       linestyle="--", linewidth=0.8,
                       label=(f"+/-2 sigma = +/-{2*sigma:.2f}"
                              if sign == +1 else None))
        ax.legend(fontsize=8, loc="upper right")
        return sigma

    ax0 = axes[0]
    ax0.plot(t, detrended, color="#1f77b4", linewidth=0.4)
    ax0.axhline(0, color="black", linewidth=0.5)
    shade_labels(ax0, labels)
    sigma0 = add_sigma_band(ax0, detrended)
    ax0.set_ylabel("raw - MA")
    ax0.set_title(f"{name} - detrended (raw minus MA-{DETREND_WINDOW}) "
                  f"+ {len(labels)} hand labels (orange)  "
                  f"[robust sigma = {sigma0:.2f}]")

    for ax, W in zip(axes[1:], SMOOTHING_WINDOWS):
        sm = centred_moving_average(detrended, W)
        ax.plot(t, sm, color="#1f77b4", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        shade_labels(ax, labels)
        sigma_W = add_sigma_band(ax, sm)
        ax.set_ylabel(f"MA-{W}")
        ax.set_title(f"detrended + low-pass (window = {W} samples = "
                     f"{W/60:.1f} hours)  "
                     f"[robust sigma = {sigma_W:.3f}]", fontsize=9)

    axes[-1].set_xlabel("sample index (original, un-trimmed)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def plot_month_interactive_detrended(name, series, labels, out_path):
    baseline = centred_moving_average(series, DETREND_WINDOW)
    detrended = series - baseline

    # Pre-compute smoothed signals and robust sigmas per panel
    series_per_panel = [detrended] + [
        centred_moving_average(detrended, W) for W in SMOOTHING_WINDOWS
    ]
    sigmas = [robust_std(s) for s in series_per_panel]

    n_panels = 1 + len(SMOOTHING_WINDOWS)
    subplot_titles = [
        f"detrended (raw - MA-{DETREND_WINDOW})  [sigma={sigmas[0]:.2f}]"
    ] + [
        f"detrended + MA-{W} ({W/60:.1f} h)  [sigma={s:.3f}]"
        for W, s in zip(SMOOTHING_WINDOWS, sigmas[1:])
    ]
    fig = make_subplots(
        rows=n_panels, cols=1, shared_xaxes=True,
        vertical_spacing=0.015,
        subplot_titles=subplot_titles,
    )
    t = np.arange(len(series))

    for k, (s_panel, sigma) in enumerate(zip(series_per_panel, sigmas),
                                          start=1):
        fig.add_trace(
            go.Scatter(x=t, y=s_panel, mode="lines",
                       line=dict(color="#1f77b4", width=0.8),
                       hoverinfo="x+y", showlegend=False),
            row=k, col=1,
        )

    for row, sigma in enumerate(sigmas, start=1):
        fig.add_hline(y=0, line=dict(color="black", width=1),
                      row=row, col=1)
        for sign in (+1, -1):
            fig.add_hline(
                y=sign * 2 * sigma,
                line=dict(color="#d62728", width=1, dash="dash"),
                row=row, col=1,
                annotation_text=(f"+2sigma={2*sigma:.2f}" if sign == +1
                                 else f"-2sigma={2*sigma:.2f}"),
                annotation_position="top right" if sign == +1 else "bottom right",
                annotation_font=dict(size=9, color="#d62728"),
            )
        for s, e in labels:
            fig.add_vrect(
                x0=s, x1=e,
                fillcolor="orange", opacity=0.25,
                line_width=0, layer="below",
                row=row, col=1,
            )

    fig.update_layout(
        title=(f"{name} - detrended view "
               f"({len(labels)} current labels)"),
        height=200 + 180 * n_panels,
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(title_text="sample index (original)",
                     row=n_panels, col=1)
    fig.write_html(out_path, include_plotlyjs="cdn")


def plot_zoomed_around_labels(name, series, labels, out_path,
                              pad_samples=600):
    """One subplot per labelled event, zoomed in."""
    if not labels:
        return
    n = len(labels)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5 * rows),
                             squeeze=False)
    for k, (s, e) in enumerate(labels):
        ax = axes[k // cols][k % cols]
        lo = max(0, s - pad_samples)
        hi = min(len(series), e + pad_samples)
        ax.plot(np.arange(lo, hi), series[lo:hi],
                color="#1f77b4", linewidth=0.5, label="raw")
        for W in [30, 120]:
            sm = centred_moving_average(series, W)
            ax.plot(np.arange(lo, hi), sm[lo:hi],
                    linewidth=1.2,
                    label=f"MA-{W}")
        ax.axvspan(s, e, color="orange", alpha=0.3)
        ax.set_title(f"sample {s}-{e} (n={e - s})", fontsize=10)
        if k == 0:
            ax.legend(fontsize=8, loc="upper right")
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"{name} - zoomed view around each label "
                 f"(+/- {pad_samples} samples)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    labels = {k: v for k, v in labels.items() if not k.startswith("_")}

    for col in df.columns:
        col_labels = labels.get(col, [])
        series = pd.to_numeric(df[col], errors="coerce")\
            .interpolate(method="linear", limit_direction="both")\
            .values.astype(np.float64)

        safe = col.replace("/", "_").replace(" ", "_").replace(":", "")
        static_path = os.path.join(OUT_DIR, f"labeling_{safe}.png")
        html_path = os.path.join(OUT_DIR, f"labeling_{safe}.html")
        static_dt_path = os.path.join(OUT_DIR, f"labeling_detrended_{safe}.png")
        html_dt_path = os.path.join(OUT_DIR, f"labeling_detrended_{safe}.html")
        zoom_path = os.path.join(OUT_DIR, f"labeling_zoomed_{safe}.png")

        print(f"\n=== {col} ===  ({len(col_labels)} current labels)")
        plot_month_static(col, series, col_labels, static_path)
        print(f"  static (raw)        -> {static_path}")
        plot_month_interactive(col, series, col_labels, html_path)
        print(f"  html (raw)          -> {html_path}")
        plot_month_static_detrended(col, series, col_labels, static_dt_path)
        print(f"  static (detrended)  -> {static_dt_path}")
        plot_month_interactive_detrended(col, series, col_labels, html_dt_path)
        print(f"  html (detrended)    -> {html_dt_path}")
        plot_zoomed_around_labels(col, series, col_labels, zoom_path)
        print(f"  zoomed              -> {zoom_path}")

    print(f"\nAll outputs in {OUT_DIR}")
    print("\nTo refine labels:")
    print(f"  edit {LABELS_PATH}")
    print("  then re-run this script.")


if __name__ == "__main__":
    main()
