"""
Interactive Real Data Explorer — Labeling Helper
=================================================
This script loads the real 2015 gamma dose data and generates a highly interactive
Plotly HTML file. The purpose is to help you visually identify anomalies in the
real data and measure their properties (amplitude, period) so you can tune the
CONFIG parameters in `generate_custom_dataset.py`.

Usage:
    py interactive_labeling_helper.py

Output:
    data-gen/logs/real_data_explorer.html  — Open in browser, hover over peaks to
    read their index and value, then compute:
      - Period  = end_index - start_index
      - Amplitude = peak_value - baseline_value (shown in orange)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    root_dir = script_dir
    data_path = os.path.join(root_dir, 'data', '2015_months_DebitDoseA.txt')
    logs_dir = os.path.join(root_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # --- Configuration ---
    BASELINE_WINDOW = 480   # Moving average window for baseline estimation
    DEVIATION_THRESHOLD = 2  # Number of std deviations above baseline to flag as "suspicious"

    print("=" * 60)
    print("REAL DATA EXPLORER — LABELING HELPER")
    print("=" * 60)

    # 1. Load Real Data
    print("\nLoading real data...")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    columns = df.columns.tolist()
    print(f"Found {len(columns)} time series columns: {columns}")

    # 2. Create Plotly Figure with one subplot per column
    fig = make_subplots(
        rows=len(columns), cols=1,
        shared_xaxes=False,
        vertical_spacing=0.04,
        subplot_titles=[f"Time Period: {col}" for col in columns]
    )

    stats_report = []

    for col_idx, col_name in enumerate(columns):
        print(f"\nProcessing column {col_idx+1}/{len(columns)}: {col_name}")

        # Parse numeric values (handle corrupted entries like "97..3123")
        series = pd.to_numeric(df[col_name], errors='coerce')
        series = series.interpolate(method='linear', limit_direction='both')
        values = series.values.astype(np.float64)
        n = len(values)
        indices = np.arange(n)

        # Compute moving average baseline
        kernel = np.ones(BASELINE_WINDOW) / BASELINE_WINDOW
        baseline_valid = np.convolve(values, kernel, mode='valid')
        # Pad to match original length
        pad_left = BASELINE_WINDOW // 2
        pad_right = n - len(baseline_valid) - pad_left
        baseline = np.concatenate([
            np.full(pad_left, baseline_valid[0]),
            baseline_valid,
            np.full(pad_right, baseline_valid[-1])
        ])

        # Compute residuals and noise statistics
        residuals = values - baseline
        noise_std = np.std(residuals)
        global_mean = np.mean(values)

        # Identify suspicious points (above threshold)
        suspicious_mask = residuals > (DEVIATION_THRESHOLD * noise_std)
        suspicious_indices = indices[suspicious_mask]
        suspicious_values = values[suspicious_mask]

        # Identify suspicious "events" (contiguous regions)
        events = []
        if len(suspicious_indices) > 0:
            event_start = suspicious_indices[0]
            prev_idx = suspicious_indices[0]
            for i in range(1, len(suspicious_indices)):
                if suspicious_indices[i] - prev_idx > 5:  # gap of 5+ points = new event
                    events.append((event_start, prev_idx))
                    event_start = suspicious_indices[i]
                prev_idx = suspicious_indices[i]
            events.append((event_start, prev_idx))

        stats_report.append({
            'column': col_name,
            'mean': global_mean,
            'noise_std': noise_std,
            'num_suspicious_events': len(events)
        })

        print(f"  Global Mean: {global_mean:.2f}")
        print(f"  HF Noise Std: {noise_std:.2f}")
        print(f"  Suspicious events (>{DEVIATION_THRESHOLD} sigma): {len(events)}")

        # Build hover text showing index, raw value, baseline value, and deviation
        hover_texts = [
            f"Index: {i}<br>Raw: {values[i]:.2f}<br>Baseline: {baseline[i]:.2f}<br>Deviation: {residuals[i]:.2f} ({residuals[i]/noise_std:.1f}σ)"
            for i in range(n)
        ]

        # Raw signal trace
        fig.add_trace(
            go.Scatter(
                x=indices, y=values, mode='lines',
                name=f'{col_name} Raw' if col_idx == 0 else f'{col_name} Raw',
                line=dict(color='teal', width=1),
                text=hover_texts, hoverinfo='text',
                showlegend=(col_idx == 0),
                legendgroup='raw'
            ),
            row=col_idx+1, col=1
        )

        # Baseline trace
        fig.add_trace(
            go.Scatter(
                x=indices, y=baseline, mode='lines',
                name='Baseline (moving avg)' if col_idx == 0 else 'Baseline',
                line=dict(color='orange', width=2, dash='dash'),
                hoverinfo='skip',
                showlegend=(col_idx == 0),
                legendgroup='baseline'
            ),
            row=col_idx+1, col=1
        )

        # Threshold band (baseline + 3σ)
        upper_band = baseline + DEVIATION_THRESHOLD * noise_std
        fig.add_trace(
            go.Scatter(
                x=indices, y=upper_band, mode='lines',
                name=f'{DEVIATION_THRESHOLD}σ threshold' if col_idx == 0 else f'{DEVIATION_THRESHOLD}σ',
                line=dict(color='red', width=1, dash='dot'),
                hoverinfo='skip',
                showlegend=(col_idx == 0),
                legendgroup='threshold'
            ),
            row=col_idx+1, col=1
        )

        # Suspicious points highlighted
        if len(suspicious_indices) > 0:
            suspicious_hover = [
                f"⚠ SUSPICIOUS<br>Index: {i}<br>Raw: {values[i]:.2f}<br>Baseline: {baseline[i]:.2f}<br>Amplitude above baseline: {residuals[i]:.2f}<br>Deviation: {residuals[i]/noise_std:.1f}σ"
                for i in suspicious_indices
            ]
            fig.add_trace(
                go.Scatter(
                    x=suspicious_indices, y=suspicious_values, mode='markers',
                    name=f'Suspicious points' if col_idx == 0 else 'Suspicious',
                    marker=dict(color='magenta', size=4, symbol='circle'),
                    text=suspicious_hover, hoverinfo='text',
                    showlegend=(col_idx == 0),
                    legendgroup='suspicious'
                ),
                row=col_idx+1, col=1
            )

        # Annotate events with their measured properties
        for evt_start, evt_end in events:
            evt_slice = values[evt_start:evt_end+1]
            evt_baseline = baseline[evt_start:evt_end+1]
            peak_amplitude = np.max(evt_slice) - np.mean(evt_baseline)
            period = evt_end - evt_start + 1

            fig.add_annotation(
                x=(evt_start + evt_end) / 2,
                y=np.max(evt_slice) + noise_std * 0.5,
                text=f"A={peak_amplitude:.1f} P={period}",
                showarrow=False,
                font=dict(size=8, color='yellow'),
                row=col_idx+1, col=1
            )

        fig.update_yaxes(title_text="Gamma Dose", row=col_idx+1, col=1)
        fig.update_xaxes(title_text="Time Step Index", row=col_idx+1, col=1)

    fig.update_layout(
        title='Real Data Explorer — Hover over peaks to measure anomaly properties',
        height=500 * len(columns),
        template='plotly_dark',
        hovermode='closest',
        legend=dict(yanchor="top", y=1.0, xanchor="right", x=1.05)
    )

    output_path = os.path.join(logs_dir, 'real_data_explorer.html')
    fig.write_html(output_path)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    for s in stats_report:
        print(f"  {s['column']}: mean={s['mean']:.2f}, noise_std={s['noise_std']:.2f}, suspicious_events={s['num_suspicious_events']}")
    print(f"\nExplorer saved to: {output_path}")
    print(f"Open it in your browser and hover over peaks to read amplitude and period values.")
    print(f"Then update CONFIG in generate_custom_dataset.py with the measured values.")

if __name__ == "__main__":
    main()
