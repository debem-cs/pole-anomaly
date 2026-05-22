import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch.nn.functional as F

from model_selection import MODEL_DISPLAY_NAMES, get_model, select_model


def main():
    model_name = select_model()
    display_name = MODEL_DISPLAY_NAMES[model_name]

    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    data_path = os.path.join(script_dir, '..', 'data-gen', 'data', '2015_months_DebitDoseA.txt')
    model_path = os.path.join(script_dir, 'saved_models', model_name, f'best_{model_name}.pth')
    stats_path = os.path.join(script_dir, 'saved_models', model_name, 'normalization_stats.json')
    logs_dir = os.path.join(script_dir, 'logs', model_name)
    os.makedirs(logs_dir, exist_ok=True)

    print("=" * 60)
    print(f"{display_name} REAL DATA INFERENCE")
    print("=" * 60)

    # 1. Load Data
    print("\nLoading data...")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Dataset loaded. Shape: {df.shape}")
    columns = df.columns.tolist()
    print(f"Time series columns found: {columns}")

    # Read num_classes from class_mapping.txt (authoritative source)
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
        print("ERROR: class_mapping.txt not found. Cannot determine number of classes.")
        return

    print(f"Number of classes: {num_classes}")
    print(f"Class mapping: {class_names}")

    # 2. Load normalization stats from training
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        train_std = stats['std']
        print(f"\nLoaded training normalization stats: std={train_std:.4f}")
    else:
        print("\n[WARNING] normalization_stats.json not found! Cannot normalize safely.")
        return

    # 3. Setup Model
    window_size = 512
    stride = 10
    smoothing_window = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Environment ---")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {device}")

    model = get_model(model_name, num_classes=num_classes, in_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 4. Prepare Matplotlib Figure (Only 1 month, 10k snapshot for PNG)
    print(f"\nGenerating Matplotlib graph for 1 month (10k snapshot)...")
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(15, 6))
        
    print(f"\nGenerating Plotly graph with {len(columns)} subplots...")
    fig_plotly = make_subplots(
        rows=len(columns), cols=1,
        shared_xaxes=False,
        vertical_spacing=0.05,
        subplot_titles=[f"Time Period starting: {col}" for col in columns],
        specs=[[{"secondary_y": True}] for _ in range(len(columns))]
    )

    colors = ['gray', 'magenta', 'orange', 'green', 'lime', 'cyan', 'dodgerblue', 'purple', 'pink', 'gold', 'olive']

    # 5. Process Each Time Series
    for col_idx, col_name in enumerate(columns):
        print(f"\nProcessing column {col_idx+1}/{len(columns)}: {col_name}")
        # Handle string artifacts like "97..3123" by coercing and interpolating
        gamma_series = pd.to_numeric(df[col_name], errors='coerce')
        gamma_series = gamma_series.interpolate(method='linear', limit_direction='both')
        gamma_dose = gamma_series.values.astype(np.float32)
        segment_size = len(gamma_dose)

        from inference_engine import run_sliding_window_inference
        
        # We don't have true anomaly labels, so we skip evaluation and just predict
        probabilities, counts, num_windows, inference_duration = run_sliding_window_inference(
            gamma_dose, model, device, train_std, num_classes, window_size, stride, batch_size=256
        )
        print(f"  Processed {num_windows} windows in {inference_duration}.")

        # 6. Add to Matplotlib Figure (Only first month)
        if col_idx == 0:
            limit = min(10000, segment_size)
            time_steps_png = np.arange(limit)

            # Plot Background Gamma (Primary Y)
            ax1.plot(time_steps_png, gamma_dose[:limit], color='#1f77b4', linewidth=1, alpha=0.7, label=f'{col_name} Dose')
            ax1.set_title(f"Time Period starting: {col_name} (10k Snapshot)")
            ax1.set_ylabel("Gamma Dose Level")
            ax1.set_xlabel("Time Steps")

            ax2 = ax1.twinx()

            # Plot Confidence Curves (Secondary Y)
            for c in range(0, num_classes):
                df_probs = pd.DataFrame({'prob': probabilities[:, c]})
                df_probs['prob'] = df_probs['prob'].interpolate(method='linear', limit_direction='both')
                df_probs['prob'] = df_probs['prob'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
                smooth_probs = df_probs['prob'].values

                color = colors[c % len(colors)]
                class_label = class_names.get(c, f"Class {c}")
                
                alpha = 0.4 if c == 0 else 1.0
                lw = 1.5 if c == 0 else 2

                ax2.plot(time_steps_png, smooth_probs[:limit] * 100, color=color, linewidth=lw, alpha=alpha, label=f'{class_label} Confidence')

            ax2.set_ylabel("CNN Confidence Probability (%)")
            ax2.set_ylim(0, 105)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))
            
        # Add to Plotly Figure
        time_steps = np.arange(segment_size)
        fig_plotly.add_trace(
            go.Scatter(x=time_steps, y=gamma_dose, mode='lines', name=f'{col_name} Dose',
                       line=dict(color='teal', width=1), opacity=0.7, showlegend=(col_idx == 0)),
            row=col_idx+1, col=1, secondary_y=False,
        )

        for c in range(0, num_classes):
            df_probs = pd.DataFrame({'prob': probabilities[:, c]})
            df_probs['prob'] = df_probs['prob'].interpolate(method='linear', limit_direction='both')
            df_probs['prob'] = df_probs['prob'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
            smooth_probs = df_probs['prob'].values

            color = colors[c % len(colors)]
            class_label = class_names.get(c, f"Class {c}")
            show_leg = (col_idx == 0)
            
            opacity = 0.4 if c == 0 else 1.0
            lw = 1.5 if c == 0 else 3

            fig_plotly.add_trace(
                go.Scatter(x=time_steps, y=smooth_probs * 100, mode='lines', name=f'{class_label} Confidence',
                           line=dict(color=color, width=lw), opacity=opacity, showlegend=show_leg),
                row=col_idx+1, col=1, secondary_y=True,
            )

        fig_plotly.update_yaxes(title_text="Gamma Dose Level", row=col_idx+1, col=1, secondary_y=False)
        fig_plotly.update_yaxes(range=[0, 105], showgrid=False, row=col_idx+1, col=1, secondary_y=True)

    plt.suptitle(f'{display_name} Inference on Real Data (Unlabelled)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.85, 0.98])  # Adjust for legend outside

    output_png = os.path.join(logs_dir, 'real_data_inference.png')
    plt.savefig(output_png, bbox_inches='tight', dpi=200)
    plt.close()

    fig_plotly.update_layout(
        title=f'{display_name} Inference on Real Data (Unlabelled)',
        height=400 * len(columns),  # Make it tall enough to fit all subplots
        template='plotly_white',
        hovermode='x unified',
        legend=dict(yanchor="top", y=1.0, xanchor="right", x=1.05)
    )

    output_html = os.path.join(logs_dir, 'real_data_inference.html')
    fig_plotly.write_html(output_html)

    print(f"\nVisualization saved to {output_png} and {output_html}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
