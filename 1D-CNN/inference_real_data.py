import os
import json
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch.nn.functional as F

# Import the model architecture
from model import Anomaly1DCNN

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    data_path = os.path.join(script_dir, '..', 'data-gen', 'data', '2015_months_DebitDoseA.txt')
    model_path = os.path.join(script_dir, 'saved_models', 'best_1d_cnn_pytorch.pth')
    stats_path = os.path.join(script_dir, 'saved_models', 'normalization_stats.json')
    logs_dir = os.path.join(script_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    print("=" * 60)
    print("1D-CNN REAL DATA INFERENCE")
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
    
    model = Anomaly1DCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    
    # 4. Prepare Plotly Subplots
    print(f"\nGenerating Plotly graph with {len(columns)} subplots...")
    fig = make_subplots(
        rows=len(columns), cols=1, 
        shared_xaxes=False, 
        vertical_spacing=0.05,
        subplot_titles=[f"Time Period starting: {col}" for col in columns],
        specs=[[{"secondary_y": True}] for _ in range(len(columns))]
    )
    
    colors = ['magenta', 'orange', 'yellow', 'lime', 'cyan', 'dodgerblue', 'white', 'pink', 'gold', 'lightgreen']
    
    # 5. Process Each Time Series
    for col_idx, col_name in enumerate(columns):
        print(f"\nProcessing column {col_idx+1}/{len(columns)}: {col_name}")
        # Handle string artifacts like "97..3123" by coercing and interpolating
        gamma_series = pd.to_numeric(df[col_name], errors='coerce')
        gamma_series = gamma_series.interpolate(method='linear', limit_direction='both')
        gamma_dose = gamma_series.values.astype(np.float32)
        segment_size = len(gamma_dose)
        
        # We don't have true anomaly labels, so we skip evaluation and just predict
        windows = []
        indices = []
        
        idx = 0
        while idx + window_size <= segment_size:
            w = gamma_dose[idx:idx + window_size].copy()
            # Same normalization as training
            w = (w - np.mean(w)) / train_std
            windows.append(w)
            indices.append(idx)
            idx += stride
            
        print(f"  Extracted {len(windows)} windows.")
        
        # Accumulated probabilities mapping
        # probabilities[t, c] = list of probability predictions for time t, class c
        prob_accum = {i: {c: [] for c in range(num_classes)} for i in range(segment_size)}
        
        # Batch inference
        batch_size = 256
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch_windows = windows[i:i + batch_size]
                batch_indices = indices[i:i + batch_size]
                
                inputs = torch.tensor(np.array(batch_windows), dtype=torch.float32).unsqueeze(1).to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                
                for b_idx, start_idx in enumerate(batch_indices):
                    window_probs = probs[b_idx]
                    for t in range(start_idx, start_idx + window_size):
                        for c in range(num_classes):
                            prob_accum[t][c].append(window_probs[c])
                            
        # Average probabilities per time step
        probabilities = np.full((segment_size, num_classes), np.nan)
        for t in range(segment_size):
            if len(prob_accum[t][0]) > 0:
                for c in range(num_classes):
                    probabilities[t, c] = np.mean(prob_accum[t][c])
                    
        # 6. Add to Plotly Figure
        time_steps = np.arange(segment_size)
        
        # Plot Background Gamma (Primary Y)
        fig.add_trace(
            go.Scatter(x=time_steps, y=gamma_dose, mode='lines', name=f'{col_name} Dose',
                       line=dict(color='Teal', width=1), opacity=0.7, showlegend=(col_idx == 0)),
            row=col_idx+1, col=1, secondary_y=False,
        )
        
        # Plot Confidence Curves (Secondary Y)
        for c in range(1, num_classes):
            df_probs = pd.DataFrame({'prob': probabilities[:, c]})
            df_probs['prob'] = df_probs['prob'].interpolate(method='linear', limit_direction='both')
            df_probs['prob'] = df_probs['prob'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
            smooth_probs = df_probs['prob'].values
            
            color = colors[(c - 1) % len(colors)]
            class_label = class_names.get(c, f"Class {c}")
            
            # Only show legend once per class
            show_leg = (col_idx == 0)
            
            fig.add_trace(
                go.Scatter(x=time_steps, y=smooth_probs * 100, mode='lines', name=f'{class_label} Confidence',
                           line=dict(color=color, width=2), showlegend=show_leg),
                row=col_idx+1, col=1, secondary_y=True,
            )
            
        fig.update_yaxes(title_text="Gamma Dose Level", row=col_idx+1, col=1, secondary_y=False)
        fig.update_yaxes(range=[0, 105], showgrid=False, row=col_idx+1, col=1, secondary_y=True)

    fig.update_layout(
        title='1D-CNN Inference on Real Data (Unlabelled)',
        height=400 * len(columns),  # Make it tall enough to fit all subplots
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(yanchor="top", y=1.0, xanchor="right", x=1.05)
    )
    
    output_html = os.path.join(logs_dir, 'real_data_inference.html')
    fig.write_html(output_html)
        
    print(f"\nVisualization saved to {output_html}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
