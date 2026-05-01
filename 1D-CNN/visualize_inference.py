import os
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

# Import the model architecture
from model import Anomaly1DCNN

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    data_path = os.path.join(script_dir, '..', 'data-gen', 'data', 'synthetic_custom_dataset.csv')
    model_path = os.path.join(script_dir, 'saved_models', 'best_1d_cnn_pytorch.pth')
    logs_dir = os.path.join(script_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("Missing dataset or trained model!")
        return

    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    segment_size = min(50000, len(df))
    df_segment = df.iloc[:segment_size]
    
    gamma_dose = df_segment['gamma_dose'].values
    is_anomaly = df_segment['is_anomaly'].values
    time_steps = df_segment['time_step'].values
    
    num_classes = len(np.unique(df['is_anomaly'].values))
    print(f"Visualizing for {num_classes} classes.")
    
    class_mapping_path = os.path.join(script_dir, '..', 'data-gen', 'data', 'class_mapping.txt')
    class_names = {i: f"Class {i}" for i in range(num_classes)}
    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, 'r') as f:
            for line in f:
                if ':' in line:
                    c_id, name = line.split(':')
                    class_names[int(c_id.strip())] = name.strip()
    
    # 2. Setup Model
    window_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Anomaly1DCNN(in_channels=1, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    scaler = StandardScaler()
    scaler.fit(df['gamma_dose'].values.reshape(-1, 1))
    
    # 3. Continuous Inference via Sliding Window
    print("Running inference...")
    probabilities = np.zeros((segment_size, num_classes))
    counts = np.zeros(segment_size)
    
    stride = 10
    
    for start in range(0, segment_size - window_size + 1, stride):
        end = start + window_size
        window_data = gamma_dose[start:end]
        
        window_scaled = scaler.transform(window_data.reshape(-1, 1)).flatten()
        x_tensor = torch.tensor(window_scaled, dtype=torch.float32).view(1, 1, window_size).to(device)
        
        with torch.no_grad():
            output = model(x_tensor)
            prob = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        center = start + window_size // 2
        probabilities[center] = prob
        counts[center] = 1

    for i in range(len(counts)):
        if counts[i] == 0:
            probabilities[i] = np.nan
            
    # 4. Visualization
    print("Generating Plotly graph...")
    
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
        df_probs['prob'] = df_probs['prob'].rolling(window=30, center=True, min_periods=1).mean()
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
        
    print(f"Visualization saved to {output_html}")

if __name__ == "__main__":
    main()
