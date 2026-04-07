import os
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
anomalies_dir = os.path.join(script_dir, '..', 'anomalies')
logs_dir = os.path.join(script_dir, '..', 'logs')
os.makedirs(logs_dir, exist_ok=True)

def visualize_anomalies():
    # Find all csv files in the anomalies directory
    anomaly_files = glob.glob(os.path.join(anomalies_dir, '*.csv'))
    
    if not anomaly_files:
        print(f"No anomaly templates found in {anomalies_dir}")
        return

    num_files = len(anomaly_files)
    titles = [os.path.basename(f).replace('.csv', '').capitalize() for f in anomaly_files]
    
    fig = make_subplots(rows=num_files, cols=1, subplot_titles=titles)

    for i, file_path in enumerate(anomaly_files, 1):
        anomaly_name = os.path.basename(file_path).replace('.csv', '')
        
        try:
            # Read the CSV format (time, value)
            df = pd.read_csv(file_path)
            
            if 'time' not in df.columns or 'value' not in df.columns:
                print(f"Warning: {anomaly_name}.csv must have 'time' and 'value' columns")
                continue
                
            # Build piece-wise dense line for plotting the mathematical shape
            t_dense = []
            v_dense = []
            
            for index_row in range(len(df) - 1):
                row = df.iloc[index_row]
                next_row = df.iloc[index_row+1]
                t0, v0 = row['time'], row['value']
                t1, v1 = next_row['time'], next_row['value']
                
                seg_steps = max(10, int(100 * (t1 - t0))) if t1 > t0 else 2
                t_seg = np.linspace(t0, t1, seg_steps)
                
                mode = row['interp'] if 'interp' in df.columns else 'linear'
                if mode == 'exp' and t1 > t0:
                    x_norm = (t_seg - t0) / (t1 - t0)
                    alpha = 3.0
                    y_norm = (np.exp(alpha * x_norm) - 1) / (np.exp(alpha) - 1)
                    v_seg = v0 + (v1 - v0) * y_norm
                else:
                    v_seg = np.linspace(v0, v1, seg_steps)
                    
                if index_row < len(df) - 2:
                    t_dense.extend(t_seg[:-1])
                    v_dense.extend(v_seg[:-1])
                else:
                    t_dense.extend(t_seg)
                    v_dense.extend(v_seg)
                    
            # Plot the continuous mathematical shape
            fig.add_trace(go.Scatter(
                x=t_dense, 
                y=v_dense, 
                mode='lines', 
                name=anomaly_name.capitalize(),
                line=dict(color='cyan')
            ), row=i, col=1)
            
            # Plot the control keypoints purely as markers
            fig.add_trace(go.Scatter(
                x=df['time'], 
                y=df['value'], 
                mode='markers', 
                name=f'{anomaly_name.capitalize()} (Keypoints)',
                marker=dict(color='white', size=6),
                showlegend=False
            ), row=i, col=1)
            
            # If boolean quadrant flags exist, plot representative crosshairs to indicate allowed movements
            if all(col in df.columns for col in ['t_minus', 't_plus', 'v_minus', 'v_plus']):
                # Dummy magnitudes purely for visualization purposes
                MAG_T = 0.05
                MAG_V = 0.1
                
                fig.add_trace(go.Scatter(
                    x=df['time'],
                    y=df['value'],
                    mode='markers',
                    name=f'{anomaly_name.capitalize()} (Allowed Quadrants)',
                    marker=dict(color='rgba(255,165,0,0.8)', size=1),
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=df['t_plus'] * MAG_T,
                        arrayminus=df['t_minus'] * MAG_T,
                        color='rgba(255, 165, 0, 0.5)',
                        thickness=3
                    ),
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=df['v_plus'] * MAG_V,
                        arrayminus=df['v_minus'] * MAG_V,
                        color='rgba(255, 165, 0, 0.4)',
                        thickness=2
                    ),
                    showlegend=False
                ), row=i, col=1)
            print(f"Loaded: {anomaly_name} ({len(df)} keypoints)")
            
            # Ensure amplitude and time ranges are clearly visible per subplot
            fig.update_xaxes(range=[-0.1, 1.1], title_text="Normalized Time", row=i, col=1)
            fig.update_yaxes(range=[-0.1, 1.2], title_text="Amplitude", row=i, col=1)
            
        except Exception as e:
            print(f"Error loading {anomaly_name}: {e}")

    # Layout configuration
    fig.update_layout(
        title='Anomaly Templates (Normalized Unity Period and Amplitude)',
        height=500 * num_files,
        width=800, # Limiting width to prevent shapes from stretching horizontally
        hovermode='x unified',
        template='plotly_dark'
    )
    
    output_path = os.path.join(logs_dir, 'anomaly_templates.html')
    fig.write_html(output_path)
    print(f"\nVisualization saved successfully to: {output_path}")

if __name__ == "__main__":
    visualize_anomalies()
