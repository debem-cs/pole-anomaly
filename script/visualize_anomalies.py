import os
import glob
import pandas as pd
import plotly.graph_objects as go

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
anomalies_dir = os.path.join(script_dir, '..', 'anomalies')
logs_dir = os.path.join(script_dir, '..', 'logs')
os.makedirs(logs_dir, exist_ok=True)

def visualize_anomalies():
    # Find all txt files in the anomalies directory
    anomaly_files = glob.glob(os.path.join(anomalies_dir, '*.txt'))
    
    if not anomaly_files:
        print(f"No anomaly templates found in {anomalies_dir}")
        return

    fig = go.Figure()

    for file_path in anomaly_files:
        anomaly_name = os.path.basename(file_path).replace('.txt', '')
        
        try:
            # Read the CSV format (time, value)
            df = pd.read_csv(file_path)
            
            if 'time' not in df.columns or 'value' not in df.columns:
                print(f"Warning: {anomaly_name}.txt must have 'time' and 'value' columns")
                continue
                
            # Plot the base shape
            fig.add_trace(go.Scatter(
                x=df['time'], 
                y=df['value'], 
                mode='lines+markers', 
                name=anomaly_name.capitalize(),
                line_shape='linear' # We use linear to connect the keypoints shape
            ))
            
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
                        color='rgba(255, 165, 0, 0.5)',
                        thickness=3
                    ),
                    showlegend=True
                ))
            print(f"Loaded: {anomaly_name} ({len(df)} keypoints)")
            
        except Exception as e:
            print(f"Error loading {anomaly_name}: {e}")

    # Layout configuration
    fig.update_layout(
        title='Anomaly Templates (Normalized Unity Period and Amplitude)',
        xaxis_title='Normalized Time (0.0 to 1.0)',
        yaxis_title='Normalized Amplitude (0.0 to 1.0)',
        hovermode='x unified',
        template='plotly_dark'
    )
    
    # Ensure amplitude and time ranges are clearly visible
    fig.update_xaxes(range=[-0.1, 1.1])
    fig.update_yaxes(range=[-0.1, 1.2])

    output_path = os.path.join(logs_dir, 'anomaly_templates.html')
    fig.write_html(output_path)
    print(f"\nVisualization saved successfully to: {output_path}")

if __name__ == "__main__":
    visualize_anomalies()
