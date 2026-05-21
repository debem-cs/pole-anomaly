import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_dataset(csv_path, output_html=None, output_png=None, plot_limit=100000):
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    root_dir = script_dir
    
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if len(df) > plot_limit:
        print(f"Dataset has {len(df)} points. Truncating to first {plot_limit} points for plotting to avoid memory issues.")
        df = df.iloc[:plot_limit]
        
    time_step = df['time_step'].values
    gamma_dose = df['gamma_dose'].values
    is_anomaly = df['is_anomaly'].values.astype(int)
    
    # Try to load class names
    class_mapping_path = os.path.join(root_dir, 'data', 'class_mapping.txt')
    class_names = {}
    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, 'r') as f:
            for line in f:
                if ':' in line:
                    c_id, name = line.split(':')
                    class_names[int(c_id.strip())] = name.strip()
                    
    plot_classes_text = ["Normal Background" if val == 0 else f"Anomaly: {class_names.get(val, f'Class {val}')}" for val in is_anomaly]
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.plot(time_step, gamma_dose, color='#1f77b4', linewidth=1, label='Gamma Dose')

    anomalous_points = np.where(is_anomaly > 0)[0]
    anomalous_texts = []
    if len(anomalous_points) > 0:
        anomalous_texts = [plot_classes_text[i] for i in anomalous_points]
        # Red dots removed from PNG as requested

    ax.set_title(f'DATASET PLOT: {os.path.basename(csv_path)}', fontsize=14)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Gamma Dose')
    ax.legend(loc='upper right')

    # Limit x-axis to first 3 anomalies for the static PNG snapshot
    is_anom_bool = is_anomaly > 0
    diffs = np.diff(is_anom_bool.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    if len(is_anom_bool) > 0 and is_anom_bool[0]: starts = np.insert(starts, 0, 0)
    if len(is_anom_bool) > 0 and is_anom_bool[-1]: ends = np.append(ends, len(is_anom_bool))
    
    if len(starts) >= 3:
        padding = 2000
        xlim_start = time_step[max(0, starts[0] - padding)]
        xlim_end = time_step[min(len(time_step)-1, ends[2] + padding)]
        ax.set_xlim(xlim_start, xlim_end)
    elif len(starts) > 0:
        padding = 2000
        xlim_start = time_step[max(0, starts[0] - padding)]
        xlim_end = time_step[min(len(time_step)-1, ends[-1] + padding)]
        ax.set_xlim(xlim_start, xlim_end)

    if output_png is None:
        logs_dir = os.path.join(root_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_png = os.path.join(logs_dir, f'{base_name}.png')
        
    plt.tight_layout()
    plt.savefig(output_png, bbox_inches='tight', dpi=200)
    plt.close()
    
    fig_plotly = go.Figure()
    fig_plotly.add_trace(go.Scatter(
        x=time_step, y=gamma_dose, mode='lines', name='Gamma Dose',
        line=dict(color='teal', width=1), hovertext=plot_classes_text, hoverinfo="x+y+text"
    ))

    if len(anomalous_points) > 0:
        fig_plotly.add_trace(go.Scatter(
            x=time_step[anomalous_points], y=gamma_dose[anomalous_points], mode='markers',
            name='Anomalies', marker=dict(color='red', size=5),
            hovertext=anomalous_texts, hoverinfo="x+y+text"
        ))

    fig_plotly.update_layout(
        title=f'DATASET PLOT: {os.path.basename(csv_path)}',
        xaxis_title='Time Steps', yaxis_title='Gamma Dose',
        hovermode='x unified', template='plotly_white'
    )
    
    if output_html is None:
        output_html = os.path.join(logs_dir, f'{base_name}.html')
    fig_plotly.write_html(output_html)

    print(f"Static plot saved to: {output_png}")
    print(f"Interactive plot saved to: {output_html}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot a dataset CSV file.')
    parser.add_argument('csv_path', nargs='?', default=None, help='Path to the dataset CSV file')
    parser.add_argument('--limit', type=int, default=100000, help='Maximum number of points to plot')
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    
    target_csv = args.csv_path
    if target_csv is None:
        target_csv = os.path.join(script_dir, 'data', 'synthetic_custom_dataset.csv')
        
    plot_dataset(target_csv, plot_limit=args.limit)
