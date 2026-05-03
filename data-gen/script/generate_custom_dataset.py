import numpy as np
import plotly.graph_objects as go
import pandas as pd
import glob
import os
import sys

# Ensure src is in the python path to import the generator library
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

from src.anomaly_generator import generate_anomaly

def create_synthetic_dataset():
    data_dir = os.path.join(root_dir, 'data')
    logs_dir = os.path.join(root_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Evaluate REAL background characteristics
    filename = os.path.join(root_dir, 'data', '2015_months_DebitDoseA.txt')
    try:
        data_gamma = np.genfromtxt(filename, delimiter=',', skip_header=1)
        mois = 3 
        real_background = data_gamma[:, mois]
        
        # Clean boundaries
        Nh = 400
        N = len(real_background)
        real_background = real_background[Nh:N-Nh]
        
        mean_noise = np.nanmean(real_background)
        std_noise = np.nanstd(real_background)
        print(f"Sensor noise characteristics -> Mean: {mean_noise:.2f}, Standard Deviation: {std_noise:.2f}")
    except FileNotFoundError:
        print(f"Error: Could not find {filename}. Using default noise estimates.")
        mean_noise = 100.0
        std_noise = 4.5

    # 2. Load Custom Templates
    anomalies_dir = os.path.join(root_dir, 'anomalies')
    template_files = glob.glob(os.path.join(anomalies_dir, '*.csv'))
    if not template_files:
        print(f"Error: No anomaly templates found in {anomalies_dir}")
        return
        
    loaded_templates = {}
    for filepath in template_files:
        name = os.path.basename(filepath).replace('.csv', '')
        loaded_templates[name] = pd.read_csv(filepath)
        
    template_names = list(loaded_templates.keys())

    # Create mapping for multi-class
    class_map = {name: i + 1 for i, name in enumerate(template_names)}
    with open(os.path.join(data_dir, 'class_mapping.txt'), 'w') as f:
        f.write("0: Normal Background\n")
        for name, cls_id in class_map.items():
            f.write(f"{cls_id}: {name}\n")

    # 3. Configuration
    N_synthetic = 10000000
    CHUNK_SIZE = 50000  # Process this many points at a time to limit RAM usage
    num_anomalies = np.random.randint(5000, 8000)
    
    # Pre-compute all anomaly injection points and their shapes
    spacing = N_synthetic // (num_anomalies + 1)
    inject_points = [spacing * i + np.random.randint(-200, 200) for i in range(1, num_anomalies + 1)]
    
    print(f"Injecting {num_anomalies} isolated custom spikes across {N_synthetic} time steps...")
    print(f"Writing in chunks of {CHUNK_SIZE} to limit RAM usage...")
    
    # Pre-generate all anomalies (they are small, ~200-500 points each)
    anomaly_injections = []
    for idx in inject_points:
        chosen_template_name = np.random.choice(template_names)
        df_template = loaded_templates[chosen_template_name]
        
        target_amplitude = std_noise * np.random.uniform(3.0, 7.6)
        target_period = np.random.randint(200, 500)
        variance_level = np.random.uniform(0.02, 0.08)

        t_discrete, v_discrete = generate_anomaly(
            df_template, 
            amplitude=target_amplitude, 
            period=target_period, 
            variance=variance_level
        )
        
        anomaly_injections.append({
            'start': idx,
            'values': v_discrete,
            'template_name': chosen_template_name,
            'class_id': class_map[chosen_template_name]
        })

    # 4. Write CSV in chunks
    output_csv_path = os.path.join(data_dir, 'synthetic_custom_dataset.csv')
    
    # Collect a small preview segment for plotting (first 50k points max)
    plot_limit = min(50000, N_synthetic)
    plot_time = []
    plot_gamma = []
    plot_anomaly_vals = []
    plot_labels = []
    plot_classes_text = []
    
    with open(output_csv_path, 'w') as csv_file:
        csv_file.write("time_step,gamma_dose,is_anomaly\n")
        
        num_chunks = (N_synthetic + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * CHUNK_SIZE
            chunk_end = min(chunk_start + CHUNK_SIZE, N_synthetic)
            chunk_len = chunk_end - chunk_start
            
            # Generate noise for this chunk only
            chunk_background = np.random.normal(loc=mean_noise, scale=std_noise, size=chunk_len)
            chunk_background = np.clip(chunk_background, 0, None)
            chunk_pure_anomalies = np.zeros(chunk_len, dtype=float)
            chunk_labels = np.zeros(chunk_len, dtype=int)
            chunk_classes_text = ["Normal Background"] * chunk_len
            
            # Inject anomalies that overlap with this chunk
            for anom in anomaly_injections:
                anom_start = anom['start']
                anom_end = anom_start + len(anom['values'])
                
                # Check if this anomaly overlaps with current chunk
                if anom_end <= chunk_start or anom_start >= chunk_end:
                    continue
                
                # Calculate overlap region
                overlap_start = max(anom_start, chunk_start)
                overlap_end = min(anom_end, chunk_end)
                
                for global_idx in range(overlap_start, overlap_end):
                    local_idx = global_idx - chunk_start
                    anom_idx = global_idx - anom_start
                    
                    v = anom['values'][anom_idx]
                    chunk_background[local_idx] += v
                    chunk_pure_anomalies[local_idx] += v
                    
                    if v > 1e-2:
                        chunk_labels[local_idx] = anom['class_id']
                        chunk_classes_text[local_idx] = f"Anomaly: {anom['template_name'].capitalize()}"
            
            # Write chunk to CSV
            for i in range(chunk_len):
                global_t = chunk_start + i
                csv_file.write(f"{global_t},{chunk_background[i]},{chunk_labels[i]}\n")
            
            # Collect data for plotting (only up to plot_limit)
            if chunk_start < plot_limit:
                plot_end = min(chunk_end, plot_limit)
                plot_len = plot_end - chunk_start
                plot_time.extend(range(chunk_start, plot_end))
                plot_gamma.extend(chunk_background[:plot_len].tolist())
                plot_anomaly_vals.extend(chunk_pure_anomalies[:plot_len].tolist())
                plot_labels.extend(chunk_labels[:plot_len].tolist())
                plot_classes_text.extend(chunk_classes_text[:plot_len])
            
            print(f"  Chunk {chunk_idx + 1}/{num_chunks} written ({chunk_start:,} - {chunk_end:,})")
    
    print(f"Dataset CSV saved: {output_csv_path}")

    # 5. Plot (using only the preview segment, not the full dataset)
    print(f"Generating plot from first {len(plot_time):,} points...")
    
    plot_time = np.array(plot_time)
    plot_gamma = np.array(plot_gamma)
    plot_anomaly_vals = np.array(plot_anomaly_vals)
    plot_labels = np.array(plot_labels)
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_time, 
        y=plot_gamma, 
        mode='lines', 
        name='Simulated Gamma Noise',
        line=dict(color='Teal', width=1),
        hovertext=plot_classes_text,
        hoverinfo="x+y+text"
    ))

    anomalous_points = np.where(plot_labels > 0)[0]
    anomalous_texts = [plot_classes_text[i] for i in anomalous_points]
    fig.add_trace(go.Scatter(
        x=plot_time[anomalous_points],
        y=plot_gamma[anomalous_points],
        mode='markers',
        name='Injected Anomalies',
        marker=dict(color='red', size=5),
        hovertext=anomalous_texts,
        hoverinfo="x+y+text"
    ))

    fig.add_trace(go.Scatter(
        x=plot_time, 
        y=plot_anomaly_vals + mean_noise, 
        mode='lines', 
        name='Pure Anomaly Shapes (No Noise)',
        line=dict(color='orange', width=2),
        hoverinfo="skip"
    ))

    fig.update_layout(
        title='SYNTHETIC DATASET: Simulated Sensor Noise + Custom Anomaly Forms',
        xaxis_title='Time Steps',
        yaxis_title='Simulated Gamma Dose',
        hovermode='x unified',
        template='plotly_dark'
    )

    output_plot_path = os.path.join(logs_dir, 'synthetic_custom_dataset.html')
    fig.write_html(output_plot_path)
    output_png_path = os.path.join(logs_dir, 'synthetic_custom_dataset.png')
    fig.write_image(output_png_path)

    print(f"\nFinal interactive plot saved to: {output_plot_path}")
    print(f"Final dataset CSV saved for training to: {output_csv_path}")

if __name__ == "__main__":
    create_synthetic_dataset()
