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
        
        # Clean boundaries and remove NaNs
        Nh = 400
        real_background = real_background[Nh:-Nh]
        real_background = real_background[~np.isnan(real_background)]
        
        global_mean = np.mean(real_background)
        
        # Isolate baseline using a moving average
        window_size = 480
        kernel = np.ones(window_size) / window_size
        baseline = np.convolve(real_background, kernel, mode='valid')
        
        # Calculate high-frequency noise from residuals
        hf_residual = real_background[window_size//2 : -window_size//2 + 1] - baseline
        std_noise = np.std(hf_residual)
        
        # Baseline OU parameters
        baseline_diffs = np.diff(baseline)
        sigma = np.std(baseline_diffs)
        # Empirical low reversion rate from analysis
        theta = 0.000004 
        
        print(f"Sensor noise characteristics -> Global Mean: {global_mean:.2f}, HF Noise Std: {std_noise:.2f}")
        print(f"Baseline characteristics -> Theta: {theta:.6f}, Sigma: {sigma:.6f}")
    except FileNotFoundError:
        print(f"Error: Could not find {filename}. Using default noise estimates.")
        global_mean = 100.0
        std_noise = 2.55
        theta = 0.000004
        sigma = 0.010

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
    N_synthetic = 1000000
    CHUNK_SIZE = 50000  # Process this many points at a time to limit RAM usage
    num_anomalies = np.random.randint(500, 800)
    
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
    plot_baseline = []
    
    with open(output_csv_path, 'w') as csv_file:
        csv_file.write("time_step,gamma_dose,is_anomaly\n")
        
        num_chunks = (N_synthetic + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        current_baseline_val = global_mean
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * CHUNK_SIZE
            chunk_end = min(chunk_start + CHUNK_SIZE, N_synthetic)
            chunk_len = chunk_end - chunk_start
            
            # Generate wandering baseline using OU process
            chunk_baseline = np.zeros(chunk_len, dtype=float)
            noise_steps = np.random.normal(scale=sigma, size=chunk_len)
            
            val = current_baseline_val
            for i in range(chunk_len):
                val = val + theta * (global_mean - val) + noise_steps[i]
                chunk_baseline[i] = val
            current_baseline_val = val
            
            # Generate high-frequency noise around the wandering baseline
            chunk_background = np.random.normal(loc=chunk_baseline, scale=std_noise)
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
                plot_baseline.extend(chunk_baseline[:plot_len].tolist())
            
            print(f"  Chunk {chunk_idx + 1}/{num_chunks} written ({chunk_start:,} - {chunk_end:,})")
    
    print(f"Dataset CSV saved: {output_csv_path}")

    # 5. Plot (using only the preview segment, not the full dataset)
    print(f"Generating plot from first {len(plot_time):,} points...")
    
    plot_time = np.array(plot_time)
    plot_gamma = np.array(plot_gamma)
    plot_anomaly_vals = np.array(plot_anomaly_vals)
    plot_labels = np.array(plot_labels)
    plot_baseline = np.array(plot_baseline)
    
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
        y=plot_anomaly_vals + global_mean, 
        mode='lines', 
        name='Pure Anomaly Shapes (No Noise)',
        line=dict(color='orange', width=2),
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter(
        x=plot_time, 
        y=plot_baseline, 
        mode='lines', 
        name='Wandering Baseline',
        line=dict(color='yellow', width=2, dash='dash'),
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
