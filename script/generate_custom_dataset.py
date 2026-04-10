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
    filename = os.path.join(data_dir, '2015_months_DebitDoseA.txt')
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

    # 2. Generate completely SYNTHETIC background noise
    N_synthetic = 50000
    time_steps = np.arange(N_synthetic)
    synthetic_background = np.random.normal(loc=mean_noise, scale=std_noise, size=N_synthetic)
    synthetic_background = np.clip(synthetic_background, 0, None) # physical radiation can't be negative
    labels = np.zeros(N_synthetic, dtype=int)
    anomaly_classes_text = np.array(["Normal Background"] * N_synthetic, dtype=object)

    # 3. Load Custom Templates
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

    # 4. Inject
    num_anomalies = np.random.randint(20, 30) # 20 to 30 anomalies
    spacing = N_synthetic // (num_anomalies + 1)
    inject_points = [spacing * i + np.random.randint(-200, 200) for i in range(1, num_anomalies + 1)]

    print(f"Injecting {num_anomalies} isolated custom spikes...")

    for idx in inject_points:
        chosen_template_name = np.random.choice(template_names)
        df_template = loaded_templates[chosen_template_name]
        
        target_amplitude = std_noise * np.random.uniform(3.0, 7.6)
        target_period = np.random.randint(100, 300)
        variance_level = np.random.uniform(0.04, 0.08) # deformation jitter amount

        t_discrete, v_discrete = generate_anomaly(
            df_template, 
            amplitude=target_amplitude, 
            period=target_period, 
            variance=variance_level
        )
        
        length_spike = len(v_discrete)
        
        # Inject
        for j in range(length_spike):
            if idx + j < N_synthetic:
                synthetic_background[idx + j] += v_discrete[j]
                
                # Ground truth label
                if v_discrete[j] > (std_noise * 1.5): # Mark core anomaly points as 1
                    labels[idx + j] = 1
                    anomaly_classes_text[idx + j] = f"Anomaly: {chosen_template_name.capitalize()}"

    # 5. Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_steps, 
        y=synthetic_background, 
        mode='lines', 
        name='Simulated Gamma Noise',
        line=dict(color='Teal', width=1),
        hovertext=anomaly_classes_text,
        hoverinfo="x+y+text"
    ))

    anomalous_points = np.where(labels == 1)[0]
    fig.add_trace(go.Scatter(
        x=time_steps[anomalous_points],
        y=synthetic_background[anomalous_points],
        mode='markers',
        name='Injected Anomalies',
        marker=dict(color='red', size=5),
        hovertext=anomaly_classes_text[anomalous_points],
        hoverinfo="x+y+text"
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

    # Save CSV
    output_csv_path = os.path.join(data_dir, 'synthetic_custom_dataset.csv')
    data_stack = np.column_stack((time_steps, synthetic_background, labels))
    np.savetxt(output_csv_path, data_stack, delimiter=',', header='time_step,gamma_dose,is_anomaly', comments='')

    print(f"\nFinal interactive plot saved to: {output_plot_path}")
    print(f"Final dataset CSV saved for training to: {output_csv_path}")

if __name__ == "__main__":
    create_synthetic_dataset()
