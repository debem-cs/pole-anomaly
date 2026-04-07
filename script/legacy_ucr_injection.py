import numpy as np
import plotly.graph_objects as go
import os
try:
    from tslearn.datasets import UCR_UEA_datasets
except ImportError:
    print("Please install tslearn and scipy: pip install tslearn scipy")
    exit(1)

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
data_dir = os.path.join(script_dir, '..', 'data')
logs_dir = os.path.join(script_dir, '..', 'logs')

# 1. Evaluate REAL background characteristics
filename = os.path.join(data_dir, '2015_months_DebitDoseA.txt')
try:
    data_gamma = np.genfromtxt(filename, delimiter=',', skip_header=1)
    mois = 2 
    real_background = data_gamma[:, mois]
except FileNotFoundError:
    print(f"Error: Could not find {filename}.")
    exit(1)

# Clean boundaries
Nh = 60
N = len(real_background)
real_background = real_background[Nh:N-Nh]

mean_noise = np.nanmean(real_background)
std_noise = np.nanstd(real_background)
print(f"Sensor noise characteristics -> Mean: {mean_noise:.2f}, Standard Deviation: {std_noise:.2f}")

# 2. Generate completely SYNTHETIC background noise
# The user wants to see the COMPLETE dataset behavior, so we make it long (10,000 steps)
N_synthetic = 10000
time_steps = np.arange(N_synthetic)
synthetic_background = np.random.normal(loc=mean_noise, scale=std_noise, size=N_synthetic)
synthetic_background = np.clip(synthetic_background, 0, None)
labels = np.zeros(N_synthetic, dtype=int)
anomaly_classes_text = np.array(["Normal Background"] * N_synthetic, dtype=object)
cbf_class_names = {1: 'Cylinder', 2: 'Bell', 3: 'Funnel'}

# 3. Fetch REAL CBF Anomalies (UCR CBF)
print("Downloading CBF UCR dataset to extract isolated anomalies...")
dl = UCR_UEA_datasets()
X_train, y_train, _, _ = dl.load_dataset('CBF')

# 4. Inject ISOLATED CBF anomalies from DIFFERENT classes
num_anomalies = 6
spacing = N_synthetic // (num_anomalies + 1)
inject_points = [spacing * i + np.random.randint(-200, 200) for i in range(1, num_anomalies + 1)]

print(f"Injecting {num_anomalies} isolated CBF spikes...")

for i, idx in enumerate(inject_points):
    # There are 3 CBF classes (Cylinder, Bell, Funnel)
    # We guarantee diversity by iterating through them
    anomaly_class = (i % 3) + 1
    class_pool = X_train[y_train == anomaly_class]
    if len(class_pool) == 0: class_pool = X_train
    
    raw_anomaly = class_pool[np.random.randint(0, len(class_pool)), :, 0]
    
    # Use the full 128-step pre-cut CBF sequence
    isolated_spike = raw_anomaly.copy()
    
    # CRITICAL FIX: The CBF dataset already has inherent mathematical noise N(0,1)
    # If we amplify this shape by 15x, the noise becomes N(0,15) which looks awful.
    # We must apply a smoothing filter to extract the pure geometric shape BEFORE scaling.
    from scipy.ndimage import gaussian_filter1d
    isolated_spike = gaussian_filter1d(isolated_spike, sigma=3)
    
    # Ensure it's positioned as a positive spike
    # Find if the major feature is positive or negative
    if np.abs(np.min(isolated_spike)) > np.max(isolated_spike):
        isolated_spike = -isolated_spike
        
    isolated_spike -= np.min(isolated_spike)
    
    # Smooth boundaries using Hanning window so it blends perfectly into the noise
    taper = np.hanning(len(isolated_spike) + 2)[1:-1]
    isolated_spike = isolated_spike * taper
    
    # Scale to match radiation anomalies (massive spike compared to 3.56 std deviation)
    target_peak = std_noise * np.random.uniform(10.0, 25.0)
    current_peak = np.max(isolated_spike)
    
    if current_peak > 0:
        scaled_spike = isolated_spike * (target_peak / current_peak)
    else:
        scaled_spike = isolated_spike

    length_spike = len(scaled_spike)
    
    # Inject
    for j in range(length_spike):
        if idx + j < N_synthetic:
            synthetic_background[idx + j] += scaled_spike[j]
            # Ground truth label
            if scaled_spike[j] > (std_noise * 1.5):
                labels[idx + j] = 1
                anomaly_classes_text[idx + j] = f"Anomaly: {cbf_class_names[anomaly_class]}"

# 5. Plot the final CBF hybrid dataset COMPLETELY
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
    name='Isolated Anomalies (CBF Classes)',
    marker=dict(color='red', size=5),
    hovertext=anomaly_classes_text[anomalous_points],
    hoverinfo="x+y+text"
))

fig.update_layout(
    title='SYNTHETIC DATASET: Simulated Sensor Noise + Isolated CBF Spikes',
    xaxis_title='Time Steps (10,000 total)',
    yaxis_title='Simulated Gamma Dose',
    hovermode='x unified'
)

# Rename to synthetic_dataset.html as requested
output_plot_path = os.path.join(logs_dir, 'synthetic_dataset.html')
fig.write_html(output_plot_path)

# Also save the data for training later
output_csv_path = os.path.join(data_dir, 'synthetic_dataset.csv')
data_stack = np.column_stack((time_steps, synthetic_background, labels))
np.savetxt(output_csv_path, data_stack, delimiter=',', header='time_step,gamma_dose,is_anomaly', comments='')

print(f"\nFinal interactive plot saved to: {output_plot_path}")
print(f"Final dataset CSV saved for training to: {output_csv_path}")
