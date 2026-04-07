import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import urllib.request
import traceback

try:
    from tslearn.datasets import UCR_UEA_datasets
    import statsmodels.api as sm
except ImportError:
    print("Installing dependencies...")
    os.system("pip install tslearn statsmodels plotly pandas")
    from tslearn.datasets import UCR_UEA_datasets
    import statsmodels.api as sm

script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
logs_dir = os.path.join(script_dir, '..', 'logs')
os.makedirs(logs_dir, exist_ok=True)

def save_plot(fig, filename):
    out_path = os.path.join(logs_dir, filename)
    fig.write_html(out_path)
    print(f"Saved: {filename}")

# --- 1 to 4. UCR DATASETS ---
dl = UCR_UEA_datasets()
ucr_datasets = {
    'StarLightCurves': 'Starlight Curves (Photon Flux)',
    'CBF': 'Cylinder-Bell-Funnel (Synthetic)',
    'ECG5000': 'Heartbeats (Biological)',
    'Epilepsy': 'EEG Brainwaves (Neuroscience)'
}

for ds_name, desc in ucr_datasets.items():
    try:
        X_train, y_train, _, _ = dl.load_dataset(ds_name)
        fig = go.Figure()
        # Increase the number of samples plotted to 15 to show more variety
        num_samples_to_plot = min(15, len(X_train))
        for i in range(num_samples_to_plot):
            series = X_train[i, :, 0]
            label = y_train[i]
            fig.add_trace(go.Scatter(y=series, mode='lines', name=f'Sample {i+1} (Class {label})'))
            
        fig.update_layout(title=f'Original Dataset: {ds_name} - {desc}',
                          xaxis_title='Time Steps', yaxis_title='Value', hovermode='x unified')
        save_plot(fig, f'dataset_{ds_name.lower()}.html')
    except Exception as e:
        print(f"Error in {ds_name}: {e}")

# --- 5. SUNSPOTS DATASET ---
try:
    sunspots = sm.datasets.sunspots.load_pandas().data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sunspots['YEAR'], y=sunspots['SUNACTIVITY'], mode='lines', name='Solar Activity', line=dict(color='orange')))
    fig.update_layout(title='Original Dataset: Sunspots (Natural Asymmetric Spikes)',
                      xaxis_title='Year', yaxis_title='Activity', hovermode='x unified')
    save_plot(fig, 'dataset_sunspots.html')
except Exception as e:
    print(f"Error in Sunspots: {e}")

print("Done plotting datasets.")
