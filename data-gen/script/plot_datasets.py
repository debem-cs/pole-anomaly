import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

def plot_legacy_datasets():
    dl = UCR_UEA_datasets()
    ucr_datasets = {
        'StarLightCurves': 'Starlight Curves (Photon Flux)',
        'CBF': 'Cylinder-Bell-Funnel (Synthetic)',
        'ECG5000': 'Heartbeats (Biological)',
        'Epilepsy': 'EEG Brainwaves (Neuroscience)'
    }
    
    titles = [f"{name} - {desc}" for name, desc in ucr_datasets.items()] + ["Sunspots - Solar Activity (Natural Asymmetric Spikes)"]
    num_plots = len(titles)
    
    fig = make_subplots(rows=num_plots, cols=1, subplot_titles=titles)
    row_idx = 1
    
    # 1 to 4. UCR DATASETS
    for ds_name, desc in ucr_datasets.items():
        try:
            X_train, y_train, _, _ = dl.load_dataset(ds_name)
            num_samples_to_plot = min(10, len(X_train)) # reduced slightly to avoid extreme clutter
            for i in range(num_samples_to_plot):
                series = X_train[i, :, 0]
                label = y_train[i]
                fig.add_trace(go.Scatter(y=series, mode='lines', name=f'{ds_name} Sample {i+1}'), row=row_idx, col=1)
        except Exception as e:
            print(f"Error in {ds_name}: {e}")
        row_idx += 1
            
    # 5. SUNSPOTS DATASET
    try:
        sunspots = sm.datasets.sunspots.load_pandas().data
        fig.add_trace(go.Scatter(
            x=sunspots['YEAR'], 
            y=sunspots['SUNACTIVITY'], 
            mode='lines', 
            name='Solar Activity', 
            line=dict(color='orange')
        ), row=row_idx, col=1)
    except Exception as e:
        print(f"Error in Sunspots: {e}")
        
    fig.update_layout(
        title='Legacy Datasets Overview',
        height=350 * num_plots,
        hovermode='x unified',
        template='plotly_dark'
    )
    
    out_path = os.path.join(logs_dir, 'legacy_datasets_overview.html')
    fig.write_html(out_path)
    print(f"Saved merged visual to: {out_path}")
    
    # Built-in cleanup of old individual html files
    for f in os.listdir(logs_dir):
        if f.startswith('dataset_') and f != 'legacy_datasets_overview.html' and f.endswith('.html'):
            try:
                os.remove(os.path.join(logs_dir, f))
                print(f"Cleaned up old log: {f}")
            except Exception as e:
                pass

if __name__ == "__main__":
    plot_legacy_datasets()
