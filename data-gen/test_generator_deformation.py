import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure src is in the python path to import the generator library
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
root_dir = script_dir
sys.path.append(root_dir)

from anomaly_generator import generate_anomaly

def test_anomaly_deformation():
    anomalies_dir = os.path.join(root_dir, 'anomalies')
    template_files = glob.glob(os.path.join(anomalies_dir, '*.csv'))
    
    if not template_files:
        print(f"No templates found in {anomalies_dir}")
        return
        
    num_templates = len(template_files)
    titles = [os.path.basename(f).replace('.csv', '').capitalize() for f in template_files]
    
    plt.style.use('default')
    fig, axes = plt.subplots(num_templates, 1, figsize=(10, 4 * num_templates))
    if num_templates == 1:
        axes = [axes]
        
    fig_plotly = make_subplots(rows=num_templates, cols=1, subplot_titles=titles)
    
    for i, template_path in enumerate(template_files, 1):
        df = pd.read_csv(template_path)
        ax = axes[i-1]
        ax.set_title(titles[i-1])
        
        # Test generations straight from the template to visualize deformation randomness
        for j in range(6):
            std_noise = 2.53
            target_amplitude = std_noise * np.random.uniform(2.5, 7.6)
            target_period = np.random.randint(200, 800)
            variance_level = np.random.uniform(0.02, 0.10)

            t, v = generate_anomaly(df, amplitude=target_amplitude, period=target_period, variance=variance_level)
            ax.plot(t, v, label=f'Var {j+1}')
            
            fig_plotly.add_trace(go.Scatter(
                x=t, y=v, mode='lines', name=f'{titles[i-1]} Var {j+1}'
            ), row=i, col=1)
            
        ax.legend()
            
    plt.suptitle('Anomaly Template Deformation Test (All Templates)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    logs_dir = os.path.join(root_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    out_png = os.path.join(logs_dir, 'anomaly_template_deformation_test.png')
    plt.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close()
    
    fig_plotly.update_layout(
        title='Anomaly Template Deformation Test (All Templates)', 
        height=400 * num_templates,
        template="plotly_white",
        hovermode='x unified'
    )
    
    out = os.path.join(logs_dir, 'anomaly_template_deformation_test.html')
    fig_plotly.write_html(out)
    
    print(f"Test visualization saved successfully to: {out_png} and {out}")

if __name__ == "__main__":
    test_anomaly_deformation()
