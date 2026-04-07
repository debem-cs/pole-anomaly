import os
import sys
import glob
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure src is in the python path to import the generator library
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

from src.anomaly_generator import generate_anomaly

def test_anomaly_deformation():
    anomalies_dir = os.path.join(root_dir, 'anomalies')
    template_files = glob.glob(os.path.join(anomalies_dir, '*.csv'))
    
    if not template_files:
        print(f"No templates found in {anomalies_dir}")
        return
        
    num_templates = len(template_files)
    titles = [os.path.basename(f).replace('.csv', '').capitalize() for f in template_files]
    
    fig = make_subplots(rows=num_templates, cols=1, subplot_titles=titles)
    
    for i, template_path in enumerate(template_files, 1):
        df = pd.read_csv(template_path)
        
        # Test generations straight from the template to visualize deformation randomness
        for j in range(5):
            t, v = generate_anomaly(df, amplitude=50, period=200, variance=0.02)
            fig.add_trace(go.Scatter(
                x=t, 
                y=v, 
                mode='lines', 
                name=f'{titles[i-1]} Var {j+1}'
            ), row=i, col=1)
            
    fig.update_layout(
        title='Anomaly Template Deformation Test (All Templates)', 
        height=400 * num_templates,
        template="plotly_dark",
        hovermode='x unified'
    )
    
    logs_dir = os.path.join(root_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    out = os.path.join(logs_dir, 'anomaly_template_deformation_test.html')
    fig.write_html(out)
    print(f"Test visualization saved successfully to: {out}")

if __name__ == "__main__":
    test_anomaly_deformation()
