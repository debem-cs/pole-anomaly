import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

def load_template(filepath):
    return pd.read_csv(filepath)

def generate_anomaly(template_df, amplitude, period, variance_t, variance_v, sample_rate=1.0):
    """
    Generates a discrete anomaly array based on a template and parameters.
    template_df: pandas DataFrame with time, value, t_minus, t_plus, v_minus, v_plus
    amplitude: scaling factor for the anomaly values
    period: total time span of the anomaly (e.g., number of steps)
    variance_t: std deviation for time jitter (normalized scale 0-1)
    variance_v: std deviation for value jitter (normalized scale 0-1)
    sample_rate: distance between discrete points (default 1 step)
    """
    n_points = len(template_df)
    t_jitter = np.zeros(n_points)
    v_jitter = np.zeros(n_points)
    
    for i in range(n_points):
        row = template_df.iloc[i]
        
        # Jitter Time depending on enabled quadrants
        if row['t_minus'] == 1 and row['t_plus'] == 1:
            t_jitter[i] = np.random.normal(0, variance_t)
        elif row['t_plus'] == 1:
            t_jitter[i] = np.abs(np.random.normal(0, variance_t))
        elif row['t_minus'] == 1:
            t_jitter[i] = -np.abs(np.random.normal(0, variance_t))
            
        # Jitter Value depending on enabled quadrants
        if row['v_minus'] == 1 and row['v_plus'] == 1:
            v_jitter[i] = np.random.normal(0, variance_v)
        elif row['v_plus'] == 1:
            v_jitter[i] = np.abs(np.random.normal(0, variance_v))
        elif row['v_minus'] == 1:
            v_jitter[i] = -np.abs(np.random.normal(0, variance_v))

    # Apply jitter to the normalized space
    t_new = template_df['time'].values + t_jitter
    v_new = template_df['value'].values + v_jitter
    
    # Failsafe: Ensure strict monotonicity in time so points never cross backwards
    for i in range(1, len(t_new)):
        if t_new[i] <= t_new[i-1]:
            # Force it to be strictly after the previous point
            t_new[i] = t_new[i-1] + 1e-5
            
    # Failsafe: Ensure time stays loosely within extreme bounds (e.g., no negative time)
    t_new = np.clip(t_new, 0.0, None)
            
    # Scale from Normalized to physical absolute sizes
    t_physical = t_new * period
    v_physical = v_new * amplitude
    
    # Evaluate on discrete sample grid
    max_time = np.max(t_physical)
    # create discrete time steps [0, 1, 2... max_time]
    t_discrete = np.arange(0, int(np.ceil(max_time)), sample_rate)
    
    # Linear interpolation to build the 1D signal
    v_discrete = np.interp(t_discrete, t_physical, v_physical)
    
    return t_discrete, v_discrete

if __name__ == "__main__":
    # Test execution
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    template_path = os.path.join(script_dir, '..', 'anomalies', 'square.txt')
    
    if os.path.exists(template_path):
        df = load_template(template_path)
        
        # Test 10 generations to see the randomness
        fig = go.Figure()
        
        for i in range(5):
            t, v = generate_anomaly(df, amplitude=50, period=200, variance_t=0.03, variance_v=0.1)
            fig.add_trace(go.Scatter(x=t, y=v, mode='lines', name=f'Anomalia Aleatória {i+1}'))
            
        fig.update_layout(
            title="Teste da Engine de Geração (Amplitude=50, Período=200, Jitter)", 
            template="plotly_dark",
            xaxis_title="Tempo Discreto (Steps)",
            yaxis_title="Dose Gama"
        )
        
        out = os.path.join(script_dir, '..', 'logs', 'generated_test.html')
        fig.write_html(out)
        print(f"Teste salvo com sucesso em: {out}")
    else:
        print(f"Template não encontrado: {template_path}")
