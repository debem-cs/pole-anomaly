import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

def load_template(filepath):
    return pd.read_csv(filepath)

def generate_anomaly(template_df, amplitude, period, variance, sample_rate=1.0):
    """
    Generates a discrete anomaly array based on a template and parameters.
    template_df: pandas DataFrame with time, value, t_minus, t_plus, v_minus, v_plus
    amplitude: scaling factor for the anomaly values
    period: total time span of the anomaly (e.g., number of steps)
    variance: std deviation for time and value jitter (normalized scale 0-1)
    sample_rate: distance between discrete points (default 1 step)
    """
    n_points = len(template_df)
    t_jitter = np.zeros(n_points)
    v_jitter = np.zeros(n_points)
    
    for i in range(n_points):
        row = template_df.iloc[i]
        
        # Base independent samples representing the intrinsic variance energy
        raw_t = np.random.normal(0, variance)
        raw_v = np.random.normal(0, variance)
        
        # Time direction enforcement: Guarantee movement if a single direction is open
        if row['t_minus'] <= 0 and row['t_plus'] > 0:
            raw_t = np.abs(raw_t)
        elif row['t_plus'] <= 0 and row['t_minus'] > 0:
            raw_t = -np.abs(raw_t)
            
        # Value direction enforcement: Guarantee movement if a single direction is open
        if row['v_minus'] <= 0 and row['v_plus'] > 0:
            raw_v = np.abs(raw_v)
        elif row['v_plus'] <= 0 and row['v_minus'] > 0:
            raw_v = -np.abs(raw_v)
            
        # Apply the explicit multipliers from the CSV template
        t_jitter[i] = raw_t * row['t_plus'] if raw_t > 0 else raw_t * row['t_minus']
        v_jitter[i] = raw_v * row['v_plus'] if raw_v > 0 else raw_v * row['v_minus']

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
    t_discrete = np.arange(0, int(np.ceil(max_time)), sample_rate)
    v_discrete = np.zeros_like(t_discrete, dtype=float)
    
    # Piecewise interpolation across physical values
    modes = template_df['interp'].values if 'interp' in template_df.columns else ['linear'] * len(template_df)
    
    for k, t_val in enumerate(t_discrete):
        # find localized segment
        for index_row in range(len(t_physical) - 1):
            t0, t1 = t_physical[index_row], t_physical[index_row+1]
            if (t0 <= t_val <= t1) or (index_row == len(t_physical)-2 and t_val >= t1):
                v0, v1 = v_physical[index_row], v_physical[index_row+1]
                mode = modes[index_row]
                
                if t1 == t0:
                    v_discrete[k] = v1
                elif mode == 'exp':
                    x_norm = (t_val - t0) / (t1 - t0)
                    alpha = 3.0
                    y_norm = (np.exp(alpha * x_norm) - 1) / (np.exp(alpha) - 1)
                    v_discrete[k] = v0 + (v1 - v0) * y_norm
                else: # linear fallback
                    x_norm = (t_val - t0) / (t1 - t0)
                    v_discrete[k] = v0 + (v1 - v0) * x_norm
                break
    
    return t_discrete, v_discrete

if __name__ == "__main__":
    # Test execution
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    template_path = os.path.join(script_dir, '..', 'anomalies', 'exp_square.csv')
    
    if os.path.exists(template_path):
        df = load_template(template_path)
        
        # Test 10 generations to see the randomness
        fig = go.Figure()
        
        for i in range(5):
            t, v = generate_anomaly(df, amplitude=50, period=200, variance=0.05)
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
