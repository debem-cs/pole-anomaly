import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go

# Set the working directory to access the 'data' directory correctly
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
base_dir = os.path.join(script_dir, '..', 'data')

# Nomes dos meses referenciados nas colunas
meses_nomes = {0: 'Fevereiro', 1: 'Abril', 2: 'Junho', 3: 'Outubro'}

# Load data files
data_dict = {}
file_mappings = {
    'Gamma': '2015_months_DebitDoseA.txt',
    'Temperatura': '2015_months_TEMP.txt',
    'Hygrometria': '2015_months_HYGR.txt',
    'Pression': '2015_months_PATM.txt'
}

for label, fname in file_mappings.items():
    filepath = os.path.join(base_dir, fname)
    try:
        data_dict[label] = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        data_dict[label] = None

# Create Plotly figure
fig = go.Figure()

Nh = 60  # Eliminate first and last hour
for label, data in data_dict.items():
    if data is not None:
        N = data.shape[0]
        start = Nh
        end = N - Nh
        t = np.arange(N)
        
        # Loop over each month column
        for mois, month_name in meses_nomes.items():
            if mois < data.shape[1]:
                sig = data[:, mois]
                
                # Apply special scaling/offset if needed, e.g., Pression - 800
                plot_y = sig[start:end]
                if label == 'Pression':
                    plot_y = plot_y - 800
                    name_label = f'{label} - 800 ({month_name})'
                else:
                    name_label = f'{label} ({month_name})'
                    
                fig.add_trace(go.Scatter(x=t[start:end], y=plot_y, mode='lines', name=name_label))

fig.update_layout(
    title='Dados do Sensor ao longo de múltiplos meses (2015)',
    xaxis_title='Tempo em min',
    hovermode='x unified'
)

# Save the figure as an interactive HTML file
logs_dir = os.path.join(script_dir, '..', 'logs')
os.makedirs(logs_dir, exist_ok=True)
output_path = os.path.join(logs_dir, 'affichage_donnees.html')
fig.write_html(output_path)
print(f"Interactive plot (All Months) saved successfully to: {output_path}")

