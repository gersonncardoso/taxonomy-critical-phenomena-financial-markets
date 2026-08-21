import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.estatisticas.metrics_utils import get_network_metric_columns, plot_metric_timeseries

RESULTS_DIR = 'data/processed'
FIG_DIR = 'figures/networks'
os.makedirs(FIG_DIR, exist_ok=True)

network_metrics_files = [
    ('network_metrics_planar_long.csv', 'Planar'),
]

for fname, net_type in network_metrics_files:
    filepath = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(filepath):
        print(f"Arquivo não encontrado: {filepath}")
        continue
    df = pd.read_csv(filepath)
    metric_cols = get_network_metric_columns(df)
    for metric in metric_cols:
        plot_metric_timeseries(
            df=df, 
            metric_col=metric, 
            network_type=net_type, 
            save_dir=FIG_DIR
        )