import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde

def calcular_kdes_das_janelas(
    edgelist_pattern,
    num_points=320,
    max_distance=2.0,
    bandwidth_scale=1.6,
    smooth_sigma_distance=1.2,
):
    """
    Para cada arquivo de edgelist (um por janela), calcula o kernel da coluna 'distance'.
    Retorna:
      x_vals: array dos valores para o eixo X (distância, igual para todos)
      kdes_matrix: array 2d (shape [n_janelas, len(x_vals)]) dos KDEs
      window_ids: lista dos IDs das janelas
    """
    file_list = sorted(glob.glob(edgelist_pattern))
    kdes_matrix = []
    window_ids = []
    all_vals = []
    for fname in file_list:
        df = pd.read_csv(fname)
        if 'distance' not in df.columns:
            continue
        vals = df['distance'].dropna().values
        if len(vals) < 2:
            continue
        all_vals.append(vals)
    # Determina faixa global dos valores com base em todas janelas
    if not all_vals:
        return None, None, None
    alltogether = np.concatenate(all_vals)
    x_min = 0.0
    x_max = min(max_distance, max(float(np.max(alltogether)), 1.8))
    x_vals = np.linspace(x_min, x_max, num_points)
    for fname in file_list:
        df = pd.read_csv(fname)
        if 'distance' not in df.columns:
            continue
        vals = df['distance'].dropna().values
        if len(vals) < 2:
            continue
        kde = gaussian_kde(vals, bw_method=lambda kde_obj: kde_obj.scotts_factor() * bandwidth_scale)
        kde_vals = kde(x_vals)
        kde_vals = gaussian_filter1d(kde_vals, sigma=smooth_sigma_distance)
        kdes_matrix.append(kde_vals)
        win_id = Path(fname).stem.split('_')[-1].replace('.csv', '')
        window_ids.append(win_id)
    kdes_matrix = np.array(kdes_matrix)
    return x_vals, kdes_matrix, window_ids