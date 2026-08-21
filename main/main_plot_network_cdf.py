import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Para src/

import os
import json
import numpy as np
import pandas as pd
from src.estatisticas.metrics_utils import CRISIS_DATES, PRIMARY_BLUE, CRISIS_LINE_COLOR
from src.visualization.network_distance_kde_utils import calcular_kdes_das_janelas
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter, zoom

RESULTS_DIR = 'data/processed'
FIG_DIR = 'figures/statistics'
os.makedirs(FIG_DIR, exist_ok=True)

EXAMPLE_CRISIS_DATES = {
    '2008': pd.Timestamp('2008-10-06'),
    '2020': pd.Timestamp('2020-03-09'),
}

METRICS_FILES = {
    'Planar': 'network_metrics_planar_long.csv',
    'MST': 'network_metrics_mst_long.csv',
    'FilteredPval': 'cocitation_network_filtered_metrics_long.csv',
}

DISPLAY_NAMES = {
    'Planar': 'PMFG',
    'MST': 'MST',
    'FilteredPval': 'p-value-filtered network',
}


def add_crisis_planes_and_labels(ax, x_dense, y_dense, y_label):
    # Após rotação, marcar as datas de crise no eixo X com linhas verticais e texto no rodapé
    if y_label != 'Window end date' or len(CRISIS_DATES) == 0:
        return []

    # Detectar se datas estão no eixo X (rotacionado)
    # Se x_dense cobre datas (valores grandes tipo 7xxxx), então planos verticais em X
    is_dates_on_x = np.mean(x_dense) > 70000
    visible_crises = []

    if is_dates_on_x:
        x_min = float(np.min(x_dense))
        x_max = float(np.max(x_dense))
        y_min = float(np.min(y_dense))
        y_max = float(np.max(y_dense))
        z_min = 0.0
        z_max = 5.0
        for crisis_date in CRISIS_DATES:
            crisis_x = mdates.date2num(pd.to_datetime(crisis_date))
            if x_min <= crisis_x <= x_max:
                # Mantém marcação de crise dentro dos limites do eixo para evitar
                # expansão do bounding box e espaços em branco no topo.
                ax.plot(
                    [crisis_x, crisis_x],
                    [y_max, y_max],
                    [z_min, z_max],
                    color='red',
                    linestyle='--',
                    linewidth=1.0,
                    alpha=0.45,
                    zorder=8,
                )
    else:
        # Modo antigo: datas no eixo Y
        y_min = float(np.min(y_dense))
        y_max = float(np.max(y_dense))
        x_plane = np.array([[x_dense.min(), x_dense.max()], [x_dense.min(), x_dense.max()]])
        z_plane = np.array([[0.0, 0.0], [5.0, 5.0]])
        for crisis_date in CRISIS_DATES:
            crisis_y = mdates.date2num(pd.to_datetime(crisis_date))
            if y_min <= crisis_y <= y_max:
                y_plane = np.array([[crisis_y, crisis_y], [crisis_y, crisis_y]])
                ax.plot_surface(
                    x_plane,
                    y_plane,
                    z_plane,
                    color='red',
                    alpha=0.14,
                    linewidth=0,
                    shade=False,
                )
                visible_crises.append((pd.to_datetime(crisis_date), crisis_y))
        for idx, (crisis_date, crisis_y) in enumerate(visible_crises):
            x_text = float(x_dense.max()) * (0.985 - 0.045 * (idx % 2))
            z_text = 5.05 + 0.16 * (idx % 3)
            ax.text(
                x_text,
                crisis_y,
                z_text,
                str(crisis_date.year),
                color='darkred',
                fontsize=8,
                ha='right',
                va='bottom',
            )

    return visible_crises


def slice_surface_by_time_window(X, Y, Z, y_lower, y_upper):
    row_mask = (Y[:, 0] >= y_lower) & (Y[:, 0] <= y_upper)
    if np.count_nonzero(row_mask) < 2:
        return X, Y, Z
    return X[row_mask, :], Y[row_mask, :], Z[row_mask, :]


def _load_window_end_dates(window_ids):
    metadata_path = Path(RESULTS_DIR) / 'janelas_metadata.json'
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    except Exception:
        return None

    if isinstance(metadata, dict):
        rows = metadata.get('windows', [])
    elif isinstance(metadata, list):
        rows = metadata
    else:
        rows = []

    end_dates = {}
    for row in rows:
        try:
            wid_raw = row.get('janela_id', row.get('id'))
            end_raw = row.get('fim', row.get('end'))
            wid = str(int(wid_raw))
            end_dates[wid] = end_raw
        except Exception:
            continue

    y_labels = [pd.to_datetime(end_dates.get(str(int(wid))), errors='coerce') for wid in window_ids]
    if any(pd.isna(lbl) for lbl in y_labels):
        return None
    return mdates.date2num(y_labels)


def save_crisis_example_views(net_type, X, Y, Z, y_vals, y_label):
    if net_type != 'Planar' or y_label != 'Window end date':
        return

    if len(y_vals) == 0:
        return

    y_min = float(np.min(y_vals))
    y_max = float(np.max(y_vals))
    window_half_span_days = 420

    for label, crisis_date in EXAMPLE_CRISIS_DATES.items():
        crisis_y = mdates.date2num(crisis_date)
        if not (y_min <= crisis_y <= y_max):
            continue

        fig = plt.figure(figsize=(22, 14), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        y_lower = max(y_min, crisis_y - window_half_span_days)
        y_upper = min(y_max, crisis_y + window_half_span_days)
        X_zoom, Y_zoom, Z_zoom = slice_surface_by_time_window(X, Y, Z, y_lower, y_upper)

        surf = ax.plot_surface(
            X_zoom,
            Y_zoom,
            Z_zoom,
            cmap=cm.viridis,
            linewidth=0,
            antialiased=True,
            shade=True,
            rcount=min(500, Z_zoom.shape[0]),
            ccount=min(500, Z_zoom.shape[1]),
        )
        ax.set_xlabel('Correlation distance', fontsize=16, labelpad=12)
        ax.set_ylabel(y_label, fontsize=16, labelpad=12)
        ax.set_zlabel('KDE density', fontsize=16, labelpad=12)
        ax.set_title(f'3D distance KDE for the {DISPLAY_NAMES[net_type]} around the {label} crisis', fontsize=18, pad=18)
        ax.set_xlim(0, 2)
        ax.set_zlim(0, 5)
        add_crisis_planes_and_labels(ax, X_zoom[0], Y_zoom[:, 0], y_label)
        ax.set_ylim(y_lower, y_upper)
        ax.invert_yaxis()
        ax.view_init(elev=28, azim=-118)
        ax.set_box_aspect((1.1, 1.6, 1.0))
        ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.08, label='KDE density')
        plt.tight_layout()
        save_path = os.path.join(FIG_DIR, f'cdf3d_distancia_{net_type}_crise_{label}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Crisis example generated: {save_path}')

# Padrões dos arquivos edgelist por rede (ajuste nomes conforme seu pipeline)
edgelist_patterns = [
    (f'{RESULTS_DIR}/grafo_filtrado_pval_janela_*.csv', 'FilteredPval'),
    (f'{RESULTS_DIR}/grafo_planar_janela_*.csv', 'Planar'),
    (f'{RESULTS_DIR}/grafo_mst_janela_*.csv', 'MST'),
]


# Apenas PMFG (Planar) — estilo KS/MP: subplots verticais, largura total, altura generosa
pattern, net_type = edgelist_patterns[1]  # PMFG/Planar
x_vals, kdes_matrix, window_ids = calcular_kdes_das_janelas(pattern)
if kdes_matrix is None or kdes_matrix.shape[0] == 0:
    print(f"Nenhum kernel construído para {net_type}.")
else:
    try:
        metrics_file = os.path.join(RESULTS_DIR, METRICS_FILES[net_type])
        df_metrics = pd.read_csv(metrics_file)
        df_metrics['Janela_ID_str'] = df_metrics['Janela_ID'].astype(str)
        id_to_date = (
            df_metrics.drop_duplicates(subset=['Janela_ID_str'])
            .set_index('Janela_ID_str')['Window_End']
            .to_dict()
        )
        y_labels = [pd.to_datetime(id_to_date.get(str(wid), None), errors='coerce') for wid in window_ids]
        if any(pd.isna(lbl) for lbl in y_labels):
            y_vals = _load_window_end_dates(window_ids)
            if y_vals is None:
                y_vals = np.arange(len(window_ids))
                y_label = 'Window (index/ID)'
            else:
                y_label = 'Window end date'
        else:
            y_vals = mdates.date2num(y_labels)
            y_label = 'Window end date'
    except Exception:
        y_vals = _load_window_end_dates(window_ids)
        if y_vals is None:
            y_vals = np.arange(len(window_ids))
            y_label = 'Window (index/ID)'
        else:
            y_label = 'Window end date'

    Z_base = gaussian_filter(kdes_matrix, sigma=(2.2, 2.8))
    Z_dense = zoom(Z_base, zoom=(3.0, 2.0), order=3)
    x_dense = np.linspace(float(x_vals.min()), float(x_vals.max()), Z_dense.shape[1])
    y_dense = np.linspace(float(np.min(y_vals)), float(np.max(y_vals)), Z_dense.shape[0])
    # Rotacionar: datas no eixo X, distâncias no eixo Y
    X, Y = np.meshgrid(y_dense, x_dense)
    Z = np.clip(Z_dense.T, 0, 5)  # Transpor Z para alinhar

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(22, 13), dpi=300)
    ax3d = fig.add_subplot(111, projection='3d')
    surf = ax3d.plot_surface(
        X, Y, Z,
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True,
        shade=True,
        rcount=min(500, Z.shape[0]),
        ccount=min(500, Z.shape[1]),
    )
    ax3d.set_xlabel(y_label, fontsize=15, labelpad=8)
    ax3d.set_ylabel('Correlation distance', fontsize=15, labelpad=8)
    ax3d.set_zlabel('KDE density', fontsize=15, labelpad=8)
    # Keep panel clean for manuscript layout: no in-figure title.
    ax3d.set_xlim(float(np.min(y_dense)), float(np.max(y_dense)))
    ax3d.set_ylim(0, 2)
    ax3d.set_zlim(0, 5)
    # Restaurar proporção visual original (mais próximo de 1:1:0.7)
    ax3d.set_box_aspect((2.2, 1.0, 0.7))
    # Planos de crise agora no eixo X (datas)
    add_crisis_planes_and_labels(ax3d, y_dense, x_dense, y_label)
    if y_label == 'Window end date':
        ax3d.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3d.invert_yaxis()  # Mantém ordem das datas
    fig.colorbar(surf, ax=ax3d, shrink=0.72, pad=0.03, label='KDE density')
    ax3d.set_position([0.03, 0.08, 0.87, 0.88])
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.06)
    # Salva como o nome esperado pelo artigo (sobrescreve)
    save_path = os.path.join(FIG_DIR, f'cdf3d_distancia_{net_type}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    if save_path.lower().endswith('.png'):
        plt.savefig(save_path[:-4] + '.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"PMFG 3D KDE figure (rotated: dates on X) generated and overwritten: {save_path}")