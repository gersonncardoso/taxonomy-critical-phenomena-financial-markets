import matplotlib.pyplot as plt
import os
import pandas as pd

from src.utils.config import config


def get_network_metric_columns(df):
    metrics = [col for col in df.columns if any(key in col.lower() for key in [
        'mean', 'var', 'skew', 'kurt', 'degree', 'betw', 'clos', 'eigen', 'modular', 'num_comm', 'ricci', 'q10', 'q50', 'q90', 'neg_share',
        'clustering', 'transitivity', 'assort', 'diametro', 'distancia'
    ])]
    return metrics


def _get_crisis_dates():
    """Obtem lista de datas de crise a partir do config (se definida).

    Espera uma lista de strings em eventos.datas_crise, por exemplo:
      eventos:
        datas_crise:
          - "2015-01-01"
          - "2020-03-09"
    """
    try:
        datas = config.get('eventos.datas_crise', []) or []
        return [pd.to_datetime(d) for d in datas]
    except Exception:
        return []


CRISIS_DATES = _get_crisis_dates()

# Okabe-Ito palette for color-vision accessibility.
PRIMARY_BLUE = '#0072B2'
SECONDARY_ORANGE = '#E69F00'
CRISIS_LINE_COLOR = '#7A7A7A'

# Appendix-friendly canvas: wide and short so three figures can fit per page.
APPENDIX_FIGSIZE = (16, 10.2)
MAIN_TEXT_PAIR_METRICS = {'ricci_fr_mean', 'dist_mean'}


METRIC_LABELS = {
    'dist_mean': 'Mean financial distance',
    'dist_var': 'Distance variance',
    'dist_skew': 'Distance skewness',
    'dist_kurt': 'Distance kurtosis',
    'modularidade': 'Modularity',
    'num_communities': 'Number of communities',
    'clustering_medio': 'Mean clustering coefficient',
    'transitivity_global': 'Global transitivity',
    'assortatividade': 'Degree assortativity',
    'distancia_media': 'Mean shortest-path distance',
    'diametro': 'Network diameter',
    'degree': 'Degree',
    'betweenness_centrality': 'Betweenness centrality',
    'closeness_centrality': 'Closeness centrality',
    'eigenvector_centrality': 'Eigenvector centrality',
    'ricci_fr_mean': 'Mean Forman-Ricci curvature',
    'ricci_fr_var': 'Forman-Ricci curvature variance',
    'ricci_fr_std': 'Forman-Ricci curvature std. dev.',
    'ricci_fr_skew': 'Forman-Ricci curvature skewness',
    'ricci_fr_kurt': 'Forman-Ricci curvature kurtosis',
    'ricci_fr_q10': 'Forman-Ricci curvature Q10',
    'ricci_fr_q50': 'Forman-Ricci curvature median',
    'ricci_fr_q90': 'Forman-Ricci curvature Q90',
    'ricci_fr_iqr': 'Forman-Ricci curvature interquartile range',
    'ricci_fr_min': 'Forman-Ricci curvature min',
    'ricci_fr_max': 'Forman-Ricci curvature max',
}


def get_metric_label(metric_col):
    return METRIC_LABELS.get(metric_col, metric_col.replace('_', ' ').title())


def plot_metric_timeseries(df, metric_col, network_type, save_dir):
    if metric_col not in df.columns:
        return

    metric_label = get_metric_label(metric_col)

    # Garante ordenacao temporal consistente antes de plotar
    if 'Window_End' in df.columns:
        df = df.sort_values('Window_End')
    elif 'Window_Start' in df.columns:
        df = df.sort_values('Window_Start')
    elif 'Janela_ID' in df.columns:
        df = df.sort_values('Janela_ID')

    is_main_text_pair = (
        str(network_type).strip().lower() == 'planar'
        and metric_col in MAIN_TEXT_PAIR_METRICS
    )
    fig_size = (16, 6.2) if is_main_text_pair else APPENDIX_FIGSIZE
    title_fs = 16 if is_main_text_pair else 12
    label_fs = 13 if is_main_text_pair else 10
    tick_fs = 11 if is_main_text_pair else 8
    marker_sz = 0.0 if is_main_text_pair else 3.2
    line_w = 1.35 if is_main_text_pair else 1.9

    fig, ax = plt.subplots(figsize=fig_size, dpi=220)

    # Define eixo temporal baseado em datas, se disponivel
    x_label = "Janela ID"
    if 'Window_End' in df.columns:
        x_vals = pd.to_datetime(df['Window_End'])
        x_label = "Window end date"
    elif 'Window_Start' in df.columns:
        x_vals = pd.to_datetime(df['Window_Start'])
        x_label = "Window start date"
    else:
        x_vals = df['Janela_ID']
    if is_main_text_pair:
        ax.plot(
            x_vals,
            df[metric_col],
            color=PRIMARY_BLUE,
            linewidth=line_w,
            solid_capstyle='round',
        )
    else:
        ax.plot(
            x_vals,
            df[metric_col],
            color=PRIMARY_BLUE,
            linewidth=line_w,
            marker='o',
            markersize=marker_sz,
        )

    # Linhas verticais em datas de crise (se definidas e eixo for temporal)
    if isinstance(x_vals, (pd.Series, pd.DatetimeIndex)) and len(CRISIS_DATES) > 0:
        for d in CRISIS_DATES:
            try:
                ax.axvline(d, color=CRISIS_LINE_COLOR, linestyle='--', alpha=0.55, linewidth=1.0)
            except Exception:
                continue

    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.tick_params(axis='y', labelcolor=PRIMARY_BLUE)

    ax.set_title(f"Rolling {metric_label} — {network_type} network", fontsize=title_fs)
    ax.set_xlabel(x_label, fontsize=label_fs)
    ax.set_ylabel(metric_label, color=PRIMARY_BLUE, fontsize=label_fs)
    ax.tick_params(axis='y', labelcolor=PRIMARY_BLUE, labelsize=tick_fs)
    ax.tick_params(axis='x', labelsize=tick_fs)
    ax_right.set_ylabel(metric_label, color=PRIMARY_BLUE, fontsize=label_fs)
    ax_right.tick_params(axis='y', labelsize=tick_fs)
    ax.grid(True, alpha=0.3)
    # Fix y-axis for share metrics: always 0–1
    if 'neg_share' in metric_col or metric_col.endswith('_share'):
        ax.set_ylim(0, 1.05)
        ax_right.set_ylim(0, 1.05)
    # Format date x-axis when using dates
    if isinstance(x_vals, pd.Series) and pd.api.types.is_datetime64_any_dtype(x_vals):
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=tick_fs)
    fig.tight_layout()

    fname = f"{network_type}_{metric_col}_evolucao.png".replace(" ", "_")
    out_png = os.path.join(save_dir, fname)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    if out_png.lower().endswith('.png'):
        out_pdf = out_png[:-4] + '.pdf'
        fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
