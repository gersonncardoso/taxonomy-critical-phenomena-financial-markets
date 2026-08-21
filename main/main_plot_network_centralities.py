import os
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.estatisticas.metrics_utils import CRISIS_DATES


MEAN_COLOR = '#0072B2'
MAX_COLOR = '#6C757D'
CRISIS_LINE_COLOR = '#7A7A7A'


RESULTS_DIR = 'data/processed'
FIG_DIR = 'figures/networks'
os.makedirs(FIG_DIR, exist_ok=True)
PLANAR_ADJUSTED_PATH = Path(RESULTS_DIR) / 'planar_centrality_size_adjusted_by_window.csv'
PLANAR_ADJUSTED_SUMMARY_PATH = Path(RESULTS_DIR) / 'planar_centrality_size_adjusted_summary.json'
PLANAR_PHASE_SUMMARY_PATH = Path(RESULTS_DIR) / 'planar_centrality_regime_summary.csv'
CONFIG_PATH = Path('configs/config.yaml')


CENTRALITY_SPECS = [
    ('degree', 'Degree'),
    ('betweenness_centrality', 'Betweenness centrality'),
    ('closeness_centrality', 'Closeness centrality'),
    ('eigenvector_centrality', 'Eigenvector centrality'),
]

CENTRALITY_FILES = [
    ('network_centralities_planar_long.csv', 'Planar'),
]


def _window_summary(df, metric_col):
    summary = (
        df.groupby(['Janela_ID', 'Window_Start', 'Window_End'], as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: 'observed'})
    )
    summary['Window_End'] = pd.to_datetime(summary['Window_End'])
    summary['n_nodes'] = df.groupby('Janela_ID').size().reindex(summary['Janela_ID']).to_numpy()
    return summary.sort_values('Window_End')


def _size_adjusted_summary(df, metric_col, network_type):
    summary = _window_summary(df, metric_col)
    if network_type == 'Planar' and metric_col == 'degree':
        summary['expected'] = 6.0 - 12.0 / summary['n_nodes']
        model = 'PMFG identity: 6 - 12/N'
    else:
        log_nodes = np.log1p(summary['n_nodes'].to_numpy(dtype=float))
        slope, intercept = np.polyfit(log_nodes, summary['observed'].to_numpy(dtype=float), deg=1)
        summary['expected'] = intercept + slope * log_nodes
        model = 'empirical expectation: a + b log(1 + N)'
    summary['residual'] = summary['observed'] - summary['expected']
    residual_std = float(summary['residual'].std(ddof=1))
    summary['residual_zscore'] = summary['residual'] / residual_std if residual_std > 0 else 0.0
    summary['network_type'] = network_type
    summary['metric'] = metric_col
    summary['model'] = model
    return summary


def _plot_centrality_timeseries(summary, network_type, metric_col, metric_label):

    fig, ax = plt.subplots(figsize=(16, 10.2))
    mean_line = ax.plot(
        summary['Window_End'],
        summary['residual'],
        label='Observed minus size-only expected',
        linewidth=2.2,
        color=MEAN_COLOR,
        marker='o',
        markersize=3.8,
        markeredgewidth=0,
    )[0]
    ax.fill_between(summary['Window_End'], summary['residual'], alpha=0.10, color=MEAN_COLOR)

    for crisis_date in CRISIS_DATES:
        ax.axvline(crisis_date, color=CRISIS_LINE_COLOR, linestyle='--', alpha=0.55, linewidth=1.0)

    y_min = float(summary['residual'].min())
    y_max = float(summary['residual'].max())
    y_span = max(y_max - y_min, 1.0e-9)
    y_pad = max(y_span * 0.12, max(abs(y_max), 1.0) * 0.02)

    ax.axhline(0.0, color='#495057', linestyle=':', linewidth=1.1)
    ax.set_title(f'Size-adjusted rolling {metric_label} — {network_type} network')
    ax.set_xlabel('Window end date')
    ax.set_ylabel('Observed minus N-expected')
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.grid(True, alpha=0.3)
    ax.legend([mean_line], ['Observed minus size-only expected'], loc='upper center')
    fig.tight_layout()

    filename = f'{network_type}_{metric_col}_evolucao.png'.replace(' ', '_')
    out_png = os.path.join(FIG_DIR, filename)
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    if out_png.lower().endswith('.png'):
        fig.savefig(out_png[:-4] + '.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _build_phase_summary(planar_adjusted):
    """Summarize residual centralities by explicit event-relative phases.

    Pre: window end in the six months before an event; during: window contains
    the event date; post: window start in the six months after the event.
    Rows remain event-specific so close events are never silently pooled.
    """
    if not CONFIG_PATH.exists():
        return pd.DataFrame()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8')) or {}
    event_dates = pd.to_datetime(config.get('eventos', {}).get('datas_crise', []), errors='coerce')
    event_dates = [date for date in event_dates if pd.notna(date)]
    if not event_dates:
        return pd.DataFrame()

    working = planar_adjusted.copy()
    working['Window_Start'] = pd.to_datetime(working['Window_Start'], errors='coerce')
    working['Window_End'] = pd.to_datetime(working['Window_End'], errors='coerce')
    rows = []
    for event_date in event_dates:
        phase = pd.Series(pd.NA, index=working.index, dtype='object')
        phase.loc[(working['Window_End'] < event_date) & (working['Window_End'] >= event_date - pd.DateOffset(months=6))] = 'pre'
        phase.loc[(working['Window_Start'] <= event_date) & (working['Window_End'] >= event_date)] = 'during'
        phase.loc[(working['Window_Start'] > event_date) & (working['Window_Start'] <= event_date + pd.DateOffset(months=6))] = 'post'
        event_frame = working.assign(event_date=event_date.strftime('%Y-%m-%d'), phase=phase).dropna(subset=['phase'])
        if event_frame.empty:
            continue
        summary = event_frame.groupby(['event_date', 'phase', 'metric'], as_index=False).agg(
            windows=('Janela_ID', 'nunique'),
            n_nodes_mean=('n_nodes', 'mean'),
            observed_mean=('observed', 'mean'),
            expected_mean=('expected', 'mean'),
            residual_mean=('residual', 'mean'),
            residual_zscore_mean=('residual_zscore', 'mean'),
        )
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


all_adjusted = []
for file_name, network_type in CENTRALITY_FILES:
    file_path = os.path.join(RESULTS_DIR, file_name)
    if not os.path.exists(file_path):
        print(f'Arquivo nao encontrado: {file_path}')
        continue

    df = pd.read_csv(file_path)
    if df.empty:
        print(f'Arquivo vazio: {file_path}')
        continue

    for metric_col, metric_label in CENTRALITY_SPECS:
        if metric_col not in df.columns:
            print(f'Coluna ausente em {file_name}: {metric_col}')
            continue
        summary = _size_adjusted_summary(df, metric_col, network_type)
        _plot_centrality_timeseries(summary, network_type, metric_col, metric_label)
        if network_type == 'Planar':
            all_adjusted.append(summary)

if all_adjusted:
    planar_adjusted = pd.concat(all_adjusted, ignore_index=True)
    wide = planar_adjusted[['Janela_ID', 'Window_Start', 'Window_End', 'n_nodes']].drop_duplicates()
    for metric, group in planar_adjusted.groupby('metric'):
        base_name = f'planar_{metric}_mean'
        metric_frame = group[
            ['Janela_ID', 'observed', 'expected', 'residual', 'residual_zscore']
        ].rename(
            columns={
                'observed': base_name,
                'expected': f'expected_{base_name}_size_only',
                'residual': f'{base_name}_residual_size_only',
                'residual_zscore': f'{base_name}_residual_size_only_zscore',
            }
        )
        wide = wide.merge(metric_frame, on='Janela_ID', how='left')
    wide = wide.sort_values('Janela_ID')
    wide.to_csv(PLANAR_ADJUSTED_PATH, index=False)
    diagnostics = {
        metric: {
            'model': group['model'].iloc[0],
            'windows': int(len(group)),
            'residual_mean': float(group['residual'].mean()),
            'residual_std': float(group['residual'].std(ddof=1)),
            'spearman_nodes_vs_residual': float(group['n_nodes'].corr(group['residual'], method='spearman')),
        }
        for metric, group in planar_adjusted.groupby('metric')
    }
    PLANAR_ADJUSTED_SUMMARY_PATH.write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
    phase_summary = _build_phase_summary(planar_adjusted)
    if not phase_summary.empty:
        phase_summary.to_csv(PLANAR_PHASE_SUMMARY_PATH, index=False)
