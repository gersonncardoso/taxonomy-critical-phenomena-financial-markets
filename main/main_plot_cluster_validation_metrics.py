import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.visualization.plots import plot_metric_evolution


FIG_DIR = Path('figures/validation')
CROSSMETHOD_CSV = FIG_DIR / 'crossmethod_summary.csv'
CLUSTER_STABILITY_CSV = FIG_DIR / 'cluster_stability.csv'


def _prepare_labels(df):
    if 'Window_End' in df.columns:
        return pd.to_datetime(df['Window_End']).dt.strftime('%Y-%m').tolist()
    if 'Window_Start' in df.columns:
        return pd.to_datetime(df['Window_Start']).dt.strftime('%Y-%m').tolist()
    return df['Janela_ID'].astype(str).tolist()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not CROSSMETHOD_CSV.exists():
        print(f'Arquivo nao encontrado: {CROSSMETHOD_CSV}')
        return

    df = pd.read_csv(CROSSMETHOD_CSV)
    if df.empty:
        print(f'Arquivo vazio: {CROSSMETHOD_CSV}')
        return

    if 'method' in df.columns:
        ward_df = df[df['method'].astype(str).str.lower() == 'ward'].copy()
    else:
        ward_df = df.copy()

    if ward_df.empty:
        print('Nenhuma linha do metodo ward encontrada em crossmethod_summary.csv')
        return

    ward_df['Janela_ID'] = pd.to_numeric(ward_df['Janela_ID'], errors='coerce')
    ward_df = ward_df.sort_values('Janela_ID')
    labels = _prepare_labels(ward_df)

    plot_metric_evolution(
        labels,
        ward_df['silhouette'].tolist(),
        'Silhouette score',
        FIG_DIR / 'silhouette_evolucao.png',
        '',
    )
    plot_metric_evolution(
        labels,
        ward_df['best_k'].tolist(),
        'Best K',
        FIG_DIR / 'bestk_evolucao.png',
        '',
    )
    plot_metric_evolution(
        labels,
        ward_df['modularity'].tolist(),
        'Modularity',
        FIG_DIR / 'modularidade_evolucao.png',
        '',
    )

    if CLUSTER_STABILITY_CSV.exists():
        ari_df = pd.read_csv(CLUSTER_STABILITY_CSV)
        if not ari_df.empty and 'ARI' in ari_df.columns:
            if 'Window_End' in ari_df.columns:
                ari_labels = pd.to_datetime(ari_df['Window_End']).dt.strftime('%Y-%m').tolist()
            else:
                janela_col = 'Janela_ID' if 'Janela_ID' in ari_df.columns else ari_df.columns[0]
                ari_labels = ari_df[janela_col].astype(str).tolist()
            plot_metric_evolution(
                ari_labels,
                ari_df['ARI'].tolist(),
                'ARI',
                FIG_DIR / 'ari_evolucao.png',
                'Cluster stability (ARI)',
            )


if __name__ == '__main__':
    main()