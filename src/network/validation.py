"""
Análise de Redes de Correlação — EVOLUÇÃO VERSÃO v3.0
Automático — reaproveitamento de funções do pacote src.network
Resultados para MST e Planar, validação estatística, comunidades, modularidade, métricas

Autor: Gerson Nassor Cardoso
Instituição: Universidade Federal de São Paulo (UNIFESP)
Data: 2026-02-21
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.logger import setup_logger

# IMPORTAÇÕES DE FUNÇÕES DE REDE
from src.network.graph_builder import construir_grafo_mst, construir_grafo_planar
from src.network.metrics import calcular_metricas_globais, calcular_metricas_centralidade
from src.network.communities import detectar_comunidades_louvain, analisar_modularidade
from src.network.network_validation import validar_rede
from src.visualization.plots import plot_network_with_pvalues

logger = setup_logger('main_network_analysis')

def salvar_metricas_em_csv_long(metrics_dict, window_id, rede_tipo, window_start, window_end, csv_path):
    """
    Adiciona métricas globais + comunidades ao CSV long.
    """
    metrics_row = metrics_dict.copy()
    metrics_row["Janela_ID"] = window_id
    metrics_row["Tipo_Rede"] = rede_tipo
    metrics_row["Window_Start"] = window_start
    metrics_row["Window_End"] = window_end

    df_row = pd.DataFrame([metrics_row])
    with open(csv_path, 'a') as f:
        df_row.to_csv(f, index=False, header=f.tell()==0)

def salvar_centralidades_csv_long(df_centralidade, window_id, rede_tipo, window_start, window_end, csv_path):
    """
    Salva centralidade dos nós em formato long.
    """
    df_centralidade_long = df_centralidade.reset_index().rename(columns={"index": "Node"})
    df_centralidade_long["Janela_ID"] = window_id
    df_centralidade_long["Tipo_Rede"] = rede_tipo
    df_centralidade_long["Window_Start"] = window_start
    df_centralidade_long["Window_End"] = window_end

    with open(csv_path, 'a') as f:
        df_centralidade_long.to_csv(f, index=False, header=f.tell()==0)

def main(): 
    print("\n" + "="*80)
    print("🕸️  ANÁLISE DE REDES SISTÊMICAS — EVOLUÇÃO v3.0")
    print("="*80)
    print("  • Reaproveita funções de src.network")
    print("  • Resultados para MST e Planar")
    print("  • Validação estatística e comunidades")
    print("  • Salva CSVs long por rede")
    print("="*80)
    print()

    consolidado_file = Path('data/processed/dados_consolidados.csv')
    if not consolidado_file.exists():
        print("❌ Arquivo consolidado não encontrado!")
        print(f"Esperado: {consolidado_file}")
        return False

    # Define diretórios/output
    results_dir = Path('data/processed/')
    figures_dir = Path('figures/networks')
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_long_mst = results_dir / 'network_metrics_mst_long.csv'
    csv_long_planar = results_dir / 'network_metrics_planar_long.csv'
    csv_centralidade_mst = results_dir / 'network_centralities_mst_long.csv'
    csv_centralidade_planar = results_dir / 'network_centralities_planar_long.csv'
    csv_validacao_planar = results_dir / 'network_validation_planar_long.csv'
    csv_validacao_mst = results_dir / 'network_validation_mst_long.csv'

    # Limpa CSVs long para nova execução
    for file in [csv_long_mst, csv_long_planar, csv_centralidade_mst, csv_centralidade_planar, csv_validacao_planar, csv_validacao_mst]:
        if file.exists(): file.unlink()

    print("📄 Carregando dados...")
    df = pd.read_csv(consolidado_file, parse_dates=['Date'])
    window_ids = sorted(df['Janela_ID'].dropna().unique())
    print(f"   Registros: {len(df):,}, Janelas: {len(window_ids)}, Tickers: {df['Ticker'].nunique()}")
    print()

    for window_id in tqdm(window_ids, desc="Janelas de análise", unit="janela"):
        df_window = df[df['Janela_ID'] == window_id].copy()
        window_start = df_window['Window_Start'].iloc[0] if 'Window_Start' in df_window.columns else None
        window_end = df_window['Window_End'].iloc[0] if 'Window_End' in df_window.columns else None

        tickers = sorted(df_window['Ticker'].unique())
        corrs = df_window.pivot(index='Date', columns='Ticker', values='Retorno_Log').corr()
        # Distância de Gower a partir da correlação: d_ij = sqrt(2 * (1 - rho_ij))
        dist_matrix = (2 * (1 - corrs.fillna(0))).pow(0.5)

        # -- MST --
        G_mst = construir_grafo_mst(dist_matrix)
        metrics_mst = calcular_metricas_globais(G_mst)
        df_cent_mst = calcular_metricas_centralidade(G_mst)
        comm_mst = detectar_comunidades_louvain(G_mst)
        modularidade_mst = analisar_modularidade(G_mst, comm_mst)
        metrics_mst["modularidade"] = modularidade_mst
        metrics_mst["num_communities"] = len(set(comm_mst.values()))

        salvar_metricas_em_csv_long(metrics_mst, window_id, 'MST', window_start, window_end, csv_long_mst)
        salvar_centralidades_csv_long(df_cent_mst, window_id, 'MST', window_start, window_end, csv_centralidade_mst)

        # -- Planar --
        G_planar = construir_grafo_planar(dist_matrix)
        metrics_planar = calcular_metricas_globais(G_planar)
        df_cent_planar = calcular_metricas_centralidade(G_planar)
        comm_planar = detectar_comunidades_louvain(G_planar)
        modularidade_planar = analisar_modularidade(G_planar, comm_planar)
        metrics_planar["modularidade"] = modularidade_planar
        metrics_planar["num_communities"] = len(set(comm_planar.values()))

        salvar_metricas_em_csv_long(metrics_planar, window_id, 'Planar', window_start, window_end, csv_long_planar)
        salvar_centralidades_csv_long(df_cent_planar, window_id, 'Planar', window_start, window_end, csv_centralidade_planar)

        # -- Validação estatística --
        try:
            valid_planar = validar_rede(G_planar, n_amostras=100, seed=42)
            pd.DataFrame([{'Janela_ID': window_id, 'Window_Start': window_start, 'Window_End': window_end, **valid_planar['small_world']}]).to_csv(csv_validacao_planar, mode='a', index=False, header=not csv_validacao_planar.exists())
        except Exception as e:
            logger.warning(f'Falha validação planar janela {window_id}: {e}')
        try:
            valid_mst = validar_rede(G_mst, n_amostras=100, seed=42)
            pd.DataFrame([{'Janela_ID': window_id, 'Window_Start': window_start, 'Window_End': window_end, **valid_mst['small_world']}]).to_csv(csv_validacao_mst, mode='a', index=False, header=not csv_validacao_mst.exists())
        except Exception as e:
            logger.warning(f'Falha validação mst janela {window_id}: {e}')

        # -- Visualizações de rede --
        if window_id == window_ids[0] or window_id == window_ids[-1] or int(window_id) % 10 == 0:
            figures_dir.mkdir(parents=True, exist_ok=True)
            try:
                plot_network_with_pvalues(
                    G_planar,
                    pval_dict={},
                    threshold=0.05,
                    output_path=figures_dir / f'rede_planar_janela_{int(window_id):03d}.png',
                    title=f'Rede Planar - Janela {window_id}',
                    color_mode='ricci_groups',
                    show_pvalue_styles=False,
                )
            except Exception as e:
                logger.warning(f"Falha ao salvar figura Planar {window_id}: {e}")

            try:
                plot_network_with_pvalues(
                    G_mst,
                    pval_dict={},
                    threshold=0.05,
                    output_path=figures_dir / f'rede_mst_janela_{int(window_id):03d}.png',
                    title=f'Rede MST - Janela {window_id}',
                    color_mode='ricci_groups',
                    show_pvalue_styles=False,
                )
            except Exception as e:
                logger.warning(f"Falha ao salvar figura MST {window_id}: {e}")

    print("\n✅ Processamento completo!")
    print(f"   Métricas MST long:    {csv_long_mst}")
    print(f"   Métricas Planar long: {csv_long_planar}")
    print(f"   Centralidades MST:    {csv_centralidade_mst}")
    print(f"   Centralidades Planar: {csv_centralidade_planar}")
    print(f"   Validação Planar:     {csv_validacao_planar}")
    print(f"   Validação MST:        {csv_validacao_mst}")
    print(f"   Figuras:              {figures_dir}/*.png")

if __name__ == "__main__":
    main()