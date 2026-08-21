import sys
from pathlib import Path

# Cirúrgico: garanta que src/ está no sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import json
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import concurrent.futures
import shutil
import gc
import warnings
warnings.filterwarnings('ignore')

import matplotlib

# Usa backend não interativo para evitar dependência de Tk/Tcl
matplotlib.use("Agg")

from src.utils.logger import setup_logger
from src.utils.gpu_utils import GPU_AVAILABLE, cp

from src.network.graph_builder import construir_grafo_completo, construir_grafo_mst, construir_grafo_planar
from src.network.metrics import calcular_metricas_globais, calcular_metricas_centralidade
from src.network.communities import detectar_comunidades_louvain, analisar_modularidade
from src.network.net_utils import (
    salvar_metricas_em_csv_long,
    salvar_centralidades_csv_long,
    salvar_grafo_csv_edgelist,
    construir_grafo_filtrado_pvalue
)
# Importa visualização científica (arestas pontilhadas para p >= threshold)
from src.visualization.plots import plot_network_with_pvalues

logger = setup_logger('main_network_analysis')


def _format_window_label(
    window_id: int | None = None,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
    prefix: str | None = None,
) -> str:
    """Monta rótulo temporal priorizando datas da janela."""
    start_ts = pd.to_datetime(window_start, errors='coerce') if window_start is not None else pd.NaT
    end_ts = pd.to_datetime(window_end, errors='coerce') if window_end is not None else pd.NaT

    if pd.notna(start_ts) and pd.notna(end_ts):
        label = f"{start_ts:%Y-%m-%d} to {end_ts:%Y-%m-%d}"
    elif pd.notna(end_ts):
        label = f"until {end_ts:%Y-%m-%d}"
    elif pd.notna(start_ts):
        label = f"from {start_ts:%Y-%m-%d}"
    elif window_id is not None:
        label = f"ID {int(window_id)}"
    else:
        label = "date unavailable"

    return f"{prefix} ({label})" if prefix else label


def _load_window_bounds_from_metadata(metadata_path: Path) -> dict[int, tuple[pd.Timestamp, pd.Timestamp]]:
    """Carrega universo de janelas (ID -> início/fim) a partir do metadata."""
    if not metadata_path.exists():
        return {}
    try:
        obj = json.loads(metadata_path.read_text(encoding='utf-8'))
    except Exception:
        return {}

    windows = obj.get('windows', []) if isinstance(obj, dict) else []
    out: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for w in windows:
        try:
            wid = int(w['id'])
            ws = pd.to_datetime(w['start'])
            we = pd.to_datetime(w['end'])
            out[wid] = (ws, we)
        except Exception:
            continue
    return out


def _empty_network_metrics(reason: str) -> dict:
    """Métricas placeholder para janelas sem dados úteis."""
    return {
        'n_nodes': 0,
        'n_edges': 0,
        'densidade': 0.0,
        'grau_medio': np.nan,
        'grau_max': np.nan,
        'n_componentes': 0,
        'tamanho_maior_componente': 0,
        'clustering_medio': np.nan,
        'transitivity_global': np.nan,
        'assortatividade': np.nan,
        'distancia_media': np.nan,
        'diametro': np.nan,
        'dist_mean': np.nan,
        'dist_var': np.nan,
        'dist_std': np.nan,
        'dist_skew': np.nan,
        'dist_kurt': np.nan,
        'modularidade': np.nan,
        'num_communities': 0,
        'modeled': 0,
        'model_reason': reason,
    }


def _analise_grafos_filtrados(pvals_df: pd.DataFrame, window_ids, pvalue_threshold: float = 0.05):
    """Explora o tamanho esperado dos grafos FILTRADOS por p-valor.

    Calcula, por janela, número de arestas significativas (p < limiar) e
    número de nós envolvidos nessas arestas. Retorna um resumo estatístico
    e uma recomendação qualitativa de custo de processamento.
    """
    # Garante que temos uma cópia apenas das janelas relevantes
    df_sub = pvals_df[pvals_df['Janela_ID'].isin(window_ids)].copy()
    df_sub = df_sub[(df_sub['pvalue'] < pvalue_threshold) & (df_sub['Ticker1'] != df_sub['Ticker2'])]

    if df_sub.empty:
        print("\n📊 Análise exploratória dos grafos FILTRADOS por p-valor:")
        print("   Nenhuma aresta com p-value abaixo do limiar especificado.")
        return False, "Não há arestas significativas; não faz sentido calcular métricas completas.", False

    # Arestas por janela
    edges_per_win = df_sub.groupby('Janela_ID').size()

    # Nós por janela (considerando apenas nós com ao menos uma aresta significativa)
    nodes_long = pd.concat([
        df_sub[['Janela_ID', 'Ticker1']].rename(columns={'Ticker1': 'Ticker'}),
        df_sub[['Janela_ID', 'Ticker2']].rename(columns={'Ticker2': 'Ticker'}),
    ], ignore_index=True).drop_duplicates()
    nodes_per_win = nodes_long.groupby('Janela_ID')['Ticker'].nunique()

    # Alinha em relação às janelas efetivamente usadas na Fase 5
    edges_per_win = edges_per_win.reindex(window_ids, fill_value=0)
    nodes_per_win = nodes_per_win.reindex(window_ids, fill_value=0)

    # Estatísticas básicas
    e_min, e_max, e_mean = edges_per_win.min(), edges_per_win.max(), edges_per_win.mean()
    n_min, n_max, n_mean = nodes_per_win.min(), nodes_per_win.max(), nodes_per_win.mean()

    # Estimativa grosseira de esforço relativo ~ |V| * |E|
    complexity = nodes_per_win * edges_per_win
    c_max = complexity.max()

    first_win = window_ids[0]
    last_win = window_ids[-1]
    e_first, n_first = edges_per_win.loc[first_win], nodes_per_win.loc[first_win]
    e_last, n_last = edges_per_win.loc[last_win], nodes_per_win.loc[last_win]

    print("\n📊 Análise exploratória dos grafos FILTRADOS por p-valor (para métricas completas):")
    print(f"   Janelas avaliadas: {len(window_ids)}")
    print(f"   Arestas filtradas por janela (min/máx/média): {e_min} / {e_max} / {e_mean:.1f}")
    print(f"   Nós com pelo menos uma aresta (min/máx/média): {n_min} / {n_max} / {n_mean:.1f}")
    print(f"   Primeira janela (ID {first_win}): nós={n_first}, arestas={e_first}")
    print(f"   Última  janela (ID {last_win}): nós={n_last}, arestas={e_last}")
    print(f"   Estimativa grosseira de trabalho (máx |V|*|E|): {int(c_max):,}")

    # Heurística simples de recomendação
    if e_max <= 2_000:
        recomendacao = "Carga leve: recomendação geral = SIM (rodar métricas completas para grafos filtrados)."
        rec_flag = True
    elif e_max <= 10_000:
        recomendacao = (
            "Carga moderada: recomendação = depende — rode se precisar mesmo das métricas "
            "detalhadas para grafos filtrados."
        )
        rec_flag = False
    else:
        recomendacao = (
            "Carga pesada: recomendação geral = NÃO rodar métricas completas para grafos filtrados, "
            "pois os grafos ainda são grandes."
        )
        rec_flag = False

    print(f"\n💡 {recomendacao}")
    return True, recomendacao, rec_flag


def _salvar_grafo_imagem(
    G: nx.Graph,
    window_id: int,
    tipo_rede: str,
    posicao: str,
    output_dir: Path,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
):
    """Salva visualização do grafo (primeira/última janela) com estilo unificado."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        window_label = _format_window_label(
            window_id=window_id,
            window_start=window_start,
            window_end=window_end,
            prefix=f"{posicao.capitalize()} janela",
        )

        fname = output_dir / f"{tipo_rede}_{posicao}_janela_{int(window_id)}.png"
        plot_network_with_pvalues(
            G,
            pval_dict={},
            threshold=0.05,
            output_path=fname,
            title=f"{tipo_rede} – {window_label}",
            color_mode='ricci_groups',
            show_pvalue_styles=False,
        )

        # Versão com nome canônico (sem ID) para uso direto no paper
        canonical_fname = output_dir / f"{tipo_rede}_{posicao}_janela.png"
        try:
            plot_network_with_pvalues(
                G,
                pval_dict={},
                threshold=0.05,
                output_path=canonical_fname,
                title=f"{tipo_rede} – {window_label}",
                color_mode='ricci_groups',
                show_pvalue_styles=False,
            )
        except Exception as e:
            logger.error(f"Erro ao salvar imagem canônica do grafo {tipo_rede} ({posicao}) da janela {window_id}: {e}")

        logger.info(f"Grafo {tipo_rede} ({posicao}) salvo em: {fname} (canônico: {canonical_fname})")
    except Exception as e:
        logger.error(f"Erro ao salvar imagem do grafo {tipo_rede} ({posicao}) da janela {window_id}: {e}")


def calcular_metricas_todas(G):
    """Calcula métricas globais + centralidades + comunidades.

    Quando a pilha RAPIDS/cuGraph está disponível, paralelizamos as partes
    intensivas em GPU (centralidades/Louvain) e as métricas globais em CPU
    usando threads, para melhor aproveitar CPU+GPU na FASE 5.
    """
    results = {}

    # Sem GPU/cuGraph: mantém fluxo sequencial simples.
    if not GPU_AVAILABLE:
        metricas = calcular_metricas_globais(G)
        df_cent = calcular_metricas_centralidade(G)
        comm = detectar_comunidades_louvain(G)
        modularidade = analisar_modularidade(G, comm)
    else:
        # Com GPU disponível: roda globais (CPU) e centrais+comunidades (GPU/CPU)
        # em paralelo com threads.
        import concurrent.futures

        def _globais():
            return calcular_metricas_globais(G)

        def _centrais_e_comunidades():
            df_c = calcular_metricas_centralidade(G)
            comm_local = detectar_comunidades_louvain(G)
            mod_local = analisar_modularidade(G, comm_local)
            return df_c, comm_local, mod_local

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_globais = executor.submit(_globais)
            fut_cent_comm = executor.submit(_centrais_e_comunidades)

            metricas = fut_globais.result()
            df_cent, comm, modularidade = fut_cent_comm.result()

    # Preenche estrutura de saída padronizada
    if 'modularidade' not in metricas:
        metricas["modularidade"] = modularidade
    metricas["num_communities"] = len(set(comm.values())) if comm else 0

    results["metrics"] = metricas
    results["df_cent"] = df_cent
    results["comm"] = comm
    results["modularidade"] = modularidade
    return results

def processar_janela(args):
    (window_id, df_window, pvals_win, output_dir, temp_dir, calc_metricas_filtrado, window_bounds) = args
    win_start_meta, win_end_meta = window_bounds
    window_start = win_start_meta
    window_end = win_end_meta
    if (window_start is None) and ('Window_Start' in df_window.columns) and (not df_window.empty):
        window_start = df_window['Window_Start'].iloc[0]
    if (window_end is None) and ('Window_End' in df_window.columns) and (not df_window.empty):
        window_end = df_window['Window_End'].iloc[0]
    tickers = sorted(df_window['Ticker'].unique()) if ('Ticker' in df_window.columns and not df_window.empty) else []

    # Logs detalhados por janela são opcionais para não poluir a saída do tqdm.
    # NETWORK_ANALYSIS_STEP_LOG=1 ativa; por padrão fica silencioso.
    step_log_enabled = os.getenv('NETWORK_ANALYSIS_STEP_LOG', '0') == '1'
    step_log_transient = os.getenv('NETWORK_ANALYSIS_STEP_LOG_TRANSIENT', '1') == '1'

    def _step_log(message: str, final: bool = False):
        if not step_log_enabled:
            return
        text = f"[Janela {window_id}] {message}"
        if step_log_transient:
            if final:
                print(text)
            else:
                # Atualização em linha única para não criar centenas de linhas.
                pad = ' ' * max(0, 160 - len(text))
                print(f"\r{text}{pad}", end='', flush=True)
        else:
            print(text)

    _step_log(f"Início do processamento - registros={len(df_window)}, tickers_unicos={len(tickers)}")

    if df_window.empty:
        _step_log("Janela sem registros após filtros da fase de consolidação.")

        G_empty = nx.Graph()
        salvar_grafo_csv_edgelist(G_empty, output_dir / f'grafo_planar_janela_{window_id}.csv')

        empty_metrics = _empty_network_metrics('empty_window_after_filters')
        salvar_metricas_em_csv_long(
            empty_metrics, window_id, 'Planar', window_start, window_end,
            temp_dir / f'metrics_planar_{window_id}.csv', append=False, include_header=True
        )

        empty_cent = pd.DataFrame(columns=[
            'degree_centrality',
            'betweenness_centrality',
            'closeness_centrality',
            'eigenvector_centrality',
            'degree',
        ])
        salvar_centralidades_csv_long(
            empty_cent, window_id, 'Planar', window_start, window_end,
            temp_dir / f'centralities_planar_{window_id}.csv', append=False, include_header=True
        )

        (temp_dir / f'done_{window_id}.flag').touch()
        _step_log("Janela marcada como concluída (placeholder).", final=True)
        return (
            window_id,
            {
                'planar': G_empty,
            }
        )

    # 1) MATRIZ DE CORRELAÇÃO: reaproveita correlações já calculadas na FASE 4
    _step_log("1/6 - Reconstruindo matriz de correlação a partir de pvalues_long.csv...")

    if pvals_win.empty:
        # Fallback de segurança: se por algum motivo não houver p-values para a janela,
        # recalcule a correlação diretamente dos retornos (com aviso).
        _step_log("[WARN] Nenhum registro em pvalues_long para esta janela; recalculando correlação a partir dos retornos.")
        pivot = df_window.pivot_table(
            index='Date',
            columns='Ticker',
            values='Retorno_Log',
            aggfunc='mean',
        )
        corrs = pivot.corr()
    else:
        # Reconstrói a matriz de correlação usando as colunas 'Ticker1', 'Ticker2' e 'Correlacao'
        tickers_corr = sorted(set(pvals_win['Ticker1']).union(pvals_win['Ticker2']))
        corr_mat = pvals_win.pivot(index='Ticker1', columns='Ticker2', values='Correlacao')
        corr_mat = corr_mat.reindex(index=tickers_corr, columns=tickers_corr).fillna(1.0)
        corrs = corr_mat

    _step_log(f"1/6 - Matriz de correlação pronta (shape={corrs.shape}).")
    # Distância de Gower: d_ij = sqrt(2 * (1 - rho_ij))
    corrs_filled = corrs.fillna(0)
    dist_matrix = (2 * (1 - corrs_filled)).pow(0.5)
    
    # COMPLETO: construído apenas como base para o grafo filtrado por p-valor.
    # Não salvamos edgelist, métricas nem figuras para o grafo completo.
    _step_log("2/6 - Construindo grafo completo (base para filtrado)...")
    G_completo = construir_grafo_completo(dist_matrix)
    _step_log("2/6 - Grafo completo em memória (não persistido).")

    # FILTRADO PVALUE: aplica filtragem de p-value de forma vetorizada,
    # evitando loops Python sobre cada aresta do grafo completo.
    _step_log("3/6 - Aplicando filtro de p-value às arestas...")

    # Apenas pares significativos (p < 0.05 e Ticker1 != Ticker2)
    pvals_sig = pvals_win[(pvals_win['pvalue'] < 0.05) & (pvals_win['Ticker1'] != pvals_win['Ticker2'])]

    tickers_all = dist_matrix.index

    if pvals_sig.empty:
        # Nenhuma aresta significativa: grafo filtrado vazio (apenas nós)
        G_filtrado = nx.Graph()
        G_filtrado.add_nodes_from(tickers_all)
        dist_matrix_filtrada = pd.DataFrame(0.0, index=tickers_all, columns=tickers_all)
    else:
        # Matriz de p-values alinhada aos tickers da matriz de distância
        pval_mat = pvals_sig.pivot(index='Ticker1', columns='Ticker2', values='pvalue')
        pval_mat = pval_mat.reindex(index=tickers_all, columns=tickers_all)

        # Simetriza (caso tenhamos apenas metade da matriz)
        pval_sym = pval_mat.combine_first(pval_mat.T)

        # Máscara booleana de arestas mantidas (cópia mutável para evitar array read-only).
        mask_values = (pval_sym < 0.05).to_numpy(copy=True)
        # Remove diagonal
        mask_values[np.arange(len(tickers_all)), np.arange(len(tickers_all))] = False
        mask = pd.DataFrame(mask_values, index=tickers_all, columns=tickers_all)

        # Matriz de distâncias filtrada: mantém distâncias apenas onde mask=True
        dist_matrix_filtrada = dist_matrix.where(mask, 0.0)

        # Constrói grafo filtrado diretamente da matriz de distâncias filtrada
        G_filtrado = construir_grafo_completo(dist_matrix_filtrada)

    salvar_grafo_csv_edgelist(G_filtrado, output_dir / f'grafo_filtrado_pval_janela_{window_id}.csv')
    _step_log("3/6 - Grafo filtrado por p-value salvo.")

    # Removido: cálculo de métricas para grafos filtrados e completos. Apenas planar.

    # NOTA: Planar opera sobre a matriz de distâncias COMPLETA (sem filtro p-valor).
    # O filtro p-valor é aplicado apenas ao grafo filtrado auxiliar.
    _step_log("4/5 - Etapa de pré-filtro ignorada para planar (usa matriz completa).")

    # PLANAR — usa matriz de distâncias completa (PMFG clássico sem pré-filtro)
    _step_log("5/5 - Construindo grafo planar e calculando métricas/centralidades...")
    G_planar = construir_grafo_planar(dist_matrix)
    metricas_planar = calcular_metricas_todas(G_planar)
    metricas_planar["metrics"]["modeled"] = 1
    metricas_planar["metrics"]["model_reason"] = ""
    salvar_metricas_em_csv_long(metricas_planar["metrics"], window_id, 'Planar', window_start, window_end,
                               temp_dir / f'metrics_planar_{window_id}.csv', append=False, include_header=True)
    salvar_centralidades_csv_long(metricas_planar["df_cent"], window_id, 'Planar', window_start, window_end,
                                 temp_dir / f'centralities_planar_{window_id}.csv', append=False, include_header=True)

    salvar_grafo_csv_edgelist(G_planar, output_dir / f'grafo_planar_janela_{window_id}.csv')
    _step_log("5/5 - Grafo planar salvo (métricas e centralidades registradas).")

    # --- Visualização científica: PMFG com p-valor (arestas pontilhadas para p >= threshold) ---
    # Monta dicionário {(u,v): pvalue} para todas as arestas do grafo planar
    pval_dict = {}
    if not pvals_win.empty:
        # Cria dicionário de p-valor para lookup rápido
        pval_lookup = {(row['Ticker1'], row['Ticker2']): row['pvalue'] for _, row in pvals_win.iterrows()}
        pval_lookup.update({(row['Ticker2'], row['Ticker1']): row['pvalue'] for _, row in pvals_win.iterrows()})
        for u, v in G_planar.edges():
            pval = pval_lookup.get((u, v), np.nan)
            pval_dict[(u, v)] = pval
    else:
        for u, v in G_planar.edges():
            pval_dict[(u, v)] = np.nan
    fig_net_dir = Path('figures/networks')
    fig_net_dir.mkdir(parents=True, exist_ok=True)
    img_path = fig_net_dir / f'planar_pval_janela_{window_id}.png'
    plot_network_with_pvalues(
        G_planar,
        pval_dict,
        threshold=0.05,
        output_path=img_path,
        title=f"PMFG (Planar) – {_format_window_label(window_id, window_start, window_end)}",
        node_color='skyblue',
    )

    # Marca janela como concluída para suporte a retomada (checkpointing)
    (temp_dir / f'done_{window_id}.flag').touch()

    result = {
        'planar': G_planar,
        'filtrado': G_filtrado,
    }
    _step_log("Fim do processamento.", final=True)
    return (window_id, result)

def main():
    print("\n" + "="*80)
    print("🕸️  ANÁLISE DE REDES DE CORRELAÇÃO FINANCEIRA")
    print("="*80)
    print()

    consolidado_file = Path('data/processed/dados_consolidados.csv')
    results_dir = Path('data/processed/')
    correlation_pvals_file = Path("data/correlation/pvalues_long.csv")
    metadata_file = Path('data/processed/janelas_metadata.json')
    networks_dir = Path('data/networks')
    networks_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("📄 Carregando dados...")
    df = pd.read_csv(consolidado_file, parse_dates=['Date'])
    pvals_df = pd.read_csv(correlation_pvals_file)
    window_bounds = _load_window_bounds_from_metadata(metadata_file)

    if window_bounds:
        window_ids = sorted(window_bounds.keys())
    else:
        window_ids = sorted(df['Janela_ID'].dropna().astype(int).unique())
        window_bounds = {
            int(w): (
                pd.to_datetime(df[df['Janela_ID'] == w]['Window_Start'].iloc[0]) if 'Window_Start' in df.columns else None,
                pd.to_datetime(df[df['Janela_ID'] == w]['Window_End'].iloc[0]) if 'Window_End' in df.columns else None,
            )
            for w in window_ids
        }

    modeled_windows = set(df['Janela_ID'].dropna().astype(int).unique())
    print(
        f"   Registros: {len(df):,}, Janelas (universo): {len(window_ids)}, "
        f"Janelas com dados: {len(modeled_windows)}, Tickers: {df['Ticker'].nunique()}"
    )

    # Estatística rápida sobre número de tickers por janela
    try:
        ticker_counts = df.groupby('Janela_ID')['Ticker'].nunique()
        print(
            "   Tickers por janela (min/máx/média): "
            f"{ticker_counts.min()} / {ticker_counts.max()} / {ticker_counts.mean():.1f}"
        )
    except Exception as e:
        print(f"   [Aviso] Não foi possível calcular estatísticas de tickers por janela: {e}")

    # Informação única sobre uso de GPU para correlações
    if GPU_AVAILABLE and cp is not None:
        print("   GPU disponível (CuPy): matrizes de correlação calculadas na GPU quando possível.")
    else:
        print("   GPU não disponível: matrizes de correlação calculadas na CPU.")

    # Verificação básica: se há p-values para as janelas carregadas
    if 'Janela_ID' in pvals_df.columns:
        pval_windows = set(pvals_df['Janela_ID'].unique())
        missing_pvals = [w for w in window_ids if w not in pval_windows]
        if missing_pvals:
            print(f"   ⚠️  Atenção: {len(missing_pvals)} janelas não possuem p-values associados.")
        else:
            print("   ✅ Todas as janelas possuem p-values associados.")

    # Análise exploratória para ajudar a decidir se vale rodar métricas completas
    # também para os grafos FILTRADOS por p-valor.
    has_edges, recomendacao, rec_flag = _analise_grafos_filtrados(pvals_df, window_ids)

    calc_metricas_filtrado = False
    if has_edges:
        try:
            default_str = 's' if rec_flag else 'n'
            resp = input(
                f"\n❓ Deseja calcular MÉTRICAS COMPLETAS também para os grafos FILTRADOS por p-valor? "
                f"[s/n] (recomendação: {default_str.upper()}): "
            ).strip().lower()
        except EOFError:
            resp = ''

        if resp == 's' or (resp == '' and rec_flag):
            calc_metricas_filtrado = True
            print("   ▶️ Métricas completas para grafos filtrados: ATIVADAS.")
        else:
            print("   ⏭️ Métricas completas para grafos filtrados: DESATIVADAS.")
    else:
        print("   ⏭️ Nenhuma aresta significativa nos grafos filtrados; pulando métricas completas para eles.")

    print("\n▶️  Processando janelas (construindo redes e métricas)...")

    # ── Pré-agrupa p-values por janela: evita passar 2.3 GB a cada worker via pickle ──
    print("   Pré-agrupando p-values por janela...")
    pvals_grouped = {
        int(win): pvals_df[pvals_df['Janela_ID'] == win].copy()
        for win in window_ids
    }
    del pvals_df
    gc.collect()
    print(f"   ✅ {len(pvals_grouped)} janelas agrupadas.")

    # ── Diretório temporário para métricas por janela (sem race condition) ──────────
    temp_dir = results_dir / '_temp_network_metrics'
    temp_dir.mkdir(parents=True, exist_ok=True)

    # ── Bootstrap: recupera progresso de execução anterior interrompida ─────────────
    if not list(temp_dir.glob('done_*.flag')):
        print("   Verificando dados de execução anterior (bootstrap)...")
        done_from_planar: set = set()

        _p = results_dir / 'network_metrics_planar_long.csv'
        if _p.exists():
            try:
                for wv, grp in pd.read_csv(_p).groupby('Janela_ID'):
                    grp.to_csv(temp_dir / f'metrics_planar_{int(wv)}.csv', index=False)
                    done_from_planar.add(int(wv))
            except Exception as _e:
                print(f"   [Aviso] Bootstrap Planar metrics: {_e}")

        _p = results_dir / 'network_centralities_planar_long.csv'
        if _p.exists():
            try:
                for wv, grp in pd.read_csv(_p).groupby('Janela_ID'):
                    grp.to_csv(temp_dir / f'centralities_planar_{int(wv)}.csv', index=False)
            except Exception as _e:
                print(f"   [Aviso] Bootstrap Planar centralities: {_e}")

        bootstrapped_done = done_from_planar
        for wv in bootstrapped_done:
            (temp_dir / f'done_{wv}.flag').touch()
        if bootstrapped_done:
            print(f"   ♻️  Bootstrap: {len(bootstrapped_done)} janelas recuperadas de execução anterior.")

    # ── Checkpoint: pula janelas já concluídas ───────────────────────────────────────
    done_windows = {
        int(f.stem[5:])
        for f in temp_dir.glob('done_*.flag')
        if f.stem[5:].isdigit()
    }
    pending_window_ids = [int(w) for w in window_ids if int(w) not in done_windows]
    if done_windows:
        print(f"   ♻️  Checkpoint: {len(done_windows)}/{len(window_ids)} janelas já concluídas; processando {len(pending_window_ids)} restantes.")

    results_grafos_finais = {}
    results_grafos_iniciais = {}
    processed_count = 0
    ultima_janela_id = None
    primeira_janela_id = None

    # Uso de paralelismo em nível de janela com limite para caber na memória
    cpu_count = os.cpu_count() or 1
    total_janelas = len(pending_window_ids)
    total_rows = len(df)

    # Limita via env var FASE5_MAX_WORKERS (padrão 8) para evitar OOM com grafos grandes
    MAX_WORKERS_CAP = int(os.getenv('FASE5_MAX_WORKERS', '8'))
    cpu_limit = max(1, int(cpu_count * 0.9))
    num_workers = min(max(total_janelas, 1), cpu_limit, MAX_WORKERS_CAP)

    # Ajuste geral quando há GPU disponível: limitar o número de
    # processos concorrentes reduz a competição por memória/VRAM
    # sem sacrificar demais o desempenho.
    if GPU_AVAILABLE and num_workers > 3:
        GPU_MAX_WORKERS = 3
        num_workers = GPU_MAX_WORKERS
        print(
            "   ⚙️ GPU detectada; limitando processamento paralelo a "
            f"{num_workers} worker(s) para evitar pressão excessiva em RAM/VRAM."
        )

    # Otimização adicional para o caso em que o usuário força o cálculo de
    # MÉTRICAS COMPLETAS nos grafos FILTRADOS por p-valor mesmo com
    # recomendação de "carga pesada" (rec_flag=False). Nessa situação,
    # reduzir ainda mais o paralelismo ajuda a manter estabilidade.
    if calc_metricas_filtrado and not rec_flag and num_workers > 2:
        num_workers = 2
        print(
            "   ⚙️ Carga pesada com métricas completas em grafos filtrados; "
            f"ajustando processamento paralelo para {num_workers} worker(s)."
        )
    if num_workers > 1:
        print(f"   🧵 Processando em paralelo com {num_workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for win in pending_window_ids:
                df_window = df[df['Janela_ID'] == win].copy()
                bounds = window_bounds.get(int(win), (None, None))
                args = (win, df_window, pvals_grouped[int(win)], results_dir, temp_dir, calc_metricas_filtrado, bounds)
                fut = executor.submit(processar_janela, args)
                futures[fut] = win

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Janelas", unit="janela"):
                window_id, result = future.result()
                processed_count += 1
                # Guarda resultados para primeira e última janelas
                if primeira_janela_id is None or window_id < primeira_janela_id:
                    primeira_janela_id = window_id
                    results_grafos_iniciais = result
                if ultima_janela_id is None or window_id > ultima_janela_id:
                    ultima_janela_id = window_id
                    results_grafos_finais = result
    else:
        # Fallback sequencial caso só haja 1 janela ou CPU reportada seja 1
        for win in tqdm(pending_window_ids, desc="Janelas", unit="janela"):
            df_window = df[df['Janela_ID'] == win].copy()
            bounds = window_bounds.get(int(win), (None, None))
            args = (win, df_window, pvals_grouped[int(win)], results_dir, temp_dir, calc_metricas_filtrado, bounds)
            window_id, result = processar_janela(args)
            del df_window
            processed_count += 1
            if primeira_janela_id is None or window_id < primeira_janela_id:
                primeira_janela_id = window_id
                results_grafos_iniciais = result
            if ultima_janela_id is None or window_id > ultima_janela_id:
                ultima_janela_id = window_id
                results_grafos_finais = result

    # ── Mescla arquivos temporários nos CSVs finais ────────────────────────────────
    print("\n   Mesclando métricas por janela nos arquivos finais...")
    for tipo_lower, final_metrics, final_centralities in [
        ('planar', results_dir / 'network_metrics_planar_long.csv', results_dir / 'network_centralities_planar_long.csv'),
    ]:
        mfiles = sorted(temp_dir.glob(f'metrics_{tipo_lower}_*.csv'))
        if mfiles:
            dfs_m = [pd.read_csv(f) for f in mfiles if f.stat().st_size > 0]
            if dfs_m:
                df_m = pd.concat(dfs_m, ignore_index=True).sort_values('Janela_ID').drop_duplicates(subset=['Janela_ID'])
                df_m.to_csv(final_metrics, index=False)
                print(f"   ✅ {final_metrics.name}: {len(df_m)} janelas")
        cfiles = sorted(temp_dir.glob(f'centralities_{tipo_lower}_*.csv'))
        if cfiles:
            dfs_c = [pd.read_csv(f) for f in cfiles if f.stat().st_size > 0]
            if dfs_c:
                df_c = pd.concat(dfs_c, ignore_index=True).sort_values('Janela_ID')
                df_c.to_csv(final_centralities, index=False)
                print(f"   ✅ {final_centralities.name}: {len(df_c)} linhas")
    mfiles_f = sorted(temp_dir.glob('metrics_filtrado_*.csv'))
    if mfiles_f:
        dfs_m = [pd.read_csv(f) for f in mfiles_f if f.stat().st_size > 0]
        if dfs_m:
            df_m = pd.concat(dfs_m, ignore_index=True).sort_values('Janela_ID').drop_duplicates(subset=['Janela_ID'])
            df_m.to_csv(results_dir / 'network_metrics_filtrado_long.csv', index=False)
    cfiles_f = sorted(temp_dir.glob('centralities_filtrado_*.csv'))
    if cfiles_f:
        dfs_c = [pd.read_csv(f) for f in cfiles_f if f.stat().st_size > 0]
        if dfs_c:
            df_c = pd.concat(dfs_c, ignore_index=True).sort_values('Janela_ID')
            df_c.to_csv(results_dir / 'network_centralities_filtrado_long.csv', index=False)

    # Salva grafo planar final em GraphML/CSV.
    # O grafo filtrado é intermediário; completo e MST foram removidos.
    for tipo, G in results_grafos_finais.items():
        if not G or tipo != 'planar':
            continue
        nx.write_graphml(G, networks_dir / f"grafo_{tipo}_final.graphml")
        salvar_grafo_csv_edgelist(G, networks_dir / f"grafo_{tipo}_final.csv")

    # Salva imagens para primeira e última janelas (MST e Planar)
    fig_net_dir = Path('figures/networks')

    # Salva imagens para primeira e última janelas (PMFG/Planar) com p-valor (arestas pontilhadas)
    if primeira_janela_id is not None and results_grafos_iniciais:
        G_planar_ini = results_grafos_iniciais.get('planar')
        if G_planar_ini:
            # Busca p-valor da primeira janela
            pvals_df_ini = pvals_grouped.get(primeira_janela_id, pd.DataFrame())
            pval_dict_ini = {}
            if not pvals_df_ini.empty:
                pval_lookup_ini = {(row['Ticker1'], row['Ticker2']): row['pvalue'] for _, row in pvals_df_ini.iterrows()}
                pval_lookup_ini.update({(row['Ticker2'], row['Ticker1']): row['pvalue'] for _, row in pvals_df_ini.iterrows()})
                for u, v in G_planar_ini.edges():
                    pval = pval_lookup_ini.get((u, v), np.nan)
                    pval_dict_ini[(u, v)] = pval
            else:
                for u, v in G_planar_ini.edges():
                    pval_dict_ini[(u, v)] = np.nan
            img_path_ini = fig_net_dir / f'planar_pval_primeira_janela.png'
            bounds_ini = window_bounds.get(int(primeira_janela_id), (None, None))
            plot_network_with_pvalues(
                G_planar_ini,
                pval_dict_ini,
                threshold=0.05,
                output_path=img_path_ini,
                title=f"PMFG (Planar) – {_format_window_label(primeira_janela_id, bounds_ini[0], bounds_ini[1], 'First window')}",
                node_color='skyblue',
            )
    if ultima_janela_id is not None and results_grafos_finais:
        G_planar_fim = results_grafos_finais.get('planar')
        if G_planar_fim:
            pvals_df_fim = pvals_grouped.get(ultima_janela_id, pd.DataFrame())
            pval_dict_fim = {}
            if not pvals_df_fim.empty:
                pval_lookup_fim = {(row['Ticker1'], row['Ticker2']): row['pvalue'] for _, row in pvals_df_fim.iterrows()}
                pval_lookup_fim.update({(row['Ticker2'], row['Ticker1']): row['pvalue'] for _, row in pvals_df_fim.iterrows()})
                for u, v in G_planar_fim.edges():
                    pval = pval_lookup_fim.get((u, v), np.nan)
                    pval_dict_fim[(u, v)] = pval
            else:
                for u, v in G_planar_fim.edges():
                    pval_dict_fim[(u, v)] = np.nan
            img_path_fim = fig_net_dir / f'planar_pval_ultima_janela.png'
            bounds_fim = window_bounds.get(int(ultima_janela_id), (None, None))
            plot_network_with_pvalues(
                G_planar_fim,
                pval_dict_fim,
                threshold=0.05,
                output_path=img_path_fim,
                title=f"PMFG (Planar) – {_format_window_label(ultima_janela_id, bounds_fim[0], bounds_fim[1], 'Last window')}",
                node_color='skyblue',
            )

    print("\n📊 Resumo do processamento de redes:")
    print(f"   Janelas processadas: {processed_count} (de {len(window_ids)})")
    print(f"   Pasta de métricas: {results_dir.resolve()}")
    print(f"   Pasta de grafos finais: {networks_dir.resolve()}")

    # Verificação simples de arquivos principais gerados
    arquivos_esperados = [
        results_dir / 'network_metrics_planar_long.csv',
        results_dir / 'network_centralities_planar_long.csv',
    ]

    print("\n🔎 Verificando arquivos principais gerados:")
    for f in arquivos_esperados:
        status = "OK" if f.exists() else "NÃO ENCONTRADO"
        print(f"   - {f.name}: {status}")

    print("\n✅ Processamento de redes concluído! Verifique os arquivos CSV e grafos exportados.\n")

if __name__ == "__main__":
    main()