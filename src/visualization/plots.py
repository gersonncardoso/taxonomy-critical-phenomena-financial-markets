import matplotlib
import networkx as nx

# Usa backend "Agg" (não interativo) para evitar dependência de Tkinter.
# Isso é mais robusto em scripts batch/multiprocessados como o pipeline.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

try:
    from src.estatisticas.metrics_utils import CRISIS_DATES
except ImportError:
    # Import relativo para compatibilidade em execução direta
    import sys, os
    CURR = os.path.dirname(os.path.abspath(__file__))
    ESTAT_PATH = os.path.abspath(os.path.join(CURR, '..', 'estatisticas'))
    if ESTAT_PATH not in sys.path:
        sys.path.insert(0, ESTAT_PATH)
    try:
        from metrics_utils import CRISIS_DATES
    except ImportError:
        CRISIS_DATES = []


def _flatten_setores_tree(setores_cfg):
    flat = {}
    if not isinstance(setores_cfg, dict):
        return flat

    for setor, subsetores in setores_cfg.items():
        bucket = flat.setdefault(setor, set())
        if isinstance(subsetores, dict):
            for segmentos in subsetores.values():
                if isinstance(segmentos, dict):
                    for tickers in segmentos.values():
                        if isinstance(tickers, list):
                            bucket.update(tickers)
                elif isinstance(segmentos, list):
                    bucket.update(segmentos)
        elif isinstance(subsetores, list):
            bucket.update(subsetores)

    return {setor: sorted(tickers) for setor, tickers in flat.items()}


def _normalize_ticker_label(value):
    s = str(value).strip().upper()
    if s.endswith('.SA'):
        s = s[:-3]
    return s


def _edge_strength_from_attributes(data, eps=1.0e-9):
    for key in ('weight', 'strength', 'abs_corr', 'corr', 'correlation'):
        if key in data and data[key] is not None:
            try:
                return max(float(data[key]), eps)
            except Exception:
                continue
    return 1.0


def _compute_ricci_node_groups(G):
    """Classifica nós em grupos de Ricci (baixo/médio/alto) com fallback robusto.

    Usa atributos de curvatura de aresta quando disponíveis; caso contrário,
    aplica um proxy simples baseado em distância (menor distância => score maior).
    """
    if G is None or G.number_of_nodes() == 0:
        return {}, {}

    scores = {}
    for node in G.nodes():
        vals = []
        for _, _, data in G.edges(node, data=True):
            v = None
            for key in ("forman_ricci", "ricci_fr", "ricci", "curvature"):
                if key in data and data[key] is not None:
                    try:
                        v = float(data[key])
                    except Exception:
                        v = None
                    if v is not None and np.isfinite(v):
                        break

            if v is None:
                # Proxy: distâncias menores indicam conexões mais "coesas".
                dist = data.get("distance", data.get("weight", None))
                try:
                    dist = float(dist)
                    if np.isfinite(dist) and dist > 0:
                        v = -dist
                except Exception:
                    v = None

            if v is not None and np.isfinite(v):
                vals.append(v)

        scores[node] = float(np.mean(vals)) if vals else 0.0

    arr = np.array(list(scores.values()), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {n: "Ricci médio" for n in G.nodes()}, scores

    q1 = float(np.quantile(finite, 0.33))
    q2 = float(np.quantile(finite, 0.66))

    groups = {}
    for node, score in scores.items():
        if score <= q1:
            groups[node] = "Ricci baixo"
        elif score >= q2:
            groups[node] = "Ricci alto"
        else:
            groups[node] = "Ricci médio"

    return groups, scores


def _reduce_node_overlap(pos, *, min_dist=0.08, max_iter=180):
    """Aplica repulsão leve em 2D para reduzir sobreposição de nós no layout."""
    if not pos:
        return pos

    nodes = list(pos.keys())
    coords = np.array([pos[n] for n in nodes], dtype=float)
    n = len(nodes)
    if n <= 1:
        return pos

    min_dist2 = float(min_dist) ** 2
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = coords[j, 0] - coords[i, 0]
                dy = coords[j, 1] - coords[i, 1]
                d2 = dx * dx + dy * dy
                if d2 >= min_dist2:
                    continue

                moved = True
                if d2 < 1.0e-12:
                    # Evita divisão por zero quando dois nós ficam exatamente no mesmo ponto.
                    angle = (i * 37 + j * 17) % 360
                    rad = np.deg2rad(angle)
                    ux, uy = np.cos(rad), np.sin(rad)
                    dist = 1.0e-6
                else:
                    dist = np.sqrt(d2)
                    ux, uy = dx / dist, dy / dist

                delta = 0.5 * (min_dist - dist)
                coords[i, 0] -= ux * delta
                coords[i, 1] -= uy * delta
                coords[j, 0] += ux * delta
                coords[j, 1] += uy * delta

        if not moved:
            break

    # Normaliza para caixa [-1, 1] preservando proporções.
    cmin = coords.min(axis=0)
    cmax = coords.max(axis=0)
    span = np.maximum(cmax - cmin, 1.0e-9)
    coords = (coords - cmin) / span
    coords = 2.0 * coords - 1.0

    return {n: coords[k] for k, n in enumerate(nodes)}

def plot_heatmap(corr_matrix, output_path="outputs/heatmap.png", title="Heatmap"):
    n_assets = int(corr_matrix.shape[0])
    fig_size = max(18, min(46, 0.31 * n_assets + 12))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    show_tick_labels = True
    label_font_size = max(6.5, min(12.0, 700.0 / max(n_assets, 1)))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        xticklabels=corr_matrix.columns if show_tick_labels else False,
        yticklabels=corr_matrix.index if show_tick_labels else False,
        cbar_kws={'shrink': 0.85, 'label': 'Correlacao'},
        ax=ax,
    )
    ax.set_title(title)
    if show_tick_labels:
        ax.tick_params(axis='x', rotation=90, labelsize=label_font_size)
        ax.tick_params(axis='y', rotation=0, labelsize=label_font_size)
    else:
        ax.set_xlabel(f'{n_assets} ativos')
        ax.set_ylabel(f'{n_assets} ativos')
    plt.tight_layout()
    # Padroniza saída para figures/ se não for absoluto
    out_path = output_path
    import os
    if not os.path.isabs(out_path):
        out_path = os.path.join('figures', os.path.basename(out_path))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=360, bbox_inches='tight', facecolor='white')
    if out_path.lower().endswith('.png'):
        plt.savefig(out_path[:-4] + '.pdf', dpi=360, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Heatmap salvo em {out_path}")

def plot_dendrogram(corr_matrix, output_path="outputs/dendrograma.png", title="Dendrograma"):
    # Garante matriz simétrica; se vier incompleta, preenche (evita erros do linkage)
    mat_full = corr_matrix.combine_first(corr_matrix.transpose()).fillna(1)
    try:
        Z = linkage(mat_full.values, method='ward')
    except Exception:
        Z = linkage(mat_full.values, method='ward')
    n_assets = int(mat_full.shape[0])
    fig_width = max(12, min(24, 0.11 * n_assets + 7))
    fig, ax = plt.subplots(figsize=(fig_width, 9.2))
    show_labels = n_assets <= 50
    dendrogram(
        Z,
        labels=mat_full.index if show_labels else None,
        leaf_rotation=90,
        leaf_font_size=6 if show_labels else 0,
        no_labels=not show_labels,
        color_threshold=0.7 * float(np.max(Z[:, 2])) if len(Z) else None,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel('Distancia')
    if not show_labels:
        ax.set_xlabel(f'{n_assets} ativos')
    plt.tight_layout()
    # Padroniza saída para figures/ se não for absoluto
    out_path = output_path
    import os
    if not os.path.isabs(out_path):
        out_path = os.path.join('figures', os.path.basename(out_path))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    if out_path.lower().endswith('.png'):
        plt.savefig(out_path[:-4] + '.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Dendrograma salvo em {out_path}")


def plot_dendrogram_with_groups(
    corr_matrix,
    output_path="outputs/dendrograma_grupos.png",
    title="Dendrograma com grupos",
    max_k=8,
):
    mat_full = corr_matrix.combine_first(corr_matrix.transpose()).fillna(1)
    data = mat_full.values
    n_assets = int(mat_full.shape[0])

    try:
        Z = linkage(mat_full.values, method='ward')
    except Exception:
        Z = linkage(mat_full.values, method='ward')

    best_k = 2
    best_labels = np.ones(n_assets, dtype=int)
    best_silhouette = float('nan')

    upper_k = min(max_k, max(2, n_assets - 1))
    if n_assets >= 3:
        current_best = -np.inf
        for k in range(2, upper_k + 1):
            try:
                labels = AgglomerativeClustering(
                    n_clusters=k,
                    metric='euclidean',
                    linkage='ward',
                ).fit_predict(data)
                sil = silhouette_score(data, labels)
                if sil > current_best:
                    current_best = sil
                    best_k = k
                    best_labels = labels
                    best_silhouette = sil
            except Exception:
                continue

    modularidade_value = float('nan')
    try:
        positive_weights = np.clip(mat_full.values, a_min=0.0, a_max=None)
        G = nx.from_numpy_array(positive_weights)
        communities = {}
        for idx, label in enumerate(best_labels):
            communities.setdefault(int(label), []).append(idx)
        if communities:
            from networkx.algorithms.community.quality import modularity
            modularidade_value = modularity(G, communities.values(), weight='weight')
    except Exception:
        pass

    fig_width = max(14, min(30, 0.16 * n_assets + 10))
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    show_labels = n_assets <= 120
    dendro = dendrogram(
        Z,
        labels=mat_full.index if show_labels else None,
        leaf_rotation=90,
        leaf_font_size=5 if show_labels else 0,
        no_labels=not show_labels,
        ax=ax,
    )

    color_palette = sns.color_palette('tab10', n_colors=max(3, best_k))
    label_to_color = {
        ticker: color_palette[(int(cluster_id) - 1) % len(color_palette)]
        for ticker, cluster_id in zip(mat_full.index, best_labels)
    }

    if show_labels:
        for tick_label in ax.get_xmajorticklabels():
            label = tick_label.get_text()
            if label in label_to_color:
                tick_label.set_color(label_to_color[label])

    ordered_labels = dendro.get('ivl', list(mat_full.index))
    if ordered_labels:
        x_positions = np.arange(5, 10 * len(ordered_labels) + 5, 10)
        strip_y = np.full(len(ordered_labels), -0.035 * float(np.max(Z[:, 2])) if len(Z) else -0.05)
        strip_colors = [label_to_color.get(label, (0.5, 0.5, 0.5)) for label in ordered_labels]
        ax.scatter(x_positions, strip_y, c=strip_colors, marker='s', s=18, clip_on=False)

    metrics_text = f'grupos={best_k} | silhouette={best_silhouette:.3f} | modularidade={modularidade_value:.3f}'
    ax.set_title(f'{title}\n{metrics_text}')
    ax.set_ylabel('Distancia')
    if not show_labels:
        ax.set_xlabel(f'{n_assets} ativos')

    legend_handles = [
        Patch(facecolor=color_palette[idx], edgecolor=color_palette[idx], label=f'Grupo {idx + 1}')
        for idx in range(min(best_k, len(color_palette)))
    ]
    ax.legend(handles=legend_handles, loc='upper right', ncol=min(4, max(1, best_k)), frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Dendrograma com grupos salvo em {output_path}")

def plot_validation_histogram(real_corrs, random_corrs, output_path="outputs/validacao.png", title="Validação"):
    plt.figure(figsize=(8, 5))
    plt.hist(real_corrs, bins=30, alpha=0.7, label="Real")
    plt.hist(random_corrs, bins=30, alpha=0.5, label="Aleatória")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Histograma de validação salvo em {output_path}")
    
def plot_metric_evolution(janela_labels, metric_values, ylabel, output_path, title):
    """Plota evolução temporal de uma métrica por janela.

    Usa um eixo numérico sequencial (0,1,2,...) para garantir que as
    linhas sejam ligadas na ordem correta das janelas, e aplica rótulos
    de data apenas em alguns pontos para evitar poluição visual.
    """
    import matplotlib.pyplot as plt

    labels = list(janela_labels)
    values = np.asarray(metric_values, dtype=float)
    n = len(labels)
    x_seq = np.arange(n)

    parsed_dates = pd.to_datetime(labels, errors='coerce')
    has_date_axis = not pd.isna(parsed_dates).all()
    x_vals = parsed_dates if has_date_axis else x_seq

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16.0, 10.2), dpi=220)
    line_color = '#1F4E79'

    ax.plot(
        x_vals,
        values,
        color=line_color,
        linewidth=2.2,
        marker='o',
        markersize=2.6,
        markeredgewidth=0,
        zorder=3,
    )

    finite_values = values[np.isfinite(values)]
    if finite_values.size:
        baseline = float(finite_values.min())
        ax.fill_between(x_vals, values, baseline, color=line_color, alpha=0.10, zorder=1)

    ax.set_xlabel('Window end date' if has_date_axis else 'Rolling window end', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, pad=10)

    if len(CRISIS_DATES) > 0 and not pd.isna(parsed_dates).all():
        valid_dates = parsed_dates.dropna()
        if len(valid_dates) > 0:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            for crisis_date in CRISIS_DATES:
                if min_date <= crisis_date <= max_date:
                    ax.axvline(crisis_date, color='red', linestyle=':', alpha=0.45, linewidth=1.0, zorder=2)

    if has_date_axis:
        import matplotlib.dates as mdates

        ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    else:
        if n <= 20:
            xticks = x_seq
        else:
            step = max(1, n // 12)
            xticks = x_seq[::step]
        ax.set_xticks(xticks)
        ax.set_xticklabels([labels[i] for i in xticks], rotation=35, ha='right', fontsize=9)

    ax.grid(True, axis='y', alpha=0.22)
    ax.grid(False, axis='x')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.grid(True, axis='x', alpha=0.12)
    ax.spines['left'].set_color('#aeb8c4')
    ax.spines['bottom'].set_color('#aeb8c4')

    if finite_values.size:
        ymin = float(finite_values.min())
        ymax = float(finite_values.max())
        span = max(ymax - ymin, 1e-6)
        ax.set_ylim(ymin - 0.08 * span, ymax + 0.10 * span)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    out_str = str(output_path)
    if out_str.lower().endswith('.png'):
        fig.savefig(out_str[:-4] + '.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Evolução de {ylabel} salva em {output_path}")


def plot_ks_mp_dual(
    labels,
    ks_stats,
    mp_signal_frac,
    mp_market_frac,
    output_path,
    title_ks="KS: correlações empíricas vs. nulo Gaussiano",
    title_mp="Marchenko-Pastur: fração de variância com sinal",
):
    """Gráfico duplo (dois painéis empilhados) com marcadores de crise.

    Painel superior  — KS statistic por janela.
    Painel inferior  — fração da variância explicada por fatores de sinal (MP)
                       e fração do fator de mercado (autovalor dominante).

    Linhas vermelhas verticais marcam as datas de crise definidas em
    CRISIS_DATES (via configs/config.yaml > eventos.datas_crise).
    """
    labels = list(labels)
    ks_arr = np.asarray(ks_stats, dtype=float)
    sig_arr = np.asarray(mp_signal_frac, dtype=float)
    mkt_arr = np.asarray(mp_market_frac, dtype=float)
    n = len(labels)
    x = np.arange(n)

    parsed_dates = pd.to_datetime(labels, errors='coerce')

    def _add_crisis_lines(ax):
        if len(CRISIS_DATES) == 0 or pd.isna(parsed_dates).all():
            return
        valid_pos = [(i, d) for i, d in enumerate(parsed_dates) if not pd.isna(d)]
        if not valid_pos:
            return
        vidx = np.array([i for i, _ in valid_pos])
        vdates = pd.to_datetime([d for _, d in valid_pos])
        for cd in CRISIS_DATES:
            if cd < vdates.min() or cd > vdates.max():
                continue
            cx = int(vidx[int(np.argmin(np.abs(vdates - cd)))])
            ax.axvline(cx, color='red', linestyle=':', alpha=0.45, linewidth=1.0,
                       label='Crise' if cx == vidx[int(np.argmin(np.abs(vdates - CRISIS_DATES[0])))] else None)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 7.0), sharex=True,
                                    gridspec_kw={'hspace': 0.16})

    # ── Painel superior: KS ──────────────────────────────────────────────────
    ax1.plot(x, ks_arr, color='steelblue', linewidth=1.6, label='KS stat')
    ax1.fill_between(x, ks_arr, alpha=0.12, color='steelblue')
    _add_crisis_lines(ax1)
    ax1.set_ylabel("KS statistic", fontsize=10)
    ax1.set_title(title_ks, fontsize=11, pad=6)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, axis='y', alpha=0.25)
    ax1.set_ylim(bottom=0)

    # ── Painel inferior: MP ──────────────────────────────────────────────────
    ax2.plot(x, sig_arr * 100, color='darkorange', linewidth=1.6,
             label='% variância sinal (MP)')
    ax2.fill_between(x, sig_arr * 100, alpha=0.12, color='darkorange')
    ax2.plot(x, mkt_arr * 100, color='crimson', linewidth=1.2, linestyle=':',
             label='% variância fator de mercado (λ₁)')
    _add_crisis_lines(ax2)
    ax2.set_ylabel("Variância explicada (%)", fontsize=10)
    ax2.set_title(title_mp, fontsize=11, pad=6)
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, axis='y', alpha=0.25)
    ax2.set_ylim(0, 100)

    # ── Eixo X ───────────────────────────────────────────────────────────────
    step = max(1, n // 12)
    xticks = x[::step]
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([labels[i] for i in xticks], rotation=45,
                         ha='right', fontsize=8)
    ax2.set_xlabel("Janela", fontsize=10)

    plt.tight_layout(pad=0.8, h_pad=1.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    out_str = str(output_path)
    if out_str.lower().endswith('.png'):
        plt.savefig(out_str[:-4] + '.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Gráfico KS + MP salvo em {output_path}")


def plot_hist_pvalues(pval_mat, output_path, title="Distribuição dos p-values"):
    plt.figure(figsize=(8,5))
    vals = pval_mat[np.triu_indices(pval_mat.shape[0], k=1)]
    sns.histplot(vals, bins=30, kde=True, color='royalblue')
    plt.title(title)
    plt.xlabel('p-value')
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Histograma de p-values salvo em {output_path}")

def plot_network_with_pvalues(
    G,
    pval_dict,
    threshold,
    output_path,
    title="Network with p-values",
    node_color='skyblue',
    color_mode='ricci_groups',
    show_pvalue_styles=True,
    significant_style='solid',
):
    """
    Plota um grafo NetworkX com codificação visual orientada à legibilidade:
        - estilo configurável por significância (`significant_style`):
            `solid` => p < threshold sólido, `dashed` => p < threshold tracejado;
    - espessura de aresta proporcional à força (1/distância);
    - tamanho de nó proporcional a centralidade (betweenness + grau);
    - cores de nós por grupos de Ricci (padrão) ou por setor.
    """
    if G is None or G.number_of_nodes() == 0:
        print(f"⚠️ Grafo vazio em {output_path}; figura não gerada.")
        return

    # Escala visual para deixar os grafos ~30% maiores no render final.
    VISUAL_SCALE = 1.30
    # A legenda fica ainda mais destacada: +20% além da escala do grafo.
    LEGEND_SCALE = 1.20

    # Prepara mapeamento de setor (modo opcional) e fallback de cor.
    ticker_setor = {}
    setores_ordenados = []
    try:
        import yaml
        with open('configs/tickers.yaml', encoding='utf-8') as f:
            tickers_cfg = yaml.safe_load(f)
        flat = _flatten_setores_tree(tickers_cfg.get('setores', {}))
        setores_ordenados = sorted(flat.keys())
        for setor, lista in flat.items():
            for ticker in lista:
                ticker_setor[_normalize_ticker_label(ticker)] = setor
    except Exception:
        setores_ordenados = []

    palette = sns.color_palette('tab20', n_colors=max(len(setores_ordenados), 1))
    setor_cores = {setor: palette[i % len(palette)] for i, setor in enumerate(setores_ordenados)}
    default_node_color = '#8f8f8f'

    ricci_palette = {
        'Ricci baixo': '#c44e52',
        'Ricci médio': '#7f8c8d',
        'Ricci alto': '#4c72b0',
    }

    deg = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    max_deg = max(deg.values()) if deg else 1
    max_betw = max(betweenness.values()) if betweenness else 1

    # Layout oficial em duas etapas: Circular -> Kamada-Kawai.
    # Depois aplica ajuste radial para trazer nós com maior betweenness ao centro.
    try:
        pos0 = nx.circular_layout(G)
        pos = nx.kamada_kawai_layout(G, pos=pos0)
    except Exception:
        # Fallback determinístico caso haja instabilidade numérica.
        pos = nx.kamada_kawai_layout(G)

    for n, pxy in pos.items():
        betw_norm = (betweenness.get(n, 0.0) / max_betw) if max_betw else 0.0
        # Maior betweenness -> menor raio (mais próximo do centro).
        radial_scale = 1.0 - 0.48 * betw_norm
        pos[n] = np.asarray(pxy) * radial_scale

    # Pós-processamento para reduzir sobreposição de nós e rótulos.
    pos = _reduce_node_overlap(pos, min_dist=0.085, max_iter=220)
    pos = {n: np.asarray(pxy) * VISUAL_SCALE for n, pxy in pos.items()}

    # Combina grau e betweenness para dar contexto local + global.
    node_sizes = []
    node_colors = []
    ricci_groups, _ricci_scores = _compute_ricci_node_groups(G)
    for n in G.nodes:
        deg_norm = (deg.get(n, 0) / max_deg) if max_deg else 0
        betw_norm = (betweenness.get(n, 0.0) / max_betw) if max_betw else 0
        size = 140 + 1550 * (0.72 * betw_norm + 0.28 * deg_norm)
        node_sizes.append(size * VISUAL_SCALE)
        if color_mode == 'sector':
            setor = ticker_setor.get(_normalize_ticker_label(n))
            node_colors.append(setor_cores.get(setor, default_node_color))
        else:
            grp = ricci_groups.get(n, 'Ricci médio')
            node_colors.append(ricci_palette.get(grp, default_node_color))

    if significant_style not in {'solid', 'dashed'}:
        significant_style = 'dashed'

    # Separar arestas por significância e escalar largura por força 1/distância.
    solid_edges, dashed_edges = [], []
    solid_widths, dashed_widths = [], []
    sig_count = 0

    # Normaliza chaves do dicionário de p-values para evitar mismatch de labels
    # (ex.: PETR4 vs PETR4.SA), que pode fazer arestas significativas virarem tracejadas.
    norm_pval_dict = {}
    for (a, b), pv in (pval_dict or {}).items():
        ka = _normalize_ticker_label(a)
        kb = _normalize_ticker_label(b)
        norm_pval_dict[(ka, kb)] = pv
        norm_pval_dict[(kb, ka)] = pv

    for u, v in G.edges():
        if show_pvalue_styles:
            p = pval_dict.get((u, v), pval_dict.get((v, u), np.nan))
            if pd.isna(p):
                ku = _normalize_ticker_label(u)
                kv = _normalize_ticker_label(v)
                p = norm_pval_dict.get((ku, kv), np.nan)
        else:
            p = np.nan
        dist = G[u][v].get('distance', G[u][v].get('weight', np.nan))
        if pd.notna(dist) and float(dist) > 0:
            strength = 1.0 / float(dist)
        else:
            strength = 1.0
        width = 0.8 + 2.6 * min(strength, 1.8)
        width = width * 1.12
        is_significant = pd.notna(p) and float(p) < threshold
        if is_significant:
            sig_count += 1

        if not show_pvalue_styles:
            solid_edges.append((u, v))
            solid_widths.append(width)
        else:
            sig_in_solid = (significant_style == 'solid')
            if (is_significant and sig_in_solid) or ((not is_significant) and (not sig_in_solid)):
                solid_edges.append((u, v))
                solid_widths.append(width)
            else:
                dashed_edges.append((u, v))
                dashed_widths.append(width)

    fig, ax = plt.subplots(figsize=(17.5, 13.65))
    ax.set_facecolor('#fafafa')

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.94,
        linewidths=0.45,
        edgecolors='#2f2f2f',
        ax=ax,
    )

    if solid_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=solid_edges,
            width=solid_widths,
            edge_color='#1f2937',
            style='solid',
            alpha=0.72,
            ax=ax,
        )
    if dashed_edges and show_pvalue_styles:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=dashed_edges,
            width=dashed_widths,
            edge_color='#9ca3af',
            style='dashed',
            alpha=0.62,
            ax=ax,
        )

    # Rótulos para todos os nós, em preto, com fundo translúcido.
    label_nodes = list(G.nodes)
    labels = {n: _normalize_ticker_label(n) for n in label_nodes}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=10.5,
        font_color='#111827',
        font_weight='normal',
        bbox=dict(facecolor=(1, 1, 1, 0.5), edgecolor='none', boxstyle='round,pad=0.14'),
        ax=ax,
    )

    legend_items = []
    if show_pvalue_styles:
        if significant_style == 'solid':
            legend_items.extend([
                Line2D([0], [0], color='#1f2937', lw=2.2, linestyle='solid', label=f'p < {threshold}'),
                Line2D([0], [0], color='#9ca3af', lw=2.2, linestyle='dashed', label=f'p >= {threshold}'),
            ])
        else:
            legend_items.extend([
                Line2D([0], [0], color='#1f2937', lw=2.2, linestyle='solid', label=f'p >= {threshold}'),
                Line2D([0], [0], color='#9ca3af', lw=2.2, linestyle='dashed', label=f'p < {threshold}'),
            ])

    if color_mode == 'sector':
        legend_items.append(Patch(facecolor=default_node_color, edgecolor='#2f2f2f', label='unmapped sector'))
    else:
        legend_items.extend([
            Patch(facecolor=ricci_palette['Ricci baixo'], edgecolor='#2f2f2f', label='low Ricci group'),
            Patch(facecolor=ricci_palette['Ricci médio'], edgecolor='#2f2f2f', label='mid Ricci group'),
            Patch(facecolor=ricci_palette['Ricci alto'], edgecolor='#2f2f2f', label='high Ricci group'),
        ])

    legend = ax.legend(
        handles=legend_items,
        loc='upper right',
        frameon=True,
        fontsize=10 * LEGEND_SCALE,
        title='Visual encoding',
    )
    legend.get_title().set_fontsize(10 * LEGEND_SCALE)
    legend.get_frame().set_alpha(0.94)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(0.8)
    try:
        legend.get_frame().set_boxstyle('round,pad=0.28')
    except Exception:
        pass

    ax.set_title(title, fontsize=12.5, fontweight='normal')
    ax.text(
        0.01,
        0.02,
        f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | labels shown: {len(label_nodes)}",
        transform=ax.transAxes,
        fontsize=9,
        color='#374151',
    )
    ax.axis('off')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        out_str = str(output_path)
        if out_str.lower().endswith('.png'):
            plt.savefig(out_str[:-4] + '.pdf', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        if show_pvalue_styles:
            if significant_style == 'solid':
                print(
                    f"✅ Rede salva em {output_path} "
                    f"(solid = p < {threshold}; dashed = p >= {threshold}) | "
                    f"significant_edges={sig_count}/{G.number_of_edges()}"
                )
            else:
                print(
                    f"✅ Rede salva em {output_path} "
                    f"(solid = p >= {threshold}; dashed = p < {threshold}) | "
                    f"significant_edges={sig_count}/{G.number_of_edges()}"
                )
        else:
            print(f"✅ Rede salva em {output_path} (colors by Ricci groups)")
    plt.close(fig)