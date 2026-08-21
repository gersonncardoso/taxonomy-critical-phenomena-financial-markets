"""
Funções de Clustering Hierárquico
==================================
"""

import numpy as np
from typing import TYPE_CHECKING
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import AgglomerativeClustering

if TYPE_CHECKING:
    import pandas as pd

SETORES = {
    # Exemplo: mapeamento real de setores, substitua pelo seu completo
    'PETR4': 'Petróleo/Gás', 'VALE3': 'Mineração',
    'ITUB4': 'Bancos', 'ELET3': 'Energia', 'LREN3': 'Varejo',
    # ... (restante dos tickers/setores)
}

def obter_setor(ticker: str) -> str:
    return SETORES.get(ticker, 'Outros')

def identificar_clusters(dist_matrix: "pd.DataFrame", n_clusters: int = 5) -> "pd.Series":
    import pandas as pd

    dist_condensed = squareform(dist_matrix.values)
    Z = linkage(dist_condensed, method='average')
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    clusters = pd.Series(labels, index=dist_matrix.index, name='Cluster')
    return clusters

def analisar_composicao_cluster(clusters: "pd.Series") -> "pd.DataFrame":
    df = clusters.to_frame()
    df['Setor'] = df.index.map(obter_setor)
    composicao = df.groupby(['Cluster', 'Setor']).size().reset_index(name='N_Tickers')
    total_por_cluster = df.groupby('Cluster').size()
    composicao['Pct'] = composicao.apply(
        lambda row: row['N_Tickers'] / total_por_cluster[row['Cluster']] * 100,
        axis=1
    )
    return composicao

def calcular_estabilidade_clusters(clusters_anterior: "pd.Series", clusters_atual: "pd.Series") -> dict:
    tickers_comuns = clusters_anterior.index.intersection(clusters_atual.index)
    if len(tickers_comuns) < 2:
        return {'ari': np.nan, 'tickers_comuns': 0}
    ari = adjusted_rand_score(
        clusters_anterior[tickers_comuns],
        clusters_atual[tickers_comuns]
    )
    return {'ari': ari, 'tickers_comuns': len(tickers_comuns)}

# -------- Funções adicionais para Fase 4 --------

def calc_silhouette(data, labels):
    """
    Calcula o silhouette score para uma matriz de dados e labels de cluster.
    """
    try:
        return silhouette_score(data, labels)
    except Exception:
        return float('nan')

def calc_ari(labels_1, labels_2):
    """
    Calcula o Adjusted Rand Index (ARI) entre dois agrupamentos.
    """
    return adjusted_rand_score(labels_1, labels_2)

def run_multiple_clustering_methods(corr_matrix, max_k=10, linkage_methods=None):
    """
    Executa diferentes métodos de linkage/aglomeração sobre a matriz de correlação.
    Busca melhor k (pelo silhouette) para cada, e calcula modularidade se possível.
    """
    if linkage_methods is None:
        linkage_methods = ['ward', 'average', 'complete']
    resumo = {}
    try:
        from networkx.algorithms.community.quality import modularity
        import networkx as nx
    except ImportError:
        modularity = None
        nx = None
    n = corr_matrix.shape[0]
    for method in linkage_methods:
        best_k = 2
        best_silhouette = -1
        best_labels = None
        silhouettes = []
        for k in range(2, min(max_k+1, n)):
            clustering = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage=method)
            try:
                labels = clustering.fit_predict(corr_matrix)
                sil = silhouette_score(corr_matrix, labels)
                silhouettes.append((k, sil, labels))
                if sil > best_silhouette:
                    best_silhouette = sil
                    best_k = k
                    best_labels = labels
            except Exception:
                continue
        # Modularidade usando rede
        modularidade_value = np.nan
        if nx is not None and modularity is not None and best_labels is not None:
            try:
                G = nx.from_numpy_array(corr_matrix)
                communities = {}
                for idx, label in enumerate(best_labels):
                    communities.setdefault(label, []).append(idx)
                modularidade_value = modularity(G, communities.values())
            except Exception:
                pass
        resumo[method] = {
            "best_k": best_k,
            "best_silhouette": best_silhouette,
            "modularity": modularidade_value,
            "labels": best_labels
        }
    return {"resumo": resumo}