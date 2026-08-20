"""
Métricas de Redes
=================

Calcula métricas de centralidade e características de rede

Autor: Gerson Nassor Cardoso
Instituição: Universidade Federal de São Paulo (UNIFESP)
Data: 2026-02-13

Copyright (c) 2026 Gerson Nassor Cardoso - UNIFESP
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict
import concurrent.futures

from .cugraph_utils import HAS_CUGRAPH, nx_to_cugraph

# Limite a partir do qual o grafo é considerado "pesado" para
# fins de centralidades e métricas globais completas. Acima desse
# valor de arestas, evitamos cálculos \mathcal{O}(|V||E|) como
# betweenness/closeness em grafos muito densos (caso da rede completa).
HEAVY_EDGE_THRESHOLD = 30_000


def _edge_strength_from_data(data: dict, eps: float = 1.0e-9) -> float:
    """Converte atributo de aresta em força positiva para FR.

    Prioriza atributo `distance` (forca inversa da distancia) e, na ausencia,
    usa `weight` quando positivo.
    """
    if 'distance' in data and pd.notna(data['distance']):
        d = float(data['distance'])
        if d > 0:
            return 1.0 / (d + eps)
    if 'weight' in data and pd.notna(data['weight']):
        w = float(data['weight'])
        if w > 0:
            return w
    return 1.0


def calcular_forman_ricci_stats(G: nx.Graph) -> Dict:
    """Calcula estatisticas de curvatura Forman-Ricci por aresta.

    Implementacao discreta ponderada (versao para grafos sem faces),
    adequada para monitoramento temporal de fragilidade em redes financeiras.
    """
    if G.number_of_edges() == 0:
        return {
            'ricci_fr_mean': np.nan,
            'ricci_fr_var': np.nan,
            'ricci_fr_std': np.nan,
            'ricci_fr_skew': np.nan,
            'ricci_fr_kurt': np.nan,
            'ricci_fr_q10': np.nan,
            'ricci_fr_q50': np.nan,
            'ricci_fr_q90': np.nan,
            'ricci_fr_iqr': np.nan,
            'ricci_fr_min': np.nan,
            'ricci_fr_max': np.nan,
        }

    edge_strength = {}
    for u, v, data in G.edges(data=True):
        key = (u, v) if u <= v else (v, u)
        edge_strength[key] = _edge_strength_from_data(data)

    def _key(a, b):
        return (a, b) if a <= b else (b, a)

    fr_values = []
    node_weight = 1.0

    for u, v in G.edges():
        e_key = _key(u, v)
        w_e = edge_strength.get(e_key, 1.0)
        if w_e <= 0:
            continue

        # Termos de incidência no nó u (exclui a própria aresta e=(u,v))
        sum_u = 0.0
        for nbr in G.neighbors(u):
            if nbr == v:
                continue
            w_ue = edge_strength.get(_key(u, nbr), 1.0)
            sum_u += node_weight / np.sqrt(max(w_e * w_ue, 1.0e-12))

        # Termos de incidência no nó v (exclui a própria aresta e=(u,v))
        sum_v = 0.0
        for nbr in G.neighbors(v):
            if nbr == u:
                continue
            w_ve = edge_strength.get(_key(v, nbr), 1.0)
            sum_v += node_weight / np.sqrt(max(w_e * w_ve, 1.0e-12))

        # Curvatura Forman-Ricci ponderada para grafos sem faces.
        fr_e = w_e * ((node_weight / w_e) + (node_weight / w_e) - sum_u - sum_v)
        fr_values.append(fr_e)

    if not fr_values:
        return {
            'ricci_fr_mean': np.nan,
            'ricci_fr_var': np.nan,
            'ricci_fr_std': np.nan,
            'ricci_fr_skew': np.nan,
            'ricci_fr_kurt': np.nan,
            'ricci_fr_q10': np.nan,
            'ricci_fr_q50': np.nan,
            'ricci_fr_q90': np.nan,
            'ricci_fr_iqr': np.nan,
            'ricci_fr_min': np.nan,
            'ricci_fr_max': np.nan,
        }

    s = pd.Series(fr_values, dtype='float64')
    return {
        'ricci_fr_mean': float(s.mean()),
        'ricci_fr_var': float(s.var()),
        'ricci_fr_std': float(s.std()),
        'ricci_fr_skew': float(s.skew()),
        'ricci_fr_kurt': float(s.kurt()),
        'ricci_fr_q10': float(s.quantile(0.10)),
        'ricci_fr_q50': float(s.quantile(0.50)),
        'ricci_fr_q90': float(s.quantile(0.90)),
        'ricci_fr_iqr': float(s.quantile(0.75) - s.quantile(0.25)),
        'ricci_fr_min': float(s.min()),
        'ricci_fr_max': float(s.max()),
    }


def calcular_metricas_basicas(G: nx.Graph) -> Dict:
    """
    Calcula métricas básicas da rede
    
    Args:
        G: Grafo NetworkX
    
    Returns:
        Dict com métricas
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    # Densidade: proporção de arestas existentes / possíveis
    densidade = nx.density(G)
    
    # Grau médio
    degrees = dict(G.degree())
    grau_medio = np.mean(list(degrees.values()))
    grau_max = np.max(list(degrees.values()))
    
    # Componentes conectados
    n_componentes = nx.number_connected_components(G)
    
    # Tamanho do maior componente
    if n_componentes > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        tamanho_maior_componente = len(largest_cc)
    else:
        tamanho_maior_componente = 0
    
    return {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'densidade': densidade,
        'grau_medio': grau_medio,
        'grau_max': grau_max,
        'n_componentes': n_componentes,
        'tamanho_maior_componente': tamanho_maior_componente
    }


def calcular_metricas_centralidade(G: nx.Graph) -> pd.DataFrame:
    """
    Calcula métricas de centralidade para cada nó
    
    Args:
        G: Grafo NetworkX
    
    Returns:
        DataFrame com métricas por nó
    """
    n_edges = G.number_of_edges()

    # Degree centrality (grau normalizado) – sempre calculado em CPU,
    # pois é barato mesmo para grafos moderados.
    degree_cent = nx.degree_centrality(G)

    # Para grafos muito grandes/densos (ex.: rede completa com centenas
    # de milhares de arestas), evitar centralidades \mathcal{O}(|V||E|)
    # que tornam a FASE 5 impraticável. Nesses casos, preenchemos as
    # demais métricas com NaN.
    nodes = list(G.nodes())

    if n_edges > HEAVY_EDGE_THRESHOLD:
        # Grafos muito densos: continuamos evitando métricas pesadas,
        # mesmo que cuGraph esteja disponível, para não arriscar
        # estouro de memória em GPUs menores.
        nan_map = {node: np.nan for node in nodes}
        df = pd.DataFrame({
            'degree_centrality': degree_cent,
            'betweenness_centrality': nan_map,
            'closeness_centrality': nan_map,
            'eigenvector_centrality': nan_map,
            'degree': dict(G.degree())
        })
        # Garante mesma ordem dos nós
        df = df.reindex(nodes)
        return df

    # Grafos moderados: tentamos usar cuGraph para as métricas mais
    # pesadas (betweenness, eigenvector). Closeness permanece em
    # NetworkX, pois é bem mais leve nos grafos filtrados (MST/Planar).
    betweenness_cent = None
    eigenvector_cent = None

    if HAS_CUGRAPH:
        try:
            G_cu = nx_to_cugraph(G, directed=False)
        except Exception:
            G_cu = None

        if G_cu is not None:
            try:
                import cugraph  # type: ignore

                # Betweenness centrality em GPU
                bc_df = cugraph.betweenness_centrality(G_cu)  # type: ignore[union-attr]
                bc_s = bc_df.set_index('vertex')['betweenness_centrality']
                betweenness_cent = bc_s.to_pandas().to_dict()

                # Eigenvector centrality em GPU
                ev_res = cugraph.eigenvector_centrality(G_cu, max_iter=1000, tol=1.0e-5)  # type: ignore[union-attr]
                if isinstance(ev_res, tuple):
                    ev_df = ev_res[0]
                else:
                    ev_df = ev_res
                ev_s = ev_df.set_index('vertex')['eigenvector_centrality']
                eigenvector_cent = ev_s.to_pandas().to_dict()
            except Exception:
                # Qualquer falha na pilha RAPIDS → volta para NetworkX
                betweenness_cent = None
                eigenvector_cent = None
                pagerank = None

    # Fallback para implementação CPU (NetworkX) caso cuGraph não
    # esteja disponível ou tenha falhado em tempo de execução.
    if betweenness_cent is None:
        betweenness_cent = nx.betweenness_centrality(G)
    if eigenvector_cent is None:
        try:
            eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)
        except Exception:
            eigenvector_cent = {node: 0 for node in nodes}

    # Métricas mais leves continuam em NetworkX
    closeness_cent = nx.closeness_centrality(G)

    # Monta DataFrame apenas com as métricas necessárias para a análise:
    # grau (degree), betweenness, proximidade (closeness), autovalor (eigenvector)
    # e grau normalizado (degree_centrality).
    df = pd.DataFrame({
        'degree_centrality': degree_cent,
        'betweenness_centrality': betweenness_cent,
        'closeness_centrality': closeness_cent,
        'eigenvector_centrality': eigenvector_cent,
        'degree': dict(G.degree())
    })

    df = df.reindex(nodes)
    return df


def calcular_metricas_globais(G: nx.Graph) -> Dict:
    """
    Calcula métricas globais da rede
    
    Args:
        G: Grafo NetworkX
    
    Returns:
        Dict com métricas globais
    """
    metricas = calcular_metricas_basicas(G)

    n_edges = G.number_of_edges()

    # Para grafos menores/moderados, calcula clustering, assortatividade e
    # distâncias normalmente. Para grafos muito grandes (ex.: rede completa
    # com centenas de milhares de arestas), esses cálculos podem ser
    # proibitivamente caros, então são omitidos (NaN) para não travar a FASE 5.
    if n_edges <= HEAVY_EDGE_THRESHOLD:
        # Coeficiente de clustering médio
        clustering_medio = nx.average_clustering(G)
        metricas['clustering_medio'] = clustering_medio
        metricas['transitivity_global'] = nx.transitivity(G)

        # Assortatividade (tendência de nós similares se conectarem)
        try:
            assortatividade = nx.degree_assortativity_coefficient(G)
            metricas['assortatividade'] = assortatividade
        except Exception:
            metricas['assortatividade'] = np.nan

        # Distância média (caminho mais curto médio)
        if nx.is_connected(G):
            dist_media = nx.average_shortest_path_length(G)
            diametro = nx.diameter(G)
            metricas['distancia_media'] = dist_media
            metricas['diametro'] = diametro
        else:
            # Se não conectado, calcular para maior componente
            largest_cc = max(nx.connected_components(G), key=len)
            G_cc = G.subgraph(largest_cc).copy()
            if len(G_cc) > 1:
                dist_media = nx.average_shortest_path_length(G_cc)
                diametro = nx.diameter(G_cc)
                metricas['distancia_media'] = dist_media
                metricas['diametro'] = diametro
            else:
                metricas['distancia_media'] = np.nan
                metricas['diametro'] = np.nan
    else:
        metricas['clustering_medio'] = np.nan
        metricas['transitivity_global'] = np.nan
        metricas['assortatividade'] = np.nan
        metricas['distancia_media'] = np.nan
        metricas['diametro'] = np.nan

    # Curvatura Forman-Ricci: útil para distinguir arestas internas de
    # comunidade vs pontes entre blocos, especialmente nas redes filtradas/planar.
    # Em grafos muito grandes, evita custo elevado e registra NaN.
    if n_edges <= HEAVY_EDGE_THRESHOLD:
        metricas.update(calcular_forman_ricci_stats(G))
    else:
        metricas['ricci_fr_mean'] = np.nan
        metricas['ricci_fr_var'] = np.nan
        metricas['ricci_fr_std'] = np.nan
        metricas['ricci_fr_skew'] = np.nan
        metricas['ricci_fr_kurt'] = np.nan
        metricas['ricci_fr_q10'] = np.nan
        metricas['ricci_fr_q50'] = np.nan
        metricas['ricci_fr_q90'] = np.nan
        metricas['ricci_fr_iqr'] = np.nan
        metricas['ricci_fr_min'] = np.nan
        metricas['ricci_fr_max'] = np.nan

    # Estatísticas da distribuição de pesos/distâncias das arestas
    if G.number_of_edges() > 0:
        dist_values = []
        for _, _, data in G.edges(data=True):
            # Preferencialmente usa atributo 'distance', senão 'weight'
            val = None
            if 'distance' in data:
                val = data['distance']
            elif 'weight' in data:
                val = data['weight']
            if val is not None and not pd.isna(val):
                dist_values.append(val)

        if dist_values:
            s = pd.Series(dist_values)
            metricas['dist_mean'] = s.mean()
            metricas['dist_var'] = s.var()
            metricas['dist_std'] = s.std()
            metricas['dist_skew'] = s.skew()
            metricas['dist_kurt'] = s.kurt()
    
    return metricas