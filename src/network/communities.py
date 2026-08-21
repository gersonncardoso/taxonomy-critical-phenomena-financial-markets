"""
Detecção de Comunidades
========================

Identifica comunidades (grupos) em redes

Autor: Gerson Nassor Cardoso
Instituição: Universidade Federal de São Paulo (UNIFESP)
Data: 2026-02-13

Copyright (c) 2026 Gerson Nassor Cardoso - UNIFESP
"""

import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import community
from typing import Dict, List

from .cugraph_utils import HAS_CUGRAPH, nx_to_cugraph


def detectar_comunidades_louvain(G: nx.Graph) -> Dict[str, int]:
    """
    Detecta comunidades usando algoritmo de Louvain
    
    Maximiza modularidade (qualidade da divisão em comunidades)
    
    Args:
        G: Grafo NetworkX
    
    Returns:
        Dict {nó: comunidade}
    """
    # Primeiro tenta usar cuGraph, se disponível, pois a detecção de
    # comunidades Louvain é uma das partes mais custosas da FASE 5.
    if HAS_CUGRAPH:
        try:
            G_cu = nx_to_cugraph(G, directed=False)
        except Exception:
            G_cu = None

        if G_cu is not None:
            try:
                import cugraph  # type: ignore

                parts_df, _mod = cugraph.louvain(G_cu)  # type: ignore[union-attr]
                parts_pd = parts_df.to_pandas()
                node_to_community = dict(
                    zip(parts_pd["vertex"], parts_pd["partition"])
                )
                return node_to_community
            except Exception:
                # Qualquer problema na pilha RAPIDS → fallback para NetworkX
                pass

    # Fallback: implementação em NetworkX
    if not G.is_directed():
        communities = community.louvain_communities(G, seed=42)
    else:
        G_undirected = G.to_undirected()
        communities = community.louvain_communities(G_undirected, seed=42)
    
    # Converter para dict
    node_to_community: Dict[str, int] = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_community[node] = i
    
    return node_to_community


def detectar_comunidades_label_propagation(G: nx.Graph) -> Dict[str, int]:
    """
    Detecta comunidades usando Label Propagation
    
    Args:
        G: Grafo NetworkX
    
    Returns:
        Dict {nó: comunidade}
    """
    communities = community.label_propagation_communities(G)
    
    node_to_community = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_community[node] = i
    
    return node_to_community


def analisar_modularidade(G: nx.Graph, communities_dict: Dict[str, int]) -> float:
    """
    Calcula modularidade de uma partição em comunidades
    
    Modularidade varia de -1 a 1:
    - Próximo de 1: boa divisão em comunidades
    - Próximo de 0: divisão aleatória
    - Negativo: pior que aleatório
    
    Args:
        G: Grafo NetworkX
        communities_dict: Dict {nó: comunidade}
    
    Returns:
        Modularidade
    """
    # Casos degenerados: sem arestas (ou sem nós) não possuem modularidade informativa.
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return 0.0

    # Converter dict para lista de sets
    unique_communities = set(communities_dict.values())
    communities_list = []
    for comm_id in unique_communities:
        nodes_in_comm = [node for node, c in communities_dict.items() if c == comm_id]
        communities_list.append(set(nodes_in_comm))

    if not communities_list:
        return 0.0
    
    # Calcular modularidade
    try:
        modularity = community.modularity(G, communities_list)
    except ZeroDivisionError:
        modularity = 0.0
    
    return modularity


def comparar_comunidades(comm1: Dict[str, int], comm2: Dict[str, int]) -> Dict:
    """
    Compara duas detecções de comunidades
    
    Args:
        comm1: Comunidades da primeira detecção
        comm2: Comunidades da segunda detecção
    
    Returns:
        Dict com métricas de comparação
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    # Garantir mesma ordem de nós
    nodes = sorted(set(comm1.keys()) & set(comm2.keys()))
    labels1 = [comm1[node] for node in nodes]
    labels2 = [comm2[node] for node in nodes]
    
    # Adjusted Rand Index
    ari = adjusted_rand_score(labels1, labels2)
    
    # Normalized Mutual Information
    nmi = normalized_mutual_info_score(labels1, labels2)
    
    return {
        'ari': ari,
        'nmi': nmi,
        'n_nodes_comuns': len(nodes)
    }