"""
Construção de Grafos
=====================

Converte matrizes de distância em grafos (redes)

Métodos implementados:
- Grafo Completo: todas as conexões possíveis
- MST (Minimum Spanning Tree): árvore geradora mínima
- PMFG (Planar Maximally Filtered Graph): grafo planar máximo
- KNN (K-Nearest Neighbors): k vizinhos mais próximos

Autor: Gerson Nassor Cardoso
Instituição: Universidade Federal de São Paulo (UNIFESP)
Data: 2026-02-13

Copyright (c) 2026 Gerson Nassor Cardoso - UNIFESP
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Union, Tuple
import warnings


def construir_grafo_completo(dist_matrix: pd.DataFrame) -> nx.Graph:
    """
    Constrói grafo completo ponderado
    
    Todas as conexões possíveis, peso = distância
    
    Args:
        dist_matrix: Matriz de distância (NxN)
    
    Returns:
        Grafo NetworkX completo
    """
    G = nx.Graph()
    
    # Adicionar nós
    G.add_nodes_from(dist_matrix.index)
    
    # Adicionar todas as arestas (triângulo superior)
    n = len(dist_matrix)
    for i in range(n):
        for j in range(i+1, n):
            dist = dist_matrix.iloc[i, j]
            
            if dist > 0 and not np.isnan(dist):
                node_i = dist_matrix.index[i]
                node_j = dist_matrix.index[j]
                G.add_edge(node_i, node_j, weight=dist, distance=dist)
    
    return G


def construir_grafo_mst(dist_matrix: pd.DataFrame) -> nx.Graph:
    """
    Constrói Minimum Spanning Tree (MST) - Árvore Geradora Mínima
    
    Conecta todos os nós minimizando a soma total das distâncias.
    Resultado: grafo conexo sem ciclos (árvore)
    
    Propriedades:
    - N nós → N-1 arestas
    - Conecta todos os nós
    - Minimiza soma das distâncias
    
    Args:
        dist_matrix: Matriz de distância (NxN)
    
    Returns:
        Grafo NetworkX (árvore)
    """
    # Criar grafo completo
    G_complete = construir_grafo_completo(dist_matrix)
    
    # Calcular MST usando algoritmo de Kruskal
    mst = nx.minimum_spanning_tree(G_complete, weight='weight')
    
    return mst


def construir_grafo_knn(dist_matrix: pd.DataFrame, 
                       k: int = 5,
                       simetrico: bool = True) -> nx.Graph:
    """
    Constrói grafo K-Nearest Neighbors (KNN)
    
    Cada nó conecta aos k vizinhos mais próximos
    
    Args:
        dist_matrix: Matriz de distância (NxN)
        k: Número de vizinhos mais próximos
        simetrico: Se True, torna grafo simétrico (A→B implica B→A)
    
    Returns:
        Grafo NetworkX
    """
    G = nx.Graph()
    G.add_nodes_from(dist_matrix.index)
    
    # Para cada nó
    for node in dist_matrix.index:
        # Pegar distâncias para todos os outros nós
        distances = dist_matrix.loc[node].copy()
        distances = distances[distances > 0]  # Remover self-loop
        
        # Pegar k menores distâncias
        if len(distances) >= k:
            k_nearest = distances.nsmallest(k)
        else:
            k_nearest = distances
        
        # Criar arestas
        for neighbor, dist in k_nearest.items():
            G.add_edge(node, neighbor, weight=dist, distance=dist)
    
    # Se simetrico, garantir que arestas sejam bidirecionais
    if simetrico:
        # Já é simétrico por usar nx.Graph (não direcionado)
        pass
    
    return G


def construir_grafo_planar(dist_matrix: pd.DataFrame, 
                          max_edges: int = None) -> nx.Graph:
    """
    Constrói grafo planar máximo (PMFG - Planar Maximally Filtered Graph)
    
    Grafo planar: pode ser desenhado no plano sem arestas se cruzando
    PMFG: máximo de arestas mantendo planaridade
    
    Propriedades:
    - N nós → max 3N-6 arestas (para N ≥ 3)
    - Pode ser desenhado sem cruzamentos
    
    Args:
        dist_matrix: Matriz de distância (NxN)
        max_edges: Número máximo de arestas (padrão: 3N-6)
    
    Returns:
        Grafo NetworkX planar
    """
    n = len(dist_matrix)
    
    if max_edges is None:
        max_edges = 3 * n - 6 if n >= 3 else n * (n - 1) // 2
    
    # Criar lista de todas as arestas com distâncias
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            dist = dist_matrix.iloc[i, j]
            if dist > 0 and not np.isnan(dist):
                node_i = dist_matrix.index[i]
                node_j = dist_matrix.index[j]
                edges.append((node_i, node_j, dist))
    
    # Ordenar por distância (menor primeiro)
    edges.sort(key=lambda x: x[2])
    
    # Construir grafo adicionando arestas mantendo planaridade
    G = nx.Graph()
    G.add_nodes_from(dist_matrix.index)
    
    for node_i, node_j, dist in edges:
        if G.number_of_edges() >= max_edges:
            break
        
        # Adicionar aresta temporariamente
        G.add_edge(node_i, node_j, weight=dist, distance=dist)
        
        # Verificar se ainda é planar
        if not nx.is_planar(G):
            # Remover aresta se quebrar planaridade
            G.remove_edge(node_i, node_j)
    
    return G


def validar_grafo(G: nx.Graph, dist_matrix: pd.DataFrame) -> dict:
    """
    Valida propriedades do grafo construído
    
    Args:
        G: Grafo NetworkX
        dist_matrix: Matriz de distância original
    
    Returns:
        Dict com resultados da validação
    """
    n_nodes_esperado = len(dist_matrix)
    n_nodes_grafo = G.number_of_nodes()
    
    validacao = {
        'n_nodes_esperado': n_nodes_esperado,
        'n_nodes_grafo': n_nodes_grafo,
        'n_edges': G.number_of_edges(),
        'nodes_faltando': n_nodes_esperado - n_nodes_grafo,
        'conectado': nx.is_connected(G),
        'n_componentes': nx.number_connected_components(G),
        'planar': nx.is_planar(G),
        'aciclico': nx.is_tree(G),
        'densidade': nx.density(G)
    }
    
    # Verificar se pesos correspondem às distâncias
    discrepancias = []
    for u, v, data in G.edges(data=True):
        if 'weight' in data:
            peso_grafo = data['weight']
            dist_matriz = dist_matrix.loc[u, v]
            
            if not np.isclose(peso_grafo, dist_matriz, rtol=1e-5):
                discrepancias.append((u, v, peso_grafo, dist_matriz))
    
    validacao['n_discrepancias_peso'] = len(discrepancias)
    validacao['valido'] = (
        validacao['nodes_faltando'] == 0 and
        validacao['n_discrepancias_peso'] == 0
    )
    
    return validacao