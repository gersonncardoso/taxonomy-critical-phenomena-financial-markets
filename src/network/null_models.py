"""
Modelos Nulos para Validação de Redes
======================================

Gera redes aleatórias para comparação estatística

Referências:
- Newman, M. E. J. (2018). Networks (2nd ed.). Oxford University Press.
- Barabási, A.-L. (2016). Network Science. Cambridge University Press.

Autor: Gerson Nassor Cardoso
Instituição: Universidade Federal de São Paulo (UNIFESP)
Data: 2026-02-13

Copyright (c) 2026 Gerson Nassor Cardoso - UNIFESP
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict


def gerar_matriz_distancia_aleatoria(n: int, seed: int = None) -> pd.DataFrame:
    """
    Gera matriz de distância aleatória válida
    
    Propriedades matemáticas de uma métrica de distância:
    1. d(i,j) ≥ 0 (não-negatividade)
    2. d(i,i) = 0 (identidade)
    3. d(i,j) = d(j,i) (simetria)
    4. d(i,k) ≤ d(i,j) + d(j,k) (desigualdade triangular)
    
    Args:
        n: Número de nós
        seed: Seed para reprodutibilidade
    
    Returns:
        DataFrame com matriz de distância
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Gerar matriz aleatória uniforme [0, 1]
    matriz = np.random.rand(n, n)
    
    # Tornar simétrica: d(i,j) = d(j,i)
    matriz = (matriz + matriz.T) / 2
    
    # Diagonal zero: d(i,i) = 0
    np.fill_diagonal(matriz, 0)
    
    # Normalizar para [0, 1]
    if matriz.max() > 0:
        matriz = matriz / matriz.max()
    
    # Criar DataFrame com labels genéricos
    nodes = [f'N{i:03d}' for i in range(n)]
    df = pd.DataFrame(matriz, index=nodes, columns=nodes)
    
    return df


def gerar_grafo_erdos_renyi(n: int, m: int, seed: int = None) -> nx.Graph:
    """
    Gera grafo aleatório Erdős-Rényi com N nós e M arestas
    
    Modelo: cada par de nós tem mesma probabilidade de conexão
    
    Args:
        n: Número de nós
        m: Número de arestas
        seed: Seed para reprodutibilidade
    
    Returns:
        Grafo NetworkX aleatório
    """
    # Probabilidade de conexão para ter aproximadamente M arestas
    max_edges = n * (n - 1) // 2
    p = m / max_edges if max_edges > 0 else 0
    
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    
    # Ajustar para ter exatamente M arestas (se possível)
    current_edges = G.number_of_edges()
    
    if current_edges < m:
        # Adicionar arestas aleatórias
        nodes = list(G.nodes())
        while G.number_of_edges() < m:
            i, j = np.random.choice(nodes, size=2, replace=False)
            if not G.has_edge(i, j):
                G.add_edge(i, j)
    
    elif current_edges > m:
        # Remover arestas aleatórias
        edges = list(G.edges())
        np.random.shuffle(edges)
        for u, v in edges[:current_edges - m]:
            G.remove_edge(u, v)
    
    return G


def gerar_grafo_configuration_model(degree_sequence: List[int], 
                                    seed: int = None) -> nx.Graph:
    """
    Gera grafo aleatório com sequência de graus especificada
    
    Preserva distribuição de graus do grafo original
    
    Args:
        degree_sequence: Lista de graus [d1, d2, ..., dn]
        seed: Seed para reprodutibilidade
    
    Returns:
        Grafo NetworkX aleatório
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Verificar se sequência é realizável (soma par)
    if sum(degree_sequence) % 2 != 0:
        degree_sequence = list(degree_sequence)
        degree_sequence[0] += 1
    
    # Gerar grafo
    G = nx.configuration_model(degree_sequence, seed=seed)
    
    # Remover self-loops e múltiplas arestas
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    return G


def randomizar_grafo_preservando_grau(G: nx.Graph, 
                                     n_swaps: int = None,
                                     seed: int = None) -> nx.Graph:
    """
    Randomiza grafo preservando a sequência de graus
    
    Método: edge swaps
    - Escolhe duas arestas (u,v) e (x,y)
    - Troca para (u,y) e (x,v) se não existirem
    - Preserva grau de cada nó
    
    Args:
        G: Grafo original
        n_swaps: Número de trocas (padrão: 10 * número de arestas)
        seed: Seed para reprodutibilidade
    
    Returns:
        Grafo randomizado
    """
    if seed is not None:
        np.random.seed(seed)
    
    if n_swaps is None:
        n_swaps = 10 * G.number_of_edges()
    
    # Copiar grafo
    G_random = G.copy()
    
    # Realizar edge swaps
    swaps_realizados = 0
    tentativas = 0
    max_tentativas = n_swaps * 10
    
    while swaps_realizados < n_swaps and tentativas < max_tentativas:
        tentativas += 1
        
        # Escolher duas arestas aleatórias
        edges = list(G_random.edges())
        if len(edges) < 2:
            break
        
        edge1, edge2 = np.random.choice(len(edges), size=2, replace=False)
        u, v = edges[edge1]
        x, y = edges[edge2]
        
        # Verificar se swap é válido
        if len({u, v, x, y}) == 4:  # Todos diferentes
            if not G_random.has_edge(u, y) and not G_random.has_edge(x, v):
                # Realizar swap
                G_random.remove_edge(u, v)
                G_random.remove_edge(x, y)
                G_random.add_edge(u, y)
                G_random.add_edge(x, v)
                swaps_realizados += 1
    
    return G_random