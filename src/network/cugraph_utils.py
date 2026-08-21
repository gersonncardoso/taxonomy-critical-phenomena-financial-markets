"""Integração opcional com cuGraph.

Este módulo centraliza a detecção da disponibilidade do cuGraph/cuDF e
fornece um helper para converter grafos NetworkX em grafos cuGraph.

Se cuGraph/cuDF não estiverem instalados (ou o ambiente não suportar
RAPIDS), ``HAS_CUGRAPH`` será ``False`` e as funções aqui definidas
retornarão ``None`` sem levantar exceções, permitindo fallback limpo
para a implementação em NetworkX.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx

try:  # Importação preguiçosa/robusta do stack RAPIDS
    import cudf  # type: ignore
    import cugraph  # type: ignore

    HAS_CUGRAPH: bool = True
except Exception:  # ImportError + quaisquer erros de runtime
    cudf = None  # type: ignore
    cugraph = None  # type: ignore
    HAS_CUGRAPH = False


def nx_to_cugraph(G: nx.Graph, *, directed: Optional[bool] = None):
    """Converte um grafo NetworkX para um grafo cuGraph.

    Retorna ``None`` se cuGraph não estiver disponível ou se o grafo não
    tiver arestas (caso degenerado onde centralidades não são úteis).
    """
    if not HAS_CUGRAPH:
        return None

    if directed is None:
        directed = G.is_directed()

    # Para muitas rotinas (Louvain, métricas globais) trabalhamos em
    # grafos não direcionados; quem chama pode forçar ``directed=False``.
    G_use = G if directed else G.to_undirected()

    import pandas as pd

    # Cria edgelist: columns = ["source", "target", ...]
    edges_pd: pd.DataFrame = nx.to_pandas_edgelist(G_use)
    if edges_pd.empty:
        return None

    # Renomeia para convenção do cuGraph
    edges_pd = edges_pd.rename(columns={"source": "src", "target": "dst"})

    # Garante coluna de peso
    if "weight" not in edges_pd.columns:
        if "distance" in edges_pd.columns:
            edges_pd["weight"] = edges_pd["distance"]
        else:
            edges_pd["weight"] = 1.0

    # Converte para cuDF e cria grafo em GPU
    gdf = cudf.from_pandas(edges_pd)  # type: ignore[union-attr]

    G_cu = cugraph.Graph(directed=directed)  # type: ignore[call-arg]
    G_cu.from_cudf_edgelist(  # type: ignore[union-attr]
        gdf,
        source="src",
        destination="dst",
        edge_attr="weight",
        renumber=True,
    )
    return G_cu
