import pandas as pd
import networkx as nx
import glob
import os

def salvar_metricas_em_csv_long(metrics_dict, window_id, rede_tipo, window_start, window_end, csv_path, append=False, include_header=True):
    """
    Salva as métricas globais em formato long no arquivo csv_path.
    Se append=True, adiciona ao final do arquivo (modo 'a'); caso contrário, sobrescreve.
    O cabeçalho será incluído apenas se include_header=True.
    """
    metrics_row = metrics_dict.copy()
    metrics_row["Janela_ID"] = window_id
    metrics_row["Tipo_Rede"] = rede_tipo
    metrics_row["Window_Start"] = window_start
    metrics_row["Window_End"] = window_end
    df_row = pd.DataFrame([metrics_row])
    modo = 'a' if append else 'w'
    with open(csv_path, modo, newline='') as f:
        df_row.to_csv(f, index=False, header=include_header)

def salvar_centralidades_csv_long(df_centralidade, window_id, rede_tipo, window_start, window_end, csv_path, append=False, include_header=True):
    """
    Salva as centralidades em formato long no arquivo csv_path.
    Se append=True, adiciona ao final do arquivo (modo 'a'); caso contrário, sobrescreve.
    O cabeçalho será incluído apenas se include_header=True.
    """
    df_centralidade_long = df_centralidade.reset_index().rename(columns={"index": "Node"})
    df_centralidade_long["Janela_ID"] = window_id
    df_centralidade_long["Tipo_Rede"] = rede_tipo
    df_centralidade_long["Window_Start"] = window_start
    df_centralidade_long["Window_End"] = window_end
    modo = 'a' if append else 'w'
    with open(csv_path, modo, newline='') as f:
        df_centralidade_long.to_csv(f, index=False, header=include_header)

def salvar_grafo_csv_edgelist(G, path):
    edgelist = nx.to_pandas_edgelist(G)
    edgelist.to_csv(path, index=False)

def construir_grafo_filtrado_pvalue(pvals_df, window_id, pvalue_threshold=0.05):
    rows = pvals_df[pvals_df["Janela_ID"] == window_id]
    G = nx.Graph()
    tickers = sorted(set(rows['Ticker1']).union(rows['Ticker2']))
    G.add_nodes_from(tickers)
    for _, row in rows.iterrows():
        if row["pvalue"] < pvalue_threshold and row["Ticker1"] != row["Ticker2"]:
            G.add_edge(row["Ticker1"], row["Ticker2"], weight=abs(row["Correlacao"]))
    return G

def concatena_csvs(temp_dir, glob_in, csv_out):
    files = sorted(glob.glob(str(temp_dir / glob_in)))
    dfs = [pd.read_csv(f) for f in files if os.path.getsize(f) > 0]
    if dfs:
        pd.concat(dfs, ignore_index=True).to_csv(csv_out, index=False)