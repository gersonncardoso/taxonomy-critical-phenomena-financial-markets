"""
Funções auxiliares para correlação
==================================
"""

import pandas as pd
import numpy as np

def calcular_correlacao_janela(df: pd.DataFrame, janela: dict, min_obs: float = 0.80) -> pd.DataFrame:
    """
    Calcula matriz de correlação para uma janela.
    Usa coluna 'Date', 'Ticker', 'Retorno_Log'.
    """
    df_janela = df[
        (df['Date'] >= janela['data_inicio']) &
        (df['Date'] < janela['data_fim'])
    ].copy()
    if df_janela.empty:
        return None
    dias_janela = df_janela['Date'].nunique()
    completude = df_janela.groupby('Ticker')['Date'].nunique() / dias_janela
    tickers_validos = completude[completude >= min_obs].index.tolist()
    df_janela = df_janela[df_janela['Ticker'].isin(tickers_validos)]
    if df_janela['Ticker'].nunique() < 2:
        return None
    df_pivot = df_janela.pivot(index='Date', columns='Ticker', values='Retorno_Log')
    corr_matrix = df_pivot.corr()
    return corr_matrix

def extract_correlation_stats(corr_matrix: pd.DataFrame) -> dict:
    """Extrai estatísticas do triângulo superior da matriz de correlação."""
    corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    return {
        'Num_Tickers': len(corr_matrix),
        'Num_Pares': len(corr_values),
        'Correlacao_Media': float(np.mean(corr_values)),
        'Correlacao_Mediana': float(np.median(corr_values)),
        'Correlacao_Std': float(np.std(corr_values)),
        'Correlacao_Min': float(np.min(corr_values)),
        'Correlacao_Max': float(np.max(corr_values)),
        'Correlacao_Q25': float(np.percentile(corr_values, 25)),
        'Correlacao_Q75': float(np.percentile(corr_values, 75)),
    }

def matrix_to_long(corr_matrix, window_id, window_start, window_end):
    """Converte matriz de correlação para formato long."""
    rows = []
    tickers = corr_matrix.index
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            cor = corr_matrix.iloc[i, j]
            rows.append({
                "Janela_ID": window_id,
                "Window_Start": window_start,
                "Window_End": window_end,
                "Ticker1": t1, "Ticker2": t2,
                "Correlacao": cor
            })
    return rows