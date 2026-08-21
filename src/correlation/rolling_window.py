"""
Funções de Janelas Móveis (Rolling Windows)
===========================================

Autor: Gerson Nassor Cardoso
Data: 2026-02-13
"""

import pandas as pd

def criar_janelas_temporais(df: pd.DataFrame, janela_meses: int = 12, step_meses: int = 1) -> list:
    """
    Cria janelas temporais com step mensal.
    """
    data_min = df['Date'].min()
    data_max = df['Date'].max()
    janelas = []
    data_inicio = data_min

    while True:
        data_fim = data_inicio + pd.DateOffset(months=janela_meses)
        if data_fim > data_max:
            break
        janelas.append({
            'id': len(janelas) + 1,
            'data_inicio': data_inicio,
            'data_fim': data_fim,
            'label': f"{data_inicio.strftime('%Y-%m')} a {data_fim.strftime('%Y-%m')}"
        })
        data_inicio = data_inicio + pd.DateOffset(months=step_meses)
    return janelas