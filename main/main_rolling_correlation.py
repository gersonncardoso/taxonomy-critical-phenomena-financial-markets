"""
Cálculo de Correlações Rolling - VERSÃO 2.0 AUTOMÁTICA + MATRIZ LONG ÚNICA
===========================================================================

Calcula correlações em janelas móveis e salva:
- Apenas o resumo (stats) em correlacoes_summary.csv
- Todas as matrizes de correlação, formato LONG, em correlacoes_matrizes_long.csv (um único arquivo flat)

Autor: Gerson Nassor Cardoso
Data: 2026-02-19 (mod. Copilot)
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logger import setup_logger
from src.utils.gpu_utils import GPU_AVAILABLE, cp

logger = setup_logger('main_rolling_correlation')

_CORR_GPU_INFO_PRINTED = False

def calculate_correlation_matrix(df_window: pd.DataFrame) -> pd.DataFrame:
    """Calcula matriz de correlação para uma janela.

    - Usa pivot_table com agregação (média) para lidar com possíveis
      duplicatas Date/Ticker sem quebrar com o erro
      "Index contains duplicate entries, cannot reshape".
    - Quando GPU (CuPy) está disponível, calcula a correlação na GPU,
      com fallback automático para CPU (pandas.corr).
    """
    global _CORR_GPU_INFO_PRINTED

    df_wide = df_window.pivot_table(
        index='Date',
        columns='Ticker',
        values='Retorno_Log',
        aggfunc='mean',
    )

    # Se matriz muito pequena, não vale a pena ir para GPU
    if df_wide.shape[0] <= 1 or df_wide.shape[1] <= 1:
        return df_wide.corr(method='pearson')

    # Caminho GPU (CuPy) quando disponível
    if GPU_AVAILABLE and cp is not None:
        try:
            if not _CORR_GPU_INFO_PRINTED:
                logger.info("[GPU] Usando CuPy para matrizes de correlação em main_rolling_correlation.")
                _CORR_GPU_INFO_PRINTED = True

            x = cp.asarray(df_wide.to_numpy(dtype=float))
            n_obs = x.shape[0]

            # Centraliza por coluna
            x = x - cp.mean(x, axis=0, keepdims=True)
            # Covariância amostral
            cov = (x.T @ x) / (n_obs - 1)
            std = cp.sqrt(cp.diag(cov))
            denom = std[:, None] * std[None, :]
            corr = cov / (denom + 1e-12)
            corr = cp.clip(corr, -1.0, 1.0)

            corr_np = cp.asnumpy(corr)
            return pd.DataFrame(corr_np, index=df_wide.columns, columns=df_wide.columns)
        except Exception as e:
            logger.warning(f"[GPU] Falha ao calcular correlação em GPU na Fase 2; voltando para CPU. Detalhe: {e}")

    # Fallback seguro: CPU
    return df_wide.corr(method='pearson')

def calculate_long_matrix(corr_matrix, window_id, window_start, window_end):
    # Converte para long apenas triângulo superior sem diagonal
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

def extract_correlation_stats(corr_matrix: pd.DataFrame) -> dict:
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
        'Correlacao_Q75': float(np.percentile(corr_values, 75))
    }

def main():
    print("\n" + "="*80)
    print("📊 CORRELAÇÕES ROLLING - VERSÃO 2.0 AUTOMÁTICA")
    print("="*80)
    print("\nSalvando: correlacoes_summary.csv (stats) e correlacoes_matrizes_long.csv (todas as janelas, formato long)\n")
    consolidado_file = Path('data/processed/dados_consolidados.csv')
    if not consolidado_file.exists():
        print("❌ ERRO: Arquivo consolidado não encontrado! Execute main_preprocessing primeiro.")
        return False
    try:
        df = pd.read_csv(consolidado_file, parse_dates=['Date'])
        logger.info(f"Dados carregados: {len(df):,} registros")
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        print(f"\n❌ ERRO ao carregar dados: {e}")
        return False

    window_ids = sorted(df['Janela_ID'].dropna().unique())
    print(f"Janelas a processar: {len(window_ids)}\n")

    summary = []
    matrix_long_rows = []

    for window_id in tqdm(window_ids, desc="Processando janelas", unit="win"):
        try:
            df_window = df[df['Janela_ID'] == window_id].copy()
            if df_window.empty or df_window['Ticker'].nunique() < 2:
                logger.warning(f"Janela {window_id} vazia ou <2 tickers")
                continue
            corr_matrix = calculate_correlation_matrix(df_window)
            if corr_matrix.isna().all().all():
                logger.warning(f"Janela {window_id} matriz inválida")
                continue
            win_start = df_window['Window_Start'].iloc[0] if 'Window_Start' in df_window else pd.NaT
            win_end = df_window['Window_End'].iloc[0] if 'Window_End' in df_window else pd.NaT
            stats = extract_correlation_stats(corr_matrix)
            stats['Janela_ID'] = window_id
            stats['Window_Start'] = win_start
            stats['Window_End'] = win_end
            summary.append(stats)

            # Adiciona todas as combinações (long)
            matrix_long_rows.extend(
                calculate_long_matrix(corr_matrix, window_id, win_start, win_end)
            )
        except Exception as e:
            logger.error(f"Erro em janela {window_id}: {e}")
            continue

    # Salvar summary
    output_summary = Path('data/processed/correlacoes_summary.csv')
    pd.DataFrame(summary).sort_values("Janela_ID").to_csv(output_summary, index=False)
    print(f"\n✅ Stats resumidos salvos em: {output_summary}")

    # Salvar matriz long (append-friendly)
    output_long = Path('data/processed/correlacoes_matrizes_long.csv')
    pd.DataFrame(matrix_long_rows).to_csv(output_long, index=False)
    print(f"✅ Matrizes long salvas em: {output_long}  ({len(matrix_long_rows)} linhas)")

    print("\n✔️ Tudo pronto!")
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Processamento interrompido pelo usuário!")
        logger.warning("Processamento interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERRO FATAL: {str(e)}")
        logger.error(f"Erro fatal: {str(e)}", exc_info=True)
        raise