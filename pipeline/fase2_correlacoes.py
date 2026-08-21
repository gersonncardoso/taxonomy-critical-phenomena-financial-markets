"""
FASE 2: Correlações Rolling - GARANTE CONSOLIDAÇÃO PRÉVIA!
----------------------------------------------------------

2.1: Rolling correlação e geração de matrizes long e resumos
"""

from pathlib import Path
import pandas as pd

from pipeline._runner import run_python

def fase2_correlacoes(force: bool = False):
    print(f"\n{'#'*80}\nFASE 2: CORRELAÇÕES\n{'#'*80}")

    consolidado_file = Path('data/processed/dados_consolidados.csv')
    summary_file = Path('data/processed/correlacoes_summary.csv')
    long_file = Path('data/processed/correlacoes_matrizes_long.csv')


    # Agora calcula as correlações
    if not summary_file.exists() or not long_file.exists() or force:
        print("\n▶️  Calculando correlações rolling e salvando matrizes long e resumo...")
        run_python(["main/main_rolling_correlation.py"])
        # Checa e mostra um resumo dos resultados
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            print("\nResumo das primeiras 3 janelas:")
            print(df.head(3).to_string(index=False))
    else:
        print("\n⏭️  2.1. Correlações Rolling - JÁ CONCLUÍDO")
        df = pd.read_csv(summary_file)
        print("Janelas já calculadas (amostra):")
        print(df.head(3).to_string(index=False))

    print(f"\nMatrizes salvas em: {long_file}\nResumo em: {summary_file}")