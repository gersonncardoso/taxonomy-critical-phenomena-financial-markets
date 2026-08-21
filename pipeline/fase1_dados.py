"""
FASE 1: Dados
1.1: Download incremental (API B3)
1.2: Filtro de ativos
1.3: Pré-processamento/consolidação (gera dados_consolidados.csv e metadados)
1.4: Data quality (após consolidado; relatório de gaps/datas úteis)
"""

from pathlib import Path

import pandas as pd

from main.main_filter_assets import load_local_equity_universe_from_cvm

from pipeline._runner import run_python


def _filtered_file_needs_refresh(filtered_file: Path) -> bool:
    if not filtered_file.exists():
        return True

    try:
        whitelist = load_local_equity_universe_from_cvm(Path('configs/cvm_company_map.yaml'))
        if not whitelist:
            return False

        sample = pd.read_csv(filtered_file, usecols=['Ticker'], nrows=200000)
        sample['Ticker'] = sample['Ticker'].fillna('').astype(str).str.upper().str.strip()
        sample = sample[sample['Ticker'] != '']
        invalid = sorted(set(sample['Ticker']) - whitelist)
        if invalid:
            print(f"\n⚠️  Arquivo filtrado contém tickers fora do universo de ações locais: {invalid[:10]}")
            return True
    except Exception as e:
        print(f"\n⚠️  Não foi possível validar artefato filtrado existente: {e}")

    return False

def fase1_dados(force: bool = False):
    print(f"\n{'#'*80}\nFASE 1: DADOS\n{'#'*80}")

    # 1.1 Download (API B3)
    if not Path('data/raw/b3_dados_completos.csv').exists() or force:
        print(f"\n📊 1.1. Download incremental (API B3)...")
        run_python(["main/main_download.py"])
    else:
        print(f"\n⏭️  1.1. Download - JÁ CONCLUÍDO")

    # 1.2 Filtro de Ativos
    filtered_file = Path('data/raw/b3_dados_filtrados.csv')
    filter_needs_refresh = _filtered_file_needs_refresh(filtered_file)
    if not filtered_file.exists() or force or filter_needs_refresh:
        print("\n▶️  1.2. Filtro de ativos...")
        run_python(["main/main_filter_assets.py"])
    else:
        print(f"\n⏭️  1.2. Filtro de Ativos - JÁ CONCLUÍDO")

    # 1.3 Consolidação/Pré-processamento
    consolidado_file = Path('data/processed/dados_consolidados.csv')
    metadata_file = Path('data/processed/janelas_metadata.json')
    if not consolidado_file.exists() or not metadata_file.exists() or force:
        print("\n▶️  1.3. Executando pré-processamento e consolidação...")
        run_python(["main/main_preprocessing.py"])
    else:
        print("\n⏭️  1.3. Consolidação - JÁ CONCLUÍDA")

    # 1.4 Data quality (após consolidação)
    if consolidado_file.exists():
        print("\n📋  1.4. Data quality (relatório de gaps/datas úteis)...")
        try:
            from src.utils.data_quality import DataQualityChecker
            ref_file = Path('data/raw/b3_dados_filtrados.csv')
            if not ref_file.exists():
                ref_file = Path('data/raw/b3_dados_completos.csv')

            checker = DataQualityChecker(
                str(consolidado_file),
                reference_file=str(ref_file) if ref_file.exists() else None,
            )
            if checker.load_data():
                checker.print_summary()
                monthly_path = checker.save_monthly_report('data/processed/data_quality_monthly.csv')
                print(f"\n📄 Consolidado mensal de qualidade salvo em: {monthly_path}")
                # Se quiser também salvar em arquivo JSON (opcional):
                # checker.generate_report(output_file='data/processed/qualidade_dados.json')
        except Exception as e:
            print(f"\n❌ ERRO no relatório de qualidade pós-consolidação: {e}")

    # 1.5 Contexto ticker/CEO por janela (facilita consumo downstream)
    context_file = Path('data/processed/ticker_context_by_window.csv')
    if consolidado_file.exists() and (not context_file.exists() or force):
        print("\n🧩  1.5. Gerando contexto de ticker/CEO por janela...")
        run_python(["main/main_build_ticker_window_context.py"])
    elif context_file.exists():
        print("\n⏭️  1.5. Contexto por janela - JÁ CONCLUÍDO")