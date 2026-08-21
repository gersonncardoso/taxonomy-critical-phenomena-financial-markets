"""
Pré-processamento e Consolidação de Dados - VERSÃO 2.0

Consolida dados brutos com informações de janelas temporais em UM ÚNICO arquivo.

FUNCIONALIDADES:
- Detecção automática de janelas baseada em rolling de 12 meses civis (lag de 1 mês)
- Cálculo de retornos logarítmicos
- Atribuição de ID de janela para cada registro
- Geração de metadados de janelas
- Consolidação em formato LONG com todas as informações

SAÍDA:
- data/processed/dados_consolidados.csv (formato LONG com janelas)
- data/processed/janelas_metadata.json (metadados das janelas)

Autor: Gerson Nassor Cardoso
Instituição: Universidade Federal de São Paulo (UNIFESP)
Data: 2026-02-18
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.consolidator import DataConsolidator
from src.utils.window_manager import WindowManager
from src.utils.logger import setup_logger

logger = setup_logger('main_preprocessing')

def verificar_dados_existentes():
    possiveis_arquivos = [
        Path('data/raw/b3_dados_filtrados.csv'),
        Path('data/raw/b3_dados_ibovespa.csv'),
    ]
    for arquivo in possiveis_arquivos:
        if arquivo.exists():
            return arquivo
    return None

def main():
    print("\n" + "="*80)
    print("🔄 PRÉ-PROCESSAMENTO E CONSOLIDAÇÃO - VERSÃO 2.0")
    print("="*80)
    print()
    print("Este script executa um pipeline completo:")
    print("  1. Detecta janelas temporais baseadas em 12 meses civis (lag 1 mês)")
    print("  2. Calcula retornos logarítmicos")
    print("  3. Atribui ID de janela para cada observação")
    print("  4. Consolida tudo em UM ÚNICO arquivo CSV")
    print()
    
    # Verificar se dados brutos existem
    input_file = verificar_dados_existentes()
    if input_file is None:
        print("="*80)
        print("❌ ERRO: Nenhum arquivo de dados encontrado!")
        print("="*80)
        print("\nArquivos esperados:")
        print("  - data/raw/b3_dados_filtrados.csv")
        print("  - data/raw/b3_dados_ibovespa.csv")
        print("\n💡 Execute primeiro:")
        print("  python main/main_download.py")
        print("="*80)
        return
    
    print(f"✅ Arquivo de dados encontrado: {input_file}\n")
    window_num_months = 12

    print("="*80)
    print("⚙️  PARÂMETROS DE JANELAS")
    print("="*80)
    print(f"   Arquivo de entrada: {input_file}")
    print(f"   Tamanho da janela: {window_num_months} meses civis")
    print(f"   Rolling mensal: lag de 1 mês")
    print("="*80 + "\n")

    print("🪟 1. Detectando janelas temporais...")
    window_manager = WindowManager(
        data_file=str(input_file),
        window_num_months=window_num_months
    )
    
    try:
        window_manager.calculate_windows()
        num_windows = len(window_manager.windows)
        if num_windows == 0:
            print("="*80)
            print("❌ ERRO: Nenhuma janela detectada!")
            print("="*80)
            print(f"\nO arquivo possui dados suficientes?")
            print(f"  Mínimo necessário: {window_num_months} meses completos")
            print(f"\n💡 Verifique:")
            print(f"  - Período dos dados em {input_file}")
            print(f"  - Configuração de janela em configs/config.yaml")
            print("="*80)
            return
        print(f"\n✅ Detectadas {num_windows} janelas temporais de 12 meses com lag mensal.\n")
    except Exception as e:
        logger.error(f"Erro ao calcular janelas: {e}", exc_info=True)
        print(f"\n❌ ERRO ao calcular janelas: {e}")
        return

    print("📋 2. Gerando metadados das janelas...")
    try:
        window_manager.generate_metadata()
        print(f"   ✅ Metadados salvos em: data/processed/janelas_metadata.json")
    except Exception as e:
        logger.error(f"Erro ao gerar metadados: {e}")
        print(f"   ⚠️  Aviso: Erro ao salvar metadados")
    
    window_manager.print_summary()

    # Carregue todos os dados para processamento e janelas
    df_raw = pd.read_csv(input_file)
    if "Date" in df_raw.columns:
        date_col = "Date"
    elif "Data" in df_raw.columns:
        date_col = "Data"
    else:
        print("❌ ERRO: Coluna de data não encontrada no arquivo de entrada.")
        logger.error("Coluna de data não encontrada no arquivo de entrada")
        return
    df_raw[date_col] = pd.to_datetime(df_raw[date_col])

    print("\n🔄 3. Consolidação de dados:")
    print("   - Carregando e processando toda a base, calculando retornos e IDs de janela...")

    consolidator = DataConsolidator(
        input_file=str(input_file),
        output_file='data/processed/dados_consolidados.csv'
    )
    
    try:
        df = consolidator.consolidate()
        if df.empty:
            logger.error("Consolidação retornou DataFrame vazio")
            print("="*80)
            print("❌ ERRO: Consolidação retornou dados vazios!")
            print("="*80)
            return

        logger.info("Consolidação concluída com sucesso")
        consolidator.print_summary()
        
        print("="*80)
        print("📊 4. ESTATÍSTICAS POR JANELA")
        print("="*80)
        # Resumo por janela: usa o universo completo de janelas do metadata
        # para manter consistência temporal mesmo quando alguma janela fica
        # sem registros após filtros de completude.
        has_empresa = 'Empresa' in df.columns and df['Empresa'].notna().any()

        stats_data = df.groupby('Janela_ID').agg({
            'Window_Start': 'first',
            'Window_End': 'first',
            'Ticker': 'nunique',
            'Date': 'nunique',
            'Retorno_Log': ['mean', 'std']
        }).reset_index()
        stats_data.columns = [
            'Janela_ID', 'Window_Start', 'Window_End',
            'Num_Tickers', 'Num_Dias', 'Retorno_Medio', 'Volatilidade'
        ]

        if has_empresa:
            num_empresas = (
                df.groupby('Janela_ID')['Empresa']
                .nunique()
                .rename('Num_Empresas')
                .reset_index()
            )
            stats_data = stats_data.merge(num_empresas, on='Janela_ID', how='left')
            stats_data['Num_Empresas'] = stats_data['Num_Empresas'].fillna(0).astype(int)

        windows_universe = pd.DataFrame([
            {
                'Janela_ID': int(w['id']),
                'Window_Start': pd.to_datetime(w['start']),
                'Window_End': pd.to_datetime(w['end']),
            }
            for w in window_manager.windows
        ]).sort_values('Janela_ID')

        stats_by_window = windows_universe.merge(
            stats_data.drop(columns=['Window_Start', 'Window_End'], errors='ignore'),
            on='Janela_ID',
            how='left',
        )

        stats_by_window['Num_Tickers'] = stats_by_window['Num_Tickers'].fillna(0).astype(int)
        stats_by_window['Num_Dias'] = stats_by_window['Num_Dias'].fillna(0).astype(int)
        if 'Num_Empresas' in stats_by_window.columns:
            stats_by_window['Num_Empresas'] = stats_by_window['Num_Empresas'].fillna(0).astype(int)
        stats_by_window['modeled'] = (stats_by_window['Num_Dias'] > 0).astype(int)
        stats_by_window['model_reason'] = np.where(
            stats_by_window['modeled'] == 1,
            '',
            'empty_window_after_filters'
        )

        print("Primeiras 10 janelas (início/fim, #tickers, #dias úteis):")
        print(stats_by_window.head(10).to_string(index=False))
        if len(stats_by_window) > 10:
            print(f"\n... e mais {len(stats_by_window) - 10} janelas")

        # Resumo compacto de dias por janela (sem listar janela a janela)
        ndias_todos = list(stats_by_window['Num_Dias'])
        print(f"\nMínimo de dias por janela: {min(ndias_todos)}")
        print(f"Máximo de dias por janela: {max(ndias_todos)}")
        print(f"Média de dias por janela: {sum(ndias_todos)/len(ndias_todos):.1f}")

        n_modeled = int((stats_by_window['modeled'] == 1).sum())
        n_empty = int((stats_by_window['modeled'] == 0).sum())
        print(f"Janelas com dados após filtros: {n_modeled}/{len(stats_by_window)}")
        if n_empty > 0:
            print(f"Janelas sem dados após filtros: {n_empty} (mantidas no resumo para padronização)")

        max_tickers = max(1, int(stats_by_window['Num_Tickers'].max()))
        janelas_completas = stats_by_window[stats_by_window['Num_Tickers'] >= (0.9 * max_tickers)]
        print()
        print(f"📈 Qualidade:")
        print(f"   Janelas completas (>90% tickers): {len(janelas_completas)}/{len(stats_by_window)}")
        print(f"   Completude média: {(len(janelas_completas)/len(stats_by_window)*100):.1f}%")
        janelas_problematicas = stats_by_window[stats_by_window['Num_Tickers'] < (0.5 * max_tickers)]
        if not janelas_problematicas.empty:
            print()
            print(f"⚠️  ATENÇÃO: {len(janelas_problematicas)} janelas com <50% dos tickers")
            print(f"   IDs: {janelas_problematicas['Janela_ID'].tolist()}")

        # Salva relatório detalhado por janela para inspeção externa
        janelas_resumo_path = Path('data/processed/janelas_resumo.csv')
        janelas_resumo_path.parent.mkdir(parents=True, exist_ok=True)
        stats_by_window.sort_values('Janela_ID').to_csv(janelas_resumo_path, index=False)
        print(f"\n📄 Resumo detalhado de janelas salvo em: {janelas_resumo_path}")

        # ── Tabela de conferência: Ticker vs Empresa por janela ──────────────────
        print()
        print("="*80)
        if 'Num_Empresas' in stats_by_window.columns:
            print("📋 TABELA DE CONFERÊNCIA: TICKERS E EMPRESAS POR JANELA")
            print("   (filtro 80% por empresa: dias agregados sob todos os tickers da empresa)")
        else:
            print("📋 TABELA DE CONFERÊNCIA: TICKERS SELECIONADOS POR JANELA")
            print("   (apenas janelas com dados; filtro 80% completude já aplicado)")
        print("="*80)

        _tbl = stats_by_window[stats_by_window['modeled'] == 1].copy()
        _tbl['Window_Start'] = pd.to_datetime(_tbl['Window_Start']).dt.strftime('%Y-%m-%d')
        _tbl['Window_End']   = pd.to_datetime(_tbl['Window_End']).dt.strftime('%Y-%m-%d')

        if 'Num_Empresas' in _tbl.columns:
            _cols = ['Janela_ID', 'Window_Start', 'Window_End', 'Num_Tickers', 'Num_Empresas', 'Num_Dias']
            _tbl_show = _tbl[_cols].copy()
            # Renomeia para cabeçalho mais legível
            _tbl_show = _tbl_show.rename(columns={
                'Num_Tickers':  'N_Tickers',
                'Num_Empresas': 'N_Empresas',
            })
        else:
            _tbl_show = _tbl[['Janela_ID', 'Window_Start', 'Window_End', 'Num_Tickers', 'Num_Dias']].copy()

        print(_tbl_show.to_string(index=False))

        _nt = _tbl['Num_Tickers']
        _summary = (f"\nTotal janelas com dados: {len(_tbl)}  |  "
                    f"Tickers — Min: {_nt.min()}  Max: {_nt.max()}  Média: {_nt.mean():.1f}")
        if 'Num_Empresas' in _tbl.columns:
            _ne = _tbl['Num_Empresas']
            _summary += (f"\n                            "
                         f"Empresas — Min: {_ne.min()}  Max: {_ne.max()}  Média: {_ne.mean():.1f}")
        print(_summary)
        print("="*80)

        print("="*80)
        print("✅ CONSOLIDAÇÃO CONCLUÍDA COM SUCESSO")
        print("="*80)
        print()
        print("📄 Arquivos gerados:")
        print(f"   - data/processed/dados_consolidados.csv")
        print(f"     • Formato: LONG (uma linha por ticker/data)")
        print(f"     • Tamanho: {Path('data/processed/dados_consolidados.csv').stat().st_size / 1024 / 1024:.2f} MB")
        print(f"     • Registros: {len(df):,}")
        print(f"     • Janelas: {df['Janela_ID'].nunique()}")
        print()
        print(f"   - data/processed/janelas_metadata.json")
        print(f"     • Metadados de {num_windows} janelas")
        print("="*80)
    except Exception as e:
        logger.error(f"Erro durante consolidação: {e}", exc_info=True)
        print("="*80)
        print(f"❌ ERRO: {e}")
        print("="*80)
        print("\nVerifique:")
        print("  - Memória disponível")
        print("  - Espaço em disco")
        print("  - Permissões de escrita")
        print("="*80)
        return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Processamento interrompido pelo usuário!")
        logger.warning("Processamento interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ ERRO FATAL: {str(e)}")
        logger.error(f"Erro fatal: {str(e)}", exc_info=True)
        raise