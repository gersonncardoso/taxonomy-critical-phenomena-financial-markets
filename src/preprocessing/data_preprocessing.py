"""
Módulo de Preprocessamento de Dados
====================================

Funções para processar dados históricos da B3:
- Preenchimento de dias faltantes (business days)
- Análise de dados faltantes por ticker
- Preenchimento de valores nulos
- Cálculo de log-retornos
- Teste de estacionariedade (ADF)

Autor: Gerson Nassor Cardoso - UNIFESP
Data: 2026-02-12
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


def carregar_dados(filepath: str) -> pd.DataFrame:
    """
    Carrega dados do arquivo CSV
    
    Args:
        filepath: Caminho do arquivo
    
    Returns:
        pd.DataFrame com dados carregados
    """
    print(f"📥 Carregando dados: {filepath}")
    
    df = pd.read_csv(filepath, parse_dates=['Data'])
    
    print(f"   ✅ Dados carregados")
    print(f"   Shape: {df.shape}")
    print(f"   Período: {df['Data'].min().date()} até {df['Data'].max().date()}")
    print(f"   Tickers: {df['Ticker'].nunique()}")
    
    return df


def preencher_dias_faltantes(df: pd.DataFrame, coluna_preco: str = 'Preco_Fechamento') -> pd.DataFrame:
    """
    Preenche dias úteis faltantes (business days)
    
    Para cada ticker:
    - Cria range completo de business days no período
    - Preenche com NaN onde não há dados
    
    Args:
        df: DataFrame no formato LONG (Data | Ticker | Preco_Fechamento | ...)
        coluna_preco: Nome da coluna de preço
    
    Returns:
        pd.DataFrame com todos os business days preenchidos
    """
    print("📅 Preenchendo dias faltantes (business days)...")
    
    # Range de business days
    data_min = df['Data'].min()
    data_max = df['Data'].max()
    business_days = pd.bdate_range(start=data_min, end=data_max)
    
    print(f"   Período: {data_min.date()} até {data_max.date()}")
    print(f"   Business days totais: {len(business_days)}")
    print(f"   Dias com dados originais: {df['Data'].nunique()}")
    
    # Processar cada ticker
    tickers = df['Ticker'].unique()
    df_list = []
    
    print(f"   Processando {len(tickers)} tickers...")
    
    for i, ticker in enumerate(tickers, 1):
        if i % 50 == 0:
            print(f"      Progresso: {i}/{len(tickers)}")
        
        # Dados do ticker
        df_ticker = df[df['Ticker'] == ticker].copy()
        
        # DataFrame com todos os business days
        df_completo = pd.DataFrame({'Data': business_days})
        df_completo['Ticker'] = ticker
        
        # Merge com dados existentes
        df_merged = df_completo.merge(
            df_ticker[['Data', 'Ticker', coluna_preco]], 
            on=['Data', 'Ticker'], 
            how='left'
        )
        
        df_list.append(df_merged)
    
    # Combinar todos
    df_final = pd.concat(df_list, ignore_index=True)
    
    print(f"   ✅ Shape após preenchimento: {df_final.shape}")
    print(f"   NaNs adicionados: {df_final[coluna_preco].isna().sum():,}")
    
    return df_final


def analisar_dados_faltantes(df: pd.DataFrame, coluna_preco: str = 'Preco_Fechamento') -> pd.DataFrame:
    """
    Analisa dados faltantes por ticker
    
    Gera relatório com:
    - N° de observações totais
    - N° de observações com dados
    - N° de dados faltantes
    - % de dados faltantes
    - Primeira data com dados
    - Última data com dados
    
    Args:
        df: DataFrame com dias completos
        coluna_preco: Nome da coluna de preço
    
    Returns:
        pd.DataFrame com relatório de dados faltantes
    """
    print("📊 Analisando dados faltantes por ticker...")
    
    relatorio = []
    tickers = df['Ticker'].unique()
    
    for ticker in tickers:
        df_ticker = df[df['Ticker'] == ticker]
        
        n_total = len(df_ticker)
        n_validos = df_ticker[coluna_preco].notna().sum()
        n_faltantes = df_ticker[coluna_preco].isna().sum()
        pct_faltantes = (n_faltantes / n_total) * 100
        
        # Datas
        df_validos = df_ticker[df_ticker[coluna_preco].notna()]
        primeira_data = df_validos['Data'].min() if len(df_validos) > 0 else None
        ultima_data = df_validos['Data'].max() if len(df_validos) > 0 else None
        
        relatorio.append({
            'Ticker': ticker,
            'N_Total': n_total,
            'N_Validos': n_validos,
            'N_Faltantes': n_faltantes,
            'Pct_Faltantes': pct_faltantes,
            'Primeira_Data': primeira_data,
            'Ultima_Data': ultima_data
        })
    
    df_relatorio = pd.DataFrame(relatorio)
    df_relatorio = df_relatorio.sort_values('Pct_Faltantes', ascending=False)
    
    print(f"   ✅ Tickers analisados: {len(df_relatorio)}")
    print(f"   Média de faltantes: {df_relatorio['Pct_Faltantes'].mean():.2f}%")
    
    # Mostrar top 10 com mais faltantes
    print()
    print("   📋 Top 10 tickers com MAIS dados faltantes:")
    print(df_relatorio.head(10)[['Ticker', 'Pct_Faltantes', 'Primeira_Data']].to_string(index=False))
    
    return df_relatorio


def preencher_nulos(df: pd.DataFrame, coluna_preco: str = 'Preco_Fechamento', valor: float = 0.0) -> pd.DataFrame:
    """
    Preenche valores nulos com valor especificado
    
    Args:
        df: DataFrame
        coluna_preco: Coluna de preço
        valor: Valor para preencher NaNs (padrão: 0.0)
    
    Returns:
        pd.DataFrame com NaNs preenchidos
    """
    print(f"🔧 Preenchendo NaN com {valor}...")
    
    df = df.copy()
    nans_antes = df[coluna_preco].isna().sum()
    
    df[coluna_preco] = df[coluna_preco].fillna(valor)
    
    nans_depois = df[coluna_preco].isna().sum()
    
    print(f"   ✅ NaNs antes: {nans_antes:,}")
    print(f"   ✅ NaNs depois: {nans_depois:,}")
    
    return df


def calcular_log_retornos(df: pd.DataFrame, coluna_preco: str = 'Preco_Fechamento') -> pd.DataFrame:
    """
    Calcula log-retornos para cada ticker
    
    Log-retorno = ln(P_t / P_{t-1})
    
    Args:
        df: DataFrame com preços
        coluna_preco: Coluna de preço
    
    Returns:
        pd.DataFrame com coluna 'Log_Retorno' adicionada
    """
    print("📈 Calculando log-retornos...")
    
    df = df.copy()
    df = df.sort_values(['Ticker', 'Data'])
    
    # Calcular log-retornos por ticker
    df['Log_Retorno'] = df.groupby('Ticker')[coluna_preco].transform(
        lambda x: np.log(x / x.shift(1))
    )
    
    # Substituir infinitos por NaN (divisão por zero)
    df['Log_Retorno'] = df['Log_Retorno'].replace([np.inf, -np.inf], np.nan)
    
    nans = df['Log_Retorno'].isna().sum()
    
    print(f"   ✅ Log-retornos calculados")
    print(f"   NaNs: {nans:,} (primeira obs de cada ticker + divisões por zero)")
    
    return df


def teste_adf_serie(serie: pd.Series, nome: str = '') -> Dict:
    """
    Teste de Dickey-Fuller Aumentado (ADF) para uma série
    
    Testa se série temporal é estacionária
    
    H0: Série tem raiz unitária (NÃO estacionária)
    H1: Série é estacionária
    
    Decisão: Se p-value < 0.05 → Rejeita H0 → Série é ESTACIONÁRIA
    
    Args:
        serie: Série temporal (pd.Series)
        nome: Nome da série (para identificação)
    
    Returns:
        Dict com resultados do teste
    """
    # Remover NaNs
    serie = serie.dropna()
    
    # Verificar tamanho mínimo
    if len(serie) < 10:
        return {
            'ticker': nome,
            'n_obs': len(serie),
            'adf_stat': np.nan,
            'p_value': np.nan,
            'estacionaria': 'INSUFICIENTE',
            'conclusao': 'Dados insuficientes para teste (< 10 obs)'
        }
    
    try:
        # Executar teste ADF
        resultado = adfuller(serie, autolag='AIC')
        
        adf_stat = resultado[0]
        p_value = resultado[1]
        
        # Decisão: p-value < 0.05 → estacionária
        estacionaria = 'SIM' if p_value < 0.05 else 'NÃO'
        
        return {
            'ticker': nome,
            'n_obs': len(serie),
            'adf_stat': adf_stat,
            'p_value': p_value,
            'estacionaria': estacionaria,
            'conclusao': f"p={p_value:.4f} → {'Estacionária' if p_value < 0.05 else 'Não estacionária'}"
        }
    
    except Exception as e:
        return {
            'ticker': nome,
            'n_obs': len(serie),
            'adf_stat': np.nan,
            'p_value': np.nan,
            'estacionaria': 'ERRO',
            'conclusao': f'Erro ao executar teste: {str(e)}'
        }


def testar_estacionariedade_todos(df: pd.DataFrame, coluna: str = 'Log_Retorno') -> pd.DataFrame:
    """
    Testa estacionariedade (ADF) para todos os tickers
    
    Args:
        df: DataFrame com log-retornos
        coluna: Nome da coluna a testar (padrão: 'Log_Retorno')
    
    Returns:
        pd.DataFrame com resultados do teste ADF para cada ticker
    """
    print("🧪 Testando estacionariedade (ADF) para todos os tickers...")
    print("   H0: Série tem raiz unitária (NÃO estacionária)")
    print("   H1: Série é estacionária")
    print("   Decisão: p-value < 0.05 → Série é ESTACIONÁRIA")
    print()
    
    tickers = df['Ticker'].unique()
    resultados = []
    
    for i, ticker in enumerate(tickers, 1):
        if i % 50 == 0:
            print(f"      Progresso: {i}/{len(tickers)}")
        
        serie = df[df['Ticker'] == ticker][coluna]
        resultado = teste_adf_serie(serie, nome=ticker)
        resultados.append(resultado)
    
    df_adf = pd.DataFrame(resultados)
    
    # Renomear colunas para padronizar
    df_adf.columns = ['Ticker', 'N_Obs', 'ADF_Stat', 'P_Value', 'Estacionaria', 'Conclusao']
    
    # Estatísticas
    n_estacionarias = (df_adf['Estacionaria'] == 'SIM').sum()
    n_nao_estacionarias = (df_adf['Estacionaria'] == 'NÃO').sum()
    n_insuficientes = (df_adf['Estacionaria'] == 'INSUFICIENTE').sum()
    
    print()
    print(f"   ✅ Teste ADF concluído")
    print(f"   Total: {len(df_adf)}")
    print(f"   ✅ Estacionárias: {n_estacionarias} ({n_estacionarias/len(df_adf)*100:.1f}%)")
    print(f"   ❌ Não estacionárias: {n_nao_estacionarias} ({n_nao_estacionarias/len(df_adf)*100:.1f}%)")
    print(f"   ⚠️  Dados insuficientes: {n_insuficientes}")
    
    return df_adf


def pipeline_completo(filepath: str, 
                      coluna_preco: str = 'Preco_Fechamento',
                      valor_preenchimento: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executa pipeline completo de preprocessamento
    
    Etapas:
    1. Carregar dados brutos
    2. Preencher dias faltantes (business days)
    3. Analisar dados faltantes por ticker
    4. Preencher NaN com valor especificado
    5. Calcular log-retornos
    6. Testar estacionariedade (ADF)
    
    Args:
        filepath: Caminho do arquivo de dados brutos
        coluna_preco: Coluna de preço (padrão: 'Preco_Fechamento')
        valor_preenchimento: Valor para preencher NaNs (padrão: 0.0)
    
    Returns:
        Tuple com 3 DataFrames:
        - df_processado: Dados processados com log-retornos
        - df_missing: Relatório de dados faltantes
        - df_adf: Resultados do teste ADF
    """
    print("="*80)
    print("PIPELINE COMPLETO DE PREPROCESSAMENTO")
    print("="*80)
    print()
    
    # 1. Carregar
    df = carregar_dados(filepath)
    print()
    
    # 2. Preencher dias faltantes
    print("="*80)
    df = preencher_dias_faltantes(df, coluna_preco)
    print()
    
    # 3. Analisar faltantes
    print("="*80)
    df_missing = analisar_dados_faltantes(df, coluna_preco)
    print()
    
    # 4. Preencher nulos
    print("="*80)
    df = preencher_nulos(df, coluna_preco, valor=valor_preenchimento)
    print()
    
    # 5. Calcular log-retornos
    print("="*80)
    df = calcular_log_retornos(df, coluna_preco)
    print()
    
    # 6. Testar estacionariedade
    print("="*80)
    df_adf = testar_estacionariedade_todos(df, coluna='Log_Retorno')
    print()
    
    print("="*80)
    print("✅ PIPELINE CONCLUÍDO")
    print("="*80)
    
    return df, df_missing, df_adf