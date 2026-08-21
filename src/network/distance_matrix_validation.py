"""
Validação de Matriz de Distância
=================================

Testa se matriz de distância possui estrutura não-aleatória.

Se a matriz é significativamente diferente de aleatória,
TODOS os grafos construídos a partir dela são válidos.

Testes implementados:
1. Múltiplas Estatísticas (média, std, assimetria, curtose, entropia)
2. Kolmogorov-Smirnov (distribuição de distâncias) - USA KS MAX

Autor: Gerson Nassor Cardoso
Instituição: Universidade Federal de São Paulo (UNIFESP)
Data: 2026-02-13

Copyright (c) 2026 Gerson Nassor Cardoso - UNIFESP
"""

import numpy as np
import pandas as pd
from typing import Dict
from scipy import stats

from .null_models import gerar_matriz_distancia_aleatoria


def extrair_triangulo_superior(matriz: pd.DataFrame) -> np.ndarray:
    """
    Extrai triângulo superior da matriz (sem diagonal)
    
    Args:
        matriz: Matriz quadrada
    
    Returns:
        Array 1D com valores do triângulo superior
    """
    n = len(matriz)
    indices = np.triu_indices(n, k=1)
    return matriz.values[indices]


def calcular_estatisticas_matriz(matriz: pd.DataFrame) -> Dict:
    """
    Calcula estatísticas descritivas da matriz de distância
    
    Args:
        matriz: Matriz de distância
    
    Returns:
        Dict com estatísticas
    """
    valores = extrair_triangulo_superior(matriz)
    
    estatisticas = {
        'media': np.mean(valores),
        'mediana': np.median(valores),
        'std': np.std(valores),
        'min': np.min(valores),
        'max': np.max(valores),
        'q25': np.percentile(valores, 25),
        'q75': np.percentile(valores, 75),
        'iqr': np.percentile(valores, 75) - np.percentile(valores, 25)
    }
    
    # Assimetria e curtose
    if len(valores) > 2:
        estatisticas['skewness'] = stats.skew(valores)
        estatisticas['kurtosis'] = stats.kurtosis(valores)
    
    # Entropia (discretizar em bins)
    hist, _ = np.histogram(valores, bins=20)
    probs = hist / hist.sum()
    probs = probs[probs > 0]  # Remover bins vazios
    estatisticas['entropia'] = -np.sum(probs * np.log2(probs))
    
    # Coeficiente de variação
    if estatisticas['media'] > 0:
        estatisticas['cv'] = estatisticas['std'] / estatisticas['media']
    else:
        estatisticas['cv'] = np.nan
    
    return estatisticas


def teste_multiplas_estatisticas(dist_matrix_real: pd.DataFrame,
                                 n_amostras: int = 1000,
                                 alpha: float = 0.05,
                                 seed: int = None) -> Dict:
    """
    Testa múltiplas estatísticas da matriz de distância
    
    H₀: Matriz real ~ Matriz aleatória
    H₁: Matriz real ≠ Matriz aleatória
    
    Args:
        dist_matrix_real: Matriz de distância real
        n_amostras: Número de matrizes aleatórias
        alpha: Nível de significância
        seed: Seed para reprodutibilidade
    
    Returns:
        Dict com resultados
    """
    print(f"\n🔬 TESTE 1: MÚLTIPLAS ESTATÍSTICAS DA MATRIZ")
    print(f"   Gerando {n_amostras} matrizes aleatórias...")
    
    n = len(dist_matrix_real)
    
    # Estatísticas da matriz real
    stats_real = calcular_estatisticas_matriz(dist_matrix_real)
    
    # Gerar matrizes aleatórias e calcular estatísticas
    stats_aleatorias = {k: [] for k in stats_real.keys()}
    
    for i in range(n_amostras):
        current_seed = seed + i if seed is not None else None
        dist_random = gerar_matriz_distancia_aleatoria(n, seed=current_seed)
        stats_random = calcular_estatisticas_matriz(dist_random)
        
        for k, v in stats_random.items():
            stats_aleatorias[k].append(v)
    
    print(f"   ✅ Matrizes geradas!")
    
    # Calcular z-scores e p-values
    resultados = []
    n_significativas = 0
    
    metricas_testaveis = ['media', 'std', 'skewness', 'kurtosis', 'entropia', 'cv', 'iqr']
    
    for metrica in metricas_testaveis:
        if metrica not in stats_real or metrica not in stats_aleatorias:
            continue
        
        valor_real = stats_real[metrica]
        valores_aleatorios = [v for v in stats_aleatorias[metrica] if not np.isnan(v)]
        
        if len(valores_aleatorios) < 10 or np.isnan(valor_real):
            continue
        
        media_nula = np.mean(valores_aleatorios)
        std_nula = np.std(valores_aleatorios)
        
        if std_nula > 0:
            z_score = (valor_real - media_nula) / std_nula
            p_value = 2 * stats.norm.sf(abs(z_score))
        else:
            z_score = 0
            p_value = 1.0
        
        percentil = stats.percentileofscore(valores_aleatorios, valor_real)
        
        resultados.append({
            'metrica': metrica,
            'real': valor_real,
            'media_nula': media_nula,
            'std_nula': std_nula,
            'z_score': z_score,
            'p_value': p_value,
            'percentil': percentil
        })
    
    df_resultados = pd.DataFrame(resultados)
    
    # Correção de Bonferroni
    n_testes = len(df_resultados)
    alpha_corrigido = alpha / n_testes if n_testes > 0 else alpha
    
    df_resultados['significativo'] = df_resultados['p_value'] < alpha_corrigido
    n_significativas = df_resultados['significativo'].sum()
    
    print(f"\n   📈 RESULTADOS:")
    print(f"   Estatísticas testadas: {n_testes}")
    print(f"   Significativas (α={alpha_corrigido:.4f}): {n_significativas}")
    
    # Decisão
    if n_significativas >= 3:
        decisao = "REJEITAR H₀"
        conclusao = "Matriz é SIGNIFICATIVAMENTE DIFERENTE de aleatória"
    elif n_significativas >= 1:
        decisao = "EVIDÊNCIA FRACA"
        conclusao = "Algumas diferenças, mas não robustas"
    else:
        decisao = "NÃO REJEITAR H₀"
        conclusao = "Matriz é INDISTINGUÍVEL de aleatória"
    
    print(f"   🎯 {decisao}")
    print(f"   {conclusao}")
    
    return {
        'teste': 'Múltiplas Estatísticas',
        'decisao': decisao,
        'conclusao': conclusao,
        'n_estatisticas_testadas': n_testes,
        'n_significativas': n_significativas,
        'alpha': alpha,
        'alpha_corrigido': alpha_corrigido,
        'resultados_detalhados': df_resultados
    }


def teste_kolmogorov_smirnov(dist_matrix_real: pd.DataFrame,
                             n_amostras: int = 1000,
                             seed: int = None) -> Dict:
    """
    Teste K-S para distribuição de distâncias
    
    Compara distribuição de distâncias da matriz real com
    distribuições de matrizes aleatórias.
    
    USA MÁXIMO KS statistic (mais conservador) para decisão.
    
    Args:
        dist_matrix_real: Matriz de distância real
        n_amostras: Número de matrizes aleatórias
        seed: Seed
    
    Returns:
        Dict com resultados
    """
    print(f"\n🔬 TESTE 2: KOLMOGOROV-SMIRNOV (Distribuição)")
    print(f"   Gerando {n_amostras} matrizes aleatórias...")
    
    n = len(dist_matrix_real)
    
    # Distribuição de distâncias da matriz real
    distancias_real = extrair_triangulo_superior(dist_matrix_real)
    
    # Gerar matrizes aleatórias e testar
    ks_statistics = []
    p_values = []
    
    for i in range(n_amostras):
        current_seed = seed + i if seed is not None else None
        dist_random = gerar_matriz_distancia_aleatoria(n, seed=current_seed)
        distancias_random = extrair_triangulo_superior(dist_random)
        
        # Teste K-S
        statistic, p_value = stats.ks_2samp(distancias_real, distancias_random)
        
        ks_statistics.append(statistic)
        p_values.append(p_value)
    
    # Estatísticas agregadas
    ks_medio = np.mean(ks_statistics)
    ks_max = np.max(ks_statistics)      # MÁXIMO (usado para decisão)
    ks_min = np.min(ks_statistics)
    ks_std = np.std(ks_statistics)
    
    p_medio = np.mean(p_values)
    p_min = np.min(p_values)
    prop_significativos = np.mean([p < 0.05 for p in p_values])
    
    print(f"   ✅ Testes realizados!")
    print(f"\n   📈 RESULTADOS:")
    print(f"   KS statistic:")
    print(f"      Média: {ks_medio:.4f}")
    print(f"      Max:   {ks_max:.4f} ← (usado para decisão)")
    print(f"      Min:   {ks_min:.4f}")
    print(f"      Std:   {ks_std:.4f}")
    print(f"   P-value:")
    print(f"      Médio: {p_medio:.4f}")
    print(f"      Min:   {p_min:.4f}")
    print(f"   Proporção significativos: {prop_significativos:.2%}")
    
    # Decisão baseada no MÁXIMO KS statistic
    if ks_max > 0.3 and prop_significativos > 0.95:
        decisao = "REJEITAR H₀"
        conclusao = "Distribuição é SIGNIFICATIVAMENTE DIFERENTE (KS max > 0.3)"
    elif ks_max > 0.2 and prop_significativos > 0.90:
        decisao = "REJEITAR H₀"
        conclusao = "Distribuição é SIGNIFICATIVAMENTE DIFERENTE (KS max > 0.2)"
    elif ks_max > 0.15 and prop_significativos > 0.80:
        decisao = "EVIDÊNCIA MODERADA"
        conclusao = "Algumas diferenças na distribuição (KS max > 0.15)"
    else:
        decisao = "NÃO REJEITAR H₀"
        conclusao = "Distribuição similar ao aleatório"
    
    print(f"   🎯 {decisao}")
    print(f"   {conclusao}")
    
    return {
        'teste': 'Kolmogorov-Smirnov',
        'decisao': decisao,
        'conclusao': conclusao,
        'ks_statistic_medio': ks_medio,
        'ks_statistic_max': ks_max,
        'ks_statistic_min': ks_min,
        'ks_statistic_std': ks_std,
        'p_value_medio': p_medio,
        'p_value_min': p_min,
        'prop_significativos': prop_significativos
    }


def validar_matriz_distancia(dist_matrix: pd.DataFrame,
                             n_amostras: int = 1000,
                             alpha: float = 0.05,
                             seed: int = 42) -> Dict:
    """
    Validação completa da matriz de distância
    
    Args:
        dist_matrix: Matriz de distância a validar
        n_amostras: Número de matrizes aleatórias
        alpha: Nível de significância
        seed: Seed
    
    Returns:
        Dict com todos os resultados
    """
    print("="*80)
    print("VALIDAÇÃO DA MATRIZ DE DISTÂNCIA")
    print("="*80)
    print(f"Dimensão: {len(dist_matrix)}×{len(dist_matrix)}")
    print(f"Amostras aleatórias: {n_amostras}")
    print(f"Nível de significância: α = {alpha}")
    print("="*80)
    
    resultados = {}
    
    # Teste 1: Múltiplas estatísticas
    resultados['teste1'] = teste_multiplas_estatisticas(
        dist_matrix, n_amostras, alpha, seed
    )
    
    # Teste 2: Kolmogorov-Smirnov
    resultados['teste2'] = teste_kolmogorov_smirnov(
        dist_matrix, n_amostras, seed
    )
    
    # Decisão final
    print(f"\n{'='*80}")
    print("DECISÃO FINAL")
    print(f"{'='*80}")
    
    evidencias = 0
    
    if resultados['teste1']['n_significativas'] >= 3:
        evidencias += 1
        print("✅ Teste 1: Múltiplas estatísticas REJEITAM H₀")
    else:
        print("❌ Teste 1: Insuficiente para rejeitar H₀")
    
    if resultados['teste2']['prop_significativos'] > 0.95:
        evidencias += 1
        print("✅ Teste 2: Distribuição REJEITA H₀")
    else:
        print("❌ Teste 2: Distribuição similar")
    
    print(f"\n{'-'*80}")
    print(f"EVIDÊNCIAS PARA REJEITAR H₀: {evidencias}/2")
    print(f"{'-'*80}")
    
    if evidencias == 2:
        decisao_final = "REJEITAR H₀"
        conclusao_final = "✅ MATRIZ É SIGNIFICATIVAMENTE DIFERENTE DE ALEATÓRIA"
        implicacao = "TODOS OS GRAFOS (MST, Planar, etc.) SÃO VÁLIDOS"
        eh_aleatorio = False
    elif evidencias == 1:
        decisao_final = "EVIDÊNCIA FRACA"
        conclusao_final = "⚠️ Algumas diferenças, mas não conclusivo"
        implicacao = "Validação de grafos individuais recomendada"
        eh_aleatorio = None
    else:
        decisao_final = "NÃO REJEITAR H₀"
        conclusao_final = "❌ MATRIZ É INDISTINGUÍVEL DE ALEATÓRIA"
        implicacao = "GRAFOS PODEM SER ARTEFATOS ESTATÍSTICOS"
        eh_aleatorio = True
    
    print(f"\n🎯 DECISÃO: {decisao_final}")
    print(f"{conclusao_final}")
    print(f"\n💡 IMPLICAÇÃO: {implicacao}")
    print("="*80)
    
    resultados['decisao_final'] = {
        'decisao': decisao_final,
        'conclusao': conclusao_final,
        'implicacao': implicacao,
        'eh_aleatorio': eh_aleatorio,
        'evidencias': evidencias,
        'total_testes': 2
    }
    
    return resultados