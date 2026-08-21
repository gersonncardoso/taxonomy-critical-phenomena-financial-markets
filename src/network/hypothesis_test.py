"""
Testes de Hipótese para Redes
==============================

Testa se grafo real é significativamente diferente de aleatório

H₀: Grafo real ~ Grafo aleatório
H₁: Grafo real ≠ Grafo aleatório

Métodos:
1. Teste de múltiplas métricas (Bonferroni correction)
2. Teste de Kolmogorov-Smirnov (distribuição de grau)
3. Teste de permutação (modularidade)

Autor: Gerson Nassor Cardoso
Instituição: Universidade Federal de São Paulo (UNIFESP)
Data: 2026-02-13

Copyright (c) 2026 Gerson Nassor Cardoso - UNIFESP
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple
from scipy import stats
import warnings

from .null_models import (
    gerar_grafo_erdos_renyi,
    randomizar_grafo_preservando_grau
)
from .metrics import (
    calcular_metricas_globais,
    calcular_metricas_centralidade
)
from .communities import detectar_comunidades_louvain, analisar_modularidade


def teste_metricas_multiplas(G_real: nx.Graph,
                             tipo_modelo: str = 'erdos_renyi',
                             n_amostras: int = 1000,
                             alpha: float = 0.05,
                             seed: int = None) -> Dict:
    """
    Testa múltiplas métricas contra modelo nulo
    
    Usa correção de Bonferroni para testes múltiplos
    
    Args:
        G_real: Grafo real
        tipo_modelo: 'erdos_renyi' ou 'randomized'
        n_amostras: Número de grafos aleatórios
        alpha: Nível de significância
        seed: Seed para reprodutibilidade
    
    Returns:
        Dict com resultados do teste
    """
    print(f"\n🔬 TESTE DE HIPÓTESE: {n_amostras} amostras aleatórias")
    print(f"   H₀: Grafo real ~ Grafo {tipo_modelo}")
    print(f"   H₁: Grafo real ≠ Grafo {tipo_modelo}")
    print(f"   Significância: α = {alpha}")
    
    n_nodes = G_real.number_of_nodes()
    n_edges = G_real.number_of_edges()
    
    # Métricas do grafo real
    metricas_real = calcular_metricas_globais(G_real)
    
    print(f"\n   📊 Gerando {n_amostras} grafos aleatórios...")
    
    # Gerar distribuições nulas
    metricas_aleatorias = []
    
    for i in range(n_amostras):
        if seed is not None:
            current_seed = seed + i
        else:
            current_seed = None
        
        # Gerar grafo aleatório
        if tipo_modelo == 'erdos_renyi':
            G_random = gerar_grafo_erdos_renyi(n_nodes, n_edges, seed=current_seed)
        elif tipo_modelo == 'randomized':
            G_random = randomizar_grafo_preservando_grau(G_real, seed=current_seed)
        else:
            raise ValueError(f"Modelo desconhecido: {tipo_modelo}")
        
        # Calcular métricas
        metricas = calcular_metricas_globais(G_random)
        metricas_aleatorias.append(metricas)
    
    df_aleatorias = pd.DataFrame(metricas_aleatorias)
    
    print(f"   ✅ Grafos gerados!")
    
    # Teste para cada métrica
    resultados_metricas = []
    
    for metrica in df_aleatorias.columns:
        valor_real = metricas_real.get(metrica, np.nan)
        
        if np.isnan(valor_real):
            continue
        
        valores_aleatorios = df_aleatorias[metrica].dropna()
        
        if len(valores_aleatorios) < 10:
            continue
        
        # Estatísticas da distribuição nula
        media_nula = valores_aleatorios.mean()
        std_nula = valores_aleatorios.std()
        
        # Z-score
        if std_nula > 0:
            z_score = (valor_real - media_nula) / std_nula
        else:
            z_score = np.nan
        
        # P-value (two-tailed)
        if not np.isnan(z_score):
            p_value = 2 * stats.norm.sf(abs(z_score))
        else:
            p_value = np.nan
        
        # Percentil
        percentil = stats.percentileofscore(valores_aleatorios, valor_real)
        
        resultados_metricas.append({
            'metrica': metrica,
            'valor_real': valor_real,
            'media_nula': media_nula,
            'std_nula': std_nula,
            'z_score': z_score,
            'p_value': p_value,
            'percentil': percentil
        })
    
    df_resultados = pd.DataFrame(resultados_metricas)
    
    # Correção de Bonferroni para testes múltiplos
    n_testes = len(df_resultados)
    alpha_corrigido = alpha / n_testes if n_testes > 0 else alpha
    
    df_resultados['significativo'] = df_resultados['p_value'] < alpha_corrigido
    
    # Resumo
    n_significativas = df_resultados['significativo'].sum()
    
    print(f"\n   📈 RESULTADOS:")
    print(f"   Métricas testadas: {n_testes}")
    print(f"   Significativas (p < {alpha_corrigido:.4f}): {n_significativas}")
    
    # Decisão do teste
    if n_significativas >= 3:
        decisao = "REJEITAR H₀"
        conclusao = "Grafo real é SIGNIFICATIVAMENTE DIFERENTE de aleatório"
        eh_aleatorio = False
    elif n_significativas >= 1:
        decisao = "EVIDÊNCIA FRACA"
        conclusao = "Algumas diferenças, mas evidência limitada"
        eh_aleatorio = None
    else:
        decisao = "NÃO REJEITAR H₀"
        conclusao = "Grafo real é INDISTINGUÍVEL de aleatório"
        eh_aleatorio = True
    
    print(f"\n   🎯 DECISÃO: {decisao}")
    print(f"   {conclusao}")
    
    return {
        'decisao': decisao,
        'conclusao': conclusao,
        'eh_aleatorio': eh_aleatorio,
        'n_metricas_testadas': n_testes,
        'n_metricas_significativas': n_significativas,
        'alpha': alpha,
        'alpha_corrigido': alpha_corrigido,
        'tipo_modelo': tipo_modelo,
        'n_amostras': n_amostras,
        'resultados_detalhados': df_resultados
    }


def teste_kolmogorov_smirnov_grau(G_real: nx.Graph,
                                  tipo_modelo: str = 'erdos_renyi',
                                  n_amostras: int = 1000,
                                  seed: int = None) -> Dict:
    """
    Teste K-S para distribuição de grau
    
    Compara se distribuições de grau são diferentes
    
    Args:
        G_real: Grafo real
        tipo_modelo: Tipo de modelo nulo
        n_amostras: Número de grafos aleatórios
        seed: Seed
    
    Returns:
        Dict com resultados
    """
    print(f"\n🔬 TESTE KOLMOGOROV-SMIRNOV (Distribuição de Grau)")
    
    # Distribuição de grau do grafo real
    degrees_real = [d for n, d in G_real.degree()]
    
    # Gerar distribuições aleatórias
    p_values = []
    statistics = []
    
    n_nodes = G_real.number_of_nodes()
    n_edges = G_real.number_of_edges()
    
    for i in range(n_amostras):
        current_seed = seed + i if seed is not None else None
        
        if tipo_modelo == 'erdos_renyi':
            G_random = gerar_grafo_erdos_renyi(n_nodes, n_edges, seed=current_seed)
        elif tipo_modelo == 'randomized':
            G_random = randomizar_grafo_preservando_grau(G_real, seed=current_seed)
        else:
            raise ValueError(f"Modelo desconhecido: {tipo_modelo}")
        
        degrees_random = [d for n, d in G_random.degree()]
        
        # Teste K-S
        statistic, p_value = stats.ks_2samp(degrees_real, degrees_random)
        
        statistics.append(statistic)
        p_values.append(p_value)
    
    # Média dos p-values
    p_value_medio = np.mean(p_values)
    statistic_medio = np.mean(statistics)
    
    # Proporção de testes significativos
    prop_significativos = np.mean([p < 0.05 for p in p_values])
    
    print(f"   KS statistic (média): {statistic_medio:.4f}")
    print(f"   P-value (médio): {p_value_medio:.4f}")
    print(f"   Proporção significativos: {prop_significativos:.2%}")
    
    if prop_significativos > 0.95:
        decisao = "REJEITAR H₀"
        conclusao = "Distribuição de grau é DIFERENTE"
    elif prop_significativos > 0.5:
        decisao = "EVIDÊNCIA MODERADA"
        conclusao = "Algumas diferenças na distribuição"
    else:
        decisao = "NÃO REJEITAR H₀"
        conclusao = "Distribuição de grau similar"
    
    print(f"   🎯 {decisao}: {conclusao}")
    
    return {
        'teste': 'Kolmogorov-Smirnov',
        'decisao': decisao,
        'conclusao': conclusao,
        'statistic_medio': statistic_medio,
        'p_value_medio': p_value_medio,
        'prop_significativos': prop_significativos
    }


def teste_modularidade_permutacao(G_real: nx.Graph,
                                 n_permutacoes: int = 1000,
                                 seed: int = None) -> Dict:
    """
    Teste de permutação para modularidade
    
    Testa se modularidade (comunidades) é maior que o esperado ao acaso
    
    Args:
        G_real: Grafo real
        n_permutacoes: Número de permutações
        seed: Seed
    
    Returns:
        Dict com resultados
    """
    print(f"\n🔬 TESTE DE PERMUTAÇÃO (Modularidade)")
    
    # Detectar comunidades no grafo real
    communities_real = detectar_comunidades_louvain(G_real)
    modularity_real = analisar_modularidade(G_real, communities_real)
    
    print(f"   Modularidade real: {modularity_real:.4f}")
    
    # Gerar distribuição nula via permutação
    modularities_nulas = []
    
    for i in range(n_permutacoes):
        current_seed = seed + i if seed is not None else None
        
        # Randomizar preservando grau
        G_random = randomizar_grafo_preservando_grau(G_real, seed=current_seed)
        
        # Detectar comunidades
        communities_random = detectar_comunidades_louvain(G_random)
        modularity_random = analisar_modularidade(G_random, communities_random)
        
        modularities_nulas.append(modularity_random)
    
    # Estatísticas
    media_nula = np.mean(modularities_nulas)
    std_nula = np.std(modularities_nulas)
    
    # P-value empírico
    p_value = np.mean([m >= modularity_real for m in modularities_nulas])
    
    # Z-score
    if std_nula > 0:
        z_score = (modularity_real - media_nula) / std_nula
    else:
        z_score = np.nan
    
    print(f"   Modularidade nula (média): {media_nula:.4f} ± {std_nula:.4f}")
    print(f"   Z-score: {z_score:.2f}")
    print(f"   P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        decisao = "REJEITAR H₀"
        conclusao = "Modularidade é SIGNIFICATIVAMENTE MAIOR que aleatório"
    else:
        decisao = "NÃO REJEITAR H₀"
        conclusao = "Modularidade similar ao aleatório"
    
    print(f"   🎯 {decisao}: {conclusao}")
    
    return {
        'teste': 'Permutação (Modularidade)',
        'decisao': decisao,
        'conclusao': conclusao,
        'modularity_real': modularity_real,
        'modularity_nula_mean': media_nula,
        'modularity_nula_std': std_nula,
        'z_score': z_score,
        'p_value': p_value
    }


def testar_hipotese_completo(G_real: nx.Graph,
                            tipo_modelo: str = 'erdos_renyi',
                            n_amostras: int = 1000,
                            alpha: float = 0.05,
                            seed: int = 42) -> Dict:
    """
    Bateria completa de testes de hipótese
    
    Args:
        G_real: Grafo real
        tipo_modelo: Tipo de modelo nulo
        n_amostras: Número de amostras
        alpha: Nível de significância
        seed: Seed
    
    Returns:
        Dict com todos os resultados
    """
    print("="*80)
    print("BATERIA COMPLETA DE TESTES DE HIPÓTESE")
    print("="*80)
    print(f"Grafo: {G_real.number_of_nodes()} nós, {G_real.number_of_edges()} arestas")
    print(f"Modelo nulo: {tipo_modelo}")
    print(f"Amostras: {n_amostras}")
    print("="*80)
    
    resultados = {}
    
    # Teste 1: Múltiplas métricas
    print("\n[1/3] TESTE DE MÚLTIPLAS MÉTRICAS")
    resultados['metricas_multiplas'] = teste_metricas_multiplas(
        G_real, tipo_modelo, n_amostras, alpha, seed
    )
    
    # Teste 2: Kolmogorov-Smirnov (distribuição de grau)
    print("\n[2/3] TESTE KOLMOGOROV-SMIRNOV")
    resultados['ks_grau'] = teste_kolmogorov_smirnov_grau(
        G_real, tipo_modelo, n_amostras, seed
    )
    
    # Teste 3: Permutação (modularidade)
    print("\n[3/3] TESTE DE PERMUTAÇÃO")
    resultados['permutacao_modularidade'] = teste_modularidade_permutacao(
        G_real, n_amostras, seed
    )
    
    # DECISÃO FINAL
    print("\n" + "="*80)
    print("DECISÃO FINAL")
    print("="*80)
    
    # Contar evidências
    evidencias_rejeitar = 0
    
    if resultados['metricas_multiplas']['n_metricas_significativas'] >= 3:
        evidencias_rejeitar += 1
        print("✅ Teste 1: Múltiplas métricas REJEITAM H₀")
    else:
        print("❌ Teste 1: Insuficiente para rejeitar H₀")
    
    if resultados['ks_grau']['prop_significativos'] > 0.95:
        evidencias_rejeitar += 1
        print("✅ Teste 2: Distribuição de grau REJEITA H₀")
    else:
        print("❌ Teste 2: Distribuição similar")
    
    if resultados['permutacao_modularidade']['p_value'] < 0.05:
        evidencias_rejeitar += 1
        print("✅ Teste 3: Modularidade REJEITA H₀")
    else:
        print("❌ Teste 3: Modularidade similar")
    
    print("\n" + "-"*80)
    print(f"EVIDÊNCIAS PARA REJEITAR H₀: {evidencias_rejeitar}/3")
    print("-"*80)
    
    if evidencias_rejeitar >= 2:
        decisao_final = "REJEITAR H₀"
        conclusao_final = "🎉 SEU GRAFO É SIGNIFICATIVAMENTE DIFERENTE DE ALEATÓRIO"
        eh_aleatorio = False
    elif evidencias_rejeitar == 1:
        decisao_final = "INCONCLUSIVO"
        conclusao_final = "⚠️ Evidência mista - algumas diferenças, mas não robustas"
        eh_aleatorio = None
    else:
        decisao_final = "NÃO REJEITAR H₀"
        conclusao_final = "❌ SEU GRAFO NÃO É DISTINGUÍVEL DE ALEATÓRIO"
        eh_aleatorio = True
    
    print(f"\n🎯 DECISÃO: {decisao_final}")
    print(f"{conclusao_final}")
    print("="*80)
    
    resultados['decisao_final'] = {
        'decisao': decisao_final,
        'conclusao': conclusao_final,
        'eh_aleatorio': eh_aleatorio,
        'evidencias': evidencias_rejeitar,
        'total_testes': 3
    }
    
    return resultados