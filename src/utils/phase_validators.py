"""
Validadores de Fases do Pipeline
=================================

Verifica se fases do pipeline foram concluídas.

Autor: Gerson Nassor Cardoso - UNIFESP
Data: 2026-02-18
"""

from pathlib import Path
from typing import Tuple


class PhaseValidator:
    """Validador de fases do pipeline"""
    
    @staticmethod
    def verificar_download() -> bool:
        """Verifica se download foi concluído"""
        return (Path('data/raw/b3_dados_completos.csv').exists() or
                Path('data/raw/precos_historicos.csv').exists())
    
    @staticmethod
    def verificar_filter() -> bool:
        """Verifica se filtro foi concluído"""
        return Path('data/raw/b3_dados_filtrados.csv').exists()
    
    @staticmethod
    def verificar_qualidade() -> bool:
        """Versão rápida: considera concluído se o arquivo filtrado existe"""
        return Path("data/raw/b3_dados_filtrados.csv").exists()
    
    @staticmethod
    def verificar_preprocessing() -> bool:
        """Verifica se consolidação foi concluída"""
        consolidado = Path('data/processed/dados_consolidados.csv')
        metadata = Path('data/processed/janelas_metadata.json')
        return consolidado.exists() and metadata.exists()
    
    @staticmethod
    def verificar_rolling_correlation() -> bool:
        """Verifica se correlações foram calculadas"""
        return Path('data/processed/correlacoes_summary.csv').exists()
    
    @staticmethod
    def verificar_heatmaps(min_heatmaps: int = 5) -> bool:
        """Verifica se heatmaps foram gerados"""
        figures_dir = Path('figures/correlation')
        if not figures_dir.exists():
            return False
        heatmaps = list(figures_dir.glob('heatmap_janela_*.png'))
        return len(heatmaps) >= min_heatmaps
    
    @staticmethod
    def verificar_clustering(min_dendrogramas: int = 5) -> bool:
        """Verifica se dendrogramas foram gerados"""
        figures_dir = Path('figures/clustering')
        if not figures_dir.exists():
            return False
        dendrogramas = list(figures_dir.glob('dendrogram_janela_*.png'))
        return len(dendrogramas) >= min_dendrogramas
    
    @staticmethod
    def verificar_validation(num_janelas_esperado: int) -> bool:
        """Verifica se validação foi concluída"""
        validation_dir = Path('data/validation')
        resumo_existe = (validation_dir / 'resumo_geral_validacao_matrizes.csv').exists()
        
        if not resumo_existe or not validation_dir.exists():
            return False
        
        janelas_validadas = list(validation_dir.glob('janela_*'))
        return resumo_existe and len(janelas_validadas) >= num_janelas_esperado
    
    @staticmethod
    def verificar_plot_validation() -> bool:
        """Verifica se gráficos de validação foram gerados"""
        figures_dir = Path('figures/validation')
        if not figures_dir.exists():
            return False
        graficos = list(figures_dir.glob('*.png'))
        return len(graficos) >= 4
    
    @staticmethod
    def verificar_network_analysis() -> bool:
        """Verifica se análise de redes foi concluída"""
        return Path('data/processed/network_metrics_summary.csv').exists()
    
    @staticmethod
    def verificar_plot_networks() -> bool:
        """Verifica se gráficos de redes foram gerados"""
        figures_dir = Path('figures/networks')
        if not figures_dir.exists():
            return False
        graficos_rede = list(figures_dir.glob('*.png'))
        return len(graficos_rede) >= 4
    
    @staticmethod
    def verificar_compile_paper() -> bool:
        """Verifica se paper foi compilado"""
        return Path('paper/systemic_risk_brazil.pdf').exists()
    
    @staticmethod
    def contar_heatmaps() -> Tuple[int, str]:
        """Conta heatmaps gerados"""
        figures_dir = Path('figures/correlation')
        if not figures_dir.exists():
            return 0, "Diretório não existe"
        
        heatmaps = list(figures_dir.glob('heatmap_janela_*.png'))
        return len(heatmaps), f"{len(heatmaps)} heatmaps"
    
    @staticmethod
    def contar_dendrogramas() -> Tuple[int, str]:
        """Conta dendrogramas gerados"""
        figures_dir = Path('figures/clustering')
        if not figures_dir.exists():
            return 0, "Diretório não existe"
        
        dendrogramas = list(figures_dir.glob('dendrogram_janela_*.png'))
        return len(dendrogramas), f"{len(dendrogramas)} dendrogramas"
