"""
Systemic Risk Model - Source Package
Pacote contendo módulos de download e análise de dados da B3
"""

from .b3_downloader import B3Downloader, download_rapido

__version__ = '1.0.0'
__author__ = 'Gerson Cardoso'

# O que estará disponível quando fizer: from src import *
__all__ = ['B3Downloader', 'download_rapido']