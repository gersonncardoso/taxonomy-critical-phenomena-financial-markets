"""
Gerenciador de Configurações
=============================

Carrega e gerencia configurações do arquivo YAML

Autor: Gerson Nassor Cardoso
Instituição: Universidade Federal de São Paulo (UNIFESP)
Data: 2026-02-12

Copyright (c) 2026 Gerson Nassor Cardoso - UNIFESP
"""

import yaml
import os
from typing import Any, List


class Config:
    """Classe para gerenciar configurações do projeto"""
    
    def __init__(self, config_path: str = None):
        """
        Inicializa o gerenciador de configurações
        Args:
            config_path: Caminho para o arquivo de configuração (opcional)
        """
        # Sempre resolve o caminho do config.yaml relativo à raiz do projeto
        if config_path is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(root_dir, 'configs', 'config.yaml')
        self.config_path = config_path
        self.config_data = self._load_config()
    
    
    def _load_config(self) -> dict:
        """Carrega arquivo YAML de configuração"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtém valor de configuração usando notação de ponto
        
        Args:
            key: Chave no formato 'secao.subsecao.chave'
            default: Valor padrão se a chave não existir
        
        Returns:
            Valor da configuração ou default
        
        Example:
            >>> config.get('dados.data_inicio')
            '2014-01-01'
        """
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    
    def get_tickers(self, categoria: str = 'ibovespa') -> List[str]:
        """
        Obtém lista de tickers de uma categoria específica
        
        Args:
            categoria: Categoria de tickers
                - 'ibovespa': Tickers do IBOVESPA
                - 'todos': Todos os tickers disponíveis no YAML
                - 'small_mid_caps': Small e Mid caps
                - 'energia': Setor de energia
                - 'commodities': Setor de commodities
                - 'financeiro': Setor financeiro
                - 'etfs': ETFs
        
        Returns:
            Lista de tickers (strings)
        
        Example:
            >>> config.get_tickers('ibovespa')
            ['PETR4', 'VALE3', 'ITUB4', ...]
            
            >>> config.get_tickers('todos')
            ['PETR4', 'VALE3', ..., 'ABCD3', ...]
        """
        tickers_file = self.get('dados.tickers_file', 'configs/tickers.yaml')
        
        if not os.path.exists(tickers_file):
            print(f"⚠️  Arquivo de tickers não encontrado: {tickers_file}")
            return []
        
        with open(tickers_file, 'r', encoding='utf-8') as f:
            tickers_data = yaml.safe_load(f)
        
        # Caso 1: Arquivo no formato novo (com categorias)
        if 'tickers' in tickers_data and isinstance(tickers_data['tickers'], dict):
            
            # Se pediu 'todos', retornar união de todas as categorias
            if categoria == 'todos':
                todos = set()
                for cat_name, cat_tickers in tickers_data['tickers'].items():
                    if isinstance(cat_tickers, list):
                        todos.update(cat_tickers)
                return sorted(list(todos))
            
            # Retornar categoria específica
            if categoria in tickers_data['tickers']:
                tickers = tickers_data['tickers'][categoria]
                if isinstance(tickers, list):
                    return tickers
                else:
                    print(f"⚠️  Categoria '{categoria}' não é uma lista válida")
                    return []
            else:
                print(f"⚠️  Categoria '{categoria}' não encontrada no arquivo")
                print(f"   Categorias disponíveis: {list(tickers_data['tickers'].keys())}")
                return []
        
        # Caso 2: Arquivo no formato antigo (lista simples)
        elif isinstance(tickers_data, list):
            print(f"ℹ️  Arquivo no formato antigo (lista simples)")
            return tickers_data
        
        # Caso 3: Formato desconhecido
        else:
            print(f"⚠️  Formato do arquivo de tickers não reconhecido")
            return []


# Instância global
config = Config()