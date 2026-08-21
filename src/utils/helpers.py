"""
Funções auxiliares gerais
"""

import pandas as pd
from pathlib import Path

def ensure_dir(directory):
    """
    Garante que diretório existe, cria se necessário
    
    Args:
        directory: Caminho do diretório (str ou Path)
    
    Exemplo:
        >>> ensure_dir('data/processed')
        >>> ensure_dir(Path('figures/graphs'))
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def load_csv(filepath, **kwargs):
    """
    Carrega dados de arquivo CSV
    
    Args:
        filepath: Caminho do arquivo
        **kwargs: Argumentos adicionais para pd.read_csv
    
    Returns:
        pd.DataFrame
    
    Exemplo:
        >>> df = load_csv('data/processed/dados_limpos.csv', parse_dates=['Data'])
    """
    return pd.read_csv(filepath, **kwargs)


def save_csv(df, filepath, **kwargs):
    """
    Salva DataFrame em CSV (cria diretório se necessário)
    
    Args:
        df: DataFrame
        filepath: Caminho do arquivo
        **kwargs: Argumentos adicionais para df.to_csv
    
    Exemplo:
        >>> save_csv(df, 'data/processed/resultado.csv', index=False)
    """
    # Padroniza saídas relativas ao diretório processado do repositório.
    out_path = Path(filepath)
    if not out_path.is_absolute():
        out_path = Path('data/processed') / out_path.name
    ensure_dir(out_path.parent)
    df.to_csv(out_path, **kwargs)


def load_config_tickers(tickers_file='configs/tickers.yaml'):
    """
    Carrega lista de tickers do arquivo YAML
    
    Args:
        tickers_file: Caminho para arquivo de tickers
    
    Returns:
        list: Lista de tickers
    
    Exemplo:
        >>> tickers = load_config_tickers()
        >>> print(len(tickers))  # 88
    """
    import yaml
    
    with open(tickers_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config.get('tickers', [])
