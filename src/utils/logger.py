"""
Sistema de logging configurável
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name='systemic_risk', log_file=None, level=logging.INFO):
    """
    Configura logger para o projeto
    
    Args:
        name: Nome do logger
        log_file: Arquivo de log (opcional). Se None, não salva em arquivo
        level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        logging.Logger
    
    Exemplo:
        >>> from src.utils.logger import setup_logger
        >>> logger = setup_logger('meu_modulo')
        >>> logger.info('Processamento iniciado')
        >>> logger.warning('Atenção: dados faltantes')
        >>> logger.error('Erro ao processar arquivo')
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove handlers existentes (evita duplicação)
    logger.handlers.clear()
    
    # Formato das mensagens
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (sempre ativo)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (opcional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Logger padrão do projeto
logger = setup_logger()
