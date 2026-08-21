"""
Utilidades
"""

from .logger import setup_logger
from .dependencies import (
    verificar_e_instalar_dependencias,
    verificar_pacote_instalado,
    instalar_pacote
)
try:
    from .spark_runtime import (
        log_env_info,
        setup_logging,
        debug_spark_env,
        configurar_ambiente_linux,
        configurar_spark_linux,
        configurar_ambiente_windows,
        configurar_spark_windows,
        executar_pipeline,
    )
    _HAS_SPARK_RUNTIME = True
except Exception:
    _HAS_SPARK_RUNTIME = False

__all__ = [
    'setup_logger',
    'verificar_e_instalar_dependencias',
    'verificar_pacote_instalado',
    'instalar_pacote',
]

if _HAS_SPARK_RUNTIME:
    __all__.extend([
        'log_env_info',
        'setup_logging',
        'debug_spark_env',
        'configurar_ambiente_linux',
        'configurar_spark_linux',
        'configurar_ambiente_windows',
        'configurar_spark_windows',
        'executar_pipeline',
    ])
