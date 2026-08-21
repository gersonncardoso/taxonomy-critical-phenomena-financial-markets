"""Utilitários para detecção e uso opcional de GPU (CuPy).

Se uma GPU NVIDIA compatível e o pacote CuPy estiverem disponíveis,
``GPU_AVAILABLE`` será ``True`` e ``cp`` apontará para o módulo ``cupy``.
Caso contrário, ``GPU_AVAILABLE`` será ``False`` e ``cp`` será ``None``.

As funções aqui são usadas pelos módulos estatísticos para decidir
se devem tentar usar GPU ou ficar apenas no NumPy/CPU.

ATENÇÃO: para que o uso de GPU funcione corretamente é necessário ter
o toolkit CUDA instalado (por exemplo, CUDA 12.x), bem como a versão
compatível do pacote ``cupy-cudaXX`` instalada no ambiente Python.
"""

from __future__ import annotations

from typing import Any


# Flag global para (re)ativar GPU manualmente caso o ambiente esteja pronto.
# Como o CUDA 12.x foi instalado, deixamos ativado por padrão; se algo
# falhar (import ou ausência de GPU), caímos automaticamente para CPU.
ENABLE_GPU: bool = True

try:
    if not ENABLE_GPU:
        raise ImportError("GPU support disabled by configuration")
    import cupy as _cp  # type: ignore

    # Tenta detectar GPUs com múltiplos métodos — necessário em alguns ambientes WSL2
    _NGPU = 0
    try:
        _NGPU = _cp.cuda.runtime.getDeviceCount()
    except Exception:
        pass

    if _NGPU == 0:
        try:
            _NGPU = _cp.cuda.Device.count()
        except Exception:
            pass

    if _NGPU == 0:
        # Último recurso: tenta alocar um array pequeno na GPU
        try:
            _test = _cp.zeros(1)
            del _test
            _NGPU = 1
        except Exception:
            pass

except ImportError:
    _cp = None  # type: ignore
    _NGPU = 0

# Permite forçar GPU via variável de ambiente: FORCE_GPU=1
import os as _os
_force_gpu = _os.environ.get("FORCE_GPU", "0") == "1"

GPU_AVAILABLE: bool = bool((_cp is not None and _NGPU > 0) or (_force_gpu and _cp is not None))
cp = _cp  # reexporta para uso opcional em outros módulos (pode ser None)


def to_numpy(x: Any):
    """Converte arrays CuPy para NumPy; retorna demais objetos inalterados."""
    if cp is not None:
        try:
            from cupy import ndarray as _cupy_ndarray  # type: ignore

            if isinstance(x, _cupy_ndarray):  # type: ignore[attr-defined]
                return cp.asnumpy(x)  # type: ignore[union-attr]
        except Exception:
            # Se algo der errado com CuPy, apenas devolve o objeto original
            return x
    return x
