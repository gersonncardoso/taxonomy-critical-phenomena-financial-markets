import numpy as np
from scipy.stats import ks_2samp

from src.utils.gpu_utils import to_numpy


def calcular_ks(mat_real, mat_random):
    """Retorna estatística e p-value do teste KS-2 entre correlações empíricas e nulas.

    Compara os triângulos superiores (sem diagonal) das duas matrizes de
    correlação, garantindo que as duas amostras estejam na mesma escala [-1, 1].
    mat_real:   np.array [n, n] — matriz de correlação empírica
    mat_random: np.array [n, n] — matriz de correlação nula (N(0,1) data)
    """
    mat_real_np = to_numpy(mat_real)
    mat_random_np = to_numpy(mat_random)

    n = mat_real_np.shape[0]
    idx = np.triu_indices(n, k=1)  # triângulo superior, sem diagonal
    real_flat = mat_real_np[idx]
    random_flat = mat_random_np[idx]
    ks_stat, ks_pvalue = ks_2samp(real_flat, random_flat)
    return ks_stat, ks_pvalue
