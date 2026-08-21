import numpy as np
from scipy.stats import pearsonr, t as t_dist

from src.utils.gpu_utils import GPU_AVAILABLE, cp, to_numpy


def _calcular_pvalues_cpu(data_matrix: np.ndarray) -> np.ndarray:
    """Implementa├º├úo original baseada em pearsonr (CPU puro)."""
    n = data_matrix.shape[1]
    pval_mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            corr, pval = pearsonr(data_matrix[:, i], data_matrix[:, j])
            pval_mat[i, j] = pval_mat[j, i] = pval
    return pval_mat


def _calcular_pvalues_gpu(data_matrix: np.ndarray) -> np.ndarray:
    """Vers├úo vetorizada usando CuPy para correla├º├Áes, com p-values via SciPy.

    Calcula a matriz de correla├º├úo de Pearson em GPU e, em seguida, aplica a
    f├│rmula exata do teste t para obter p-values bilaterais, retornando um
    array NumPy compat├¡vel com o restante do pipeline.
    """
    if not (GPU_AVAILABLE and cp is not None):  # type: ignore[truthy-function]
        # salvaguarda: se GPU n├úo estiver realmente dispon├¡vel, cai para CPU
        return _calcular_pvalues_cpu(data_matrix)

    n_obs, n = data_matrix.shape
    if n_obs <= 2 or n <= 1:
        return _calcular_pvalues_cpu(data_matrix)

    try:
        x = cp.asarray(data_matrix)
        # Centraliza por coluna
        x = x - cp.mean(x, axis=0, keepdims=True)
        # Covari├óncia amostral
        cov = (x.T @ x) / (n_obs - 1)
        std = cp.sqrt(cp.diag(cov))
        denom = std[:, None] * std[None, :]
        corr = cov / (denom + 1e-12)
        corr = cp.clip(corr, -1.0, 1.0)

        df = n_obs - 2
        t_stat = corr * cp.sqrt(df / (1 - corr ** 2 + 1e-12))
        t_np = cp.asnumpy(t_stat)

        # p-value bicaudal a partir da distribui├º├úo t de Student
        p_np = 2 * (1 - t_dist.cdf(np.abs(t_np), df=df))
        np.fill_diagonal(p_np, 0.0)
        return p_np
    except Exception:
        # Em caso de qualquer problema na GPU, volta silenciosamente para CPU
        return _calcular_pvalues_cpu(data_matrix)


def calcular_pvalues_from_corr(corr_matrix: np.ndarray, n_obs: int) -> np.ndarray:
    """Calcula p-values bilaterais H0: rho=0 direto da matriz de correlação.

    Usa a estatística t exata: t_ij = rho_ij * sqrt((n_obs-2)/(1-rho_ij^2)),
    distribuída como t-Student com df = n_obs-2 sob H0.

    corr_matrix: np.array [n, n] - matriz de correlação de Pearson
    n_obs:       int - número de observações usadas para estimar cada correlação
    """
    corr = np.asarray(corr_matrix, dtype=float)
    df = max(n_obs - 2, 1)
    denom = np.clip(1.0 - corr ** 2, 1e-12, None)
    t_stat = corr * np.sqrt(df / denom)
    pval = 2.0 * (1.0 - t_dist.cdf(np.abs(t_stat), df=df))
    np.fill_diagonal(pval, 1.0)
    return pval


def calcular_pvalues(corr_matrix, data_matrix):
    """Retorna matriz de p-values para cada par de séries a partir dos dados brutos.

    corr_matrix: ignorado (mantido por compatibilidade da API).
    data_matrix: np.array ou CuPy array [dias, n] - séries de retornos

    Quando GPU + CuPy estão disponíveis, usa versão vetorizada em GPU;
    caso contrário, usa pearsonr/CPU.
    """
    data_np = np.asarray(to_numpy(data_matrix))

    if GPU_AVAILABLE and cp is not None:
        return _calcular_pvalues_gpu(data_np)

    return _calcular_pvalues_cpu(data_np)
