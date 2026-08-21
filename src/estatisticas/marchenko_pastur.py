"""
Análise espectral de Marchenko-Pastur por janela rolling.

Para uma janela com W observações e N ativos, a teoria de matrizes
aleatórias (RMT) prevê que, sob o modelo nulo i.i.d. Gaussiano, os
autovalores da matriz de correlação amostral seguem a distribuição
Marchenko-Pastur com razão q = N/W.  O limite superior da distribuição é:

    λ+ = σ² (1 + √q)²

Autovalores empíricos acima de λ+ carregam sinal econômico genuíno;
os demais são estatisticamente indistinguíveis de ruído dimensional.

Referências:
    Marchenko & Pastur (1967). Distribution of eigenvalues for some
        sets of random matrices. Mat. Sb., 72(4):507-536.
    Laloux et al. (1999). Noise Dressing of Financial Correlation
        Matrices. Physical Review Letters, 83(7):1467-1470.
    Plerou et al. (2002). Random matrix approach to cross correlations
        in financial data. Physical Review E, 65(6):066126.
"""

from __future__ import annotations

import numpy as np


def marchenko_pastur_upper(q: float, sigma: float = 1.0) -> float:
    """Limite superior λ+ da distribuição Marchenko-Pastur.

    Parameters
    ----------
    q : float
        Razão N/W (ativos / observações). Deve ser q > 0.
    sigma : float
        Variância das entradas da matriz aleatória (default=1).
    """
    return sigma ** 2 * (1.0 + np.sqrt(q)) ** 2


def marchenko_pastur_lower(q: float, sigma: float = 1.0) -> float:
    """Limite inferior λ- da distribuição Marchenko-Pastur."""
    return sigma ** 2 * (1.0 - np.sqrt(q)) ** 2


def compute_mp_stats(corr_matrix: np.ndarray, n_obs: int) -> dict:
    """Calcula estatísticas Marchenko-Pastur para uma janela rolling.

    Parameters
    ----------
    corr_matrix : np.ndarray, shape (N, N)
        Matriz de correlação amostral empírica.
    n_obs : int
        Número de observações W usadas para estimar a correlação.

    Returns
    -------
    dict com:
        n_assets         : N (número de ativos)
        n_obs            : W (observações)
        q                : N/W
        lambda_plus      : limite superior MP (λ+)
        lambda_minus     : limite inferior MP (λ-)
        lambda_max_emp   : maior autovalor empírico
        n_signal         : # autovalores > λ+  (fatores com sinal)
        n_noise          : # autovalores ≤ λ+  (ruído dimensional)
        var_signal_frac  : fração da variância total explicada por fatores de sinal
        market_mode_frac : fração de variância do autovalor dominante (fator de mercado)
    """
    corr = np.asarray(corr_matrix, dtype=float)
    n_assets = corr.shape[0]

    if n_assets < 2 or n_obs <= n_assets:
        # Regime subdeterminado: todos autovalores são ruído por construção
        return {
            "n_assets": n_assets,
            "n_obs": n_obs,
            "q": float(n_assets) / max(n_obs, 1),
            "lambda_plus": float("nan"),
            "lambda_minus": float("nan"),
            "lambda_max_emp": float("nan"),
            "n_signal": 0,
            "n_noise": n_assets,
            "var_signal_frac": 0.0,
            "market_mode_frac": float("nan"),
        }

    q = float(n_assets) / float(n_obs)
    lp = marchenko_pastur_upper(q)
    lm = marchenko_pastur_lower(q)

    # Autovalores em ordem decrescente
    eigenvalues = np.linalg.eigvalsh(corr)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)  # impede negativos numéricos

    total_var = eigenvalues.sum()  # = N para correlação
    lambda_max_emp = float(eigenvalues[0])

    signal_mask = eigenvalues > lp
    n_signal = int(signal_mask.sum())
    n_noise = n_assets - n_signal
    var_signal = float(eigenvalues[signal_mask].sum())
    var_signal_frac = var_signal / total_var if total_var > 0 else 0.0
    market_mode_frac = float(eigenvalues[0]) / total_var if total_var > 0 else float("nan")

    return {
        "n_assets": n_assets,
        "n_obs": n_obs,
        "q": q,
        "lambda_plus": lp,
        "lambda_minus": lm,
        "lambda_max_emp": lambda_max_emp,
        "n_signal": n_signal,
        "n_noise": n_noise,
        "var_signal_frac": var_signal_frac,
        "market_mode_frac": market_mode_frac,
    }
