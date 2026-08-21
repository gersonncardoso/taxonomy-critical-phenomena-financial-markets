from statsmodels.tsa.stattools import adfuller

def calcular_adf(logreturns):
    """
    Aplica o teste ADF aos log-retornos.
    Returns:
      dict com stat, p-value e se ├® estacion├írio.
    """
    result = adfuller(logreturns, autolag='AIC')
    return {
        "adf_stat": result[0],
        "adf_pvalue": result[1],
        "adf_stationary": result[1] < 0.05
    }
