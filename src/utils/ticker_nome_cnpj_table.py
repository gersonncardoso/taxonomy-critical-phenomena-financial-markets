"""
Tabela hardcoded de ticker, nome_empresa e CNPJ para empresas da B3.
Atualize manualmente ou rode o script de enriquecimento para novos tickers.
"""
TICKER_NOME_CNPJ = [
    {"ticker": "ABEV3", "nome_empresa": "Ambev", "cnpj": "07.526.557/0001-00"},
    {"ticker": "ITUB4", "nome_empresa": "Itaú Unibanco", "cnpj": "60.872.504/0001-23"},
    {"ticker": "VALE3", "nome_empresa": "Vale", "cnpj": "33.592.510/0001-54"},
    {"ticker": "PETR4", "nome_empresa": "Petrobras PN", "cnpj": "33.000.167/0001-01"},
    {"ticker": "WEGE3", "nome_empresa": "WEG", "cnpj": "84.429.695/0001-11"},
    # ...adicione mais manualmente ou rode enriquecimento...
]

def get_cnpj_by_ticker(ticker):
    for row in TICKER_NOME_CNPJ:
        if row["ticker"] == ticker:
            return row["cnpj"]
    return ""
