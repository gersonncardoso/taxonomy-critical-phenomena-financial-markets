import yfinance as yf
import pandas as pd

# Exemplo: baixar balanço patrimonial, DRE e fluxo de caixa de um ticker
# Para empresas brasileiras, use o ticker com ".SA" (ex: PETR4.SA, ITUB4.SA)
def baixar_balanco_yahoo(ticker):
    acao = yf.Ticker(ticker)
    # Balanço patrimonial
    balanco = acao.balance_sheet
    # DRE
    dre = acao.financials
    # Fluxo de caixa
    fluxo_caixa = acao.cashflow
    # Salva como CSV
    balanco.to_csv(f"data/yahoo_balanco_{ticker}.csv")
    dre.to_csv(f"data/yahoo_dre_{ticker}.csv")
    fluxo_caixa.to_csv(f"data/yahoo_fluxo_caixa_{ticker}.csv")
    print(f"Dados salvos para {ticker}")

if __name__ == "__main__":
    # Lê lista de tickers da base CVM
    try:
        tickers_cvm = pd.read_csv("data/processed/ticker_nome_empresa.csv")
        # Assume coluna 'ticker' e converte para formato Yahoo (ex: PETR4 -> PETR4.SA)
        lista_tickers = tickers_cvm['ticker'].dropna().unique()
        lista_tickers_yahoo = [f"{t.strip().upper()}.SA" for t in lista_tickers]
        for t in lista_tickers_yahoo:
            try:
                baixar_balanco_yahoo(t)
            except Exception as e:
                print(f"Erro ao baixar {t}: {e}")
    except Exception as e:
        print(f"Erro ao ler lista de tickers da CVM: {e}")
