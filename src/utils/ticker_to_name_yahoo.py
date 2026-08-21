import yaml
import yfinance as yf
import pandas as pd
import os

# Caminho do YAML de tickers
YAML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../configs/tickers_b3_completo.yaml'))
# Caminho do CSV de saída
OUT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/ticker_nome_yahoo.csv'))

def carregar_tickers(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and isinstance(data.get('tickers'), list):
        return data['tickers']
    if isinstance(data, dict):
        return list(data.keys())
    return data

def buscar_nome_yahoo(ticker):
    try:
        info = yf.Ticker(ticker + ".SA").info
        return info.get('longName') or info.get('shortName')
    except Exception:
        return None

def main():
    tickers = carregar_tickers(YAML_PATH)
    results = []
    for ticker in tickers:
        nome = buscar_nome_yahoo(ticker)
        print(f"{ticker}: {nome}")
        results.append({'ticker': ticker, 'nome_empresa': nome})
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"Tabela salva em {OUT_CSV}")

if __name__ == "__main__":
    main()