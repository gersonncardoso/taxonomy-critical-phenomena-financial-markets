import os
from ticker_company_dict import extract_ticker_company_dict
import pandas as pd

YAML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../configs/tickers_b3_completo.yaml'))
OUT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/ticker_nome_empresa.csv'))

def main():
    d = extract_ticker_company_dict(YAML_PATH)
    df = pd.DataFrame(list(d.items()), columns=['ticker', 'nome_empresa'])
    df.to_csv(OUT_CSV, index=False)
    print(f"Tabela salva em {OUT_CSV}")

if __name__ == "__main__":
    main()