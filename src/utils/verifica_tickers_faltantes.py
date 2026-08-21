"""
Verifica se todos os tickers do YAML/config estão presentes na base consolidada (CVM/B3).
Se faltar algum ticker, emite alerta com lista dos ausentes.
Pode ser agendado para rodar periodicamente ou após atualização dos dados.
"""
import csv
from src.utils.ticker_company_dict import extract_ticker_company_dict

YAML_PATH = "configs/tickers_b3_completo.yaml"
# Caminho para base consolidada (exemplo: CVM ou B3)
BASE_CSV = "data/processed/ticker_nome_cnpj.csv"  # ou outro consolidado


def get_tickers_base(csv_path):
    tickers = set()
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickers.add(row["ticker"].strip())
    return tickers

def main():
    tickers_yaml = set(extract_ticker_company_dict(YAML_PATH).keys())
    tickers_base = get_tickers_base(BASE_CSV)
    faltando = tickers_yaml - tickers_base
    if faltando:
        print("ALERTA: Faltam dados para os seguintes tickers:")
        for t in sorted(faltando):
            print(f"  - {t}")
    else:
        print("Todos os tickers do YAML possuem dados na base consolidada.")

if __name__ == "__main__":
    main()
