"""
Utilitário para extrair dicionário {ticker: nome_empresa} a partir do arquivo YAML de tickers da B3.
"""
import yaml
import re
from typing import Dict


def extract_ticker_company_dict(yaml_path: str) -> Dict[str, str]:
    """
    Lê o arquivo YAML de tickers e retorna um dicionário {ticker: nome_empresa}.
    Suporta tanto a estrutura nova em dict quanto o formato antigo com comentários.
    """
    ticker_dict = {}
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict) and isinstance(data.get('companies'), dict):
            for meta in data['companies'].values():
                nome = str(meta.get('nome_pregao', '')).strip()
                for ticker in meta.get('tickers', []) or []:
                    ticker = str(ticker).strip().upper()
                    if ticker and nome:
                        ticker_dict[ticker] = nome
            return ticker_dict
    except Exception:
        pass

    with open(yaml_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.match(r"\s*-\s*([A-Z0-9]+)\s+#\s*(.+)", line)
            if m:
                ticker = m.group(1).strip()
                nome = m.group(2).strip()
                ticker_dict[ticker] = nome
    return ticker_dict

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]
    else:
        yaml_path = "configs/tickers_b3_completo.yaml"
    d = extract_ticker_company_dict(yaml_path)
    for t, n in d.items():
        print(f"{t}: {n}")
