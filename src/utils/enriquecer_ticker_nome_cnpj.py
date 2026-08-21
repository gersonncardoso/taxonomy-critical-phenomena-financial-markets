"""
Script para enriquecer o dicionário {ticker: nome_empresa} com CNPJ usando busca automática em fontes públicas.
- Para cada empresa, busca o CNPJ pelo nome usando ReceitaWS (API pública) ou fallback via scraping do Google.
- Salva resultado em CSV: data/processed/ticker_nome_cnpj.csv
"""

import requests
import time
import csv
import urllib.parse
import sys
import os
# Permite importar ticker_company_dict.py mesmo rodando como script
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from ticker_company_dict import extract_ticker_company_dict

# Caminhos
YAML_PATH = "configs/tickers_b3_completo.yaml"
OUT_CSV = "data/processed/ticker_nome_cnpj.csv"

# Função para buscar CNPJ via ReceitaWS
# https://www.receitaws.com.br/v1/cnpj/<CNPJ> (mas não tem busca por nome, só por CNPJ)
# Alternativa: https://publica.cnpj.ws/cnpj/<CNPJ> (idem)
# Então, usar scraping do Google como fallback

def buscar_cnpj_google(nome_empresa):
    """Busca o CNPJ da empresa no Google e retorna o primeiro CNPJ encontrado na página."""
    query = f"{nome_empresa} CNPJ"
    url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        return ""
    import re
    # Procura padrão de CNPJ
    cnpj_match = re.search(r"(\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})", r.text)
    if cnpj_match:
        return cnpj_match.group(1)
    return ""

def main():
    ticker_dict = extract_ticker_company_dict(YAML_PATH)
    resultados = []
    for ticker, nome in ticker_dict.items():
        print(f"Buscando CNPJ para: {ticker} - {nome}")
        cnpj = buscar_cnpj_google(nome)
        resultados.append({"ticker": ticker, "nome_empresa": nome, "cnpj": cnpj})
        time.sleep(2)  # Evita bloqueio do Google
    # Salva CSV
    with open(OUT_CSV, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "nome_empresa", "cnpj"])
        writer.writeheader()
        writer.writerows(resultados)
    print(f"Arquivo salvo: {OUT_CSV}")

if __name__ == "__main__":
    main()
