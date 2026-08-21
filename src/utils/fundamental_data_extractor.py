"""
Script para extrair dados fundamentalistas históricos da B3 ou do site de RI da empresa.

- Primeiro tenta baixar da B3 (CVM ou B3 API)
- Se não encontrar, tenta buscar no site de RI da empresa

Requer: requests, beautifulsoup4, pandas
"""
import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import Optional
from src.utils.ticker_company_dict import extract_ticker_company_dict

# Exemplo de endpoint B3 para demonstração (ajustar conforme API real)
B3_BALANCE_URL = "https://www.b3.com.br/api/empresa/{ticker}/fundamentos"


def fetch_b3_fundamentals(ticker: str) -> Optional[pd.DataFrame]:
    """
    Tenta baixar dados fundamentalistas históricos da B3 para o ticker.
    Salva arquivo bruto JSON se sucesso. Retorna DataFrame ou None se não encontrar.
    """
    url = B3_BALANCE_URL.format(ticker=ticker)
    samples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/fundamental_samples'))
    os.makedirs(samples_dir, exist_ok=True)
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and resp.headers.get('Content-Type','').startswith('application/json'):
            data = resp.json()
            # Salva arquivo bruto
            raw_path = os.path.join(samples_dir, f"{ticker}_b3_raw.json")
            with open(raw_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[OK] Dados brutos salvos em {raw_path}")
            # Ajustar parsing conforme estrutura real da resposta
            if 'historico' in data and isinstance(data['historico'], list) and data['historico']:
                df = pd.DataFrame(data['historico'])
                return df
            else:
                print(f"[WARN] Resposta da B3 não contém dados históricos para {ticker}.")
                return None
        else:
            print(f"[FAIL] Não foi possível baixar dados da B3 para {ticker}. Status: {resp.status_code}")
    except Exception as e:
        print(f"Erro ao buscar na B3 para {ticker}: {e}")
    return None


def fetch_company_ri_fundamentals(ticker: str, nome_empresa: str) -> Optional[pd.DataFrame]:
    """
    Tenta buscar dados fundamentalistas no site de RI da empresa.
    (Implementação básica: busca página de RI e tenta encontrar tabelas de resultados)
    """
    # Exemplo: busca Google pelo site de RI
    search_url = f"https://www.google.com/search?q={nome_empresa}+RI+resultados+financeiros"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(search_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            # Procurar primeiro link de RI
            for a in soup.find_all('a', href=True):
                href = a['href']
                if 'ri' in href.lower() or 'investidor' in href.lower():
                    # Aqui pode-se tentar acessar a página e buscar tabelas
                    # (Implementação real exigirá lógica adicional)
                    print(f"Possível site de RI para {ticker}: {href}")
                    break
    except Exception as e:
        print(f"Erro ao buscar RI para {ticker}: {e}")
    return None

if __name__ == "__main__":
    tickers = extract_ticker_company_dict("../../configs/tickers_b3_completo.yaml")
    for ticker, nome in list(tickers.items())[:5]:  # Exemplo: só 5 primeiros
        print(f"Buscando dados para {ticker} - {nome}")
        df = fetch_b3_fundamentals(ticker)
        if df is not None:
            print(df.head())
        else:
            fetch_company_ri_fundamentals(ticker, nome)
