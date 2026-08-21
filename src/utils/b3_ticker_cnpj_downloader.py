"""
Script para baixar e extrair a relação Ticker, CNPJ e Nome Completo das empresas listadas na B3.
- Baixa o arquivo oficial da B3 (empresas-listadas.csv)
- Extrai as colunas Ticker, CNPJ, Nome da Empresa
- Salva um CSV pronto para cruzamento com dados da CVM
"""
import os
import pandas as pd
import requests


# URL correta para o CSV de empresas listadas na B3
B3_EMPRESAS_URL = "https://sistemaswebb3-listados.b3.com.br/companies-register/resources/CompaniesList.csv"

OUT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/empresas_b3_ticker_cnpj.csv'))



def baixar_arquivo_b3(url, out_path):
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'text/csv,application/csv,application/octet-stream',
        'Referer': 'https://www.b3.com.br/pt_br/market-data-e-indices/empresas-listadas/empresas-listadas/',
        'Origin': 'https://www.b3.com.br',
    }
    r = requests.get(url, headers=headers, timeout=60)
    if r.status_code == 200 and r.headers.get('Content-Type', '').startswith('text/csv'):
        with open(out_path, 'wb') as f:
            f.write(r.content)
        print(f"Arquivo baixado: {out_path}")
        return out_path
    else:
        print(f"[ERRO] Não foi possível baixar {url} (status {r.status_code})")
        print(f"Content-Type recebido: {r.headers.get('Content-Type')}")
        # Try scraping as fallback
        return baixar_por_scraping(out_path)


def baixar_por_scraping(out_path):
    from bs4 import BeautifulSoup
    import csv
    # URL da página de empresas listadas
    page_url = "https://www.b3.com.br/pt_br/market-data-e-indices/empresas-listadas/empresas-listadas/"
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Referer': 'https://www.b3.com.br/',
    }
    r = requests.get(page_url, headers=headers, timeout=60)
    if r.status_code != 200:
        print(f"[ERRO] Não foi possível acessar a página da B3 para scraping (status {r.status_code})")
        return None
    soup = BeautifulSoup(r.text, 'html.parser')
    # Procura por tabela de empresas (pode mudar conforme estrutura da página)
    table = soup.find('table')
    if not table:
        print("[ERRO] Tabela de empresas não encontrada na página da B3.")
        return None
    rows = table.find_all('tr')
    data = []
    for row in rows:
        cols = [col.get_text(strip=True) for col in row.find_all(['td', 'th'])]
        if cols:
            data.append(cols)
    # Salva como CSV
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    print(f"Tabela extraída por scraping e salva em: {out_path}")
    return out_path



def extrair_ticker_cnpj_nome(csv_path, out_csv):
    # O arquivo CompaniesList.csv da B3 é separado por vírgula e codificação utf-8
    df = pd.read_csv(csv_path, sep=',', encoding='utf-8')
    # Colunas típicas: 'CNPJ', 'Razão Social', 'Nome de Pregão', 'Código de Negociação'
    if 'CNPJ' not in df.columns or 'Nome de Pregão' not in df.columns or 'Código de Negociação' not in df.columns:
        print(f"[ERRO] Colunas esperadas não encontradas no arquivo da B3.")
        print(f"Colunas disponíveis: {df.columns.tolist()}")
        return
    df_out = df[['Código de Negociação', 'CNPJ', 'Nome de Pregão']].drop_duplicates()
    df_out.columns = ['ticker', 'cnpj', 'nome_completo']
    df_out.to_csv(out_csv, index=False)
    print(f"Tabela extraída e salva em: {out_csv}")

if __name__ == "__main__":
    temp_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), 'EmpresasListadas.csv'))
    arq = baixar_arquivo_b3(B3_EMPRESAS_URL, temp_csv)
    if arq:
        extrair_ticker_cnpj_nome(arq, OUT_CSV)
