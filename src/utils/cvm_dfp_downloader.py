"""
Utilitário para baixar e processar demonstrações financeiras históricas (DFP) da CVM.
- Baixa arquivos anuais .zip da CVM
- Extrai CSVs
- Filtra dados para tickers/empresas desejados
"""

import os
import requests
import zipfile
import pandas as pd
from io import BytesIO
# Importa tabela hardcoded de ticker-nome-cnpj
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from ticker_nome_cnpj_table import TICKER_NOME_CNPJ

def download_cvm_dfp_zip(year, out_dir):
    url = f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_{year}.zip"
    local_path = os.path.join(out_dir, f"dfp_cia_aberta_{year}.zip")
    if not os.path.exists(local_path):
        print(f"Baixando {url}...")
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(r.content)
            print(f"Salvo em {local_path}")
        else:
            print(f"[ERRO] Não foi possível baixar {url}")
            return None
    else:
        print(f"Já existe: {local_path}")
    return local_path

def extract_csv_from_zip(zip_path, out_dir):
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            if name.endswith('.csv'):
                z.extract(name, out_dir)
                print(f"Extraído: {name}")
                return os.path.join(out_dir, name)
    return None


def filtrar_dfp_por_cnpj(csv_path, cnpjs, n=5):
    df = pd.read_csv(csv_path, sep=';', encoding='latin1', low_memory=False)
    # A coluna 'CNPJ_CIA' contém o CNPJ da empresa
    cnpjs = list(cnpjs)[:n]
    filtrado = df[df['CNPJ_CIA'].isin(cnpjs)]
    return filtrado

if __name__ == "__main__":
    from datetime import datetime
    ano_atual = datetime.now().year
    anos = list(range(1995, ano_atual + 1))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/fundamental_samples'))
    os.makedirs(out_dir, exist_ok=True)
    # Usa tabela hardcoded de ticker-nome-cnpj
    cnpjs = [row["cnpj"] for row in TICKER_NOME_CNPJ if row["cnpj"]]
    for ano in anos:
        zip_path = download_cvm_dfp_zip(ano, out_dir)
        if zip_path:
            csv_path = extract_csv_from_zip(zip_path, out_dir)
            if csv_path:
                df_filtrado = filtrar_dfp_por_cnpj(csv_path, cnpjs)
                if not df_filtrado.empty:
                    out_csv = os.path.join(out_dir, f"dfp_filtrado_cnpj_{ano}.csv")
                    df_filtrado.to_csv(out_csv, index=False)
                    print(f"Amostra salva: {out_csv}")
                else:
                    print(f"[INFO] Nenhum dado encontrado para CNPJs selecionados em {ano}")
