"""
Script para baixar, extrair sob demanda, parsear e consolidar arquivos COTAHIST da B3 - VERSÃO 2.2 EFICIENTE + LIMPEZA
============================================================================================================

Pipeline:
1. Mantém apenas arquivos COTAHIST_A<ANO>.ZIP em data/raw/
2. Extrai o .TXT do ZIP apenas temporariamente via tempfile, durante o parse
3. Faz parsing direto do .TXT temporário; apaga imediatamente após uso
4. Consolida todos em único CSV: data/raw/b3_dados_completos.csv
5. Ao final, limpa todos os .TXT restantes na pasta raw (só os ZIP brutos ficam)

- Download incremental: só busca .ZIP se faltar localmente (ou via --force)
- Nunca mantém arquivos .TXT após parse
- Não há filtro, parsing externo, nem duplicação de arquivos raw

Autor: Gerson Nassor Cardoso
Instituição: Universidade Federal de São Paulo (UNIFESP)
Data: 2026-02-18
"""

import sys
import os
import re
import unicodedata
from difflib import SequenceMatcher
import requests
from datetime import datetime
from pathlib import Path
import zipfile
import pandas as pd
import tempfile
import shutil

RAW_DIR = Path('data/raw')
RAW_DIR.mkdir(parents=True, exist_ok=True)
LEGACY_MAP_PATH = Path('configs/ticker_legacy_map.yaml')

ANO_INICIO = 1995
ANO_FIM = datetime.now().year


def normalize_company_text(value: str) -> str:
    s = str(value or "").strip().upper()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Remove class markers from labels (from YAML comments)
    toks = [t for t in s.split() if t not in {"ON", "PN", "PNA", "PNB", "PNC", "UNIT", "UNT"}]
    return " ".join(toks)


def infer_class_from_especi(especi: str) -> str:
    e = str(especi or "").upper()
    e = unicodedata.normalize("NFKD", e)
    e = "".join(ch for ch in e if not unicodedata.combining(ch))
    e = re.sub(r"[^A-Z0-9 ]+", " ", e)
    e = re.sub(r"\s+", " ", e).strip()

    if "UNIT" in e or "UNT" in e:
        return "UNIT"
    if re.search(r"\bPN[A-Z]*\b", e):
        return "PN"
    if re.search(r"\bON\b", e):
        return "ON"
    return ""


def class_from_ticker(ticker: str) -> str:
    t = normalize_ticker_symbol(ticker)
    if t.endswith("11"):
        return "UNIT"
    if t.endswith("4"):
        return "PN"
    if t.endswith("3"):
        return "ON"
    return ""


def load_ticker_catalog(yaml_path: Path = Path("configs/tickers_b3_completo.yaml")):
    """Load canonical ticker catalog from structured YAML or legacy YAML comments.

    Returns:
      company_class_to_ticker: (company_norm, class) -> ticker
      ticker_to_company: ticker -> company_norm
      by_class: class -> [tickers]
    """
    company_class_to_ticker = {}
    ticker_to_company = {}
    by_class = {"ON": [], "PN": [], "UNIT": []}

    if not yaml_path.exists():
        return company_class_to_ticker, ticker_to_company, by_class

    try:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        data = None

    if isinstance(data, dict) and isinstance(data.get('companies'), dict):
        for meta in data['companies'].values():
            company = normalize_company_text(meta.get('nome_pregao', ''))
            tickers = meta.get('tickers', []) or []
            for raw_ticker in tickers:
                ticker = normalize_ticker_symbol(raw_ticker)
                cls = class_from_ticker(ticker)
                if not ticker or not company or not cls:
                    continue
                ticker_to_company[ticker] = company
                if ticker not in by_class[cls]:
                    by_class[cls].append(ticker)
                key = (company, cls)
                if key not in company_class_to_ticker:
                    company_class_to_ticker[key] = ticker
        return company_class_to_ticker, ticker_to_company, by_class

    pattern = re.compile(r"\s*-\s*([A-Z0-9]+)\s+#\s*(.+)")
    with open(yaml_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line)
            if not m:
                continue
            ticker = normalize_ticker_symbol(m.group(1))
            company = normalize_company_text(m.group(2))
            cls = class_from_ticker(ticker)
            if not ticker or not company or not cls:
                continue

            ticker_to_company[ticker] = company
            if ticker not in by_class[cls]:
                by_class[cls].append(ticker)

            key = (company, cls)
            if key not in company_class_to_ticker:
                company_class_to_ticker[key] = ticker

    return company_class_to_ticker, ticker_to_company, by_class


def load_legacy_ticker_map(yaml_path: Path = LEGACY_MAP_PATH):
    """Load explicit legacy->canonical mapping from YAML.

    Accepted schema:
      legacy_to_canonical:
        BBD4: BBDC4
        PET4: PETR4
    """
    mapping = {}
    if not yaml_path.exists():
        return mapping

    try:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            obj = yaml.safe_load(f) or {}
    except Exception:
        return mapping

    src = obj.get('legacy_to_canonical', obj)
    if not isinstance(src, dict):
        return mapping

    for k, v in src.items():
        lk = normalize_ticker_symbol(str(k))
        lv = normalize_ticker_symbol(str(v))
        if lk and lv:
            mapping[lk] = lv
    return mapping


def _company_similarity(raw_company: str, canonical_company: str) -> float:
    """Company similarity score with preference for prefix matches."""
    a = normalize_company_text(raw_company)
    b = normalize_company_text(canonical_company)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if b.startswith(a):
        return 0.95
    if a.startswith(b):
        return 0.90
    return SequenceMatcher(None, a, b).ratio()


def resolve_canonical_ticker(raw_ticker: str, company: str, cls: str) -> str:
    """Resolve legacy ticker symbols to canonical ones conservatively.

    Strategy order:
      1) already canonical ticker;
      2) exact (company, class) match from catalog;
      3) unique prefix match within class;
      4) class-restricted fuzzy company match with confidence margin.
    """
    t = normalize_ticker_symbol(raw_ticker)
    comp_norm = normalize_company_text(company)
    key_cache = (t, comp_norm, cls or "")
    cached = RESOLVE_CACHE.get(key_cache)
    if cached is not None:
        return cached

    if not t:
        RESOLVE_CACHE[key_cache] = ""
        return ""

    mapped_legacy = LEGACY_TICKER_MAP.get(t, "") if is_legacy_symbol(t) else ""
    if mapped_legacy:
        RESOLVE_CACHE[key_cache] = mapped_legacy
        return mapped_legacy

    if t in TICKER_TO_COMPANY:
        RESOLVE_CACHE[key_cache] = t
        return t

    if cls and comp_norm:
        direct = COMPANY_CLASS_TO_TICKER.get((comp_norm, cls), "")
        if direct:
            RESOLVE_CACHE[key_cache] = direct
            return direct

    if not cls:
        cls = class_from_ticker(t)

    candidates = list(BY_CLASS.get(cls, [])) if cls else []
    if not candidates:
        RESOLVE_CACHE[key_cache] = t
        return t

    base_letters = re.sub(r"[^A-Z]", "", t)
    prefix_candidates = [c for c in candidates if c.startswith(base_letters)] if base_letters else []
    if len(prefix_candidates) == 1:
        candidate = prefix_candidates[0]
        if comp_norm:
            sim = _company_similarity(comp_norm, TICKER_TO_COMPANY.get(candidate, ""))
            if sim >= 0.78:
                RESOLVE_CACHE[key_cache] = candidate
                return candidate
        else:
            RESOLVE_CACHE[key_cache] = candidate
            return candidate

    scored_pool = prefix_candidates if prefix_candidates else candidates
    if comp_norm:
        scored = []
        for c in scored_pool:
            comp_c = TICKER_TO_COMPANY.get(c, "")
            scored.append((c, _company_similarity(comp_norm, comp_c)))
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored:
            best_ticker, best_score = scored[0]
            second_score = scored[1][1] if len(scored) > 1 else 0.0
            # Conservative threshold + margin to avoid accidental remaps.
            if best_score >= 0.78 and (best_score - second_score) >= 0.12:
                RESOLVE_CACHE[key_cache] = best_ticker
                return best_ticker

    RESOLVE_CACHE[key_cache] = t
    return t


def normalize_ticker_symbol(value: str) -> str:
    """Normalize ticker text early in the pipeline.

    - repairs common mojibake artifacts when possible;
    - uppercases;
    - keeps only [A-Z0-9] to remove spacing/punctuation noise.
    """
    s = str(value or "").strip()
    if not s:
        return ""

    if any(ch in s for ch in ("Ã", "Â", "Ð", "Ñ")):
        try:
            s = s.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            pass

    s = s.upper()
    s = re.sub(r"[^A-Z0-9]+", "", s)
    return s


def is_legacy_symbol(ticker: str) -> bool:
    """Legacy short symbol shape from old COTAHIST files (ex.: BBD4, PET4, CMI4)."""
    t = normalize_ticker_symbol(ticker)
    return bool(re.fullmatch(r"[A-Z]{1,4}\d{1,2}", t))


COMPANY_CLASS_TO_TICKER, TICKER_TO_COMPANY, BY_CLASS = load_ticker_catalog()
LEGACY_TICKER_MAP = load_legacy_ticker_map()
RESOLVE_CACHE = {}

def arquivos_zip_ja_baixados():
    """Retorna set dos anos com .ZIP já presente no raw."""
    return {int(p.name[10:14]) for p in RAW_DIR.glob("COTAHIST_A*.ZIP") if p.name[10:14].isdigit()}

def baixar_cotahist_zip(ano):
    """Baixa o ZIP COTAHIST do ano.
    
    Política incremental:
    - Anos anteriores: baixa apenas se o ZIP ainda não existir localmente (cache)
    - Ano atual: sempre re-baixa, pois a B3 atualiza o arquivo diariamente
    """
    url = f"https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{ano}.ZIP"
    arq_dest = RAW_DIR / f"COTAHIST_A{ano}.ZIP"
    ano_atual = datetime.now().year
    if arq_dest.exists() and ano < ano_atual:
        return True  # ano histórico já em cache — não re-baixa
    if arq_dest.exists() and ano == ano_atual:
        arq_dest.unlink()  # ano atual: remove e re-baixa para pegar dados recentes
    print(f"🔽 Baixando {url} ... ", end="", flush=True)
    try:
        resp = requests.get(url, timeout=180)
        if resp.status_code == 200:
            with open(arq_dest, "wb") as fout:
                fout.write(resp.content)
            print(f"✔️ Salvo ({arq_dest.name})")
            return True
        elif resp.status_code == 404:
            print(f"✗ ERRO HTTP 404 ({url})")
            return False
        else:
            print(f"✗ ERRO HTTP {resp.status_code} ({url})")
            return False
    except Exception as e:
        print(f"✗ ERRO: {e}")
        return False

def extrair_txt_temporario(ano):
    """
    Extrai o .TXT do ZIP como arquivo temporário.
    Retorna caminho do TXT temporário e o diretório temporário.
    """
    zip_path = RAW_DIR / f"COTAHIST_A{ano}.ZIP"
    if not zip_path.exists():
        print(f"ZIP não encontrado para {ano} (bypass extração!).")
        return None, None
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Regra padrão: arquivos mais novos usam extensão .TXT explícita
        nomes = zf.namelist()
        txt_names = [n for n in nomes if n.upper().endswith('.TXT')]

        # Fallback para anos mais antigos (1995-2001), em que o arquivo vem
        # sem extensão .TXT explícita (ex.: COTAHIST.A1995, COTAHIST_A2001)
        if not txt_names:
            candidatos = [n for n in nomes if 'COTAHIST' in n.upper()]
            if not candidatos:
                print(f"✗ Nenhum arquivo COTAHIST encontrado em {zip_path.name}")
                return None, None
            txt_name = candidatos[0]
        else:
            txt_name = txt_names[0]
        temp_dir = tempfile.mkdtemp(prefix=f'cotahist_{ano}_')
        txt_temp_path = Path(temp_dir) / txt_name
        zf.extract(txt_name, temp_dir)
        return txt_temp_path, temp_dir

def parse_cotahist_txt(txt_file: Path):
    """Parse de COTAHIST com normalização de ticker e agregação diária.

    Regras aplicadas:
    - normaliza ticker removendo espaços internos (ex.: "PET 4" -> "PET4");
    - para múltiplas linhas no mesmo (Data, Ticker), agrega por média em todos
      os campos numéricos, conforme metodologia solicitada.
    """
    registros = []
    mapped_count = 0
    try:
        with open(txt_file, 'r', encoding='latin-1') as f:
            for linha in f:
                if linha.startswith("01"):
                    raw_ticker = linha[12:24].strip()
                    nomres = linha[27:39].strip()
                    especi = linha[39:49].strip()

                    cls = infer_class_from_especi(especi)
                    comp = normalize_company_text(nomres)
                    ticker_norm = normalize_ticker_symbol(raw_ticker)
                    canonical = resolve_canonical_ticker(ticker_norm, comp, cls)
                    if canonical:
                        if canonical != ticker_norm:
                            mapped_count += 1
                        ticker_value = canonical
                    else:
                        ticker_value = ticker_norm

                    reg = {
                        'Data': linha[2:10],
                        'Ticker': ticker_value,
                        'Empresa': comp,
                        'Preco_Abertura': float(linha[56:69]) / 100,
                        'Preco_Maximo': float(linha[69:82]) / 100,
                        'Preco_Minimo': float(linha[82:95]) / 100,
                        'Preco_Fechamento': float(linha[108:121]) / 100,
                        'Volume': float(linha[170:188]),
                    }
                    registros.append(reg)
        if registros:
            df = pd.DataFrame(registros)
            df['Data'] = pd.to_datetime(df['Data'])
            df['Ticker'] = df['Ticker'].map(normalize_ticker_symbol)
            df = df[df['Ticker'] != ''].copy()

            dup_count = int(df.duplicated(subset=['Data', 'Ticker']).sum())
            if dup_count > 0:
                num_cols = [
                    'Preco_Abertura',
                    'Preco_Maximo',
                    'Preco_Minimo',
                    'Preco_Fechamento',
                    'Volume',
                ]
                df = (
                    df.groupby(['Data', 'Ticker'], as_index=False)
                    .agg({**{col: 'mean' for col in num_cols}, 'Empresa': 'first'})
                )
                print(
                    f"   ↳ Ticker normalizado + média diária por (Data,Ticker): "
                    f"{dup_count:,} linhas duplicadas agregadas"
                )
            if mapped_count > 0:
                print(
                    f"   ↳ Mapeamento por nome/classe (NOMRES+ESPECI): "
                    f"{mapped_count:,} linhas convertidas para ticker canônico"
                )
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"✗ ERRO PARSING {txt_file}: {e}")
        return pd.DataFrame()

def limpar_txt_remanescentes():
    """Remove todos os .TXT restantes na pasta raw (cleanup manual/extra)."""
    print("\n🧹 Limpando arquivos .TXT remanescentes em data/raw/")
    for txt_file in RAW_DIR.glob("COTAHIST_A*.TXT"):
        try:
            txt_file.unlink()
            print(f"   Removido: {txt_file.name}")
        except Exception as e:
            print(f"   ERRO ao remover {txt_file.name}: {e}")

def main():
    print("="*80)
    print("DOWNLOAD+EXTRAÇÃO TEMPORÁRIA+CONSOLIDAÇÃO - COTAHIST B3 (RAW ZIP ONLY)")
    print("="*80)
    print(f"Período: {ANO_INICIO} até {ANO_FIM}")
    print(f"Destino (só ZIP permanece): {RAW_DIR.absolute()}")
    print("="*80)
    print()

    # Bloqueio absoluto de download se variáveis de ambiente pedirem
    bloqueio_download = os.environ.get('PAPER1_FORCE_DOWNLOAD', '1') == '0' or os.environ.get('PAPER1_ALWAYS_FULL_DOWNLOAD', '1') == '0'
    force_mode = ('--force' in sys.argv or os.environ.get('PIPELINE_FORCE') == '1') and not bloqueio_download
    anos_baixados = arquivos_zip_ja_baixados()

    print("📦 Arquivos ZIP já presentes:")
    if anos_baixados:
        print("   " + ", ".join(map(str, sorted(anos_baixados))))
    else:
        print("   Nenhum arquivo COTAHIST ZIP detectado na pasta raw")

    anos_full = list(range(ANO_INICIO, ANO_FIM + 1))
    if bloqueio_download:
        print("[INFO] Download bloqueado por configuração de ambiente. Nenhum arquivo será baixado ou removido.")
        anos_baixar = []
        anos_a_processar = anos_full
    elif not force_mode:
        anos_a_processar = anos_full
        # Incremental: histórico em cache, ano atual sempre atualizado
        ano_atual = datetime.now().year
        anos_baixar = [ano for ano in anos_full if ano not in anos_baixados or ano == ano_atual]
    else:
        print("\n⚠️  MODO FORCE: removendo todos arquivos antigos e re-baixando tudo!\n")
        for ano in anos_baixados:
            (RAW_DIR / f"COTAHIST_A{ano}.ZIP").unlink(missing_ok=True)
        anos_baixar = anos_full
        anos_a_processar = anos_full

    print(f"\nAnos a baixar: {anos_baixar if anos_baixar else 'Todos já baixados :)'}\n")

    # PASSO 1: DOWNLOAD ZIPs NECESSÁRIOS
    for ano in anos_baixar:
        baixar_cotahist_zip(ano)

    # PASSO 2+3: EXTRAI TXT TEMPORÁRIO, FAZ PARSE, DELETA TMP (LOOP POR ANO)
    print("="*80)
    print("PARSE E CONSOLIDAÇÃO DOS ARQUIVOS COTAHIST (.TXT temporário)")
    print("="*80)
    dfs = []
    for ano in anos_a_processar:
        zip_path = RAW_DIR / f"COTAHIST_A{ano}.ZIP"
        if not zip_path.exists():
            print(f"✗ ZIP ausente, pulando ano {ano}")
            continue
        txt_temp_path, temp_dir = extrair_txt_temporario(ano)
        if txt_temp_path is None:
            continue
        print(f"Parsing {txt_temp_path.name} ... ", end="", flush=True)
        df = parse_cotahist_txt(txt_temp_path)
        print(f"{len(df):,} registros")
        if not df.empty:
            dfs.append(df)
        # Remove imediatamente o diretório temporário e o TXT extraído
        shutil.rmtree(temp_dir, ignore_errors=True)

    # CONSOLIDAÇÃO FINAL
    if not dfs:
        print("Nenhum arquivo COTAHIST TXT parseado. Fase abortada!")
    else:
        df_final = pd.concat(dfs, ignore_index=True)
        df_final = df_final.sort_values(['Data', 'Ticker']).reset_index(drop=True)
        print(f"\nTotal consolidado: {len(df_final):,} registros, {df_final['Ticker'].nunique()} tickers")
        csv_path = RAW_DIR / "b3_dados_completos.csv"
        df_final.to_csv(csv_path, index=False)
        print(f"✔️ CSV consolidado salvo em {csv_path}")

    # LIMPEZA FINAL DE .TXT QUE TENHAM SOBRADO NO RAW
    limpar_txt_remanescentes()

    print("\n" + "="*80)
    print("✅ Pipeline COTAHIST otimizado — apenas ZIPs ficam no RAW!")
    print("="*80)
    print("Próximos passos sugeridos:")
    print("   • python main/main_preprocessing.py")
    print("   • python pipeline/pipeline_completo.py")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERRO FATAL: {str(e)}")
        sys.exit(1)