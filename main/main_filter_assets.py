"""
Filtro de Ativos da B3 - Versão Otimizada (Sem Relatórios)
----------------------------------------------------------

Filtra ativos com base em:
- Completude mínima
- Volume médio mínimo
- Frequência mínima
- Tipo de ativo permitido
- Uma ação por empresa (priorizar PN)

Versão otimizada:
- Sem loops por ticker
- Sem relatórios TXT/CSV/PDF
- Compatível com pipeline_completo.py
- Muito mais rápido

Autor: Gerson Nassor Cardoso - UNIFESP
Data: 2026-02-19
"""

import sys
import os
import re
import yaml
from pathlib import Path
import pandas as pd

# Corrige o problema de import "No module named src"
# === CORREÇÃO DEFINITIVA DO IMPORT "src" ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
print("DEBUG: PROJECT_ROOT =", PROJECT_ROOT)
print("DEBUG: sys.path =", sys.path[:5])

from src.utils.logger import setup_logger
from pipeline import config
logger = setup_logger("main_filter_assets")

# Configurações de filtro
COMPLETUDE_MIN = 0.80
VOLUME_MIN = 1_000_000
FREQUENCIA_MIN = 0.80
TIPOS_PERMITIDOS = {"ON", "PN", "UNIT"}

INPUT_FILE = "data/raw/b3_dados_completos.csv"
OUTPUT_FILE = "data/raw/b3_dados_filtrados.csv"
CANONICAL_TICKERS_ONLY = os.environ.get("PIPELINE_CANONICAL_TICKERS_ONLY", "0") == "1"
CANONICAL_TICKERS_YAML = Path("configs/tickers_b3_completo.yaml")
LEGACY_MAP_YAML = Path("configs/ticker_legacy_map.yaml")
LOCAL_EQUITY_ONLY = os.environ.get("PIPELINE_LOCAL_EQUITY_ONLY", "1") == "1"
CVM_COMPANY_MAP_YAML = Path("configs/cvm_company_map.yaml")


# ---------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------

def extrair_tipo_ativo(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if ticker.endswith("11"):
        return "UNIT"
    if ticker.endswith("4"):
        return "PN"
    if ticker.endswith("3"):
        return "ON"
    return "OUTRO"


def extrair_empresa(ticker: str) -> str:
    """Extrai parte alfabética do ticker."""
    return "".join([c for c in ticker if c.isalpha()])


def normalize_ticker_symbol(value: str) -> str:
    """Normaliza ticker com limpeza de ruido e tentativa de correção de mojibake."""
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
    t = normalize_ticker_symbol(ticker)
    return bool(re.fullmatch(r"[A-Z]{1,4}\d{1,2}", t))


def load_canonical_tickers_from_yaml(yaml_path: Path) -> set:
    """Carrega whitelist de tickers canônicos a partir do YAML de referência."""
    out = set()
    if not yaml_path.exists():
        return out
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        data = None
    if isinstance(data, dict) and isinstance(data.get('tickers'), list):
        for ticker in data['tickers']:
            t = normalize_ticker_symbol(ticker)
            if t:
                out.add(t)
        return out
    pat = re.compile(r"\s*-\s*([A-Z0-9]+)\s+#")
    with open(yaml_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line)
            if not m:
                continue
            t = normalize_ticker_symbol(m.group(1))
            if t:
                out.add(t)
    return out


def load_legacy_ticker_map(yaml_path: Path) -> dict:
    out = {}
    if not yaml_path.exists():
        return out
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            obj = yaml.safe_load(f) or {}
    except Exception:
        return out
    src = obj.get('legacy_to_canonical', obj)
    if not isinstance(src, dict):
        return out
    for k, v in src.items():
        lk = normalize_ticker_symbol(str(k))
        lv = normalize_ticker_symbol(str(v))
        if lk and lv:
            out[lk] = lv
    return out


def load_local_equity_universe_from_cvm(yaml_path: Path) -> set:
    """Monta whitelist ampla de ações/units de companhias locais via mapa CVM/B3.

    A whitelist é baseada nos códigos-base das companhias listadas no mapa CVM.
    Isso remove BDRs, ETFs, FIIs e outros instrumentos da B3 que não pertencem
    ao universo de ações corporativas desejado, sem derrubar classes válidas
    como PETR3/PETR4, BBDC3/BBDC4, ITUB3/ITUB4, KLBN11 etc.
    """
    if not yaml_path.exists():
        return set()
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}
    except Exception:
        return set()

    companies = obj.get("companies", obj)
    if not isinstance(companies, dict):
        return set()

    allowed_suffixes = ("3", "4", "5", "6", "7", "8", "11")
    whitelist = set()
    for base_code in companies.keys():
        base = normalize_ticker_symbol(str(base_code))
        if not re.fullmatch(r"[A-Z]{4}", base):
            continue
        for suffix in allowed_suffixes:
            whitelist.add(f"{base}{suffix}")
    return whitelist


def deduplicar_cotacoes_diarias(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Agrega múltiplas cotações do mesmo dia/ticker pela média.

    Regra metodológica aplicada: para linhas duplicadas em (data, ticker),
    calcula média dos campos numéricos e mantém o primeiro valor dos campos
    não numéricos.
    """
    before = len(df)
    dup = int(df.duplicated(subset=[date_col, "Ticker"]).sum())
    if dup == 0:
        return df

    id_cols = [date_col, "Ticker"]
    num_cols = [
        c for c in df.columns
        if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    other_cols = [c for c in df.columns if c not in id_cols + num_cols]

    agg = {c: "mean" for c in num_cols}
    agg.update({c: "first" for c in other_cols})

    df2 = df.groupby(id_cols, as_index=False).agg(agg)
    after = len(df2)

    logger.info(
        f"Deduplicação diária: {dup:,} duplicatas em (Data,Ticker) "
        f"agregadas por média | {before:,} -> {after:,} linhas"
    )
    return df2


def marcar_dias_negociados(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Marca dias com negociação efetiva usando variação do fechamento."""
    df = df.sort_values(["Ticker", date_col]).copy()
    fechamento_original = pd.to_numeric(df["Preco_Fechamento"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

    fechamento_base = fechamento_original.groupby(df["Ticker"]).ffill()
    fechamento_anterior = fechamento_base.groupby(df["Ticker"]).shift(1)

    mudou_fechamento = fechamento_base.ne(fechamento_anterior)
    primeira_obs = fechamento_anterior.isna() & fechamento_base.gt(0)

    df["Preco_Fechamento"] = fechamento_base
    df["Dia_Negociado"] = (
        fechamento_original.notna()
        & fechamento_base.gt(0)
        & volume.gt(0)
        & (mudou_fechamento | primeira_obs)
    )
    return df


# ---------------------------------------------------------
# Filtro otimizado
# ---------------------------------------------------------

def filtrar_ativos(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    logger.info("Calculando métricas por ativo (versão otimizada)")

    # Calendário de dias de negociação observado na base inteira
    df_neg = df[df["Dia_Negociado"]].copy()
    datas_unicas = pd.to_datetime(sorted(df_neg[date_col].dropna().unique()))
    mapa_data_idx = {d: i for i, d in enumerate(datas_unicas)}

    # Groupby turbo
    group = df_neg.groupby("Ticker", observed=True)

    dias_com_dados = group[date_col].nunique()
    volume_medio = group["Volume"].mean()
    data_inicio = group[date_col].min()
    data_fim = group[date_col].max()

    df_metricas = pd.DataFrame({
        "Ticker": dias_com_dados.index,
        "dias_com_dados": dias_com_dados.values,
        "volume_medio": volume_medio.values,
        "data_inicio": data_inicio.values,
        "data_fim": data_fim.values,
    })

    # Número de dias úteis em que o ativo poderia negociar (vida ativa na amostra)
    idx_inicio = df_metricas["data_inicio"].map(mapa_data_idx)
    idx_fim = df_metricas["data_fim"].map(mapa_data_idx)
    df_metricas["dias_vida"] = (idx_fim - idx_inicio) + 1
    df_metricas["dias_vida"] = df_metricas["dias_vida"].clip(lower=1)

    # Completude e frequência relativas ao período de vida do ativo na amostra
    df_metricas["completude"] = df_metricas["dias_com_dados"] / df_metricas["dias_vida"]
    df_metricas["frequencia"] = df_metricas["dias_com_dados"] / df_metricas["dias_vida"]

    # Filtros
    df_metricas = df_metricas[df_metricas["completude"] >= COMPLETUDE_MIN]
    df_metricas = df_metricas[df_metricas["volume_medio"] >= VOLUME_MIN]
    df_metricas = df_metricas[df_metricas["frequencia"] >= FREQUENCIA_MIN]

    # Tipo de ativo
    df_metricas["tipo"] = df_metricas["Ticker"].apply(extrair_tipo_ativo)
    df_metricas = df_metricas[df_metricas["tipo"].isin(TIPOS_PERMITIDOS)]

    # Selecionar 1 ação por empresa
    df_metricas["empresa"] = df_metricas["Ticker"].apply(extrair_empresa)

    prioridade = {"PN": 1, "ON": 2, "UNIT": 3, "OUTRO": 99}
    df_metricas["prioridade"] = df_metricas["tipo"].map(prioridade)

    df_metricas = (
        df_metricas.sort_values(["empresa", "prioridade", "volume_medio"], ascending=[True, True, False])
        .drop_duplicates(subset=["empresa"], keep="first")
    )

    tickers_final = df_metricas["Ticker"].unique()

    df_final = df[df["Ticker"].isin(tickers_final) & df["Dia_Negociado"]].copy()

    logger.info(f"Ativos finais selecionados: {len(tickers_final)}")

    return df_final


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("\n============================================================")
    print("FILTRO DE ATIVOS - VERSÃO OTIMIZADA (SEM RELATÓRIOS)")
    print("============================================================")

    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"ERRO: Arquivo não encontrado: {input_path}")
        return False

    print(f"\nCarregando dados de {input_path} ...")

    df = pd.read_csv(input_path)

    # Detectar coluna de data
    if "Date" in df.columns:
        date_col = "Date"
    elif "Data" in df.columns:
        date_col = "Data"
    else:
        print("ERRO: Coluna de data não encontrada.")
        print("Colunas disponíveis:", list(df.columns))
        return False

    df[date_col] = pd.to_datetime(df[date_col])

    # Alinha a base ao horizonte oficial do projeto para evitar vazamento temporal.
    project_end_date = pd.to_datetime(config.END_DATE)
    before_cutoff = len(df)
    df = df[df[date_col] <= project_end_date].copy()
    if len(df) != before_cutoff:
        logger.info(
            f"Aplicado cutoff END_DATE={config.END_DATE}: "
            f"{before_cutoff:,} -> {len(df):,} linhas"
        )

    # Normalizacao antecipada para evitar propagacao de ticker corrompido.
    df["Ticker"] = df["Ticker"].map(normalize_ticker_symbol)

    legacy_map = load_legacy_ticker_map(LEGACY_MAP_YAML)
    if legacy_map:
        df["Ticker"] = df["Ticker"].map(lambda t: legacy_map.get(t, t) if is_legacy_symbol(t) else t)

    df = df[df["Ticker"] != ""].copy()

    if LOCAL_EQUITY_ONLY:
        local_equity_universe = load_local_equity_universe_from_cvm(CVM_COMPANY_MAP_YAML)
        if local_equity_universe:
            before = len(df)
            before_tickers = df["Ticker"].nunique()
            df = df[df["Ticker"].isin(local_equity_universe)].copy()
            logger.info(
                "Modo local_equity_only ativo: restringindo a ações/units de companhias locais "
                f"via CVM/B3 ({len(local_equity_universe)} tickers candidatos). "
                f"{before:,} -> {len(df):,} linhas | {before_tickers} -> {df['Ticker'].nunique()} tickers"
            )
        else:
            logger.warning(
                f"Modo local_equity_only ativo, mas whitelist CVM vazia/inexistente: {CVM_COMPANY_MAP_YAML}"
            )

    # Modo opcional: restringe ao universo canônico do YAML (ticker estável no tempo).
    if CANONICAL_TICKERS_ONLY:
        canonical = load_canonical_tickers_from_yaml(CANONICAL_TICKERS_YAML)
        if canonical:
            before = len(df)
            df = df[df["Ticker"].isin(canonical)].copy()
            logger.info(
                f"Modo canônico ativo: restringindo ao YAML ({len(canonical)} tickers). "
                f"{before:,} -> {len(df):,} linhas"
            )
        else:
            logger.warning(
                f"Modo canônico ativo, mas YAML vazio/inexistente: {CANONICAL_TICKERS_YAML}"
            )

    # Segurança metodológica: consolida duplicatas do mesmo dia/ticker por média
    df = deduplicar_cotacoes_diarias(df, date_col)
    df = marcar_dias_negociados(df, date_col)

    print(f"Registros carregados: {len(df):,}")
    print(f"Ativos únicos: {df['Ticker'].nunique()}")
    print(f"Período: {df[date_col].min()} a {df[date_col].max()}")

    try:
        df_final = filtrar_ativos(df, date_col)
    except Exception as e:
        print(f"ERRO ao aplicar filtros: {e}")
        logger.error(f"Erro ao aplicar filtros: {e}", exc_info=True)
        return False

    # Salvar resultado
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_final.to_csv(output_path, index=False)

    print("\nFiltro concluído com sucesso.")
    print(f"Arquivo salvo em: {output_path}")

    return True


if __name__ == "__main__":
    try:
        ok = main()
        sys.exit(0 if ok else 1)
    except Exception as e:
        print(f"ERRO FATAL: {e}")
        logger.error(f"Erro fatal: {e}", exc_info=True)
        sys.exit(1)
