"""
Consolidador de Dados com Janelas
==================================

Consolida dados brutos com informações de janelas em um único arquivo.

Formato de saída (LONG):
- Date, Ticker, Open, High, Low, Close, Volume
- Janela_ID, Window_Start, Window_End
- Retorno_Log

Autor: Gerson Nassor Cardoso
Instituição: UNIFESP
Data: 2026-02-18
"""

import pandas as pd
import numpy as np
import re
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.logger import setup_logger
from src.utils.window_manager import WindowManager

logger = setup_logger('consolidator')


class DataConsolidator:
    """
    Consolida dados com informações de janelas
    
    Gera um único arquivo CSV com:
    - Dados de preços (LONG)
    - ID da janela para cada registro
    - Retornos logarítmicos
    """
    
    def __init__(self,
                 input_file: str = 'data/raw/b3_dados_filtrados.csv',
                 output_file: str = 'data/processed/dados_consolidados.csv'):
        """
        Args:
            input_file: Arquivo de entrada (LONG format, já filtrado)
            output_file: Arquivo de saída consolidado
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.legacy_map_file = Path('configs/ticker_legacy_map.yaml')
        
        self.window_manager = WindowManager(data_file=str(input_file), window_num_months=12)

    def _load_legacy_ticker_map(self) -> dict:
        out = {}
        if not self.legacy_map_file.exists():
            return out
        try:
            with open(self.legacy_map_file, 'r', encoding='utf-8') as f:
                obj = yaml.safe_load(f) or {}
        except Exception:
            return out

        src = obj.get('legacy_to_canonical', obj)
        if not isinstance(src, dict):
            return out

        for k, v in src.items():
            lk = self._normalize_ticker_symbol(str(k))
            lv = self._normalize_ticker_symbol(str(v))
            if lk and lv:
                out[lk] = lv
        return out

    @staticmethod
    def _normalize_ticker_symbol(value: str) -> str:
        """Normaliza ticker com limpeza de ruido e tentativa de correção de mojibake."""
        s = str(value or '').strip()
        if not s:
            return ''
        if any(ch in s for ch in ("Ã", "Â", "Ð", "Ñ")):
            try:
                s = s.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
            except Exception:
                pass
        s = s.upper()
        s = re.sub(r"[^A-Z0-9]+", "", s)
        return s

    @staticmethod
    def _is_legacy_symbol(ticker: str) -> bool:
        t = DataConsolidator._normalize_ticker_symbol(ticker)
        return bool(re.fullmatch(r"[A-Z]{1,4}\d{1,2}", t))

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula retornos logarítmicos com forward fill do fechamento e
        marca dias com negociação efetiva.

        Fórmula: Retorno_Log_t = ln(Preco_Fechamento_t / Preco_Fechamento_{t-1})
        Para pares inválidos, retorno = 0.

        Args:
            df: DataFrame com dados de preços

        Returns:
            DataFrame com coluna de retornos adicionada
        """
        logger.info("Calculando retornos logarítmicos (formula: ln(P_t / P_{t-1}))")
        
        df = df.sort_values(['Ticker', 'Date']).copy()

        fechamento_original = pd.to_numeric(df['Preco_Fechamento'], errors='coerce')
        if 'Volume' in df.columns:
            volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        else:
            volume = pd.Series(1.0, index=df.index)

        # Regra metodológica: fechamento ausente herda o último fechamento observado.
        df['Preco_Fechamento'] = fechamento_original.groupby(df['Ticker']).ffill()
        preco_base = df['Preco_Fechamento']
        preco_anterior = preco_base.groupby(df['Ticker']).shift(1)

        mudou_fechamento = preco_base.ne(preco_anterior)
        primeira_obs = preco_anterior.isna() & preco_base.gt(0)
        df['Dia_Negociado'] = (
            fechamento_original.notna()
            & preco_base.gt(0)
            & volume.gt(0)
            & (mudou_fechamento | primeira_obs)
        )

        df['Retorno_Log'] = np.where(
            preco_base.gt(0) & preco_anterior.gt(0),
            np.log(preco_base / preco_anterior),
            0.0,
        )

        # Qualquer NaN, inf ou -inf vira 0
        df['Retorno_Log'] = df['Retorno_Log'].replace([np.inf, -np.inf, np.nan], 0)

        # Evidência: exibir fórmulas e amostras, mostrando um print para algumas linhas
        exemplo = df[['Date', 'Ticker', 'Preco_Fechamento', 'Retorno_Log', 'Dia_Negociado']].head(8)
        logger.info("Evidência de cálculo de retornos logarítmicos (primeiros registros de cada ticker):\n"
                    f"{exemplo}")
        print("\n[Diagnóstico retornos log]\n", exemplo.to_string(), "\n")

        return df

    def assign_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Atribui ID de janela para cada registro
        
        Args:
            df: DataFrame com dados
        
        Returns:
            DataFrame com colunas de janela adicionadas
        """
        logger.info("Atribuindo janelas aos registros...")
        
        # Gerar metadados de janelas
        self.window_manager.calculate_windows()
        windows = self.window_manager.windows
        
        if not windows:
            logger.error("Nenhuma janela calculada!")
            return df
        
        # Para garantir que cada janela tenha de fato 12 meses de dados e que
        # cada observação pertença a todas as janelas que a contêm, criamos
        # um novo DataFrame com uma cópia dos registros por janela.

        windowed_dfs = []
        for window in tqdm(windows, desc="Atribuindo janelas"):
            window_id = window['id']
            start_date = pd.to_datetime(window['start'])
            end_date = pd.to_datetime(window['end'])

            mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
            df_win = df[mask].copy()
            if df_win.empty:
                continue

            df_win['Janela_ID'] = window_id
            df_win['Window_Start'] = start_date
            df_win['Window_End'] = end_date
            windowed_dfs.append(df_win)

        if not windowed_dfs:
            logger.error("Nenhum registro se enquadrou em nenhuma janela calculada")
            df['Janela_ID'] = None
            df['Window_Start'] = None
            df['Window_End'] = None
            return df

        df_out = pd.concat(windowed_dfs, ignore_index=True)

        logger.info(
            f"Atribuídas {len(windows)} janelas rolling de 12 meses; "
            f"total de registros com janelas: {len(df_out):,}"
        )

        return df_out

    def filter_by_window_completeness(self, df: pd.DataFrame, min_fraction: float = 0.80) -> pd.DataFrame:
        """Filtra empresas com presença insuficiente na janela.

        Quando a coluna 'Empresa' está presente (visão por empresa):
          1. Cobertura = UNIÃO de datas únicas de TODOS os tickers da empresa na
             janela (não soma de percentuais — evita double-count em dias onde dois
             tickers co-existiram).
          2. Ticker canônico eleito UMA ÚNICA VEZ para todo o histórico (global),
             com preferência PN > ON > UNIT > outros; dentro do mesmo tipo, maior
             total de dias negociados. Isso garante que KLABIN seja sempre KLAB4
             em todas as janelas, nunca KLA4 em algumas e KLAB4 em outras.
          3. Todas as linhas da empresa em TODAS as janelas são remapeadas para o
             ticker canônico global. Dias onde dois tickers co-existiam são
             deduplicados mantendo o registro de maior Volume.

        Sem 'Empresa': usa filtro original por ticker (80% de cobertura individual).

        Args:
            df: DataFrame com colunas 'Date', 'Ticker', 'Janela_ID' (e opcionalmente 'Empresa').
            min_fraction: Fração mínima de dias da janela (padrão 0.80).

        Returns:
            DataFrame filtrado com tickers canônicos globais por empresa.
        """
        import re as _re

        if 'Janela_ID' not in df.columns:
            logger.warning("Coluna 'Janela_ID' ausente; pulando filtro por janela")
            return df

        logger.info(
            "Aplicando filtro por janela: empresa deve negociar em pelo menos "
            f"{min_fraction:.0%} dos dias úteis de cada janela"
        )

        df_valid = df.dropna(subset=['Janela_ID']).copy()
        if df_valid.empty:
            logger.warning("Nenhum registro com 'Janela_ID' definido; nada a filtrar")
            return df

        neg_col = 'Dia_Negociado' if 'Dia_Negociado' in df_valid.columns else None
        df_ref = df_valid[df_valid[neg_col]] if neg_col else df_valid
        if df_ref.empty:
            logger.warning("Nenhum registro marcado como dia negociado; nada a filtrar")
            return df.iloc[0:0].copy()

        # Total de dias úteis em cada janela (base do denominador)
        dias_por_janela = df_ref.groupby('Janela_ID')['Date'].nunique().rename('total_dias_janela')

        usa_empresa = 'Empresa' in df.columns and df['Empresa'].notna().any()

        if usa_empresa:
            logger.info(
                "Coluna 'Empresa' detectada: visão por empresa "
                f"(cobertura = união de datas, ticker canônico global) ≥{min_fraction:.0%}"
            )

            # Função auxiliar de ordenação de tipo de ativo
            def _tipo_ord(ticker: str) -> int:
                m = _re.search(r'(\d+)$', str(ticker))
                if m:
                    n = int(m.group(1))
                    if n in (4, 6, 8):  return 1   # PN
                    if n in (3, 5, 7):  return 2   # ON
                    if n == 11:        return 3   # UNIT
                return 99

            def _class_key(ticker: str) -> str:
                ordv = _tipo_ord(ticker)
                if ordv == 1:
                    return 'PN'
                if ordv == 2:
                    return 'ON'
                if ordv == 3:
                    return 'UNIT'
                return 'OUTRO'

            def _class_ord(cls: str) -> int:
                return {'PN': 1, 'ON': 2, 'UNIT': 3}.get(str(cls), 99)

            def _build_company_ticker_map(df_src: pd.DataFrame) -> pd.DataFrame:
                """Elege ticker canônico global por empresa (PN > ON > UNIT > outros)."""
                df_base = df_src[df_src[neg_col]] if neg_col else df_src
                gd = (
                    df_base.groupby(['Empresa', 'Ticker'])['Date']
                    .nunique()
                    .rename('total_global_days')
                    .reset_index()
                )
                gd['tipo_ord'] = gd['Ticker'].map(_tipo_ord)
                return (
                    gd.sort_values(
                        ['Empresa', 'tipo_ord', 'total_global_days'],
                        ascending=[True, True, False]
                    )
                    .drop_duplicates(subset=['Empresa'], keep='first')
                    [['Empresa', 'Ticker']]
                    .rename(columns={'Ticker': 'Ticker_Canonico'})
                )

            def _build_class_ticker_map(df_src: pd.DataFrame) -> pd.DataFrame:
                """Elege ticker canônico global por empresa+classe (ON/PN/UNIT)."""
                df_base = df_src[df_src[neg_col]] if neg_col else df_src
                gd = (
                    df_base.groupby(['Empresa', 'Classe_Acao', 'Ticker'])['Date']
                    .nunique()
                    .rename('total_global_days')
                    .reset_index()
                )
                gd['tipo_ord'] = gd['Ticker'].map(_tipo_ord)
                return (
                    gd.sort_values(
                        ['Empresa', 'Classe_Acao', 'tipo_ord', 'total_global_days'],
                        ascending=[True, True, True, False]
                    )
                    .drop_duplicates(subset=['Empresa', 'Classe_Acao'], keep='first')
                    [['Empresa', 'Classe_Acao', 'Ticker']]
                    .rename(columns={'Ticker': 'Ticker_Canonico'})
                )

            # ── 0. Unificar nomes de empresa que mudaram ao longo do tempo ────────
            # Quando uma empresa muda de nome mas mantém o mesmo ticker (ex:
            # "BRF FOODS" → "BRF SA"), o COTAHIST registra ambos os nomes.
            # A eleição preliminar do ticker canônico detecta essa colisão
            # (2+ nomes → mesmo ticker). Unificamos para o nome com mais dias.
            _pre_ticker_map = _build_company_ticker_map(df_valid)
            _collision_groups = (
                _pre_ticker_map
                .groupby('Ticker_Canonico')['Empresa']
                .apply(list)
                .reset_index()
            )
            _collisions = _collision_groups[_collision_groups['Empresa'].map(len) > 1]

            if len(_collisions):
                _emp_total_days = (
                    df_valid.groupby('Empresa')['Date']
                    .nunique()
                    .rename('emp_days')
                    .reset_index()
                )
                _empresa_remap: dict = {}
                for _, _row in _collisions.iterrows():
                    _emps = _row['Empresa']
                    _days_tab = _emp_total_days[_emp_total_days['Empresa'].isin(_emps)]
                    _canonical_emp = (
                        _days_tab.sort_values('emp_days', ascending=False).iloc[0]['Empresa']
                    )
                    for _e in _emps:
                        if _e != _canonical_emp:
                            _empresa_remap[_e] = _canonical_emp
                logger.info(
                    f"Unificando {len(_empresa_remap)} alias(es) de empresa "
                    f"({len(_collisions)} ticker(s) com nomes múltiplos no histórico)"
                )
                for _old, _new in sorted(_empresa_remap.items())[:5]:
                    logger.info(f"  Alias: {_old!r} → {_new!r}")
                df_valid['Empresa'] = df_valid['Empresa'].map(
                    lambda x: _empresa_remap.get(x, x)
                )

            # ── 1. Cobertura por acao (classe) por janela ────────────────────────
            # Regra pedida: 80% em cima de cada acao (ON/PN/UNIT); se ON e PN
            # passarem para a mesma empresa na janela, escolher PN.
            df_valid['Classe_Acao'] = df_valid['Ticker'].map(_class_key)
            df_ref_empresa = df_valid[df_valid[neg_col]] if neg_col else df_valid

            dias_emp_classe_janela = (
                df_ref_empresa.groupby(['Janela_ID', 'Empresa', 'Classe_Acao'])['Date']
                .nunique()
                .rename('dias_classe')
                .reset_index()
            )
            dias_emp_classe_janela = dias_emp_classe_janela.merge(
                dias_por_janela.reset_index(), on='Janela_ID', how='left'
            )
            dias_emp_classe_janela['frac_classe'] = (
                dias_emp_classe_janela['dias_classe'] / dias_emp_classe_janela['total_dias_janela']
            ).clip(upper=1.0)

            classes_validas = dias_emp_classe_janela[
                dias_emp_classe_janela['frac_classe'] >= min_fraction
            ].copy()
            classes_validas['classe_ord'] = classes_validas['Classe_Acao'].map(_class_ord)

            # Escolha final por (janela, empresa): prioriza PN > ON > UNIT > OUTRO
            escolha_por_janela = (
                classes_validas
                .sort_values(
                    ['Janela_ID', 'Empresa', 'classe_ord', 'dias_classe'],
                    ascending=[True, True, True, False]
                )
                .drop_duplicates(subset=['Janela_ID', 'Empresa'], keep='first')
                [['Janela_ID', 'Empresa', 'Classe_Acao']]
            )

            # ── 2. Ticker canônico GLOBAL por empresa+classe ──────────────────────
            global_class_ticker_map = _build_class_ticker_map(df_valid)

            # ── 3. Filtrar df_valid e remapear para ticker canônico da classe ─────
            empresas_com_ticker = escolha_por_janela.merge(
                global_class_ticker_map,
                on=['Empresa', 'Classe_Acao'],
                how='left',
            )

            antes = len(df)
            df_valid = df_valid.merge(
                empresas_com_ticker[['Janela_ID', 'Empresa', 'Classe_Acao', 'Ticker_Canonico']],
                on=['Janela_ID', 'Empresa', 'Classe_Acao'],
                how='inner'
            )
            df_valid['Ticker'] = df_valid['Ticker_Canonico']
            df_valid = df_valid.drop(columns=['Ticker_Canonico', 'Classe_Acao'], errors='ignore')

            # ── 5. Deduplicar: mesmo (Date, Ticker, Janela_ID) → maior Volume ────
            sort_keys = ['Date', 'Ticker', 'Janela_ID']
            if 'Volume' in df_valid.columns:
                df_valid = df_valid.sort_values(
                    sort_keys + ['Volume'], ascending=[True, True, True, False]
                )
            else:
                df_valid = df_valid.sort_values(sort_keys)
            df_valid = df_valid.drop_duplicates(
                subset=['Date', 'Ticker', 'Janela_ID'], keep='first'
            )

            df_sem_janela = df[df['Janela_ID'].isna()].copy()
            df_filtrado = pd.concat([df_valid, df_sem_janela], ignore_index=True)

            removidos = antes - len(df_filtrado)
            logger.info(
                f"Filtro por ação (≥{min_fraction:.0%} por classe ON/PN/UNIT, com preferência PN): "
                f"{len(escolha_por_janela):,} pares empresa×janela elegíveis; "
                f"{removidos:,} registros removidos"
            )

        else:
            # Fallback: comportamento original por ticker
            logger.info("Coluna 'Empresa' ausente: usando filtro original por ticker")

            cobertura = (
                df_ref.groupby(['Janela_ID', 'Ticker'])['Date']
                .nunique()
                .rename('dias_com_dados')
                .reset_index()
            )
            cobertura = cobertura.merge(dias_por_janela.reset_index(), on='Janela_ID', how='left')
            cobertura['frac_presenca'] = cobertura['dias_com_dados'] / cobertura['total_dias_janela']

            pares_validos = cobertura[
                cobertura['frac_presenca'] >= min_fraction
            ][['Janela_ID', 'Ticker']]

            antes = len(df)
            df = df.merge(pares_validos.assign(_keep=1), on=['Janela_ID', 'Ticker'], how='left')
            df_filtrado = df[(df['_keep'] == 1) | df['Janela_ID'].isna()].copy()
            df_filtrado.drop(columns=['_keep'], inplace=True)

            removidos = antes - len(df_filtrado)
            logger.info(
                f"Filtro por janela (≥{min_fraction:.0%} dos dias) removeu {removidos:,} registros"
            )

        return df_filtrado

    def consolidate(self, 
                   tickers: Optional[List[str]] = None,
                   calculate_returns: bool = True) -> pd.DataFrame:
        """
        Consolida dados completos
        
        Args:
            tickers: Lista de tickers para filtrar (opcional)
            calculate_returns: Se True, calcula retornos
        
        Returns:
            DataFrame consolidado
        """
        logger.info("="*80)
        logger.info("CONSOLIDANDO DADOS")
        logger.info("="*80)
        
        # Carregar dados
        logger.info(f"Carregando dados de {self.input_file}")
        
        if not self.input_file.exists():
            logger.error(f"Arquivo não encontrado: {self.input_file}")
            return pd.DataFrame()
        
        # LEITURA ROBUSTA: ENCONTRA 'Data' OU 'Date' INSENSITIVO
        df = pd.read_csv(self.input_file)
        date_col = next((col for col in df.columns if col.lower() in ("date", "data")), None)
        if date_col is None:
            logger.error(f"Nenhuma coluna de data detectada! Colunas: {df.columns.tolist()}")
            raise ValueError(f"Nenhuma coluna de data encontrada! Colunas: {df.columns.tolist()}")
        df[date_col] = pd.to_datetime(df[date_col])
        if date_col != 'Date':
            df['Date'] = df[date_col]

        if 'Ticker' in df.columns:
            before = len(df)
            df['Ticker'] = df['Ticker'].map(self._normalize_ticker_symbol)
            legacy_map = self._load_legacy_ticker_map()
            if legacy_map:
                df['Ticker'] = df['Ticker'].map(
                    lambda t: legacy_map.get(t, t) if self._is_legacy_symbol(t) else t
                )
            df = df[df['Ticker'] != ''].copy()
            removed = before - len(df)
            if removed > 0:
                logger.info(f"Normalizacao de ticker removeu {removed:,} registros com ticker vazio/invalido")

        logger.info(f"Coluna de data padronizada: {date_col} → 'Date'.")
        
        logger.info(f"Carregados {len(df):,} registros")
        
        # Filtrar tickers se especificado
        if tickers:
            logger.info(f"Filtrando {len(tickers)} tickers...")
            df = df[df['Ticker'].isin(tickers)]
            logger.info(f"Após filtro: {len(df):,} registros")
        
        # Calcular retornos
        if calculate_returns:
            df = self.calculate_returns(df)
        
        # Atribuir janelas
        df = self.assign_windows(df)
        
        # Filtro adicional: presença mínima por janela (mínimo 80% dos dias negociados)
        df = self.filter_by_window_completeness(df, min_fraction=0.80)
        
        # Ordenar
        df = df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
        
        # Salvar
        logger.info(f"Salvando em {self.output_file}")
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_file, index=False)
        
        # Estatísticas
        logger.info("="*80)
        logger.info("CONSOLIDAÇÃO CONCLUÍDA")
        logger.info("="*80)
        logger.info(f"Arquivo: {self.output_file}")
        logger.info(f"Registros: {len(df):,}")
        logger.info(f"Tickers: {df['Ticker'].nunique()}")
        logger.info(f"Janelas: {df['Janela_ID'].nunique()}")
        logger.info(f"Período: {df['Date'].min()} até {df['Date'].max()}")
        logger.info(f"Tamanho: {self.output_file.stat().st_size / 1024 / 1024:.2f} MB")
        logger.info("="*80)
        
        return df
    
    def get_window_data(self, window_id: int) -> pd.DataFrame:
        """
        Obtém dados de uma janela específica do arquivo consolidado
        
        Args:
            window_id: ID da janela
        
        Returns:
            DataFrame com dados da janela
        """
        if not self.output_file.exists():
            logger.error(f"Arquivo consolidado não encontrado: {self.output_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(self.output_file)
        df_window = df[df['Janela_ID'] == window_id].copy()
        
        return df_window
    
    def get_summary(self) -> dict:
        """
        Obtém resumo do arquivo consolidado
        
        Returns:
            Dict com estatísticas
        """
        if not self.output_file.exists():
            return {'exists': False}
        
        df = pd.read_csv(self.output_file, usecols=['Date', 'Ticker', 'Janela_ID'])
        
        return {
            'exists': True,
            'file': str(self.output_file),
            'size_mb': self.output_file.stat().st_size / 1024 / 1024,
            'total_records': len(df),
            'tickers': df['Ticker'].nunique(),
            'windows': df['Janela_ID'].nunique(),
            'date_range': {
                'start': str(df['Date'].min()),
                'end': str(df['Date'].max())
            }
        }
    
    def print_summary(self):
        """Imprime resumo do arquivo consolidado"""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("📊 DADOS CONSOLIDADOS")
        print("="*80)
        
        if not summary['exists']:
            print("\n⚠️  Arquivo consolidado não existe ainda")
            print("\n💡 Execute: consolidator.consolidate()")
        else:
            print(f"\n📄 Arquivo: {summary['file']}")
            print(f"💾 Tamanho: {summary['size_mb']:.2f} MB")
            print(f"\n📊 Dados:")
            print(f"   Registros: {summary['total_records']:,}")
            print(f"   Tickers: {summary['tickers']}")
            print(f"   Janelas: {summary['windows']}")
            print(f"   Período: {summary['date_range']['start']} até {summary['date_range']['end']}")
        
        print("="*80)


def main():
    """Teste do módulo"""
    consolidator = DataConsolidator()
    
    # Consolidar dados
    df = consolidator.consolidate()
    
    # Mostrar resumo
    consolidator.print_summary()
    
    # Exemplo: obter dados de uma janela
    if not df.empty:
        print("\n📊 Exemplo - Janela 1:")
        df_w1 = consolidator.get_window_data(1)
        print(df_w1.head(10))


if __name__ == "__main__":
    main()