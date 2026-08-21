#!/usr/bin/env python3
"""
Helpers para manter compatibilidade com versão anterior simples (lista).
Queries na estrutura expandida retornam formato-compatível.
"""

import json
import yaml
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any

class TickerSynonymLookup:
    """Interface compatível com ticker_synonyms.yaml simples"""
    
    def __init__(self, yaml_file='configs/ticker_synonyms_expanded.yaml'):
        with open(yaml_file, encoding='utf-8') as f:
            self.data = yaml.safe_load(f)
        self.tickers = self.data.get('tickers', {})
        self._tickers_by_sigla: Dict[str, List[str]] = {}
        self._window_bounds_cache: Dict[int, Dict[str, str]] = {}
        self._window_tickers_cache: Dict[int, List[str]] = {}
        self._build_sigla_index()

    def _build_sigla_index(self):
        by_sigla: Dict[str, List[str]] = {}
        for ticker, entry in self.tickers.items():
            sigla = str(entry.get('sigla') or '').upper().strip()
            if not sigla:
                continue
            by_sigla.setdefault(sigla, []).append(ticker)
        self._tickers_by_sigla = by_sigla

    def _timeline_for_ticker(self, ticker: str) -> List[Dict[str, str]]:
        entry = self.tickers.get(ticker, {})
        own = entry.get('ceo_timeline', []) or []
        if own:
            return own

        # Fallback: reaproveita timeline de ticker irmão com mesma sigla.
        sigla = str(entry.get('sigla') or '').upper().strip()
        if not sigla:
            return []

        for sibling in self._tickers_by_sigla.get(sigla, []):
            sibling_timeline = self.tickers.get(sibling, {}).get('ceo_timeline', []) or []
            if sibling_timeline:
                return sibling_timeline
        return []
    
    def get_aliases(self, ticker: str, include_legacy=True) -> List[str]:
        """Retorna lista simples de aliases (backward-compatible)"""
        if ticker not in self.tickers:
            return []
        
        entry = self.tickers[ticker]
        
        # Filtrar por tipo se necessário
        if not include_legacy and entry.get('tipo') == 'legacy':
            return []
        
        return entry.get('aliases', [])
    
    def get_full_name(self, ticker: str, lang='pt') -> str:
        """Retorna nome completo em português ou inglês"""
        if ticker not in self.tickers:
            return None
        
        entry = self.tickers[ticker]
        if lang == 'en':
            return entry.get('nome_en')
        else:
            return entry.get('nome_pt')
    
    def get_sigla(self, ticker: str) -> str:
        """Retorna sigla (base 4-letras)"""
        if ticker not in self.tickers:
            return None
        return self.tickers[ticker].get('sigla')
    
    def get_ceo(self, ticker: str, on_date=None) -> Optional[str]:
        """Retorna CEO vigente na data especificada (ou atual se None)"""
        if ticker not in self.tickers:
            return None
        
        ceos = self._timeline_for_ticker(ticker)
        
        if not ceos:
            return None
        
        if on_date is None:
            # Retornar CEO atual (último com data_fim=None ou mais recente)
            active_ceos = [c for c in ceos if c.get('data_fim') is None or c.get('data_fim') > '2026-04-12']
            if active_ceos:
                return active_ceos[-1].get('nome')
            return ceos[-1].get('nome')
        else:
            # Procurar CEO vigente na data
            for ceo in reversed(ceos):
                if ceo['data_inicio'] <= on_date:
                    if ceo.get('data_fim') is None or ceo['data_fim'] > on_date:
                        return ceo['nome']
            return None
    
    def get_ceo_timeline(self, ticker: str) -> List[Dict[str, str]]:
        """Retorna timeline completa de CEOs"""
        if ticker not in self.tickers:
            return []
        return self._timeline_for_ticker(ticker)

    def _load_window_bounds(
        self,
        metadata_file: str = 'data/processed/janelas_metadata.json',
    ) -> Dict[int, Dict[str, str]]:
        if self._window_bounds_cache:
            return self._window_bounds_cache

        p = Path(metadata_file)
        if not p.exists():
            return {}

        obj = json.loads(p.read_text(encoding='utf-8'))
        windows = obj.get('windows', []) if isinstance(obj, dict) else []
        out: Dict[int, Dict[str, str]] = {}
        for w in windows:
            try:
                wid = int(w['id'])
                out[wid] = {
                    'start': str(w['start']),
                    'end': str(w['end']),
                }
            except Exception:
                continue

        self._window_bounds_cache = out
        return out

    def _load_window_tickers(
        self,
        consolidado_file: str = 'data/processed/dados_consolidados.csv',
    ) -> Dict[int, List[str]]:
        if self._window_tickers_cache:
            return self._window_tickers_cache

        p = Path(consolidado_file)
        if not p.exists():
            return {}

        df = pd.read_csv(p, usecols=['Janela_ID', 'Ticker'])
        df['Janela_ID'] = pd.to_numeric(df['Janela_ID'], errors='coerce').astype('Int64')
        df['Ticker'] = df['Ticker'].fillna('').astype(str).str.upper().str.strip()
        df = df.dropna(subset=['Janela_ID'])
        df = df[df['Ticker'] != '']
        df = df.drop_duplicates(['Janela_ID', 'Ticker'])

        by_window: Dict[int, List[str]] = {}
        for wid, group in df.groupby('Janela_ID'):
            by_window[int(wid)] = sorted(group['Ticker'].tolist())

        self._window_tickers_cache = by_window
        return by_window

    def get_window_bounds(self, janela_id: int) -> Optional[Dict[str, str]]:
        bounds = self._load_window_bounds()
        return bounds.get(int(janela_id))

    def get_window_tickers(self, janela_id: int) -> List[str]:
        by_window = self._load_window_tickers()
        return by_window.get(int(janela_id), [])

    def get_window_ceo_map(self, janela_id: int, reference: str = 'end') -> Dict[str, Optional[str]]:
        bounds = self.get_window_bounds(janela_id)
        if not bounds:
            return {}

        if reference == 'start':
            ref_date = bounds['start']
        else:
            ref_date = bounds['end']

        out: Dict[str, Optional[str]] = {}
        for ticker in self.get_window_tickers(janela_id):
            out[ticker] = self.get_ceo(ticker, on_date=ref_date)
        return out

    def build_window_context(self, janela_id: int) -> List[Dict[str, Any]]:
        bounds = self.get_window_bounds(janela_id)
        if not bounds:
            return []

        ws = bounds['start']
        we = bounds['end']
        rows: List[Dict[str, Any]] = []

        for ticker in self.get_window_tickers(janela_id):
            if ticker not in self.tickers:
                continue
            entry = self.tickers[ticker]
            rows.append(
                {
                    'Janela_ID': int(janela_id),
                    'Window_Start': ws,
                    'Window_End': we,
                    'Ticker': ticker,
                    'Sigla': entry.get('sigla'),
                    'Tipo': entry.get('tipo'),
                    'Nome_PT': entry.get('nome_pt'),
                    'Nome_EN': entry.get('nome_en'),
                    'CEO_Window_End': self.get_ceo(ticker, on_date=we),
                    'Aliases_Count': len(entry.get('aliases', []) or []),
                }
            )

        return rows

    def export_window_context_csv(
        self,
        output_file: str = 'data/processed/ticker_context_by_window.csv',
        only_windows: Optional[List[int]] = None,
    ) -> str:
        bounds = self._load_window_bounds()
        if not bounds:
            raise FileNotFoundError('janelas_metadata.json não encontrado ou inválido.')

        all_ids = sorted(bounds.keys())
        target_ids = sorted(set(int(w) for w in only_windows)) if only_windows else all_ids

        rows: List[Dict[str, Any]] = []
        for wid in target_ids:
            rows.extend(self.build_window_context(wid))

        df = pd.DataFrame(rows)
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding='utf-8')
        return str(out)
    
    def search(self, query: str, field='aliases') -> List[str]:
        """Busca tickers por query em campo especificado"""
        query_upper = query.upper()
        matches = []
        
        for ticker, entry in self.tickers.items():
            if field == 'aliases':
                if any(query_upper in alias.upper() for alias in entry.get('aliases', [])):
                    matches.append(ticker)
            elif field == 'nome_pt':
                if query_upper in entry.get('nome_pt', '').upper():
                    matches.append(ticker)
            elif field == 'nome_en':
                if query_upper in entry.get('nome_en', '').upper():
                    matches.append(ticker)
            elif field == 'sigla':
                if query_upper == entry.get('sigla', '').upper():
                    matches.append(ticker)
        
        return matches
    
    def get_all_by_type(self, tipo: str) -> List[str]:
        """Retorna todos os tickers de um tipo (principal, legacy, coligada_sem_ticker)"""
        return [t for t, e in self.tickers.items() if e.get('tipo') == tipo]
    
    def to_simple_dict(self) -> Dict[str, List[str]]:
        """Converte para formato simples (ticker -> aliases) para backward-compatibility"""
        return {ticker: entry['aliases'] for ticker, entry in self.tickers.items()}

# Exemplo de uso:
if __name__ == '__main__':
    lookup = TickerSynonymLookup()
    
    print("=== Exemplos ===")
    print(f"PETR3 aliases: {lookup.get_aliases('PETR3')[:5]}")
    print(f"PETR3 nome_pt: {lookup.get_full_name('PETR3')}")
    print(f"PETR3 nome_en: {lookup.get_full_name('PETR3', lang='en')}")
    print(f"PETR3 CEO atual: {lookup.get_ceo('PETR3')}")
    print(f"PETR3 CEOs: {len(lookup.get_ceo_timeline('PETR3'))} registros")
    
    print(f"\nLegacy tickers: {len(lookup.get_all_by_type('legacy'))}")
    print(f"Principal tickers: {len(lookup.get_all_by_type('principal'))}")
    
    print(f"\nBuscar 'PETROBRAS': {lookup.search('PETROBRAS')[:5]}")
    print(f"Buscar sigla 'PETR': {lookup.search('PETR', field='sigla')[:5]}")
