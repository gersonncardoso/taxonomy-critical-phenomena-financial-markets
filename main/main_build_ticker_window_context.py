#!/usr/bin/env python3
"""Gera contexto de tickers/CEOs por janela temporal.

Saida principal:
- data/processed/ticker_context_by_window.csv

Uso:
  python main/main_build_ticker_window_context.py
  python main/main_build_ticker_window_context.py 362 363
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ticker_helpers import TickerSynonymLookup


def _parse_windows(argv):
    out = []
    for token in argv:
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out


def main():
    selected = _parse_windows(sys.argv[1:])
    lookup = TickerSynonymLookup('configs/ticker_synonyms_expanded.yaml')

    out_csv = lookup.export_window_context_csv(
        output_file='data/processed/ticker_context_by_window.csv',
        only_windows=selected if selected else None,
    )

    print(f"[OK] Contexto por janela salvo em: {out_csv}")

    if selected:
        sample_ids = selected
    else:
        # Usa as últimas 2 janelas disponíveis (independente de ID fixo)
        all_bounds = lookup._load_window_bounds()
        sample_ids = sorted(all_bounds.keys())[-2:] if len(all_bounds) >= 2 else sorted(all_bounds.keys())

    for wid in sample_ids:
        bounds = lookup.get_window_bounds(wid)
        if bounds is None:
            print(f"[JANELA {wid}] não encontrada nos metadados — pulando")
            continue
        tickers = lookup.get_window_tickers(wid)
        ceo_map = lookup.get_window_ceo_map(wid, reference='end')
        with_ceo = sum(1 for v in ceo_map.values() if v)
        print(
            f"[JANELA {wid}] {bounds['start']} -> {bounds['end']} | "
            f"tickers={len(tickers)} | tickers_com_ceo={with_ceo}"
        )


if __name__ == '__main__':
    main()
