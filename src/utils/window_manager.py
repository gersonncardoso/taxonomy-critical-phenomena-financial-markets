"""
Window Manager - Rolling de 12 meses, lag de 1 mês (civil)
=========================================================
Cada janela cobre 12 meses completos.
Janela k: de primeiro dia do mês k até último dia do mês k+11.
Lag de 1 mês entre inícios das janelas.

Autor: Gerson Nassor Cardoso
Instituição: UNIFESP
"""

import pandas as pd
from pathlib import Path
from dateutil.relativedelta import relativedelta
from src.utils.logger import setup_logger
logger = setup_logger('window_manager')

class WindowManager:
    def __init__(self, data_file: str, window_num_months: int = 12):
        self.data_file = Path(data_file)
        self.window_num_months = window_num_months
        self.df = None
        self.month_starts = []
        self.windows = []

    def load_data(self):
        if not self.data_file.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.data_file}")
        self.df = pd.read_csv(self.data_file)
        date_col = next((col for col in self.df.columns if col.lower() in ("date", "data")), None)
        if date_col is None:
            raise ValueError(f"Nenhuma coluna de data encontrada! Colunas: {self.df.columns.tolist()}")
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        if date_col != "Data":
            self.df['Data'] = self.df[date_col]
        self.df = self.df.sort_values('Data').reset_index(drop=True)

    def calculate_windows(self):
        if self.df is None:
            self.load_data()
        min_month = self.df['Data'].min().replace(day=1)
        max_month = self.df['Data'].max().replace(day=1)
        month_starts = []
        curr = min_month
        last_data = self.df['Data'].max()
        while True:
            start = curr
            end = (start + relativedelta(months=self.window_num_months)) - pd.Timedelta(days=1)
            if end > last_data:
                break
            month_starts.append(start)
            curr = curr + relativedelta(months=1)
        self.month_starts = month_starts
        self.windows = []
        for i, start in enumerate(self.month_starts):
            end = (start + relativedelta(months=self.window_num_months)) - pd.Timedelta(days=1)
            # Mantém como date, serializa na saída!
            self.windows.append({'id': i+1, 'start': start.date(), 'end': end.date()})
        self._validate()
        return self.windows

    def _validate(self):
        print(f"\n🔍 Validação rolling de 12 meses:")
        for w in self.windows:
            meses = (w['end'].year - w['start'].year) * 12 + (w['end'].month - w['start'].month) + 1
            if meses != self.window_num_months:
                print(f"⚠️ Janela {w['id']} cobre apenas {meses} meses [{w['start']} -> {w['end']}] (esperado {self.window_num_months})")
        for i in range(1, len(self.windows)):
            prev = self.windows[i-1]['start']
            curr = self.windows[i]['start']
            prev_dt = pd.to_datetime(prev)
            curr_dt = pd.to_datetime(curr)
            delay = (curr_dt.year - prev_dt.year) * 12 + (curr_dt.month - prev_dt.month)
            if delay != 1:
                print(f"⚠️ Lag errado entre janela {i} e {i+1}: {delay} meses (esperado 1)")
        print(f"✅ Total de janelas de 12 meses: {len(self.windows)} (perde o ano mais antigo!)\n")

    def generate_metadata(self, output_file: str = 'data/processed/janelas_metadata.json') -> dict:
        if not self.windows:
            raise ValueError("Janelas não calculadas")
        import json
        from datetime import datetime as dt
        # Serializa cada janela start/end para string (JSON-safe)
        safe_windows = [
            {**w, 'start': str(w['start']), 'end': str(w['end'])}
            for w in self.windows
        ]
        metadata = {
            'timestamp': dt.now().isoformat(),
            'arquivo_origem': str(self.data_file),
            'parametros': {
                'window_num_months': self.window_num_months,
                'rolling_lag_meses': 1,
            },
            'summary': {
                'total_windows': len(safe_windows),
                'data_inicio': str(self.month_starts[0].date()) if self.month_starts else None,
                'data_fim': str(self.df['Data'].max().date())
            },
            'windows': safe_windows
        }
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Metadados salvos: {output_file}")
        return metadata

    def print_summary(self):
        print("="*55)
        print("Rolling 12 meses (lag 1 mês civil)")
        print(f"Total de janelas: {len(self.windows)}")
        if self.month_starts:
            print(f"Período global: {self.month_starts[0].date()} até {self.df['Data'].max().date()}")
            print(f"Primeira janela: {self.windows[-1]['start']} a {self.windows[-1]['end']}")
            print(f"Última janela: {self.windows[0]['start']} a {self.windows[0]['end']}")
        print("="*55)