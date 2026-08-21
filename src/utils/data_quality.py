import pandas as pd
from pathlib import Path
from typing import Optional
import holidays

from .logger import setup_logger
logger = setup_logger("data_quality_fast")


class DataQualityChecker:
    """Data quality focado em cobertura de dias de pregao.

    Prioriza dias observados em um arquivo de referencia (raw filtrado/completo),
    evitando falsos positivos de calendario generico.
    """

    def __init__(self, data_file: str, reference_file: Optional[str] = None):
        self.data_file = Path(data_file)
        self.reference_file = Path(reference_file) if reference_file else None
        self.df = None
        self.ref_df = None
        self.date_col = None
        self.ref_date_col = None
        self.monthly_df = None

    @staticmethod
    def _detect_date_col(columns):
        for c in columns:
            if c.lower() in ["date", "data"]:
                return c
        return None

    @staticmethod
    def _to_date_set(series):
        return set(pd.to_datetime(series).dt.date.unique())

    def load_data(self) -> bool:
        try:
            if not self.data_file.exists():
                logger.error(f"Arquivo nao encontrado: {self.data_file}")
                return False

            self.df = pd.read_csv(self.data_file)
            self.date_col = self._detect_date_col(self.df.columns)
            if not self.date_col:
                logger.error("Nenhuma coluna de data encontrada")
                return False

            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

            # Arquivo de referencia: dias em que a bolsa abriu dentro do universo
            # de dados utilizado pela fase 1.
            if self.reference_file and self.reference_file.exists():
                self.ref_df = pd.read_csv(self.reference_file)
                self.ref_date_col = self._detect_date_col(self.ref_df.columns)
                if self.ref_date_col:
                    self.ref_df[self.ref_date_col] = pd.to_datetime(self.ref_df[self.ref_date_col])
                else:
                    logger.warning(
                        f"Arquivo de referencia sem coluna de data: {self.reference_file}"
                    )
                    self.ref_df = None

            return True

        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            return False

    def verificar_datas_uteis(self):
        actual_dates = sorted(self._to_date_set(self.df[self.date_col].dropna()))
        if not actual_dates:
            return {
                "mode": "empty",
                "total_dias_esperados": 0,
                "total_dias_presentes": 0,
                "total_dias_uteis_faltantes": 0,
                "dias_faltantes": [],
                "source": "none",
            }

        start = actual_dates[0]
        end = actual_dates[-1]

        if self.ref_df is not None and self.ref_date_col is not None:
            ref_dates_all = self._to_date_set(self.ref_df[self.ref_date_col].dropna())
            # Compara no mesmo intervalo do consolidado para nao penalizar
            # dias apos o recorte de janela/lag.
            expected_dates = sorted([d for d in ref_dates_all if start <= d <= end])
            source = str(self.reference_file)
            mode = "reference_file"
        else:
            br_holidays = holidays.Brazil(years=range(start.year, end.year + 1))
            expected_dates = []
            for dia in pd.date_range(start, end, freq="D"):
                d = dia.date()
                if dia.weekday() >= 5:
                    continue
                if d in br_holidays:
                    continue
                expected_dates.append(d)
            source = "calendar_fallback"
            mode = "calendar"

        expected_set = set(expected_dates)
        actual_set = set(actual_dates)
        missing = sorted(expected_set - actual_set)

        # Consolidado mensal para acompanhamento de cobertura
        monthly = pd.DataFrame({"date": pd.to_datetime(expected_dates)})
        if not monthly.empty:
            monthly["year_month"] = monthly["date"].dt.to_period("M").astype(str)
            monthly["present"] = monthly["date"].dt.date.isin(actual_set)
            monthly_df = (
                monthly.groupby("year_month", as_index=False)
                .agg(
                    expected_days=("date", "count"),
                    present_days=("present", "sum"),
                )
            )
            monthly_df["missing_days"] = (
                monthly_df["expected_days"] - monthly_df["present_days"]
            )
            monthly_df["coverage_pct"] = (
                monthly_df["present_days"] / monthly_df["expected_days"] * 100.0
            )
        else:
            monthly_df = pd.DataFrame(
                columns=["year_month", "expected_days", "present_days", "missing_days", "coverage_pct"]
            )
        self.monthly_df = monthly_df

        return {
            "mode": mode,
            "total_dias_esperados": len(expected_dates),
            "total_dias_presentes": len(actual_dates),
            "total_dias_uteis_faltantes": len(missing),
            "dias_faltantes": [str(d) for d in missing],
            "source": source,
            "periodo": (str(start), str(end)),
        }

    def save_monthly_report(self, output_file: str):
        if self.monthly_df is None:
            self.verificar_datas_uteis()
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.monthly_df.to_csv(out, index=False)
        return out

    def print_summary(self):
        r = self.verificar_datas_uteis()
        print("\n" + "="*80)
        print("VERIFICACAO DE DIAS DE PREGAO")
        print("="*80)
        if r.get("mode") == "empty":
            print("Base consolidada sem datas para verificar.")
            print("\n" + "="*80)
            return

        print(f"\nFonte dias esperados: {r['source']}")
        print(f"Periodo analisado: {r['periodo'][0]} ate {r['periodo'][1]}")
        print(f"Dias esperados: {r['total_dias_esperados']}")
        print(f"Dias presentes: {r['total_dias_presentes']}")
        print(f"Dias faltantes: {r['total_dias_uteis_faltantes']}")

        if r["total_dias_uteis_faltantes"] > 0:
            print("\nPrimeiros 20 dias faltantes:")
            for d in r["dias_faltantes"][:20]:
                print(f" - {d}")

        if self.monthly_df is not None and not self.monthly_df.empty:
            print("\nConsolidado mensal (ultimos 12 meses):")
            tail = self.monthly_df.tail(12).copy()
            print(
                tail.to_string(
                    index=False,
                    formatters={"coverage_pct": lambda x: f"{x:.2f}%"},
                )
            )
        print("\n" + "="*80)
