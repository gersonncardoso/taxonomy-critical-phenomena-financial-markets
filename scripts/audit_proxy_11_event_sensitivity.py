"""Assess whether proxy stress anchors reproduce the configured-event regime taxonomy."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "validation"
PHASE_ORDER = ["tranquil", "pre", "during", "post"]


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("Window_Start", "Window_End"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _load_metrics() -> tuple[pd.DataFrame, dict[str, str]]:
    network = _read(ROOT / "data/processed/network_metrics_planar_long.csv")
    centrality = _read(ROOT / "data/processed/planar_centrality_size_adjusted_by_window.csv")
    planar_groups = _read(ROOT / "data/processed/planar_group_size_null_by_window.csv")
    ward_groups = _read(ROOT / "data/processed/ward_group_size_null_by_window.csv")
    ks = _read(OUT_DIR / "ks_results.csv")
    mp = _read(OUT_DIR / "mp_results.csv")

    merged = network[[
        "Janela_ID", "Window_Start", "Window_End",
        "dist_mean", "dist_var", "dist_skew", "dist_kurt",
        "modularidade", "num_communities",
        "ricci_fr_mean", "ricci_fr_var", "ricci_fr_std", "ricci_fr_skew",
        "ricci_fr_kurt", "ricci_fr_q10", "ricci_fr_q50", "ricci_fr_q90",
        "ricci_fr_iqr", "ricci_fr_min", "ricci_fr_max",
    ]].copy()
    sources = [
        centrality[["Janela_ID", "planar_betweenness_centrality_mean_residual_size_only", "planar_closeness_centrality_mean_residual_size_only", "planar_eigenvector_centrality_mean_residual_size_only"]],
        planar_groups[["Janela_ID", "community_residual_zscore"]],
        ward_groups[["Janela_ID", "ward_group_residual_zscore"]],
        ks[["Janela_ID", "ks_stat"]],
        mp[["Janela_ID", "var_signal_frac", "market_mode_frac"]],
    ]
    for source in sources:
        merged = merged.merge(source, on="Janela_ID", how="left", validate="one_to_one")

    labels = {
        "dist_mean": "PMFG mean distance (raw context)",
        "dist_var": "PMFG distance variance (raw context)",
        "dist_skew": "PMFG distance skewness (raw context)",
        "dist_kurt": "PMFG distance kurtosis (raw context)",
        "modularidade": "PMFG modularity (raw context)",
        "num_communities": "PMFG communities (raw context)",
        "ricci_fr_mean": "Forman–Ricci mean curvature (raw context)",
        "ricci_fr_var": "Forman–Ricci curvature variance (raw context)",
        "ricci_fr_std": "Forman–Ricci curvature standard deviation (raw context)",
        "ricci_fr_skew": "Forman–Ricci curvature skewness (raw context)",
        "ricci_fr_kurt": "Forman–Ricci curvature kurtosis (raw context)",
        "ricci_fr_q10": "Forman–Ricci curvature 10th percentile (raw context)",
        "ricci_fr_q50": "Forman–Ricci curvature median (raw context)",
        "ricci_fr_q90": "Forman–Ricci curvature 90th percentile (raw context)",
        "ricci_fr_iqr": "Forman–Ricci curvature interquartile range (raw context)",
        "ricci_fr_min": "Forman–Ricci minimum curvature (raw context)",
        "ricci_fr_max": "Forman–Ricci maximum curvature (raw context)",
        "planar_betweenness_centrality_mean_residual_size_only": "Betweenness residual (size-adjusted)",
        "planar_closeness_centrality_mean_residual_size_only": "Closeness residual (size-adjusted)",
        "planar_eigenvector_centrality_mean_residual_size_only": "Eigenvector residual (size-adjusted)",
        "community_residual_zscore": "PMFG community residual z-score",
        "ward_group_residual_zscore": "Ward group residual z-score",
        "ks_stat": "KS Gaussian-null statistic",
        "var_signal_frac": "MP signal-variance fraction",
        "market_mode_frac": "MP market-mode fraction",
    }
    return merged, labels


def _configured_events() -> pd.DataFrame:
    with (ROOT / "configs/config.yaml").open(encoding="utf-8") as handle:
        periods = (yaml.safe_load(handle) or {}).get("eventos", {}).get("periodos_crash", [])
    return pd.DataFrame(
        {
            "event_date": [item["shock_start"] for item in periods],
            "event_set": "configured_4",
            "status": "confirmed_configured",
        }
    )


def _proxy_events() -> pd.DataFrame:
    events = pd.read_csv(OUT_DIR / "crisis_dates_3sigma_proxy_b3.csv")
    return events.assign(event_date=events["shock_start"], event_set="proxy_11")[
        ["event_date", "event_set", "status", "crisis_date", "shock_z", "max_drawdown", "recovered"]
    ]


def _classify(frame: pd.DataFrame, event_date: pd.Timestamp) -> pd.Series:
    phase = pd.Series("tranquil", index=frame.index, dtype="object")
    phase.loc[(frame["Window_End"] < event_date) & (frame["Window_End"] >= event_date - pd.DateOffset(months=6))] = "pre"
    phase.loc[(frame["Window_Start"] <= event_date) & (frame["Window_End"] >= event_date)] = "during"
    phase.loc[(frame["Window_Start"] > event_date) & (frame["Window_Start"] <= event_date + pd.DateOffset(months=6))] = "post"
    return phase


def _summarize(events: pd.DataFrame, metrics: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_columns = list(labels)
    for event in events.itertuples(index=False):
        event_date = pd.to_datetime(event.event_date)
        working = metrics.copy()
        working["phase"] = _classify(working, event_date)
        for metric in metric_columns:
            means = working.groupby("phase")[metric].mean()
            counts = working.groupby("phase")[metric].count()
            pre = means.get("pre")
            for phase in PHASE_ORDER:
                value = means.get(phase)
                rows.append(
                    {
                        "event_set": event.event_set,
                        "event_date": event_date.strftime("%Y-%m-%d"),
                        "status": getattr(event, "status", "confirmed_configured"),
                        "metric": metric,
                        "metric_label": labels[metric],
                        "phase": phase,
                        "windows": int(counts.get(phase, 0)),
                        "mean": value,
                        "change_from_pre": value - pre if pd.notna(value) and pd.notna(pre) else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def _concordance(detail: pd.DataFrame) -> pd.DataFrame:
    changes = detail[detail["phase"].isin(["during", "post"])].copy()
    changes["sign"] = changes["change_from_pre"].apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
    summary = (
        changes.groupby(["event_set", "status", "metric", "metric_label", "phase"], dropna=False)
        .agg(events=("event_date", "nunique"), positive=("sign", lambda x: int((x > 0).sum())), negative=("sign", lambda x: int((x < 0).sum())), median_change=("change_from_pre", "median"), mean_change=("change_from_pre", "mean"))
        .reset_index()
    )
    summary["dominant_direction"] = summary.apply(
        lambda row: "positive" if row.positive > row.negative else ("negative" if row.negative > row.positive else "mixed"), axis=1
    )
    summary["concordance_share"] = summary[["positive", "negative"]].max(axis=1) / summary["events"]
    return summary


def main() -> None:
    metrics, labels = _load_metrics()
    events = pd.concat([_configured_events(), _proxy_events()], ignore_index=True, sort=False)
    detail = _summarize(events, metrics, labels)
    concordance = _concordance(detail)
    detail.to_csv(OUT_DIR / "proxy_11_phase_sensitivity_detail.csv", index=False)
    concordance.to_csv(OUT_DIR / "proxy_11_phase_sensitivity_concordance.csv", index=False)
    print(f"Wrote {len(detail)} phase rows and {len(concordance)} concordance rows.")


if __name__ == "__main__":
    main()
