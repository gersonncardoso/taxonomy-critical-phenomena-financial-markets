"""Export within-pre-event trajectories for the 11-anchor B3 sensitivity calendar.

The analysis is descriptive: every trajectory has six highly overlapping 12-month
rolling windows, ordered by Window_End. It is not an early-warning test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from audit_proxy_11_event_sensitivity import _classify, _load_metrics, _proxy_events

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "validation"

def _semantics(metric: str) -> str:
    if metric.startswith("ricci_fr_"):
        return "Forman–Ricci geometric diagnostic"
    if metric.startswith("dist_"):
        return "raw PMFG distance context"
    if metric in {"modularidade", "num_communities", "community_residual_zscore", "ward_group_residual_zscore"}:
        return "mesoscopic structural diagnostic"
    if metric.startswith("planar_"):
        return "size-conditioned centrality residual"
    if metric == "ks_stat":
        return "correlation-matrix validation diagnostic"
    return "correlation-matrix spectral diagnostic"


def _sign(value: float) -> str:
    if value > 0:
        return "up"
    if value < 0:
        return "down"
    return "flat"


def _turns(values: np.ndarray) -> int:
    signs = np.sign(np.diff(values))
    signs = signs[signs != 0]
    return int((signs[1:] != signs[:-1]).sum()) if len(signs) > 1 else 0


def main() -> None:
    metrics, labels = _load_metrics()
    events = _proxy_events().copy()
    trajectory_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for event in events.itertuples(index=False):
        event_date = pd.to_datetime(event.event_date)
        pre = metrics.loc[_classify(metrics, event_date).eq("pre")].sort_values("Window_End")
        for metric, label in labels.items():
            selected = pre[["Janela_ID", "Window_Start", "Window_End", metric]].dropna().copy()
            if len(selected) != 6:
                raise ValueError(f"Expected six pre windows for {event_date:%Y-%m-%d} and {metric}; found {len(selected)}.")
            values = selected[metric].to_numpy(dtype=float)
            std = values.std(ddof=0)
            selected["relative_position"] = range(1, 7)
            selected["value"] = values
            selected["within_anchor_z"] = 0.0 if std == 0 else (values - values.mean()) / std
            for row in selected.itertuples(index=False):
                trajectory_rows.append(
                    {
                        "event_date": event_date.strftime("%Y-%m-%d"),
                        "status": event.status,
                        "metric": metric,
                        "metric_label": label,
                        "measurement_semantics": _semantics(metric),
                        "relative_position": row.relative_position,
                        "Janela_ID": row.Janela_ID,
                        "Window_Start": row.Window_Start.strftime("%Y-%m-%d"),
                        "Window_End": row.Window_End.strftime("%Y-%m-%d"),
                        "value": row.value,
                        "within_anchor_z": row.within_anchor_z,
                    }
                )
            latest_step = values[-1] - values[-2]
            endpoint_change = values[-1] - values[0]
            summary_rows.append(
                {
                    "event_date": event_date.strftime("%Y-%m-%d"),
                    "status": event.status,
                    "metric": metric,
                    "metric_label": label,
                    "measurement_semantics": _semantics(metric),
                    "pre_windows": len(values),
                    "latest_step": latest_step,
                    "latest_step_direction": _sign(latest_step),
                    "endpoint_change": endpoint_change,
                    "endpoint_direction": _sign(endpoint_change),
                    "turning_points": _turns(values),
                    "relative_range": (values.max() - values.min()) / abs(values.mean()) if values.mean() else np.nan,
                }
            )

    trajectories = pd.DataFrame(trajectory_rows)
    summary = pd.DataFrame(summary_rows)
    recurrence = (
        summary.groupby(["metric", "metric_label", "measurement_semantics", "latest_step_direction"], dropna=False)
        .size()
        .rename("anchors")
        .reset_index()
        .pivot(index=["metric", "metric_label", "measurement_semantics"], columns="latest_step_direction", values="anchors")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    for direction in ("up", "down", "flat"):
        if direction not in recurrence:
            recurrence[direction] = 0
    trigger = summary.loc[summary["status"].eq("trigger_-3sigma")]
    trigger_counts = (
        trigger.groupby(["metric", "latest_step_direction"]).size().unstack(fill_value=0).reindex(columns=["up", "down", "flat"], fill_value=0)
    )
    recurrence = recurrence.merge(trigger_counts.add_prefix("trigger_").reset_index(), on="metric", how="left")
    recurrence["median_turning_points"] = summary.groupby("metric")["turning_points"].median().to_numpy()
    recurrence["median_relative_range"] = summary.groupby("metric")["relative_range"].median().to_numpy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(OUT_DIR / "proxy_11_pre_event_trajectories.csv", index=False)
    summary.to_csv(OUT_DIR / "proxy_11_pre_event_trajectory_summary.csv", index=False)
    recurrence.to_csv(OUT_DIR / "proxy_11_pre_event_latest_step_recurrence.csv", index=False)
    print(f"Wrote {len(trajectories)} trajectory rows, {len(summary)} anchor-metric summaries, and {len(recurrence)} recurrence rows.")


if __name__ == "__main__":
    main()
