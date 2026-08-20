"""Plot four observed phase states for all metrics and 11 sensitivity anchors."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DETAIL_PATH = ROOT / "figures" / "validation" / "proxy_11_phase_sensitivity_detail.csv"
OUTPUT_DIR = ROOT / "paper" / "paper1" / "figures" / "validation"

TRIGGER_COLOR = "#1D4ED8"
CORE_TRIGGER_COLOR = "#0072B2"
ADDITIONAL_TRIGGER_COLOR = "#D55E00"
FALLBACK_COLOR = "#6B7280"
PHASES = ["tranquil", "pre", "during", "post"]
PHASE_COLORS = ["#B9D7EA", "#F6D98B", "#E9A7A7", "#B8D8B8"]
PHASE_OFFSETS = [-0.30, -0.10, 0.10, 0.30]
CORE_TRIGGER_DATES = {
    pd.Timestamp("1997-10-27"),
    pd.Timestamp("1998-08-27"),
    pd.Timestamp("2008-09-29"),
    pd.Timestamp("2020-03-09"),
}


def _metric_class(metric: str) -> str:
    if metric.startswith("ricci_fr_"):
        return "Forman–Ricci geometry"
    if metric.startswith("dist_"):
        return "Raw distance context"
    if metric in {"modularidade", "num_communities", "community_residual_zscore", "ward_group_residual_zscore"}:
        return "Mesoscopic structure"
    if metric.startswith("planar_"):
        return "Size-conditioned centrality"
    if metric == "ks_stat":
        return "Matrix validation"
    return "Spectral diagnostic"


def _render_atlas(
    detail: pd.DataFrame,
    metrics: list[tuple[str, str, str]],
    stem: str,
    title: str,
    page_number: int,
    total_pages: int,
) -> None:
    dates = detail[["event_date", "status"]].drop_duplicates().sort_values("event_date").reset_index(drop=True)
    date_positions = {date: index for index, date in enumerate(dates["event_date"])}

    fig, axes = plt.subplots(len(metrics), 1, figsize=(16.5, 1.35 * len(metrics) + 2.8), sharex=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.21, right=0.91, top=0.86, bottom=0.16, hspace=0.52)
    fig.suptitle(f"{title} ({page_number}/{total_pages})", fontsize=15, weight="bold", y=0.975)
    fig.text(
        0.5,
        0.948,
        "Each anchor contains four observed means: tranquil reference, local pre-event, event-containing, and post-event. "
        "Within each metric row, values are min--max scaled across its 44 anchor-by-phase means.",
        ha="center",
        fontsize=9.0,
    )
    fig.text(
        0.5,
        0.926,
        "Solid blue circles identify the four configured core events; orange diamonds identify the additional 1999 −3σ trigger; "
        "gray squares identify the six fallback largest-local-drop sensitivity proxies. Lines join phase means within an anchor only.",
        ha="center",
        fontsize=8.7,
        color="#374151",
    )

    for axis, (metric, metric_label, metric_class) in zip(axes, metrics, strict=True):
        rows = detail[detail["metric"].eq(metric)].copy()
        rows["position"] = rows["event_date"].map(date_positions)
        low, high = rows["mean"].min(), rows["mean"].max()
        scale = high - low
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        rows["scaled_mean"] = (rows["mean"] - low) / scale
        axis.axhline(0.5, color="#9CA3AF", linewidth=0.65, linestyle="--", zorder=0)
        for _, event in dates.iterrows():
            event_rows = rows[rows["event_date"].eq(event["event_date"])]
            if len(event_rows) != len(PHASES):
                continue
            x = date_positions[event["event_date"]]
            positions = [x + offset for offset in PHASE_OFFSETS]
            values = [float(event_rows.loc[event_rows["phase"].eq(phase), "scaled_mean"].iloc[0]) for phase in PHASES]
            event_date = pd.Timestamp(event["event_date"])
            if event["status"] == "trigger_-3sigma" and event_date in CORE_TRIGGER_DATES:
                outline, linestyle, marker = CORE_TRIGGER_COLOR, "-", "o"
            elif event["status"] == "trigger_-3sigma":
                outline, linestyle, marker = ADDITIONAL_TRIGGER_COLOR, "-.", "D"
            else:
                outline, linestyle, marker = FALLBACK_COLOR, "--", "s"
            axis.plot(positions, values, color=outline, alpha=0.75, linewidth=1.15, linestyle=linestyle, zorder=1)
            for position, value, phase_color in zip(positions, values, PHASE_COLORS, strict=True):
                axis.scatter(position, value, s=30, marker=marker, facecolor=phase_color, edgecolor=outline, linewidth=0.95, zorder=2)
        axis.set_xlim(-0.55, len(dates) - 0.45)
        axis.set_ylim(-0.10, 1.10)
        axis.set_yticks([0, 0.5, 1], ["low", "mid", "high"], fontsize=6.8)
        display_label = metric_label.replace(" (size-adjusted)", "")
        display_label = textwrap.fill(display_label, width=28)
        axis.set_ylabel(display_label, fontsize=8.0, weight="bold", rotation=0, ha="right", va="center", labelpad=18)
        axis.text(1.02, 0.74, metric_class, transform=axis.transAxes, fontsize=6.5, color="#4B5563", va="center")
        axis.text(1.02, 0.23, f"range={low:.3g}…{high:.3g}", transform=axis.transAxes, fontsize=6.5, color="#4B5563", va="center")
        axis.grid(axis="y", alpha=0.2, linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#9CA3AF")
        axis.tick_params(axis="x", length=2, labelsize=7)

    labels = []
    for date, status in dates[["event_date", "status"]].itertuples(index=False, name=None):
        date = pd.Timestamp(date)
        suffix = "C" if date in CORE_TRIGGER_DATES else "A" if status == "trigger_-3sigma" else "F"
        labels.append(date.strftime("%Y\n%m") + suffix)
    axes[-1].set_xticks(range(len(dates)), labels)
    axes[-1].set_xlabel("Anchor date (C = configured core; A = additional 1999 trigger; F = fallback proxy)", fontsize=8.3)
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=PHASE_COLORS[0], markeredgecolor="#374151", label="Tranquil", markersize=7),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=PHASE_COLORS[1], markeredgecolor="#374151", label="Pre-event", markersize=7),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=PHASE_COLORS[2], markeredgecolor="#374151", label="Event-containing", markersize=7),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=PHASE_COLORS[3], markeredgecolor="#374151", label="Post-event", markersize=7),
            Line2D([0], [0], color=CORE_TRIGGER_COLOR, marker="o", label="Four configured core events", linewidth=1.3, markersize=6),
            Line2D([0], [0], color=ADDITIONAL_TRIGGER_COLOR, marker="D", linestyle="-.", label="Additional 1999 −3σ trigger", linewidth=1.3, markersize=5),
            Line2D([0], [0], color=FALLBACK_COLOR, marker="s", linestyle="--", label="Six fallback proxies", linewidth=1.3, markersize=6),
        ],
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=7.5,
        bbox_to_anchor=(0.5, 0.070),
    )
    fig.text(
        0.5,
        0.016,
        "Each metric is independently min--max scaled across its 44 observed phase means; compare profiles only within a row. "
        "The tranquil reference excludes local phase windows; pre, event-containing, and post contain 6, 12, and 6 rolling windows. Overlap prevents independent-event inference; this sensitivity atlas is descriptive, not causal or predictive.",
        ha="center",
        fontsize=7.7,
        wrap=True,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"{stem}_part{page_number}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    detail = pd.read_csv(DETAIL_PATH)
    detail = detail[detail["event_set"].eq("proxy_11")].copy()
    detail["event_date"] = pd.to_datetime(detail["event_date"])
    detail = detail[detail["phase"].isin(PHASES)]
    metric_order = detail[["metric", "metric_label"]].drop_duplicates().sort_values("metric")
    metrics = [(row.metric, row.metric_label.replace(" (raw context)", ""), _metric_class(row.metric)) for row in metric_order.itertuples(index=False)]
    non_ricci = [metric for metric in metrics if not metric[0].startswith("ricci_fr_")]
    ricci = [metric for metric in metrics if metric[0].startswith("ricci_fr_")]
    non_ricci_pages = [non_ricci[start : start + 7] for start in range(0, len(non_ricci), 7)]
    ricci_pages = [ricci[start : start + 6] for start in range(0, len(ricci), 6)]
    for page_number, page_metrics in enumerate(non_ricci_pages, start=1):
        _render_atlas(
            detail,
            page_metrics,
            "proxy_11_non_ricci_metric_contrasts",
            "Four-state atlas: non-Ricci metrics across 11 B3 anchors",
            page_number,
            len(non_ricci_pages),
        )
    for page_number, page_metrics in enumerate(ricci_pages, start=1):
        _render_atlas(
            detail,
            page_metrics,
            "proxy_11_ricci_metric_contrasts",
            "Four-state atlas: Forman–Ricci metrics across 11 B3 anchors",
            page_number,
            len(ricci_pages),
        )


if __name__ == "__main__":
    main()
