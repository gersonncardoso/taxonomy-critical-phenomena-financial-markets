"""Plot descriptive within-pre-event trajectories for all 25 phase metrics."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "figures" / "validation"
PAPER_FIGURES = ROOT / "paper" / "paper1" / "figures" / "validation"
OUT_STEM = "proxy_11_pre_event_trajectories"

def _render_page(
    data: pd.DataFrame,
    metrics: list[tuple[str, str, str]],
    page_number: int,
    total_pages: int,
) -> None:
    ncols = 3
    nrows = 3
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.6, 13.8), sharex=True, sharey=True)
    trigger_color = "#1f77b4"
    fallback_color = "#9aa0a6"
    for axis, (metric, label, semantics) in zip(axes.flat, metrics):
        panel = data.loc[data["metric"].eq(metric)]
        values = panel["within_anchor_z"].dropna()
        is_constant = values.empty or (values.max() - values.min() <= 1e-12)
        if is_constant:
            constant_value = 0.0 if values.empty else float(values.iloc[0])
            axis.axhline(constant_value, color="#6B7280", linewidth=1.1, linestyle="-")
            axis.text(
                0.5,
                0.52,
                f"constant at {constant_value:g}\nno within-anchor variation",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=8.5,
                color="#374151",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9},
            )
        else:
            for event in panel["event_date"].drop_duplicates():
                series = panel.loc[panel["event_date"].eq(event)].sort_values("relative_position")
                is_trigger = series["status"].iloc[0] == "trigger_-3sigma"
                axis.plot(
                    series["relative_position"],
                    series["within_anchor_z"],
                    color=trigger_color if is_trigger else fallback_color,
                    linewidth=2.0 if is_trigger else 1.1,
                    alpha=0.95 if is_trigger else 0.65,
                    marker="o" if is_trigger else None,
                    markersize=3.8,
                )
        axis.axhline(0, color="#495057", linewidth=0.8, linestyle=":")
        axis.axvspan(5.85, 6.15, color="#d62728", alpha=0.08)
        axis.set_title(label.replace(" (raw context)", "").replace(" (size-adjusted)", ""), loc="left", fontsize=9, pad=5, wrap=True)
        axis.grid(axis="y", alpha=0.20)
        axis.set_xticks(range(1, 7))
        axis.set_xlim(0.85, 6.15)

    for axis in axes[:, 0]:
        axis.set_ylabel("Within-anchor standard score")
    for axis in axes[-1, :]:
        axis.set_xlabel("Pre-event window order\n1 = oldest; 6 = latest")

    for row_index, row in enumerate(axes):
        for column_index, axis in enumerate(row):
            if row_index < nrows - 1:
                axis.tick_params(labelbottom=False)
            if column_index > 0:
                axis.tick_params(labelleft=False)

    for axis in axes.flat[len(metrics):]:
        axis.set_visible(False)

    fig.legend(
        handles=[
            Line2D([0], [0], color=trigger_color, marker="o", linewidth=1.8, label="Actual $-3\\sigma$ trigger (n=5)"),
            Line2D([0], [0], color=fallback_color, linewidth=1.1, label="Fallback local-drop proxy (n=6)"),
            Line2D([0], [0], color="#d62728", linewidth=5, alpha=0.18, label="Latest pre-event window"),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.952),
        fontsize=8.5,
    )
    fig.suptitle(
        f"Exploratory within-pre-event trajectories ({page_number}/{total_pages})",
        fontsize=13,
        y=0.992,
    )
    fig.text(
        0.5,
        0.025,
        "Each trajectory contains six monthly 12-month rolling windows. Values are standardized within anchor and metric solely to compare trajectory shape; "
        "adjacent windows and some anchor periods overlap. The display is descriptive and is not an early-warning test.",
        ha="center",
        va="bottom",
        fontsize=8.0,
        wrap=True,
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.10, hspace=0.62, wspace=0.28)
    VALIDATION.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        output = VALIDATION / f"{OUT_STEM}_part{page_number}.{extension}"
        fig.savefig(output, dpi=300, facecolor="white")
        shutil.copy2(output, PAPER_FIGURES / output.name)
    plt.close(fig)


def main() -> None:
    data = pd.read_csv(VALIDATION / "proxy_11_pre_event_trajectories.csv")
    data["event_date"] = pd.to_datetime(data["event_date"])
    data = data.sort_values(["event_date", "relative_position"])
    metric_info = data[["metric", "metric_label", "measurement_semantics"]].drop_duplicates().sort_values("metric")
    metrics = list(metric_info.itertuples(index=False, name=None))
    page_size = 9
    pages = [metrics[start : start + page_size] for start in range(0, len(metrics), page_size)]
    for page_number, page_metrics in enumerate(pages, start=1):
        _render_page(data, page_metrics, page_number, len(pages))
    print(f"Wrote {len(pages)} pre-event trajectory pages covering {len(metrics)} metrics.")


if __name__ == "__main__":
    main()
