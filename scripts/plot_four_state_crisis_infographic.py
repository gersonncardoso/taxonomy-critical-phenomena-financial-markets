"""Render four-event phase profiles using the pipeline's observed phase means."""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DETAIL_PATH = ROOT / "figures" / "validation" / "proxy_11_phase_sensitivity_detail.csv"
PRE_SUMMARY_PATH = ROOT / "figures" / "validation" / "proxy_11_pre_event_trajectory_summary.csv"
OUTPUT_DIR = ROOT / "paper" / "paper1" / "figures" / "infographics"
EVENTS = [("1997-10-27", "Asian crisis", "−11.56%"), ("1998-08-27", "Russian default / LTCM", "−8.44%"), ("2008-09-29", "Global financial crisis", "−8.74%"), ("2020-03-09", "COVID-19", "−12.54%")]
PHASES = ["tranquil", "pre", "during", "post"]
PHASE_LABELS = ["1  Tranquil\nreference", "2  Local\npre-event", "3  Event-containing\nstate", "4  Post-event\nreorganization"]
PHASE_COLORS = ["#B9D7EA", "#F6D98B", "#E9A7A7", "#B8D8B8"]
FOCAL_METRICS = [("dist_mean", "Mean PMFG\ndistance", "Raw PMFG geometry", ".3f"), ("dist_var", "Distance\nvariance", "Raw PMFG geometry", ".3f"), ("ricci_fr_mean", "Mean Forman--Ricci\ncurvature", "Geometric diagnostic", ".3f"), ("ks_stat", "KS\nstatistic", "Matrix validation", ".3f"), ("var_signal_frac", "MP signal\nfraction", "Spectral diagnostic", ".3f")]


def draw_phase_profile(axis: plt.Axes, rows: pd.DataFrame, latest_step: float | None, title: str, kind: str, spec: str, labels: bool, header: bool) -> None:
    values = [float(rows.loc[p, "mean"]) for p in PHASES]
    low, high = min(values), max(values)
    pad = max((high - low) * .23, max(abs(high), .01) * .055, .002)
    for i, color in enumerate(PHASE_COLORS): axis.axvspan(i - .46, i + .46, color=color, alpha=.38, zorder=0)
    axis.axvline(1.5, color="#A61B1B", linewidth=1.2, linestyle="--", zorder=1)
    axis.plot(range(4), values, color="#263238", linewidth=2.1, zorder=2)
    axis.scatter(range(4), values, s=48, color=PHASE_COLORS, edgecolor="#263238", linewidth=.9, zorder=3)
    for i, value in enumerate(values):
        offset = pad * (.48 if value < (low + high) / 2 else -.48)
        axis.text(i, value + offset, format(value, spec), fontsize=7.4, ha="center", va="bottom" if offset > 0 else "top", weight="bold", color="#1F2937", bbox={"facecolor":"white", "edgecolor":"none", "alpha":.72, "pad":.35})
    if latest_step is not None:
        axis.text(1, .04, f"5→6: {'+' if latest_step > 0 else ''}{format(latest_step, spec)}", transform=axis.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.7, color="#7C4A03", weight="bold")
    axis.set(xlim=(-.52, 3.52), ylim=(low - pad, high + pad))
    if header: axis.set_title(f"{title}\n{kind}", fontsize=9, weight="bold", pad=11)
    axis.tick_params(axis="y", labelsize=6.7, length=2, pad=1); axis.grid(axis="y", alpha=.24, linewidth=.55)
    axis.spines[["top", "right"]].set_visible(False); axis.spines[["left", "bottom"]].set_color("#9CA3AF")
    axis.set_xticks(range(4), PHASE_LABELS if labels else [], fontsize=6.5)


def main() -> None:
    detail = pd.read_csv(DETAIL_PATH); detail["event_date"] = pd.to_datetime(detail["event_date"]).dt.strftime("%Y-%m-%d"); detail = detail[detail.event_set.eq("configured_4")]
    pre = pd.read_csv(PRE_SUMMARY_PATH); pre["event_date"] = pd.to_datetime(pre["event_date"]).dt.strftime("%Y-%m-%d")
    fig, axes = plt.subplots(4, 5, figsize=(17.5, 11.5)); fig.subplots_adjust(left=.14, right=.99, top=.80, bottom=.12, hspace=.92, wspace=.42)
    fig.suptitle("Four observed phase states around configured B3 stress events", fontsize=19, weight="bold", y=.975)
    fig.text(.5, .944, "Every microchart contains four actual phase means: (1) tranquil reference, (2) local pre-event, (3) event-containing, and (4) post-event reorganization.", ha="center", fontsize=9.6)
    fig.text(.5, .918, "Dashed red divider = externally observed shock between pre-event and crisis. Amber label under pre-event = final within-pre step (window 5→6), not a phase mean.", ha="center", fontsize=8.5, color="#4B5563")
    fig.text(.5, .892, "Scales are metric- and event-specific to preserve observed magnitudes; compare trajectories within a microchart, not vertical positions across different metrics.", ha="center", fontsize=8.2, style="italic", color="#4B5563")
    for i, (date, name, shock) in enumerate(EVENTS):
        event = detail[detail.event_date.eq(date)]; latest = pre[pre.event_date.eq(date)].set_index("metric"); row = axes[i]
        row[0].text(-.60, .50, f"{name}\n{date}\nshock {shock}", transform=row[0].transAxes, ha="right", va="center", fontsize=9.7, weight="bold", color="#1F2937")
        for axis, (metric, title, kind, spec) in zip(row, FOCAL_METRICS, strict=True):
            latest_step = float(latest.loc[metric, "latest_step"]) if metric in latest.index else None
            draw_phase_profile(axis, event[event.metric.eq(metric)].set_index("phase"), latest_step, title, kind, spec, i == 3, i == 0)
    fig.text(.5, .032, "Five recurrent focal indicators: mean PMFG distance, distance variance, mean Forman--Ricci curvature, KS Gaussian-null statistic, and MP signal-variance fraction. The phase-concordance table retains the complete 13-metric evidence, including event-specific centrality results. This retrospective display is descriptive, not causal or predictive.", ha="center", fontsize=8.3, wrap=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"): fig.savefig(OUTPUT_DIR / f"four_state_crisis_infographic.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__": main()
