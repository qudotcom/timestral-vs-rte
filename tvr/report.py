"""Write results/summary.{csv,md} and results/backtest.png from a scored run."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .store import Store  # noqa: E402

LABELS = {
    "timestral": "Timestral-28M (zero-shot)",
    "timestral_calendar": "Timestral-28M + calendar covariates",
    "rte_j1": "RTE official day-ahead (J-1)",
    "naive_weekly": "Naive: same hour last week",
    "naive_2day": "Naive: same hour 2 days before",
}
COLORS = {"timestral": "#FA520F", "timestral_calendar": "#C2410C", "rte_j1": "#2563EB",
          "naive_weekly": "#9CA3AF", "naive_2day": "#D1D5DB"}


def markdown_table(s: pd.DataFrame) -> str:
    def pct(x):
        return "—" if pd.isna(x) else f"{100 * x:.0f}%"
    lines = ["| Model | MAE (MW) | MAPE | MASE vs weekly naive | 10–90% interval coverage | Days better than RTE J-1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for _, r in s.iterrows():
        lines.append(f"| {LABELS.get(r.model, r.model)} | {r.mae_mw:,.0f} | {r.mape_pct:.2f}% | "
                     f"{r.mase_vs_weekly:.2f} | {pct(r.coverage_10_90)} | {pct(r.beats_rte_days)} |")
    return "\n".join(lines)


def write(store: Store, run_id: str, summary: pd.DataFrame, out: Path = Path("results")) -> None:
    out.mkdir(exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    meta = store.query(f"SELECT protocol_sha256, store, created_at FROM runs WHERE run_id = '{run_id}'").iloc[0]
    hours = int(summary["hours"].iloc[0])
    (out / "summary.md").write_text(
        f"Run `{run_id}` on {meta.store}, {meta.created_at}. Protocol SHA-256 `{meta.protocol_sha256}`.\n"
        f"All models scored on the same {hours} hours.\n\n{markdown_table(summary)}\n", encoding="utf-8")
    _chart(store, run_id, summary, out / "backtest.png")


def write_sensitivity(run_id: str, summary: pd.DataFrame, out: Path = Path("results")) -> None:
    out.mkdir(exist_ok=True)
    summary.to_csv(out / "sensitivity.csv", index=False)
    (out / "sensitivity.md").write_text(
        "**Exploratory, not pre-registered.** Same test days and models, but the cutoff is moved 12 hours "
        "earlier to D-1 00:00 UTC (forecast lead 24 to 47 hours). RTE's dataset does not record when its J-1 "
        "forecast is issued, so this checks that the result does not depend on Timestral seeing more recent "
        f"data than RTE.\n\nRun `{run_id}`.\n\n{markdown_table(summary)}\n", encoding="utf-8")


def _chart(store: Store, run_id: str, summary: pd.DataFrame, path: Path) -> None:
    fc = store.query(f"SELECT model, ts, q10, q50, q90 FROM forecasts WHERE run_id = '{run_id}' "
                     f"AND model IN ('timestral_calendar', 'rte_j1')")
    act = store.query("SELECT ts, consumption FROM load_hourly")
    fc["ts"] = pd.to_datetime(fc["ts"], unit="s", utc=True)
    act["ts"] = pd.to_datetime(act["ts"], unit="s", utc=True)

    # One representative week from the middle of the test window.
    days = sorted(fc["ts"].dt.normalize().unique())
    start = days[len(days) // 2 - 3]
    end = start + pd.Timedelta(days=7)
    win = lambda d: d[(d["ts"] >= start) & (d["ts"] < end)].sort_values("ts")  # noqa: E731

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    a = win(act)
    ax1.plot(a["ts"], a["consumption"] / 1000, color="#111827", lw=1.6, label="Actual")
    t = win(fc[fc.model == "timestral_calendar"])
    ax1.fill_between(t["ts"], t["q10"] / 1000, t["q90"] / 1000, color=COLORS["timestral"], alpha=0.18,
                     label="Timestral 10–90%")
    ax1.plot(t["ts"], t["q50"] / 1000, color=COLORS["timestral"], lw=1.4, label="Timestral + calendar")
    r = win(fc[fc.model == "rte_j1"])
    ax1.plot(r["ts"], r["q50"] / 1000, color=COLORS["rte_j1"], lw=1.2, ls="--", label="RTE J-1")
    ax1.set_title("French electricity demand, one test week (day-ahead, cutoff D-1 12:00 UTC)", fontsize=10)
    ax1.set_ylabel("GW")
    ax1.legend(fontsize=8, ncol=2, loc="upper left")
    ax1.grid(alpha=0.25)

    s = summary.sort_values("mae_mw", ascending=False)
    ax2.barh([LABELS.get(m, m) for m in s.model], s.mae_mw, color=[COLORS.get(m, "#999") for m in s.model])
    for y, v in enumerate(s.mae_mw):
        ax2.text(v, y, f" {v:,.0f}", va="center", fontsize=8)
    ax2.set_title("MAE over Jun–Aug 2026 (MW, lower is better)", fontsize=10)
    ax2.tick_params(axis="y", labelsize=8)
    ax2.set_xlim(0, float(np.nanmax(s.mae_mw)) * 1.22)
    ax2.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
