"""Forecast tomorrow's French demand from the latest real-time data."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, models, protocol
from .backtest import epoch
from .store import Store


def tomorrow(store: Store, now: pd.Timestamp | None = None, out: Path = Path("results")) -> pd.DataFrame:
    p = protocol.load()
    spec = p["models"]["timestral_calendar"]
    now = (now or pd.Timestamp.now(tz="UTC")).floor("h")
    day = (now + pd.Timedelta(days=1)).normalize()

    start = (now - pd.Timedelta(hours=spec["context_hours"] + 48)).strftime("%Y-%m-%dT%H:00:00Z")
    hist = pd.concat([
        data.load_hourly(start, "2026-09-01T00:00:00Z"),
        data.load_recent("2026-08-31T22:00:00Z", (day + pd.Timedelta(days=2)).strftime("%Y-%m-%dT00:00:00Z")),
    ]).drop_duplicates("ts", keep="last").set_index("ts").sort_index()

    observed = hist["consumption"].dropna()
    cutoff = observed.index[-1] + pd.Timedelta(hours=1)   # first hour we do not know yet
    grid = pd.date_range(cutoff - pd.Timedelta(hours=spec["context_hours"]), periods=spec["context_hours"], freq="h")
    context = observed.reindex(grid).to_numpy(dtype=float)
    horizon = int((day + pd.Timedelta(days=1) - cutoff) / pd.Timedelta(hours=1))

    q10, q50, q90 = models.timestral_forecast(context[None, :], [grid], horizon, spec["checkpoint"], with_calendar=True)
    ts = pd.date_range(cutoff, periods=horizon, freq="h")
    fc = pd.DataFrame({"ts": ts, "q10": q10[0], "q50": q50[0], "q90": q90[0]})
    fc = fc[fc.ts >= day].reset_index(drop=True)
    fc["rte_j1"] = hist["rte_j1"].reindex(fc.ts).to_numpy()

    run_id = "live-" + day.strftime("%Y-%m-%d")
    store.ensure()
    store.execute(f"DELETE FROM forecasts WHERE run_id = '{run_id}'") if store.name == "duckdb" else None
    store.insert("forecasts", pd.DataFrame({
        "run_id": run_id, "model": "timestral_calendar", "day": day.strftime("%Y-%m-%d"),
        "ts": fc.ts.map(epoch).astype("int64"), "lead_h": np.arange(len(fc), dtype="int32"),
        "q10": fc.q10, "q50": fc.q50, "q90": fc.q90}))

    out.mkdir(exist_ok=True)
    fc.assign(ts=fc.ts.dt.tz_convert("Europe/Paris")).to_csv(out / "tomorrow.csv", index=False)
    print(f"cutoff {cutoff} (last observed hour {observed.index[-1]}), forecasting {day.date()} "
          f"[{dt.datetime.now(dt.timezone.utc):%H:%M} UTC]")
    return fc
