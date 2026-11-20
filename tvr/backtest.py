"""Run the pre-registered day-ahead backtest and score it in SQL."""
from __future__ import annotations

import datetime as dt
import uuid

import numpy as np
import pandas as pd

from . import data, models, protocol
from .store import Store

H = 3600


def epoch(ts) -> int:
    return int(pd.Timestamp(ts).timestamp())


def load_into_store(store: Store, p: dict) -> pd.DataFrame:
    hourly = data.load_hourly(p["data"]["start"], p["data"]["end"])
    observed = hourly.dropna(subset=["consumption"])
    store.insert("load_hourly", pd.DataFrame({
        "ts": observed["ts"].map(epoch).astype("int64"),
        "consumption": observed["consumption"].astype("float64"),
    }))
    return hourly


def context_for(store: Store, cutoff: pd.Timestamp, hours: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """The `hours` hourly values strictly before the cutoff, read from the store, NaN where missing."""
    c = epoch(cutoff)
    rows = store.query(
        f"SELECT ts, consumption FROM load_hourly WHERE ts < {c} AND ts >= {c - hours * H} ORDER BY ts")
    grid = pd.date_range(cutoff - pd.Timedelta(hours=hours), periods=hours, freq="h")
    series = pd.Series(rows["consumption"].to_numpy(), index=pd.to_datetime(rows["ts"], unit="s", utc=True))
    values = series.reindex(grid).to_numpy(dtype=float)
    assert grid[-1] < cutoff, "context must end before the cutoff"
    return values, grid


def run(store: Store, p: dict | None = None, log=print, fresh: bool = True, tag: str = "") -> str:
    p = p or protocol.load()
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + tag + uuid.uuid4().hex[:6]
    if fresh:
        store.reset()
        hourly = load_into_store(store, p).set_index("ts")
    else:
        hourly = data.load_hourly(p["data"]["start"], p["data"]["end"]).set_index("ts")
    store.insert("runs", pd.DataFrame([{
        "run_id": run_id, "protocol_sha256": protocol.sha256(), "store": store.name,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}]))

    origins = protocol.origins(p)
    rows = []

    def emit(model, o, ts, lead, q10, q50, q90):
        for t, k, a, b, c in zip(ts, lead, q10, q50, q90):
            if np.isfinite(b):
                rows.append((run_id, model, o.day.strftime("%Y-%m-%d"), epoch(t), int(k), float(a), float(b), float(c)))

    # Contexts come from SQL, one query per origin, so the cutoff is enforced by the store.
    by_model = {k: v for k, v in p["models"].items() if k.startswith("timestral")}
    ctx_hours = max(v["context_hours"] for v in by_model.values())
    contexts, grids = [], []
    for o in origins:
        values, grid = context_for(store, o.cutoff, ctx_hours)
        contexts.append(values)
        grids.append(grid)
    contexts = np.stack(contexts)
    log(f"{len(origins)} origins, context {ctx_hours} h, missing in contexts: {int(np.isnan(contexts).sum())} values")

    for name, spec in by_model.items():
        q10, q50, q90 = models.timestral_forecast(
            contexts[:, -spec["context_hours"]:], [g[-spec["context_hours"]:] for g in grids],
            horizon=origins[0].horizon, checkpoint=spec["checkpoint"],
            with_calendar=spec["covariates"] != "none")
        for i, o in enumerate(origins):
            ts = pd.date_range(o.cutoff, periods=o.horizon, freq="h")
            keep = slice(o.skipped, o.horizon)
            emit(name, o, ts[keep], np.arange(o.horizon)[keep], q10[i, keep], q50[i, keep], q90[i, keep])
        log(f"  {name}: done")

    # Baselines read only from the context (so they cannot see past the cutoff either).
    for i, o in enumerate(origins):
        ts = pd.date_range(o.first_scored, periods=24, freq="h")
        lead = np.arange(o.skipped, o.horizon)
        ctx = pd.Series(contexts[i], index=grids[i])
        for name, lag in (("naive_weekly", 168), ("naive_2day", 48)):
            v = ctx.reindex(ts - pd.Timedelta(hours=lag)).to_numpy()
            emit(name, o, ts, lead, v, v, v)
        rte = hourly["rte_j1"].reindex(ts).to_numpy()
        emit("rte_j1", o, ts, lead, rte, rte, rte)

    store.insert("forecasts", pd.DataFrame(rows, columns=["run_id", "model", "day", "ts", "lead_h", "q10", "q50", "q90"]))
    log(f"stored {len(rows)} forecast rows for run {run_id}")
    return run_id


def scores(store: Store, run_id: str) -> pd.DataFrame:
    """Metrics in SQL. Every model is scored on exactly the same hours."""
    n_models = int(store.query(
        f"SELECT count(DISTINCT model) AS n FROM forecasts WHERE run_id = '{run_id}'")["n"].iloc[0])
    common = (f"SELECT ts FROM forecasts WHERE run_id = '{run_id}' "
              f"GROUP BY ts HAVING count(DISTINCT model) = {n_models}")
    summary = store.query(f"""
        SELECT f.model AS model,
               count(*) AS hours,
               avg(abs(f.q50 - a.consumption)) AS mae_mw,
               100 * avg(abs(f.q50 - a.consumption) / a.consumption) AS mape_pct,
               avg(CASE WHEN a.consumption >= f.q10 AND a.consumption <= f.q90 THEN 1.0 ELSE 0.0 END) AS coverage_10_90
        FROM forecasts f JOIN load_hourly a ON a.ts = f.ts
        WHERE f.run_id = '{run_id}' AND f.ts IN ({common})
        GROUP BY f.model
    """)
    daily = f"""
        SELECT f.model AS model, f.day AS day, avg(abs(f.q50 - a.consumption)) AS mae
        FROM forecasts f JOIN load_hourly a ON a.ts = f.ts
        WHERE f.run_id = '{run_id}' AND f.ts IN ({common})
        GROUP BY f.model, f.day
    """
    wins = store.query(f"""
        SELECT x.model AS model, avg(CASE WHEN x.mae < r.mae THEN 1.0 ELSE 0.0 END) AS beats_rte_days
        FROM ({daily}) x JOIN (SELECT day, mae FROM ({daily}) d WHERE model = 'rte_j1') r ON x.day = r.day
        GROUP BY x.model
    """)
    out = summary.merge(wins, on="model", how="left")
    naive = float(out.loc[out["model"] == "naive_weekly", "mae_mw"].iloc[0])
    out["mase_vs_weekly"] = out["mae_mw"] / naive
    out.loc[~out["model"].str.startswith("timestral"), "coverage_10_90"] = np.nan
    out.loc[out["model"] == "rte_j1", "beats_rte_days"] = np.nan
    return out.sort_values("mae_mw").reset_index(drop=True)


def sensitivity_protocol(cutoff_hour: int = 0) -> dict:
    """EXPLORATORY, not pre-registered: the same test with an earlier cutoff on D-1.

    RTE's dataset does not say when its J-1 forecast is issued, so this checks
    whether the result survives giving Timestral less recent information.
    """
    p = protocol.load()
    p["task"] = dict(p["task"], cutoff_hour_utc_on_previous_day=cutoff_hour, lead_hours_skipped=24 - cutoff_hour)
    return p
