"""Leakage, protocol and scoring tests. They run on DuckDB, or on chDB when TVR_STORE=chdb."""
import os

import numpy as np
import pandas as pd
import pytest

from tvr import backtest, protocol
from tvr.store import open_store


@pytest.fixture
def store(tmp_path):
    s = open_store(os.environ.get("TVR_STORE", "duckdb"), workdir=tmp_path)
    s.reset()
    return s


def hourly(start="2026-05-01", hours=24 * 60, seed=0):
    ts = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"ts": ts.map(backtest.epoch).astype("int64"),
                         "consumption": 45000 + 5000 * np.sin(np.arange(hours) * 2 * np.pi / 24) + rng.normal(0, 300, hours)})


def test_context_never_reaches_the_cutoff(store):
    store.insert("load_hourly", hourly())
    cutoff = pd.Timestamp("2026-06-10 12:00", tz="UTC")
    values, grid = backtest.context_for(store, cutoff, 24 * 7)
    assert len(values) == 24 * 7
    assert grid[-1] == cutoff - pd.Timedelta(hours=1)
    assert (grid < cutoff).all()
    assert not np.isnan(values).any()


def test_missing_hours_become_nan_not_shifted(store):
    df = hourly()
    gap = pd.Timestamp("2026-06-10 05:00", tz="UTC")
    store.insert("load_hourly", df[df.ts != backtest.epoch(gap)])
    values, grid = backtest.context_for(store, pd.Timestamp("2026-06-10 12:00", tz="UTC"), 24)
    assert np.isnan(values[list(grid).index(gap)])
    assert np.isnan(values).sum() == 1


def test_protocol_origins_follow_the_cutoff_rule():
    p = protocol.load()
    origins = protocol.origins(p)
    assert len(origins) == 92
    for o in origins:
        assert o.cutoff == o.day - pd.Timedelta(hours=24 - p["task"]["cutoff_hour_utc_on_previous_day"])
        assert o.horizon - o.skipped == 24


def test_protocol_hash_ignores_line_endings(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_bytes(b'{"x": 1,\n "y": 2}\n')
    b.write_bytes(b'{"x": 1,\r\n "y": 2}\r\n')
    assert protocol.sha256(a) == protocol.sha256(b)


def test_sensitivity_protocol_moves_the_cutoff_only():
    p = backtest.sensitivity_protocol(0)
    o = protocol.origins(p)[0]
    assert o.cutoff == o.day - pd.Timedelta(days=1)
    assert o.horizon == 48 and o.skipped == 24


def test_scores_match_numpy_on_common_hours(store):
    actual = hourly(hours=48)
    store.insert("load_hourly", actual)
    ts = actual.ts.to_numpy()
    y = actual.consumption.to_numpy()
    rows = []
    for model, err in (("timestral", 100.0), ("rte_j1", 300.0), ("naive_weekly", 1000.0)):
        for t, v in zip(ts, y):
            rows.append(("r", model, "2026-05-01", int(t), 0, v - 2 * err, v + err, v + 2 * err))
    rows.append(("r", "timestral", "2026-05-03", int(ts[-1]) + 3600, 0, 1.0, 1.0, 1.0))  # hour no other model has
    store.insert("forecasts", pd.DataFrame(rows, columns=["run_id", "model", "day", "ts", "lead_h", "q10", "q50", "q90"]))
    s = backtest.scores(store, "r").set_index("model")
    assert s.loc["timestral", "hours"] == 48
    assert s.loc["timestral", "mae_mw"] == pytest.approx(100.0)
    assert s.loc["timestral", "mase_vs_weekly"] == pytest.approx(0.1)
    assert s.loc["timestral", "coverage_10_90"] == pytest.approx(1.0)
    assert s.loc["timestral", "beats_rte_days"] == pytest.approx(1.0)
