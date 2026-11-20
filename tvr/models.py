"""Timestral forecasts and calendar covariates."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd


@lru_cache(maxsize=4)
def forecaster(checkpoint: str):
    from timestral import TimestralForecaster
    return TimestralForecaster.from_pretrained(checkpoint)


def calendar(ts: pd.DatetimeIndex) -> np.ndarray:
    """Known-future calendar features, shape (5, len(ts)). Local French time drives demand."""
    local = ts.tz_convert("Europe/Paris")
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60
    dow = local.dayofweek.to_numpy()
    return np.stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        (dow >= 5).astype(float),
    ]).astype(np.float32)


def timestral_forecast(contexts: np.ndarray, context_ts: list[pd.DatetimeIndex], horizon: int,
                       checkpoint: str, with_calendar: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch forecast. contexts: (B, T) with NaN for missing. Returns q10, q50, q90 each (B, horizon)."""
    model = forecaster(checkpoint)
    kwargs = {}
    if with_calendar:
        known = []
        for ts in context_ts:
            full = ts.append(pd.date_range(ts[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h"))
            known.append(calendar(full))
        kwargs["known_covariates"] = np.stack(known)  # (B, 5, T + horizon)
    out = model.forecast(contexts.astype(np.float32), prediction_length=horizon, **kwargs)
    q = lambda level: out.quantile(level)[:, 0, :].detach().cpu().numpy()  # noqa: E731
    return q(0.1), q(0.5), q(0.9)
