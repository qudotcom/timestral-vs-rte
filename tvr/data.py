"""Download RTE eco2mix national data and resample it to hourly means."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

ODRE = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets"
FIELDS = "date_heure,consommation,prevision_j1"


def _export_url(dataset: str, start: str, end: str) -> str:
    query = {
        "select": FIELDS,
        "where": f'date_heure>="{start}" and date_heure<"{end}"',
        "order_by": "date_heure",
        "timezone": "UTC",
        "delimiter": ",",
    }
    return f"{ODRE}/{dataset}/exports/csv?{urlencode(query)}"


def download(dataset: str, start: str, end: str, cache: Path) -> pd.DataFrame:
    """Raw 15/30-minute rows, cached as CSV."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        resp = requests.get(_export_url(dataset, start, end), timeout=300)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
    raw = pd.read_csv(cache, encoding="utf-8-sig")
    raw["date_heure"] = pd.to_datetime(raw["date_heure"], utc=True)
    return raw


def to_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Hourly mean of actual consumption and of RTE's day-ahead forecast, on a complete hourly grid."""
    df = raw.set_index("date_heure").sort_index()
    hourly = df[["consommation", "prevision_j1"]].resample("1h").mean()
    hourly = hourly.rename(columns={"consommation": "consumption", "prevision_j1": "rte_j1"})
    hourly.index.name = "ts"
    return hourly.reset_index()


def load_hourly(start: str, end: str, data_dir: Path = Path("data")) -> pd.DataFrame:
    """Definitive consumption data between start and end (UTC, end exclusive)."""
    tag = f"{start[:10]}_{end[:10]}"
    raw = download("eco2mix-national-cons-def", start, end, data_dir / f"eco2mix_def_{tag}.csv")
    return to_hourly(raw)


def load_recent(start: str, end: str, data_dir: Path = Path("data")) -> pd.DataFrame:
    """Real-time data (not yet consolidated) for the live next-day forecast."""
    raw = download("eco2mix-national-tr", start, end, data_dir / f"eco2mix_tr_{start[:10]}_{end[:10]}.csv")
    return to_hourly(raw)
