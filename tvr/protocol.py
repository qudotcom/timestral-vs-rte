"""The pre-registered protocol: loading, hashing, and the cutoff rule."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "protocol.json"


def load(path: Path = PROTOCOL_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path = PROTOCOL_PATH) -> str:
    # Line endings are normalised so the hash is the same on Windows and Linux checkouts.
    text = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(text).hexdigest()


@dataclass(frozen=True)
class Origin:
    day: pd.Timestamp          # the scored day D (00:00 UTC)
    cutoff: pd.Timestamp       # last moment of information: D-1 at the cutoff hour
    first_scored: pd.Timestamp
    horizon: int               # hours forecast from the cutoff (skipped + scored)
    skipped: int


def origins(p: dict) -> list[Origin]:
    t = p["task"]
    days = pd.date_range(p["test_days"]["first"], p["test_days"]["last"], freq="D", tz="UTC")
    out = []
    for d in days:
        cutoff = d - pd.Timedelta(days=1) + pd.Timedelta(hours=t["cutoff_hour_utc_on_previous_day"])
        out.append(Origin(day=d, cutoff=cutoff, first_scored=d,
                          horizon=t["lead_hours_skipped"] + t["scored_hours"], skipped=t["lead_hours_skipped"]))
    # The first scored hour must be exactly `skipped` hours after the cutoff.
    for o in out:
        assert o.cutoff + pd.Timedelta(hours=o.skipped) == o.first_scored
    return out
