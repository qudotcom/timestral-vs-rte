"""Turn the forecast into an operator brief with a Mistral LLM, traced in Langfuse.

Forecast first, then let the LLM explain: the numbers come from Timestral and
the backtest, never from the language model. The brief is a draft for a human
to approve.

Needs MISTRAL_API_KEY. Langfuse tracing turns on when LANGFUSE_PUBLIC_KEY and
LANGFUSE_SECRET_KEY are set (LANGFUSE_HOST optional).
"""
from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path

import pandas as pd

SYSTEM = ("You are an assistant to a French electricity grid operator. You write short, factual "
          "day-ahead briefs. Use only the numbers you are given. Never invent figures. "
          "Flag uncertainty plainly.")


def facts(results: Path = Path("results")) -> dict:
    fc = pd.read_csv(results / "tomorrow.csv", parse_dates=["ts"])
    summary = pd.read_csv(results / "summary.csv")
    peak = fc.loc[fc.q50.idxmax()]
    trough = fc.loc[fc.q50.idxmin()]
    gap = (fc.q50 - fc.rte_j1).abs()
    back = summary.set_index("model")
    return {
        "day": str(fc.ts.iloc[0].date()),
        "peak": {"hour_local": peak.ts.strftime("%H:%M"), "mw": round(peak.q50), "p10": round(peak.q10), "p90": round(peak.q90)},
        "trough": {"hour_local": trough.ts.strftime("%H:%M"), "mw": round(trough.q50)},
        "daily_energy_gwh": round(fc.q50.sum() / 1000, 1),
        "largest_gap_vs_rte_j1": None if gap.isna().all() else {
            "hour_local": fc.loc[gap.idxmax()].ts.strftime("%H:%M"), "mw": round(float(gap.max()))},
        "backtest_jun_aug_2026": {
            "timestral_mae_mw": round(float(back.loc["timestral_calendar", "mae_mw"])),
            "rte_j1_mae_mw": round(float(back.loc["rte_j1", "mae_mw"])),
            "interval_10_90_coverage": round(float(back.loc["timestral_calendar", "coverage_10_90"]), 2),
        },
    }


def _tracer():
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import get_client
        return get_client()
    except Exception as exc:  # tracing must never break the brief
        print(f"(Langfuse disabled: {exc})")
        return None


def write_brief(model: str | None = None, results: Path = Path("results")) -> str:
    f = facts(results)
    prompt = ("Write a day-ahead brief (max 120 words, 3 bullet points then one line of caveats) "
              f"from these facts:\n{json.dumps(f, indent=2)}")
    if not os.environ.get("MISTRAL_API_KEY"):
        print("MISTRAL_API_KEY not set. The prompt that would be sent:\n")
        print(prompt)
        return prompt

    try:
        from mistralai.client import Mistral   # mistralai >= 2
    except ImportError:
        from mistralai import Mistral          # mistralai 1.x
    model = model or os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    lf = _tracer()
    span = (lf.start_as_current_observation(name="day-ahead-brief", as_type="generation", model=model, input=prompt)
            if lf else nullcontext())
    with span as gen:
        resp = client.chat.complete(model=model, temperature=0.2,
                                    messages=[{"role": "system", "content": SYSTEM},
                                              {"role": "user", "content": prompt}])
        text = resp.choices[0].message.content
        if lf:
            gen.update(output=text, metadata={"facts": f})
    if lf:
        lf.flush()
    (results / "brief.md").write_text(f"# Day-ahead brief, {f['day']} (draft, needs human approval)\n\n{text}\n",
                                      encoding="utf-8")
    print(text)
    return text
