"""python -m tvr {backtest,tomorrow,brief,all}"""
from __future__ import annotations

import argparse

from . import backtest, report
from .store import open_store


def main() -> None:
    ap = argparse.ArgumentParser(prog="tvr", description="Timestral vs RTE: day-ahead French electricity demand")
    ap.add_argument("command", choices=["backtest", "tomorrow", "brief", "all"])
    ap.add_argument("--store", help="chdb | clickhouse | duckdb (default: chdb if installed)")
    args = ap.parse_args()

    store = open_store(args.store)
    print(f"store: {store.name}")
    if args.command in ("backtest", "all"):
        run_id = backtest.run(store)
        summary = backtest.scores(store, run_id)
        report.write(store, run_id, summary)
        print(report.markdown_table(summary))
        # Exploratory robustness check (labelled as such in results/sensitivity.md).
        sens_id = backtest.run(store, backtest.sensitivity_protocol(0), fresh=False, tag="sens-")
        sens = backtest.scores(store, sens_id)
        report.write_sensitivity(sens_id, sens)
        print("\nSensitivity (cutoff D-1 00:00 UTC, exploratory):")
        print(report.markdown_table(sens))
    if args.command in ("tomorrow", "all"):
        from . import live
        fc = live.tomorrow(store)
        print(fc.assign(ts=fc.ts.dt.tz_convert("Europe/Paris").dt.strftime("%H:%M")).round(0).to_string(index=False))
    if args.command in ("brief", "all"):
        from . import brief
        brief.write_brief()


if __name__ == "__main__":
    main()
