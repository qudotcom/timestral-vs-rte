"""Analytical store: ClickHouse (in-process chDB or a server) with a DuckDB fallback.

All queries in this project use SQL that both engines accept, so the same
pipeline runs on ClickHouse in CI and on DuckDB on machines without it
(chDB ships Linux and macOS wheels only).

Pick the engine with TVR_STORE=chdb | clickhouse | duckdb (default: chdb when
it can be imported, otherwise duckdb).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Timestamps are stored as epoch seconds (Int64) so both engines treat them identically.
SCHEMA = {
    "load_hourly": [("ts", "Int64"), ("consumption", "Float64")],
    "forecasts": [
        ("run_id", "String"), ("model", "String"), ("day", "String"), ("ts", "Int64"),
        ("lead_h", "Int32"), ("q10", "Float64"), ("q50", "Float64"), ("q90", "Float64"),
    ],
    "runs": [("run_id", "String"), ("protocol_sha256", "String"), ("store", "String"), ("created_at", "String")],
}

_DUCK_TYPES = {"Int64": "BIGINT", "Int32": "INTEGER", "Float64": "DOUBLE", "String": "VARCHAR"}


class Store:
    name = "base"

    def execute(self, sql: str) -> None: ...
    def query(self, sql: str) -> pd.DataFrame: ...
    def insert(self, table: str, df: pd.DataFrame) -> None: ...
    def _create(self, table: str, cols) -> None: ...

    def reset(self, tables=SCHEMA) -> None:
        for table, cols in tables.items():
            self.execute(f"DROP TABLE IF EXISTS {table}")
            self._create(table, cols)

    def ensure(self, tables=SCHEMA) -> None:
        for table, cols in tables.items():
            if not self.exists(table):
                self._create(table, cols)

    def exists(self, table: str) -> bool:
        try:
            self.query(f"SELECT count(*) AS n FROM {table}")
            return True
        except Exception:
            return False


class DuckStore(Store):
    name = "duckdb"

    def __init__(self, path: str = "tvr.duckdb"):
        import duckdb
        self.con = duckdb.connect(path)

    def _create(self, table, cols):
        body = ", ".join(f"{c} {_DUCK_TYPES[t]}" for c, t in cols)
        self.execute(f"CREATE TABLE {table} ({body})")

    def execute(self, sql):
        self.con.execute(sql)

    def query(self, sql):
        return self.con.execute(sql).df()

    def insert(self, table, df):
        self.con.register("_incoming", df)
        self.con.execute(f"INSERT INTO {table} SELECT {', '.join(df.columns)} FROM _incoming")
        self.con.unregister("_incoming")


class ChdbStore(Store):
    """ClickHouse running in-process (chDB)."""
    name = "clickhouse (chdb)"

    def __init__(self, path: str = ".chdb"):
        from chdb import session
        self.sess = session.Session(path)

    def _create(self, table, cols):
        body = ", ".join(f"{c} {t}" for c, t in cols)
        self.execute(f"CREATE TABLE {table} ({body}) ENGINE = MergeTree ORDER BY tuple()")

    def execute(self, sql):
        self.sess.query(sql)

    def query(self, sql):
        return self.sess.query(sql, "DataFrame")

    def insert(self, table, df):
        incoming = df  # referenced by name in the Python() table function below
        self.sess.query(f"INSERT INTO {table} SELECT {', '.join(df.columns)} FROM Python(incoming)")


class ClickHouseServerStore(Store):
    """A ClickHouse server or ClickHouse Cloud, via clickhouse-connect.

    Configure with CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER,
    CLICKHOUSE_PASSWORD, CLICKHOUSE_SECURE (1/0).
    """
    name = "clickhouse (server)"

    def __init__(self):
        import clickhouse_connect
        self.client = clickhouse_connect.get_client(
            host=os.environ["CLICKHOUSE_HOST"],
            port=int(os.environ.get("CLICKHOUSE_PORT", "8443")),
            username=os.environ.get("CLICKHOUSE_USER", "default"),
            password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
            secure=os.environ.get("CLICKHOUSE_SECURE", "1") == "1",
        )

    def _create(self, table, cols):
        body = ", ".join(f"{c} {t}" for c, t in cols)
        self.execute(f"CREATE TABLE {table} ({body}) ENGINE = MergeTree ORDER BY tuple()")

    def execute(self, sql):
        self.client.command(sql)

    def query(self, sql):
        return self.client.query_df(sql)

    def insert(self, table, df):
        self.client.insert_df(table, df)


def open_store(kind: str | None = None, workdir: Path = Path(".")) -> Store:
    kind = (kind or os.environ.get("TVR_STORE") or "").lower()
    if not kind:
        try:
            import chdb  # noqa: F401
            kind = "chdb"
        except ImportError:
            kind = "duckdb"
    if kind == "chdb":
        return ChdbStore(str(workdir / ".chdb"))
    if kind == "clickhouse":
        return ClickHouseServerStore()
    if kind == "duckdb":
        return DuckStore(str(workdir / "tvr.duckdb"))
    raise ValueError(f"unknown store {kind!r}")
