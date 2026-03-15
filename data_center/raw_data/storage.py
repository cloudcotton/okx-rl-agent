"""
Storage layer: Parquet (fast read, compact) + SQLite (multi-symbol queries).

Parquet layout:
    data_center/parquet/{SYMBOL}_5m.parquet
    Engine: pyarrow, compression: snappy
    Typical size: BTC-USDT 5m (2020-2026) ≈ 8-12 MB

SQLite layout:
    data_center/market_data.db
    Table: ohlcv
    Primary key: (datetime, symbol)
    Index: (symbol, datetime)  — optimised for per-symbol time-range queries
"""

import sqlite3
import logging

import pandas as pd

from config import DB_PATH, PARQ_DIR

log = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS ohlcv (
    datetime    TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    vol         REAL,
    vol_quote   REAL,
    PRIMARY KEY (datetime, symbol)
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_sym_dt ON ohlcv (symbol, datetime);
"""


class StorageManager:

    def __init__(self):
        PARQ_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── SQLite setup ──────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)
        log.info(f"SQLite ready: {DB_PATH}")

    # ── Parquet ───────────────────────────────────────────────────────────────

    def save_parquet(self, df: pd.DataFrame, symbol: str) -> None:
        path = PARQ_DIR / f"{symbol.replace('-', '_')}_5m.parquet"
        df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
        size_mb = path.stat().st_size / 1e6
        log.info(
            f"[{symbol}] Parquet saved → {path}  "
            f"({size_mb:.1f} MB  /  {len(df):,} rows)"
        )

    # ── SQLite ────────────────────────────────────────────────────────────────

    def save_sqlite(self, df: pd.DataFrame, symbol: str) -> None:
        """
        Upsert strategy: delete all existing rows for the symbol, then
        bulk-insert the full clean dataset. This avoids PRIMARY KEY conflicts
        on re-runs and keeps the implementation simple.
        """
        rows = df.copy()
        rows["symbol"]   = symbol
        rows["datetime"] = rows["datetime"].astype(str)

        cols = ["datetime", "symbol", "open", "high", "low", "close", "vol", "vol_quote"]

        with sqlite3.connect(DB_PATH) as conn:
            # Remove stale rows for this symbol before re-inserting
            conn.execute("DELETE FROM ohlcv WHERE symbol = ?", (symbol,))
            rows[cols].to_sql(
                "ohlcv",
                conn,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=5_000,
            )

        log.info(f"[{symbol}] SQLite: {len(rows):,} rows written to {DB_PATH}")

    # ── convenience reader (used by downstream modules) ───────────────────────

    @staticmethod
    def load_parquet(symbol: str) -> pd.DataFrame:
        path = PARQ_DIR / f"{symbol.replace('-', '_')}_5m.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No parquet found for {symbol}: {path}")
        df = pd.read_parquet(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
