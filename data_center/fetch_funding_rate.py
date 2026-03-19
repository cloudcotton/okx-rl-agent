"""
Binance ETHUSDT perpetual funding rate downloader.

Downloads monthly CSV zips from data.binance.vision and saves a single
merged parquet to data_center/funding_rate/ETH_USDT_funding_rate.parquet.

Output columns:
    datetime      — UTC pandas Timestamp (funding settlement time)
    funding_rate  — float64, e.g. 0.0001 = 0.01 % per 8h

Funding is settled every 8 hours: 00:00 / 08:00 / 16:00 UTC.

Usage:
    python data_center/fetch_funding_rate.py
    python data_center/fetch_funding_rate.py --symbol ETHUSDT --start 2020-01 --end 2026-03
"""

from __future__ import annotations

import argparse
import io
import logging
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_BASE_URL  = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
_OUT_DIR   = Path(__file__).resolve().parent / "funding_rate"
_REQ_INTERVAL = 0.5   # seconds between requests


def _month_range(start: str, end: str) -> list[str]:
    """Return ['YYYY-MM', ...] from start to end inclusive."""
    months = []
    cur = pd.Timestamp(start + "-01")
    stop = pd.Timestamp(end + "-01")
    while cur <= stop:
        months.append(cur.strftime("%Y-%m"))
        cur += pd.DateOffset(months=1)
    return months


def _download_month(session: requests.Session, symbol: str, ym: str) -> pd.DataFrame | None:
    """
    Fetch one monthly zip from Binance vision and return a DataFrame with
    columns [datetime, funding_rate].  Returns None on 404 (month not yet published).
    """
    url = f"{_BASE_URL}/{symbol}/{symbol}-fundingRate-{ym}.zip"
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 404:
            log.warning(f"[{ym}] 404 — not yet published, skipping.")
            return None
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error(f"[{ym}] Download failed: {exc}")
        return None

    # Unzip in memory
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            raw = pd.read_csv(f)

    # Binance CSV columns vary slightly across history; normalise them.
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]

    # Timestamp column: 'calc_time' (ms integer)
    ts_col = next(
        (c for c in raw.columns if "time" in c or "calc" in c),
        raw.columns[0],
    )
    # Rate column: 'last_funding_rate' or 'fundingrate'
    rate_col = next(
        (c for c in raw.columns if "rate" in c and "interval" not in c),
        None,
    )
    if rate_col is None:
        log.error(f"[{ym}] Cannot find funding rate column. Columns: {list(raw.columns)}")
        return None

    df = pd.DataFrame({
        "datetime":     pd.to_datetime(raw[ts_col].astype("int64"), unit="ms", utc=True)
                          .dt.tz_convert(None),      # naive UTC
        "funding_rate": raw[rate_col].astype("float64"),
    })
    df.dropna(inplace=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    log.info(f"[{ym}] {len(df):4d} rows  "
             f"[{df['datetime'].min()} → {df['datetime'].max()}]  "
             f"rate range [{df['funding_rate'].min():.5f}, {df['funding_rate'].max():.5f}]")
    return df


def fetch_funding_rate(symbol: str = "ETHUSDT", start: str = "2020-01", end: str = "2026-03") -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f"{symbol}_funding_rate.parquet"

    months  = _month_range(start, end)
    session = requests.Session()
    session.headers["User-Agent"] = "okx-rl-pipeline/1.0"

    frames: list[pd.DataFrame] = []
    for ym in months:
        df = _download_month(session, symbol, ym)
        if df is not None:
            frames.append(df)
        time.sleep(_REQ_INTERVAL)

    if not frames:
        log.error("No data fetched — check symbol / date range.")
        return

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    combined.to_parquet(out_path, index=False)
    log.info(
        f"\nSaved → {out_path}  "
        f"({len(combined):,} rows, "
        f"{combined['datetime'].min()} → {combined['datetime'].max()})"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Binance perpetual funding rate history.")
    p.add_argument("--symbol", default="ETHUSDT", help="Binance perpetual symbol (default: ETHUSDT)")
    p.add_argument("--start",  default="2020-01",  help="Start month YYYY-MM (default: 2020-01)")
    p.add_argument("--end",    default="2026-03",  help="End month YYYY-MM (default: 2026-03)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fetch_funding_rate(args.symbol, args.start, args.end)
