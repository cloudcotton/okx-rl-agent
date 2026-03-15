#!/usr/bin/env python3
"""
OKX 5-Minute OHLCV Historical Data Pipeline
============================================
Period : 2020-01-01 → 2026-03-14
Default: BTC-USDT, ETH-USDT

Stages
------
  1. Download   chunked reverse-time sweep via OKX history-candles API
  2. Clean      strict reindex → gap imputation → spike filter
  3. Store      Parquet (fast ML reads) + SQLite (multi-symbol queries)

Usage
-----
  # default symbols (BTC-USDT, ETH-USDT)
  python pipeline.py

  # custom symbols
  python pipeline.py --symbols BTC-USDT ETH-USDT SOL-USDT

  # force re-download even if raw cache exists
  python pipeline.py --fresh

Outputs (all in data_center/)
-------------------------------
  parquet/{SYMBOL}_5m.parquet
  market_data.db
  raw_data/{SYMBOL}_raw.parquet     ← raw cache (deleted on success if --fresh)
  raw_data/checkpoints/*.json       ← resume cursors (deleted on completion)
  raw_data/pipeline.log
"""

import argparse
import logging
import sys

from config import SYMBOLS, RAW_DIR
from downloader import OKXDownloader
from cleaner import DataCleaner
from storage import StorageManager

# ── Logging ────────────────────────────────────────────────────────────────────
# Configure once here; all modules inherit the same handlers.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RAW_DIR / "pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("pipeline")


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run(symbols: list[str], fresh: bool = False) -> None:
    """
    Execute the full download → clean → store pipeline for each symbol.

    Parameters
    ----------
    symbols : list of OKX instId strings, e.g. ["BTC-USDT", "ETH-USDT"]
    fresh   : if True, delete any existing raw cache before downloading
    """
    storage = StorageManager()

    for symbol in symbols:
        log.info("=" * 64)
        log.info(f"  Symbol : {symbol}")
        log.info("=" * 64)

        try:
            downloader = OKXDownloader(symbol)

            # Optionally wipe the raw cache to force a full re-download
            if fresh and downloader.raw_path.exists():
                downloader.raw_path.unlink()
                log.info(f"[{symbol}] Raw cache cleared (--fresh mode)")

            # ── Stage 1: Download ──────────────────────────────────────────
            raw = downloader.fetch_all()

            # ── Stage 2: Clean ─────────────────────────────────────────────
            clean = DataCleaner(symbol).clean(raw)

            # ── Stage 3: Store ─────────────────────────────────────────────
            storage.save_parquet(clean, symbol)
            storage.save_sqlite(clean, symbol)

            log.info(f"[{symbol}] Pipeline complete ✓")

        except Exception:
            log.exception(f"[{symbol}] Pipeline FAILED — moving to next symbol")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and clean OKX 5-min OHLCV history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--symbols",
        nargs="+",
        default=SYMBOLS,
        metavar="INST_ID",
        help=f"OKX instrument IDs to process (default: {' '.join(SYMBOLS)})",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing raw cache and re-download from scratch",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(symbols=args.symbols, fresh=args.fresh)
