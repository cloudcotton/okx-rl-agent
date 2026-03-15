"""
OKX historical candle downloader.

Design:
  - Reverse-time cursor traversal using the `before` parameter (ts < before).
  - 1-candle overlap on every page boundary to guard against edge gaps;
    deduplication is done after concat.
  - Checkpoint file records the cursor after each page so a crash can resume
    exactly where it left off.
  - Exponential backoff on HTTP 429 or transient network errors.
  - Accumulated raw data is persisted to Parquet in DATA_DIR after each run.
"""

import json
import time
import logging
from datetime import datetime, timezone

import requests
import pandas as pd

from config import (
    OKX_BASE, HISTORY_ENDPOINT, RECENT_ENDPOINT,
    BAR, BAR_MS, LIMIT,
    REQ_INTERVAL, MAX_RETRIES, BACKOFF_BASE,
    START_TS, END_TS,
    DATA_DIR, CKPT_DIR,
)

log = logging.getLogger(__name__)

# OKX history-candles response columns (positional)
_OKX_COLS = ["ts", "open", "high", "low", "close", "vol", "vol_quote", "vol_ccy_quote", "confirm"]
_KEEP     = ["ts", "open", "high", "low", "close", "vol", "vol_quote"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _ms_to_str(ms: int) -> str:
    """Human-readable UTC string for a millisecond timestamp."""
    return datetime.fromtimestamp(ms / 1_000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _parse_page(data: list) -> pd.DataFrame:
    """
    Convert raw OKX list-of-lists response into a typed DataFrame.
    vol_quote  = volCcy  = USDT-denominated volume  (most useful for RL features)
    """
    df = pd.DataFrame(data, columns=_OKX_COLS)[_KEEP].copy()
    df["ts"] = df["ts"].astype("int64")
    for col in ["open", "high", "low", "close", "vol", "vol_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── main class ────────────────────────────────────────────────────────────────

class OKXDownloader:
    """
    Fetches the full 5-min OHLCV history for one symbol from OKX.

    Pagination strategy (backwards in time):
        cursor starts at  END_TS + 1 bar
        each page:  before=cursor  →  OKX returns rows where ts < cursor
        next cursor = oldest ts in the page  (exclusive, so no duplicate)
        stop when oldest ts <= START_TS  or  OKX returns an empty page
    """

    def __init__(self, symbol: str):
        self.symbol   = symbol
        self.slug     = symbol.replace("-", "_")
        self.ckpt     = CKPT_DIR / f"{self.slug}.json"
        self.raw_path = DATA_DIR / f"{self.slug}_raw.parquet"
        self.session  = requests.Session()
        self.session.headers["User-Agent"] = "okx-rl-pipeline/1.0"

    # ── checkpoint helpers ────────────────────────────────────────────────────

    def _load_cursor(self) -> int:
        if self.ckpt.exists():
            d = json.loads(self.ckpt.read_text())
            log.info(f"[{self.symbol}] Resuming from checkpoint: {_ms_to_str(d['cursor'])}")
            return int(d["cursor"])
        return END_TS + BAR_MS   # first run: start just past the end date

    def _save_cursor(self, cursor: int) -> None:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        self.ckpt.write_text(json.dumps({"symbol": self.symbol, "cursor": cursor}))

    def _clear_ckpt(self) -> None:
        if self.ckpt.exists():
            self.ckpt.unlink()

    # ── single-page fetch with retry / backoff ────────────────────────────────

    def _fetch_page(self, before_ms: int, endpoint: str = HISTORY_ENDPOINT) -> list:
        params = {
            "instId": self.symbol,
            "bar":    BAR,
            "before": str(before_ms),
            "limit":  str(LIMIT),
        }
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(
                    OKX_BASE + endpoint,
                    params=params,
                    timeout=15,
                )
                if resp.status_code == 429:
                    wait = BACKOFF_BASE ** attempt
                    log.warning(
                        f"[{self.symbol}] HTTP 429 — exponential backoff {wait:.0f}s "
                        f"(attempt {attempt}/{MAX_RETRIES})"
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                body = resp.json()

                if body.get("code") != "0":
                    raise ValueError(
                        f"OKX API error code={body.get('code')} msg={body.get('msg')}"
                    )
                return body.get("data", [])

            except (requests.RequestException, ValueError) as exc:
                wait = BACKOFF_BASE ** attempt
                log.warning(
                    f"[{self.symbol}] Attempt {attempt}/{MAX_RETRIES} failed: {exc} "
                    f"— retrying in {wait:.0f}s"
                )
                time.sleep(wait)

        raise RuntimeError(
            f"[{self.symbol}] All {MAX_RETRIES} retries exhausted (before={before_ms})"
        )

    # ── full sweep ────────────────────────────────────────────────────────────

    def fetch_all(self) -> pd.DataFrame:
        """
        Download complete history for self.symbol.

        Returns a de-duplicated, time-sorted DataFrame with columns:
            ts (int64 ms), open, high, low, close, vol, vol_quote (all float64)
        """
        CKPT_DIR.mkdir(parents=True, exist_ok=True)

        # ── load previously accumulated raw data (checkpoint resume) ──────────
        frames: list[pd.DataFrame] = []
        if self.raw_path.exists():
            existing = pd.read_parquet(self.raw_path)
            frames.append(existing)
            log.info(
                f"[{self.symbol}] Loaded {len(existing):,} existing rows from raw cache"
            )

        cursor = self._load_cursor()
        total  = sum(len(f) for f in frames)
        pages  = 0

        log.info(
            f"[{self.symbol}] Sweep: {_ms_to_str(END_TS)} ← {_ms_to_str(START_TS)} "
            f"starting cursor={_ms_to_str(cursor)}"
        )

        while cursor > START_TS:
            page = self._fetch_page(cursor)

            if not page:
                log.info(
                    f"[{self.symbol}] Empty response at cursor={_ms_to_str(cursor)} "
                    f"— OKX history exhausted."
                )
                break

            df      = _parse_page(page)
            oldest  = int(df["ts"].min())
            newest  = int(df["ts"].max())

            # Clip rows that fall before the desired start date
            df_clip = df[df["ts"] >= START_TS]
            if not df_clip.empty:
                frames.append(df_clip)
                total += len(df_clip)

            pages += 1
            log.info(
                f"[{self.symbol}] page={pages:4d}  rows={len(page):3d}  "
                f"[{_ms_to_str(oldest)} → {_ms_to_str(newest)}]  "
                f"cumulative={total:,}"
            )

            # Advance cursor backwards in time.
            # `before` is exclusive, so next page will return ts < oldest — no overlap.
            # We add 1 ms so we re-request the oldest candle itself as a safety overlap;
            # duplicates are removed during the final concat.
            cursor = oldest + 1
            self._save_cursor(cursor)

            time.sleep(REQ_INTERVAL)

            if oldest <= START_TS:
                log.info(f"[{self.symbol}] Reached target start date. Sweep complete.")
                break

        if not frames:
            raise ValueError(f"[{self.symbol}] No data was fetched — check symbol / date range.")

        combined = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["ts"])
            .sort_values("ts")
            .reset_index(drop=True)
        )

        # Persist raw data so interrupted runs can resume without re-downloading
        combined.to_parquet(self.raw_path, index=False)
        log.info(
            f"[{self.symbol}] Raw data persisted → {self.raw_path}  "
            f"({len(combined):,} rows, "
            f"{_ms_to_str(int(combined['ts'].min()))} → {_ms_to_str(int(combined['ts'].max()))})"
        )

        self._clear_ckpt()
        return combined
