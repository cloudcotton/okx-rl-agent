"""
Data cleaning pipeline for OKX OHLCV data.

Three-stage process:
  1. Reindex  — force a perfectly regular 5-min DatetimeIndex (no gaps, no skips).
  2. Impute   — forward-fill prices for exchange downtime gaps; zero-fill volumes.
  3. Despike  — detect and neutralise physically impossible price spikes using a
                rolling-window Z-score filter with a trailing window (center=False)
                to prevent future data leakage into model training.
"""

import logging

import numpy as np
import pandas as pd

from config import START_TS, END_TS, SPIKE_WINDOW, SPIKE_SIGMA

log = logging.getLogger(__name__)

PRICE_COLS = ["open", "high", "low", "close"]
VOL_COLS   = ["vol", "vol_quote"]


class DataCleaner:

    def __init__(self, symbol: str):
        self.symbol = symbol

    # ── public entry point ────────────────────────────────────────────────────

    def clean(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Input:  raw DataFrame with columns [ts (int64 ms), open, high, low, close,
                vol, vol_quote]
        Output: cleaned DataFrame with a tz-naive UTC DatetimeIndex stored as
                column 'datetime', ready for Parquet / SQLite.
        """
        log.info(f"[{self.symbol}] Cleaning {len(raw):,} raw rows …")

        # ── Stage 1: Reindex ──────────────────────────────────────────────────
        # Build the ground-truth 5-min timeline (strict, no gaps).
        idx = pd.date_range(
            start=pd.Timestamp(START_TS, unit="ms", tz="UTC"),
            end=pd.Timestamp(END_TS,   unit="ms", tz="UTC"),
            freq="5min",
            name="datetime",
        )

        df = raw.copy()
        df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.drop(columns=["ts"])
        df = df[~df.index.duplicated(keep="last")]   # remove any raw duplicates
        df = df.reindex(idx)                          # embed into the standard axis

        n_gaps = int(df["close"].isna().sum())
        pct    = n_gaps / len(idx) * 100
        log.info(
            f"[{self.symbol}] Gaps detected: {n_gaps:,} candles "
            f"({n_gaps * 5 / 60:.1f} min total  /  {pct:.3f}% of timeline)"
        )

        # ── Stage 2: Imputation ───────────────────────────────────────────────
        # Price: forward-fill (never interpolate — that leaks the future).
        #        bfill only for the rare edge case of leading NaN at t=0.
        # Volume: missing candles mean zero trading → fill with 0.
        df[PRICE_COLS] = df[PRICE_COLS].ffill().bfill()
        df[VOL_COLS]   = df[VOL_COLS].fillna(0.0)

        remaining_nan = int(df[PRICE_COLS].isna().sum().sum())
        if remaining_nan:
            log.warning(
                f"[{self.symbol}] {remaining_nan} NaN remain after imputation — "
                "check raw data coverage near START_TS"
            )

        # ── Stage 3: Despike ──────────────────────────────────────────────────
        df = self._despike(df)

        # ── Finalise ──────────────────────────────────────────────────────────
        # Cast to float32 to halve memory footprint in training.
        df = df.astype({c: "float32" for c in PRICE_COLS + VOL_COLS})

        # Strip timezone for broad Parquet / SQLite compatibility;
        # the index represents UTC throughout.
        df = df.reset_index()
        df["datetime"] = df["datetime"].dt.tz_localize(None)

        log.info(
            f"[{self.symbol}] Clean complete: {len(df):,} rows  "
            f"NaN remaining: {df[PRICE_COLS].isna().sum().sum()}"
        )
        return df

    # ── spike removal ─────────────────────────────────────────────────────────

    def _despike(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify candles where a price column deviates by more than SPIKE_SIGMA
        standard deviations from its rolling mean, then replace them with
        forward-filled values.

        Uses a trailing rolling window (center=False) so that the Z-score at
        time t is computed solely from [t-SPIKE_WINDOW+1 … t].  A centred window
        would incorporate future bars (t+1 … t+SPIKE_WINDOW/2), introducing
        look-ahead bias that inflates backtest performance and destroys live
        trading ability.
        """
        for col in PRICE_COLS:
            roll_mean = df[col].rolling(SPIKE_WINDOW, min_periods=10).mean()
            roll_std  = df[col].rolling(SPIKE_WINDOW, min_periods=10).std()

            # Avoid division by zero if std collapses to 0 in a flat region
            safe_std = roll_std.replace(0.0, np.nan)
            z_score  = (df[col] - roll_mean).abs() / safe_std
            mask     = z_score > SPIKE_SIGMA

            n = int(mask.sum())
            if n:
                log.warning(
                    f"[{self.symbol}] Spike filter: {n} anomalous candle(s) in "
                    f"'{col}' (|z| > {SPIKE_SIGMA}) — replaced with ffill"
                )
                df.loc[mask, col] = np.nan
                df[col] = df[col].ffill().bfill()

        return df
