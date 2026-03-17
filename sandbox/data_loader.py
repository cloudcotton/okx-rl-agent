"""
Data loader for the RL trading sandbox.

Loads a single symbol's feature parquet file from data_center/features/
and returns a DataFrame ready for TradingEnv consumption.

One symbol per environment instance — mixing symbols would interleave
two independent price timelines, creating nonsensical transitions at the
boundary rows (e.g., BTC bar at 09:00 followed by ETH bar at 09:00).
Train on multiple symbols by creating one TradingEnv per symbol.

Usage:
    from sandbox.data_loader import load_dataset
    df = load_dataset("BTC-USDT")
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# 完全替换原来的 FEATURE_COLS
FEATURE_COLS = [
    "f_ret_1", "f_ret_4", "f_rsi_14", "f_macd_hist_norm",
    "f_ema_20_bias", "f_ema_60_bias", "f_volatility_24",
    "f_bb_pos", "f_rvol_24"
]
OHLCV_COLS = ["datetime", "open", "high", "low", "close", "vol"]

# Default path relative to this file's location (sandbox/ → data_center/features/)
_DEFAULT_FEAT_DIR = Path(__file__).resolve().parent.parent / "data_center" / "features"


def load_dataset(
    symbol: str,
    feat_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load the feature parquet file for a single symbol.

    Parameters
    ----------
    symbol : str
        e.g. "BTC-USDT". Only one symbol is accepted per call.
        To train on multiple symbols, create one TradingEnv per symbol.
    feat_dir : Path, optional
        Directory containing {SYMBOL}_5m_features.parquet files.
        Defaults to data_center/features/.

    Returns
    -------
    pd.DataFrame
        Columns: [datetime, open, high, low, close, vol] + 14 f_* features.
        dtypes: datetime→datetime64, OHLCV→float64, features→float32.
        Sorted by datetime, index reset.
    """
    feat_dir = Path(feat_dir) if feat_dir else _DEFAULT_FEAT_DIR

    #fname = symbol.replace("-", "_") + "_5m_features.parquet"
    #fname = "ETH_USDT_1h_features.parquet"
    fname = "ETH_USDT_5m_features_new.parquet"
    path  = feat_dir / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found for {symbol}: {path}\n"
            f"Run: python data_center/feature_builder.py --symbols {symbol}"
        )

    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    log.info(f"[{symbol}] Loaded {len(df):,} rows from {path.name}")

    # Validate columns
    required = set(OHLCV_COLS + FEATURE_COLS)
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame missing required columns: {sorted(missing)}\n"
            "Re-run feature_builder.py to regenerate features."
        )

    # Keep only what the env needs
    keep_cols = OHLCV_COLS + FEATURE_COLS
    df = df[[c for c in keep_cols if c in df.columns]]

    log.info(
        f"[{symbol}] Dataset ready: {len(df):,} rows | "
        f"{df['datetime'].min()} → {df['datetime'].max()}"
    )
    return df
