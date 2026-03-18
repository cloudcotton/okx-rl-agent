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

# 15m MTF 特征集（13 个）— 与 generate_15m_features.py 保持同步
# 三层结构：15m 入场信号 | 1h 方向过滤 | 4h 大趋势
FEATURE_COLS = [
    # 15m 层
    "f_ret_1",
    "f_ret_4",
    "f_rsi_14",
    "f_macd_hist_norm",
    "f_atr_norm",
    "f_bb_pos",
    "f_rvol",
    # 1h 层（4 × 15m）
    "f_1h_trend",
    "f_1h_rsi",
    # 4h 层（16 × 15m）
    "f_4h_position",
    "f_4h_bias",
    # 时间编码
    "f_sin_hour",
    "f_cos_hour",
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
        Columns: [datetime, open, high, low, close, vol] + 13 f_* features.
        dtypes: datetime→datetime64, OHLCV→float64, features→float32.
        Sorted by datetime, index reset.
    """
    feat_dir = Path(feat_dir) if feat_dir else _DEFAULT_FEAT_DIR

    fname = symbol.replace("-", "_") + "_15m_features.parquet"
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
