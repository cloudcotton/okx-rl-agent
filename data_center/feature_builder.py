"""
Feature builder: compute 30 RL-ready features from clean 5-min OHLCV parquet.

Output:  data_center/features/{SYMBOL}_5m_features.parquet
Columns: original OHLCV + 30 f_* feature columns (float32, clipped to ±10).
NaN head rows (~350) dropped — never filled.

Usage:
    python feature_builder.py                      # all symbols
    python feature_builder.py --symbols BTC-USDT   # specific symbol(s)
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Directory layout ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent          # data_center/
PARQ_DIR = DATA_DIR / "parquet"
FEAT_DIR = DATA_DIR / "features"
SYMBOLS  = ["BTC-USDT", "ETH-USDT"]

# ── Constants ─────────────────────────────────────────────────────────────────
EPS      = 1e-8
Z_WINDOW = 288   # 24 h of 5-min bars

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """Trailing rolling Z-score (center=False, no future leakage)."""
    mu  = s.rolling(window, min_periods=window).mean()
    sig = s.rolling(window, min_periods=window).std(ddof=1)
    return (s - mu) / (sig + EPS)


def _ema(s: pd.Series, span: int) -> pd.Series:
    """Standard EMA; NaN until the first full span is available."""
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _wilder(s: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing — equivalent to EWM with alpha = 1/period."""
    return s.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    avg_gain = _wilder(delta.clip(lower=0.0), period)
    avg_loss = _wilder((-delta).clip(lower=0.0), period)
    rs       = avg_gain / (avg_loss + EPS)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return _wilder(tr, period)


# ── Feature computation ───────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 30 f_* features from a clean OHLCV DataFrame.

    Parameters
    ----------
    df : DataFrame with columns [datetime, open, high, low, close, vol, vol_quote]

    Returns
    -------
    DataFrame: original columns + 30 f_* columns (float32, clipped ±10),
               NaN head rows dropped.
    """
    d = df.copy()

    op    = d["open"]
    hi    = d["high"]
    lo    = d["low"]
    cl    = d["close"]
    vol   = d["vol"]

    hl_range  = hi - lo + EPS
    prev_cl   = cl.shift(1)
    prev_hi   = hi.shift(1)
    prev_lo   = lo.shift(1)

    # ── 4.1  Micro Price Action (7) ───────────────────────────────────────────
    d["f_dir_body_ratio"] = (cl - op) / hl_range
    d["f_abs_body_ratio"] = (cl - op).abs() / hl_range
    d["f_upper_shadow"]   = (hi - np.maximum(op, cl)) / hl_range
    d["f_lower_shadow"]   = (np.minimum(op, cl) - lo) / hl_range
    d["f_close_pos"]      = (cl - lo) / hl_range

    overlap_num           = np.minimum(hi, prev_hi) - np.maximum(lo, prev_lo)
    d["f_overlap_ratio"]  = overlap_num / hl_range

    tr = np.maximum(np.maximum(hi - lo, (hi - prev_cl).abs()), (lo - prev_cl).abs())
    d["f_true_range_ratio"] = tr / (prev_cl + EPS)

    # ── 4.2  Momentum & Returns (6) ───────────────────────────────────────────
    log_ret_1 = np.log(cl / (prev_cl + EPS))

    d["f_log_ret_1"]     = log_ret_1
    d["f_log_ret_3"]     = np.log(cl / (cl.shift(3)  + EPS))
    d["f_log_ret_12"]    = np.log(cl / (cl.shift(12) + EPS))
    d["f_log_ret_accel"] = log_ret_1 - log_ret_1.shift(1)
    d["f_rsi_norm"]      = (_rsi(cl, 14) - 50.0) / 50.0

    ema12      = _ema(cl, 12)
    ema26      = _ema(cl, 26)
    macd_line  = ema12 - ema26
    macd_hist  = macd_line - _ema(macd_line, 9)
    d["f_macd_hist_z"] = _rolling_zscore(macd_hist, Z_WINDOW)

    # ── 4.3  Volatility & Market Regime (5) ───────────────────────────────────
    atr14 = _atr(hi, lo, cl, 14)

    d["f_atr_norm"]            = atr14 / (cl + EPS)
    d["f_volatility_breakout"] = (hi - lo) / (atr14 + EPS)
    d["f_rolling_vol_24"]      = log_ret_1.rolling(24, min_periods=24).std(ddof=1)

    log_hl_sq = np.log(hi / (lo + EPS)) ** 2
    d["f_parkinson_vol"] = np.sqrt(
        log_hl_sq.rolling(14, min_periods=14).mean() / (4.0 * np.log(2))
    )

    bb_mid   = cl.rolling(20, min_periods=20).mean()
    bb_std   = cl.rolling(20, min_periods=20).std(ddof=1)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    d["f_bb_width"] = (bb_upper - bb_lower) / (bb_mid + EPS)

    # ── 4.4  Moving Average Context (5) ───────────────────────────────────────
    ema9  = _ema(cl, 9)
    ema50 = _ema(cl, 50)

    d["f_ema_fast_div_z"] = _rolling_zscore((cl - ema9)  / (ema9  + EPS), Z_WINDOW)
    d["f_ema_slow_div_z"] = _rolling_zscore((cl - ema50) / (ema50 + EPS), Z_WINDOW)
    d["f_ema_cross_dist"] = (ema9 - ema50) / (ema50 + EPS)
    d["f_bb_position"]    = (cl - bb_lower) / (bb_upper - bb_lower + EPS)

    rolling_high_24       = hi.rolling(24, min_periods=24).max()
    d["f_dist_to_high_24"] = (cl - rolling_high_24) / (rolling_high_24 + EPS)

    # ── 4.5  Volume Dynamics (5) ──────────────────────────────────────────────
    sma_vol_24 = vol.rolling(24, min_periods=24).mean()
    rvol       = vol / (sma_vol_24 + EPS)

    d["f_rvol"]        = rvol
    d["f_log_vol_chg"] = np.log(vol + 1) - np.log(vol.shift(1) + 1)
    d["f_vwpc_z"]      = _rolling_zscore(vol * log_ret_1, Z_WINDOW)

    # Divergence: current bar breaks above/below the *previous* 12-bar extreme
    prev_high_12 = hi.shift(1).rolling(12, min_periods=12).max()
    prev_low_12  = lo.shift(1).rolling(12, min_periods=12).min()
    is_new_high  = (hi > prev_high_12) & (rvol < 1.0)
    is_new_low   = (lo < prev_low_12)  & (rvol < 1.0)
    d["f_vol_divergence"] = np.where(is_new_high, -1.0,
                            np.where(is_new_low,   1.0, 0.0))

    money_flow = vol * ((cl - lo) - (hi - cl)) / hl_range
    d["f_money_flow_z"] = _rolling_zscore(money_flow, Z_WINDOW)

    # ── 4.6  Time Encoding (2) ────────────────────────────────────────────────
    dt        = pd.to_datetime(d["datetime"])
    hour_frac = dt.dt.hour + dt.dt.minute / 60.0
    angle     = 2.0 * np.pi * hour_frac / 24.0
    d["f_sin_hour"] = np.sin(angle)
    d["f_cos_hour"] = np.cos(angle)

    # ── Finalise ──────────────────────────────────────────────────────────────
    d.dropna(inplace=True)
    d.reset_index(drop=True, inplace=True)

    feat_cols = [col for col in d.columns if col.startswith("f_")]
    d[feat_cols] = d[feat_cols].clip(lower=-10.0, upper=10.0).astype("float32")

    return d


# ── I/O ───────────────────────────────────────────────────────────────────────

def build_features(symbol: str) -> None:
    parq_path = PARQ_DIR / f"{symbol.replace('-', '_')}_5m.parquet"
    if not parq_path.exists():
        log.error(f"[{symbol}] Source parquet not found: {parq_path}")
        return

    log.info(f"[{symbol}] Loading {parq_path} …")
    df = pd.read_parquet(parq_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    raw_rows = len(df)

    log.info(f"[{symbol}] {raw_rows:,} raw rows — computing features …")
    out = compute_features(df)

    feat_cols = [col for col in out.columns if col.startswith("f_")]
    dropped   = raw_rows - len(out)
    log.info(
        f"[{symbol}] {len(feat_cols)} feature columns | "
        f"{len(out):,} rows kept | {dropped:,} head rows dropped (NaN warm-up)"
    )

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEAT_DIR / f"{symbol.replace('-', '_')}_5m_features.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    size_mb = out_path.stat().st_size / 1e6
    log.info(f"[{symbol}] Saved → {out_path}  ({size_mb:.1f} MB)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate RL feature tables from clean OHLCV parquet."
    )
    p.add_argument(
        "--symbols", nargs="+", default=SYMBOLS,
        help="Symbols to process (default: all in SYMBOLS list)",
    )
    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    ok = fail = 0
    for sym in args.symbols:
        try:
            build_features(sym)
            ok += 1
        except Exception as exc:
            log.exception(f"[{sym}] Failed: {exc}")
            fail += 1
    log.info(f"Done — {ok} succeeded, {fail} failed.")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
