"""
15 分钟级别多时间框架特征生成器 (15m MTF Feature Engineering)
============================================================
读取 15 分钟 OHLCV 数据，计算三个时间层次的特征：
  - 15m 层：入场信号（RSI、MACD、ATR、波动率）
  - 1h 层：方向过滤（4 × 15m = 1h，EMA 方向 / RSI 状态）
  - 4h 层：大趋势判断（16 × 15m = 4h，价格区间位置）

特征说明
--------
  f_ret_1          15m 收益率 (1-bar pct_change)
  f_ret_4          1h  收益率 (4-bar pct_change)
  f_rsi_14         RSI(14) 缩放到 [-1, 1]，14 × 15m = 3.5h
  f_macd_hist_norm MACD 柱状图 / close × 1000（去量纲）
  f_atr_norm       ATR(14) / close（波动率水平）
  f_bb_pos         布林带相对位置 [-1, 1]（>1 突破上轨）
  f_rvol           相对成交量 (当前 / 24-bar 均量)
  f_1h_trend       1h 级别趋势方向：EMA(4) 斜率符号 ∈ {-1, 0, +1}
  f_1h_rsi         1h 级别 RSI：用 4 根 15m K 线的最高/最低还原
  f_4h_position    价格在 16-bar (4h) 高低区间内的相对位置 [0, 1]
  f_4h_bias        价格相对 16-bar EMA 的乖离率（大趋势偏离度）
  f_sin_hour       小时正弦编码（交易时段周期性）
  f_cos_hour       小时余弦编码

共 13 个特征，观测空间维度 = 15（13 特征 + 仓位 + 浮盈亏）

Usage:
    python generate_15m_features.py
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EPS = 1e-8


# ── 技术指标工具函数 ─────────────────────────────────────────────────────────

def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI，返回 0-100。"""
    delta    = close.diff()
    gain     = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    loss     = (-delta).clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs       = gain / (loss + EPS)
    return 100.0 - 100.0 / (1.0 + rs)


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ── 主特征生成函数 ────────────────────────────────────────────────────────────

def generate_15m_features(input_path: str, output_path: str) -> None:
    log.info(f"加载 15m 原始数据: {input_path}")
    df = pd.read_parquet(input_path)

    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)

    # 探测成交量列名
    vol_col = "vol"
    if "volume" in df.columns:
        vol_col = "volume"
    elif "Volume" in df.columns:
        vol_col = "Volume"

    log.info("计算 15m MTF 特征（三层时间框架）...")

    op = df["open"]
    hi = df["high"]
    lo = df["low"]
    cl = df["close"]
    vol = df[vol_col]

    # ── 1. 15m 层：入场信号 ────────────────────────────────────────────────

    # 动量
    df["f_ret_1"] = cl.pct_change(1)
    df["f_ret_4"] = cl.pct_change(4)   # 1h 收益率（4 × 15m）

    # RSI：RSI(14) 在 15m 上 = 3.5 小时，有足够的统计意义
    df["f_rsi_14"] = (_wilder_rsi(cl, 14) - 50.0) / 50.0   # 缩放到 [-1, 1]

    # MACD：12/26/9 在 15m 上分别代表 3h / 6.5h / 2.25h
    ema12      = _ema(cl, 12)
    ema26      = _ema(cl, 26)
    macd_line  = ema12 - ema26
    macd_sig   = _ema(macd_line, 9)
    df["f_macd_hist_norm"] = (macd_line - macd_sig) / (cl + EPS) * 1000.0

    # ATR
    atr14 = _atr(hi, lo, cl, 14)
    df["f_atr_norm"] = atr14 / (cl + EPS)

    # 布林带相对位置
    bb_mid = cl.rolling(20, min_periods=20).mean()
    bb_std = cl.rolling(20, min_periods=20).std()
    df["f_bb_pos"] = (cl - bb_mid) / (bb_std * 2 + EPS)

    # 相对成交量（vs 过去 24 根 15m K 线均量 = 6 小时）
    sma_vol = vol.rolling(24, min_periods=24).mean()
    df["f_rvol"] = vol / (sma_vol + EPS)

    # ── 2. 1h 层：方向过滤（4 × 15m = 1h）────────────────────────────────

    # 1h EMA 方向：EMA(4) 斜率符号（正 = 上升，负 = 下降，零 = 平盘）
    ema4 = _ema(cl, 4)
    ema4_slope = ema4 - ema4.shift(1)
    # 用 ATR 标准化斜率，避免绝对价格影响；符号截断成 {-1, 0, +1}
    slope_norm = ema4_slope / (atr14 + EPS)
    df["f_1h_trend"] = slope_norm.clip(-1.0, 1.0)   # 连续值保留，让模型自己判断强弱

    # 1h RSI：对最近 4 根 K 线的开高低收做聚合后计算 RSI
    # 用滚动 4 根的 close 均值近似 1h close，计算简易 1h RSI
    cl_1h_proxy = cl.rolling(4).mean()
    df["f_1h_rsi"] = (_wilder_rsi(cl_1h_proxy, 14) - 50.0) / 50.0

    # ── 3. 4h 层：大趋势（16 × 15m = 4h）────────────────────────────────

    # 4h 高低区间位置：[0, 1]，0 = 区间底部，1 = 区间顶部
    high_4h = hi.rolling(16, min_periods=16).max()
    low_4h  = lo.rolling(16, min_periods=16).min()
    df["f_4h_position"] = (cl - low_4h) / (high_4h - low_4h + EPS)

    # 4h EMA 乖离率：价格偏离大级别均线的程度
    ema_4h = _ema(cl, 16)   # 16-bar EMA ≈ 4h EMA
    df["f_4h_bias"] = (cl / (ema_4h + EPS)) - 1.0

    # ── 4. 时间编码 ─────────────────────────────────────────────────────

    dt = pd.to_datetime(df["datetime"])
    hour_frac = dt.dt.hour + dt.dt.minute / 60.0
    angle = 2.0 * np.pi * hour_frac / 24.0
    df["f_sin_hour"] = np.sin(angle)
    df["f_cos_hour"] = np.cos(angle)

    # ── 5. 清洗与输出 ────────────────────────────────────────────────────

    df_clean = df.dropna().reset_index(drop=True)
    log.info(f"有效数据行数: {len(df_clean):,}（丢弃 {len(df) - len(df_clean):,} 行暖启动 NaN）")

    # 特征裁剪到 ±10（与 TradingEnv 观测空间对齐）
    feat_cols = [c for c in df_clean.columns if c.startswith("f_")]
    df_clean[feat_cols] = df_clean[feat_cols].clip(-10.0, 10.0).astype("float32")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(output_path, index=False)
    log.info(f"已保存至: {output_path}")
    log.info(f"特征列（共 {len(feat_cols)} 个）: {feat_cols}")


if __name__ == "__main__":
    input_file  = "data_center/parquet/ETH_USDT_15m.parquet"
    output_file = "data_center/features/ETH_USDT_15m_features.parquet"
    generate_15m_features(input_file, output_file)
