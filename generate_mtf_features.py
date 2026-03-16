"""
多时间级别共振特征生成器 (Multi-Timeframe Feature Generator)
============================================================
不依赖外部网络，直接将 5 分钟 K 线升维，提取大级别的趋势特征。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def generate_macro_features(input_path: str):
    log.info(f"正在加载基础 5 分钟数据: {input_path}")
    df = pd.read_parquet(input_path)
    
    # 确保时间序列是排序的
    df = df.sort_values("datetime").reset_index(drop=True)
    
    log.info("开始铸造大级别共振特征 (Multi-Timeframe Context)...")
    
    # 1. 一小时级别的均线方向 (12 根 5 分钟 K 线 = 1 小时)
    # 我们看过去 12 小时的趋势 (12 * 12 = 144 根 5 分钟 K 线)
    df["close_1h_ema"] = df["close"].ewm(span=144, adjust=False).mean()
    # 距离 1 小时级别均线的乖离率（辨别是大级别顺势还是逆势）
    df["f_macro_1h_bias"] = (df["close"] / df["close_1h_ema"]) - 1.0
    
    # 2. 四小时级别的波动率 (48 根 5 分钟 K 线 = 4 小时)
    # 计算大级别的最高价和最低价差
    df["high_4h"] = df["high"].rolling(48).max()
    df["low_4h"] = df["low"].rolling(48).min()
    # 大级别 ATR (当前收盘价相对于 4 小时波幅的位置)
    df["f_macro_4h_position"] = (df["close"] - df["low_4h"]) / (df["high_4h"] - df["low_4h"] + 1e-8)
    
    # 3. 动量共振 (Momentum Resonance)
    # 5分钟涨幅 vs 1小时涨幅。如果两个都在涨，就是共振突破
    df["ret_5m"] = df["close"].pct_change(1)
    df["ret_1h"] = df["close"].pct_change(12)
    df["f_momentum_sync"] = np.sign(df["ret_5m"]) * np.sign(df["ret_1h"])
    
    # 清理计算产生的 NaN
    df = df.dropna().reset_index(drop=True)
    
    out_path = Path(input_path).parent / "ETH_USDT_5m_macro_features.parquet"
    df.to_parquet(out_path)
    
    log.info(f"✅ 大级别特征铸造完毕！已保存至: {out_path.name}")
    log.info("请将以下 3 个新特征加入到 sandbox/data_loader.py 的 FEATURE_COLS 中:")
    log.info("['f_macro_1h_bias', 'f_macro_4h_position', 'f_momentum_sync']")

if __name__ == "__main__":
    # 替换为你实际的 5 分钟特征文件路径
    generate_macro_features("data_center/ETH_USDT_5m_features.parquet")