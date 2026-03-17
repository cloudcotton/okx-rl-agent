"""
1 小时级别高维特征铸造器 (1H Feature Engineering)
================================================
读取 1 小时 OHLCV 数据，计算动量、趋势、波动率等量化特征，
输出可直接喂给 RL Agent 的高质特征表。
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """计算 RSI 并返回"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def generate_1h_features(input_path: str, output_path: str):
    log.info(f"正在加载 1H 原始数据: {input_path}")
    df = pd.read_parquet(input_path)
    
    # 确保按时间排序
    if 'datetime' in df.columns:
        df = df.sort_values('datetime').reset_index(drop=True)
    
    log.info("开始铸造 1H 级别核心量化特征...")

    # ── 1. 动量特征 (Momentum) ──
    df['f_ret_1'] = df['close'].pct_change(1)           # 1小时涨跌幅
    df['f_ret_4'] = df['close'].pct_change(4)           # 4小时动量 (小波段方向)
    
    # ── 2. 均值回归特征 (Mean Reversion) ──
    # RSI: 缩放到 -1 到 1 之间 (原值 0-100，减去50再除以50)
    df['f_rsi_14'] = (calc_rsi(df['close'], 14) - 50.0) / 50.0 
    
    # ── 3. 趋势追踪特征 (Trend Following) ──
    # MACD Histogram (柱状图代表加速或减速)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    # 用收盘价对 MACD 进行标准化，使其不受绝对币价影响
    df['f_macd_hist_norm'] = (macd - macd_signal) / df['close'] * 1000.0

    # 均线偏离度 (Bias)
    df['f_ema_20_bias'] = (df['close'] / df['close'].ewm(span=20, adjust=False).mean()) - 1.0
    df['f_ema_60_bias'] = (df['close'] / df['close'].ewm(span=60, adjust=False).mean()) - 1.0

    # ── 4. 波动率与微观结构 (Volatility & Microstructure) ──
    # 过去 24 小时波动率
    df['f_volatility_24'] = df['f_ret_1'].rolling(24).std()
    
    # 布林带相对位置 (Bollinger Band Position): >1突破上轨, <-1跌破下轨
    rolling_mean = df['close'].rolling(20).mean()
    rolling_std = df['close'].rolling(20).std()
    df['f_bb_pos'] = (df['close'] - rolling_mean) / (rolling_std * 2 + 1e-8)
    
    # 相对成交量 (RVOL): 突破时是否放量
    # 先自动探测真正的成交量列名叫什么
    vol_col = 'volume'
    if 'vol' in df.columns:
        vol_col = 'vol'
    elif 'Volume' in df.columns:
        vol_col = 'Volume'
        
    df['f_rvol_24'] = df[vol_col] / (df[vol_col].rolling(24).mean() + 1e-8)

    # ── 5. 数据清洗 ──
    # 砍掉开头因为计算均线和移动窗口产生的 NaN 废数据 (大约需要丢弃前 60 行)
    df_clean = df.dropna().reset_index(drop=True)
    
    log.info(f"特征提取完成！有效数据行数: {len(df_clean)}")
    
    df_clean.to_parquet(output_path)
    log.info(f"✅ 已保存最终特征集至: {output_path}")

if __name__ == "__main__":
    # 输入：刚才重采样出来的 1H 基础数据
    input_file = "data_center/parquet/ETH_USDT_1h.parquet" 
    # 输出：带有了全部 AI 特征的最终数据集
    output_file = "data_center/features/ETH_USDT_1h_features.parquet"
    
    generate_1h_features(input_file, output_file)