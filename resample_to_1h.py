"""
时间序列降维打击：5分钟 -> 1小时重采样 (Resampling)
"""
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def resample_data(input_path: str, output_path: str):
    log.info(f"正在读取 5 分钟原始数据: {input_path}")
    df_5m = pd.read_parquet(input_path)
    
    # 确保时间列是 datetime 格式并设为索引
    if not pd.api.types.is_datetime64_any_dtype(df_5m['datetime']):
        df_5m['datetime'] = pd.to_datetime(df_5m['datetime'])
    df_5m.set_index('datetime', inplace=True)
    
    log.info("开始重采样至 1 小时 (1H) 级别...")
    
    # 自动探测成交量列名叫什么
    vol_col = 'volume'
    if 'vol' in df_5m.columns:
        vol_col = 'vol'
    elif 'Volume' in df_5m.columns:
        vol_col = 'Volume'

    # 定义 OHLCV 的聚合规则
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        vol_col: 'sum'
    }
    
    # 执行重采样
    df_1h = df_5m.resample('1h').agg(agg_dict).dropna()
    
    # 重新生成最基础的特征 (你可以在这里补回 RSI, MACD 等)
    # log.info("正在重新计算 1 小时级别的基础特征...")
    # df_1h['ret_1'] = df_1h['close'].pct_change(1)
    # df_1h['volatility'] = df_1h['ret_1'].rolling(24).std() # 过去24小时波动率
    # df_1h['ma_20_bias'] = (df_1h['close'] / df_1h['close'].rolling(20).mean()) - 1.0
    
    # 清理 NaN 并恢复 datetime 列
    df_1h = df_1h.dropna().reset_index()
    
    # 保存结果
    df_1h.to_parquet(output_path)
    log.info(f"✅ 1 小时数据生成完毕！总行数: {len(df_1h)}")
    log.info(f"已保存至: {output_path}")

if __name__ == "__main__":
    # 注意：请将这里指向你最原始的带有 OHLCV 的 5 分钟数据文件！
    input_file = "data_center/parquet/ETH_USDT_5m.parquet" # 假设这是你的原始文件
    output_file = "data_center/parquet/ETH_USDT_1h.parquet"
    resample_data(input_file, output_file)