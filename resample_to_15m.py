"""
5分钟 -> 15分钟重采样 (Resampling to 15-minute bars)

Usage:
    python resample_to_15m.py
"""
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def resample_to_15m(input_path: str, output_path: str) -> None:
    log.info(f"读取 5 分钟原始数据: {input_path}")
    df = pd.read_parquet(input_path)

    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)

    # 探测成交量列名
    vol_col = "vol"
    if "volume" in df.columns:
        vol_col = "volume"
    elif "Volume" in df.columns:
        vol_col = "Volume"

    agg_dict = {
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
        vol_col: "sum",
    }

    log.info("重采样至 15 分钟...")
    df_15m = df.resample("15min").agg(agg_dict).dropna()

    # 统一列名为 vol
    if vol_col != "vol":
        df_15m.rename(columns={vol_col: "vol"}, inplace=True)

    df_15m = df_15m.reset_index()
    df_15m.to_parquet(output_path, index=False)
    log.info(f"完成！共 {len(df_15m):,} 行 | 已保存至: {output_path}")
    log.info(f"时间范围: {df_15m['datetime'].min()} → {df_15m['datetime'].max()}")


if __name__ == "__main__":
    input_file  = "data_center/parquet/ETH_USDT_5m.parquet"
    output_file = "data_center/parquet/ETH_USDT_15m.parquet"
    resample_to_15m(input_file, output_file)
