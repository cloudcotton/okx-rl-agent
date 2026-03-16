"""
OKX 微观订单流数据抓取与对齐脚本 (Order Flow Fetcher)
======================================================
抓取指定永续合约的持仓量(OI)和资金费率(Funding Rate)，
并与现有的 5 分钟 OHLCV 数据完美对齐。

注意: OKX 的 Rubik (持仓量) API 免费额度较高，可以直接裸连抓取。
"""

import time
import logging
from pathlib import Path
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# OKX API 基础配置
BASE_URL = "https://www.okx.com"
INST_ID = "ETH-USDT-SWAP"  # 注意：持仓和资金费率只有 SWAP(永续) 或 季度合约 才有

def fetch_funding_rate(inst_id: str, limit: int = 100) -> pd.DataFrame:
    """抓取资金费率历史 (通常为 8 小时一次)"""
    url = f"{BASE_URL}/api/v5/public/funding-rate-history"
    all_data = []
    after = ""
    
    log.info(f"开始抓取 {inst_id} 资金费率...")
    while True:
        params = {"instId": inst_id, "limit": limit}
        if after:
            params["after"] = after
            
        try:
            res = requests.get(url, params=params, timeout=10).json()
            if res["code"] != "0" or not res["data"]:
                break
                
            all_data.extend(res["data"])
            after = res["data"][-1]["fundingTime"]  # 用于下一页的分页游标
            time.sleep(0.1)  # 遵守 API 频率限制 (10次/秒)
            
            if len(all_data) % 1000 == 0:
                log.info(f"已抓取 {len(all_data)} 条资金费率数据...")
                
        except Exception as e:
            log.error(f"抓取资金费率中断: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["datetime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["fundingRate"] = df["fundingRate"].astype(float)
    # 按时间正序排列
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "fundingRate"]]


def fetch_open_interest(inst_id: str, period: str = "5m") -> pd.DataFrame:
    """
    抓取合约持仓量历史 (Open Interest)
    OKX Rubik 接口，period 支持 '5m', '1H', '1D' 等
    """
    url = f"{BASE_URL}/api/v5/rubik/stat/contracts/open-interest-history"
    all_data = []
    after = ""
    
    log.info(f"开始抓取 {inst_id} {period} 级别持仓量...")
    while True:
        params = {"instId": inst_id, "period": period}
        if after:
            params["after"] = after
            
        try:
            res = requests.get(url, params=params, timeout=10).json()
            if res["code"] != "0" or not res["data"]:
                break
                
            all_data.extend(res["data"])
            after = res["data"][-1][0]  # 数据结构: [ts, oi, oiCcy]
            time.sleep(0.2)  # Rubik 接口限频较严
            
            if len(all_data) % 5000 == 0:
                log.info(f"已抓取 {len(all_data)} 条持仓量数据...")
                
        except Exception as e:
            log.error(f"抓取持仓量中断: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    # OKX 返回格式: [ts, oi (张数), oiCcy (币数)]
    df = pd.DataFrame(all_data, columns=["ts", "oi", "oiCcy"])
    df["datetime"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    df["oi"] = df["oi"].astype(float)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "oi"]]


def merge_orderflow_features(ohlcv_path: Path):
    """将微观数据与基础 K 线融合，并生成衍生特征"""
    log.info(f"加载基础 K 线数据: {ohlcv_path.name}")
    # 假设你之前有一个最原始的带有 datetime 和 OHLCV 的 DataFrame
    df_main = pd.read_parquet(ohlcv_path)
    
    # 1. 获取订单流数据
    df_fund = fetch_funding_rate(INST_ID)
    df_oi = fetch_open_interest(INST_ID, period="5m")
    
    # 2. 合并资金费率 (Forward Fill 向前填充)
    # 因为资金费率是 8 小时一次，我们需要把这 8 小时内的每一根 5 分钟 K 线都打上相同的费率标签
    df_main = pd.merge_asof(
        df_main.sort_values("datetime"),
        df_fund.sort_values("datetime"),
        on="datetime",
        direction="backward" # 使用上一次结算的费率
    )
    
    # 3. 合并持仓量 (精准对齐)
    df_main = pd.merge(df_main, df_oi, on="datetime", how="left")
    df_main["oi"] = df_main["oi"].ffill() # 填补偶尔缺失的断点
    
    # ── 4. 铸造杀手级衍生特征 (Feature Engineering) ──
    
    # 特征 A: 持仓量变化率 (OI Delta) - 揭示资金是在进场还是离场
    df_main["f_oi_chg_1"] = df_main["oi"].pct_change(1)
    df_main["f_oi_chg_12"] = df_main["oi"].pct_change(12) # 过去1小时的资金流入流出
    
    # 特征 B: 量价与持仓量的共振 (Price-OI Divergence)
    # 价格涨 + OI涨 = 主力真金白银做多 (强看涨)
    # 价格跌 + OI跌 = 散户止损平仓 (可能反弹)
    df_main["f_trend_strength"] = np.sign(df_main["close"].pct_change(1)) * np.sign(df_main["f_oi_chg_1"])
    
    # 特征 C: 极端资金费率警告
    # 费率极高说明散户都在疯狂做多，主力准备镰刀伺候
    df_main["f_funding_z"] = (df_main["fundingRate"] - df_main["fundingRate"].rolling(288).mean()) / (df_main["fundingRate"].rolling(288).std() + 1e-8)
    
    # 清理 NaN
    df_main = df_main.dropna().reset_index(drop=True)
    
    # 保存融合后的新数据集
    out_path = ohlcv_path.parent / f"{INST_ID}_orderflow_features.parquet"
    df_main.to_parquet(out_path)
    log.info(f"微观特征生成完毕！已保存至: {out_path.name}")
    log.info(f"可用新特征: f_oi_chg_1, f_oi_chg_12, f_trend_strength, f_funding_z")

if __name__ == "__main__":
    import numpy as np
    # 指向你之前下载的 ETH 原始 K 线数据路径
    # merge_orderflow_features(Path("data_center/ETH_USDT_5m_raw.parquet"))
    
    # 你可以先取消注释下面两行，单独跑一下看看抓到的数据长什么样：
    print(fetch_open_interest(INST_ID).tail())
    print(fetch_funding_rate(INST_ID).tail())