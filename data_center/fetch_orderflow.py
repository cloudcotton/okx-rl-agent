"""
OKX 微观订单流数据抓取与对齐脚本 (Order Flow Fetcher) - V2 时间边界版
====================================================================
先读取本地 K 线数据，确定时间起止范围，
精准抓取指定区间的持仓量(OI)和资金费率(Funding Rate)，并完美对齐。
"""

import time
import logging
from pathlib import Path
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://www.okx.com"
INST_ID = "ETH-USDT-SWAP"

def fetch_funding_rate(inst_id: str, min_ts: int, max_ts: int, limit: int = 100) -> pd.DataFrame:
    """抓取资金费率历史，严格限制在 [min_ts, max_ts] 毫秒级时间戳范围内"""
    url = f"{BASE_URL}/api/v5/public/funding-rate-history"
    all_data = []
    
    # OKX API 的 after 代表“早于这个时间戳的数据”
    # 我们加上 1000 毫秒的容错，确保能抓到 max_ts 那个点的数据
    after = str(max_ts + 1000) 
    
    log.info(f"开始抓取 {inst_id} 资金费率...")
    log.info(f"目标区间: {pd.to_datetime(min_ts, unit='ms')} -> {pd.to_datetime(max_ts, unit='ms')}")
    
    while True:
        params = {"instId": inst_id, "limit": limit, "after": after}
        try:
            res = requests.get(url, params=params, timeout=10).json()
            if res.get("code") != "0" or not res.get("data"):
                break
                
            batch = res["data"]
            all_data.extend(batch)
            
            # 获取这一批数据中最老的一个时间戳，作为下一页的游标
            oldest_ts = int(batch[-1]["fundingTime"])
            after = batch[-1]["fundingTime"]
            time.sleep(0.1)  # 遵守 API 限频
            
            # 【核心边界控制】：如果最老的数据已经早于我们的最小时间，立刻停止抓取！
            if oldest_ts <= min_ts:
                break
                
            if len(all_data) % 500 == 0:
                log.info(f"已抓取 {len(all_data)} 条资金费率，当前推进至 {pd.to_datetime(oldest_ts, unit='ms')}")
                
        except Exception as e:
            log.error(f"抓取资金费率中断: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["datetime"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms")
    df["fundingRate"] = df["fundingRate"].astype(float)
    
    # 过滤掉多抓的冗余数据，并按时间正序排列
    df = df[(df["datetime"] >= pd.to_datetime(min_ts, unit="ms")) & 
            (df["datetime"] <= pd.to_datetime(max_ts, unit="ms"))]
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "fundingRate"]]


def fetch_open_interest(inst_id: str, min_ts: int, max_ts: int, period: str = "5m") -> pd.DataFrame:
    """抓取合约持仓量历史，严格限制在 [min_ts, max_ts] 毫秒级时间戳范围内"""
    url = f"{BASE_URL}/api/v5/rubik/stat/contracts/open-interest-history"
    all_data = []
    
    after = str(max_ts + 1000)
    
    log.info(f"开始抓取 {inst_id} {period} 级别持仓量...")
    
    while True:
        # Rubik 接口支持 begin 和 end 参数，但为了兼容性，用 after 分页最稳妥
        params = {"instId": inst_id, "period": period, "after": after}
        try:
            res = requests.get(url, params=params, timeout=10).json()
            if res.get("code") != "0" or not res.get("data"):
                break
                
            batch = res["data"]
            all_data.extend(batch)
            
            # OKX 返回格式: [ts, oi, oiCcy]
            oldest_ts = int(batch[-1][0])
            after = batch[-1][0]
            time.sleep(0.2)  # Rubik 接口限频较严
            
            # 【核心边界控制】
            if oldest_ts <= min_ts:
                break
                
            if len(all_data) % 2000 == 0:
                log.info(f"已抓取 {len(all_data)} 条持仓量，当前推进至 {pd.to_datetime(oldest_ts, unit='ms')}")
                
        except Exception as e:
            log.error(f"抓取持仓量中断: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["ts", "oi", "oiCcy"])
    df["datetime"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    df["oi"] = df["oi"].astype(float)
    
    # 严格过滤并排序
    df = df[(df["datetime"] >= pd.to_datetime(min_ts, unit="ms")) & 
            (df["datetime"] <= pd.to_datetime(max_ts, unit="ms"))]
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "oi"]]


def merge_orderflow_features(ohlcv_path: Path):
    """主函数：读取本地 K 线 -> 提取时间边界 -> 抓取 -> 对齐融合"""
    log.info(f"正在加载基础 K 线数据: {ohlcv_path.name}")
    df_main = pd.read_parquet(ohlcv_path)
    
    # ── 1. 提取 K 线数据的时间边界（转换为毫秒时间戳） ──
    min_time = df_main["datetime"].min()
    max_time = df_main["datetime"].max()
    min_ts = int(min_time.timestamp() * 1000)
    max_ts = int(max_time.timestamp() * 1000)
    
    log.info(f"K 线时间边界测定完毕: 最小 {min_time} | 最大 {max_time}")
    
    # ── 2. 根据边界进行精准抓取 ──
    df_fund = fetch_funding_rate(INST_ID, min_ts, max_ts)
    df_oi = fetch_open_interest(INST_ID, min_ts, max_ts, period="5m")
    
    if df_fund.empty or df_oi.empty:
        log.error("抓取到的订单流数据为空，请检查网络或 API 限制！")
        return
        
    # ── 3. 合并与对齐 ──
    log.info("开始与 K 线数据进行时间轴对齐融合...")
    
    # 资金费率 (8小时一次，使用 asof 向下填充)
    df_main = pd.merge_asof(
        df_main.sort_values("datetime"),
        df_fund.sort_values("datetime"),
        on="datetime",
        direction="backward"
    )
    
    # 持仓量 (精准对齐)
    df_main = pd.merge(df_main, df_oi, on="datetime", how="left")
    df_main["oi"] = df_main["oi"].ffill() # 填补偶尔缺失的断点
    
    # ── 4. 铸造杀手级衍生特征 (Feature Engineering) ──
    log.info("开始计算高级订单流特征...")
    
    df_main["f_oi_chg_1"] = df_main["oi"].pct_change(1)
    df_main["f_oi_chg_12"] = df_main["oi"].pct_change(12)
    df_main["f_trend_strength"] = np.sign(df_main["close"].pct_change(1)) * np.sign(df_main["f_oi_chg_1"])
    df_main["f_funding_z"] = (df_main["fundingRate"] - df_main["fundingRate"].rolling(288).mean()) / (df_main["fundingRate"].rolling(288).std() + 1e-8)
    
    # 清理头部因 pct_change 产生的 NaN
    df_main = df_main.dropna().reset_index(drop=True)
    
    # 保存融合后的新数据集
    out_path = ohlcv_path.parent / f"{INST_ID}_orderflow_features.parquet"
    df_main.to_parquet(out_path)
    
    log.info(f"✅ 微观特征生成完毕！已保存至: {out_path.name}")
    log.info(f"新增高净值特征: f_oi_chg_1, f_oi_chg_12, f_trend_strength, f_funding_z")

if __name__ == "__main__":
    # 注意：请将这里的路径替换为你服务器上真实的 ETH K 线 parquet 文件路径
    # 例如：Path("data_center/features/ETH_USDT_5m_features.parquet")
    
    merge_orderflow_features(Path("features/ETH_USDT_5m_features.parquet"))
    pass