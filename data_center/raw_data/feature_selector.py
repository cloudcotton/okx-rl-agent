"""
Feature Selector: Analyze 30 RL features for collinearity.

Output: 
  1. correlation_heatmap.png (Visual map of feature correlations)
  2. Terminal output listing highly correlated pairs and a suggested drop list.

Usage:
    python feature_selector.py --symbol ETH-USDT
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent / "data_center"
FEAT_DIR = DATA_DIR / "features"
THRESHOLD = 0.85  # 共线性剔除阈值

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

def analyze_features(symbol: str):
    feat_path = FEAT_DIR / f"{symbol.replace('-', '_')}_5m_features.parquet"
    if not feat_path.exists():
        log.error(f"找不到特征文件: {feat_path}")
        return

    log.info(f"[{symbol}] 正在读取特征文件...")
    df = pd.read_parquet(feat_path)
    
    # 提取所有以 f_ 开头的特征列
    feat_cols = [col for col in df.columns if col.startswith("f_")]
    df_features = df[feat_cols]
    log.info(f"[{symbol}] 成功加载 {len(feat_cols)} 个特征，共 {len(df_features):,} 行数据。")

    # ── 1. 计算斯皮尔曼秩相关系数 (Spearman) ──────────────────────────────────
    log.info(f"[{symbol}] 正在计算 Spearman 相关性矩阵 (这可能需要十几秒)...")
    corr_matrix = df_features.corr(method='spearman')

    # ── 2. 绘制并保存热力图 ───────────────────────────────────────────────────
    log.info(f"[{symbol}] 正在生成相关性热力图...")
    # 创建掩码，遮挡上三角和对角线
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    plt.figure(figsize=(20, 16))
    sns.set_theme(style="white")
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
        square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f", annot_kws={"size": 8}
    )
    plt.title(f"{symbol} 5m Feature Correlation (Spearman)", fontsize=20)
    plt.tight_layout()
    
    img_path = DATA_DIR / f"{symbol.replace('-', '_')}_correlation_heatmap.png"
    plt.savefig(img_path, dpi=150)
    log.info(f"[{symbol}] 热力图已保存至: {img_path}")

    # ── 3. 自动诊断高共线性特征 ───────────────────────────────────────────────
    corr_abs = corr_matrix.abs()
    upper_tri = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
    
    # 找出所有相关系数大于阈值的特征对
    high_corr_pairs = []
    for col in upper_tri.columns:
        highly_correlated_with = upper_tri.index[upper_tri[col] > THRESHOLD].tolist()
        for related_feat in highly_correlated_with:
            high_corr_pairs.append((related_feat, col, upper_tri.loc[related_feat, col]))

    log.info("=" * 60)
    log.info(f" 高共线性诊断报告 (阈值 > {THRESHOLD})")
    log.info("=" * 60)
    
    if not high_corr_pairs:
        log.info("未发现高度相关的特征对！所有特征都很健康。")
        return

    # 按相关性从高到低排序
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    drop_candidates = set()
    for feat_a, feat_b, corr_val in high_corr_pairs:
        log.info(f"高度相关: {feat_a} <--> {feat_b} (相关系数: {corr_val:.3f})")
        # 简单策略：保留先出现的特征，建议剔除后出现的特征
        drop_candidates.add(feat_b)

    log.info("-" * 60)
    log.info(f" ⚠️ 建议剔除名单 (共 {len(drop_candidates)} 个特征):")
    for feat in sorted(drop_candidates):
        log.info(f"   - {feat}")
    log.info("-" * 60)
    log.info(f" 如果剔除这些特征，特征维度将从 {len(feat_cols)} 降至 {len(feat_cols) - len(drop_candidates)}。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="ETH-USDT", help="Symbol to analyze")
    args = parser.parse_args()
    analyze_features(args.symbol)