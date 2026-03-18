"""
强化学习策略回测引擎 (Backtest Engine)
=====================================
加载训练好的 PPO 模型和特征归一化尺子，在从未见过的验证集上进行逐K线回放，
并打印详细的买卖点与盈亏日志。

使用方法:
    python backtest.py --symbol ETH-USDT
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sandbox.data_loader import load_dataset
from sandbox.trading_env import TradingEnv
from train import split_train_eval, TRAIN_RATIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

def main(args):
    # ── 1. 路径设置与环境检查 ──────────────────────────────────────────
    run_dir = Path(args.run_dir) / f"ppo_{args.symbol}"
    stats_path = run_dir / "vecnormalize.pkl"

    # 优先用 EvalCallback 保存的 best_model，降级用训练结束时的 final_model
    best_model_path  = run_dir / "best_model" / "best_model.zip"
    final_model_path = run_dir / "final_model.zip"
    if best_model_path.exists():
        model_path = best_model_path
        log.info(f"使用 best_model: {model_path}")
    elif final_model_path.exists():
        model_path = final_model_path
        log.warning(f"未找到 best_model，降级使用 final_model: {model_path}")
    else:
        log.error(f"找不到任何模型文件！请确认训练是否完成\n已查找:\n  {best_model_path}\n  {final_model_path}")
        return

    if not stats_path.exists():
        log.error(f"找不到归一化文件！路径: {stats_path}")
        return

    # ── 2. 加载数据并严格切分出验证集 ─────────────────────────────────
    log.info(f"正在加载 {args.symbol} 数据...")
    df_full = load_dataset(args.symbol)
    _, eval_df = split_train_eval(df_full, TRAIN_RATIO)
    
    # 提取时间戳和收盘价用于日志打印
    datetimes = eval_df["datetime"].values
    closes = eval_df["close"].values

    log.info(f"验证集就绪: {len(eval_df):,} 行 ({datetimes[0]} -> {datetimes[-1]})")

    # ── 3. 构建回测环境 (解除最大步数限制，一口气跑完) ────────────────
    def make_eval_env():
        # max_steps 设为验证集长度，不让它中途强制结算
        # drawdown_limit=1.0 关闭回撤熔断，让模型跑完全程，观察真实表现
        return TradingEnv(eval_df, max_steps=len(eval_df) - 10, drawdown_limit=1.0)
    
    # 必须使用 DummyVecEnv 包装，以匹配训练时的维度
    env = DummyVecEnv([make_eval_env])

    # 加载训练时保存的“特征尺子” (VecNormalize)
    # 【核心风控】：必须设置 training=False，否则尺子会被验证集数据污染
    env = VecNormalize.load(str(stats_path), env)
    env.training = False
    env.norm_reward = False

    # ── 4. 加载 AI 大脑 ───────────────────────────────────────────────
    log.info(f"正在唤醒 AI 模型: {model_path.name}")
    model = PPO.load(model_path, env=env)

    # ── 5. 开始逐 K 线回测 ────────────────────────────────────────────
    obs = env.reset()
    done = False

    # 记录状态，用于捕捉交易动作
    last_position = 0.0
    entry_price = 0.0
    entry_time = None

    log.info("\n" + "="*80)
    log.info(" 实盘推演日志 (Trade Log)")
    log.info("="*80)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        # SB3 的 VecEnv 返回的都是数组，提取第0个环境的信息
        info = infos[0]
        current_position = info["position"]
        data_idx = info["data_idx"]
        current_time = datetimes[data_idx]
        current_price = closes[data_idx]

        # ── 捕捉交易信号并打印日志 ──
        if current_position != last_position:
            # 1. 存在旧仓位，说明刚刚平仓了
            if last_position != 0.0:
                direction = "多头" if last_position > 0 else "空头"
                pnl_ratio = (current_price / entry_price - 1.0) if last_position > 0 else (1.0 - current_price / entry_price)
                pnl_pct = pnl_ratio * 100

                # 扣除大致的手续费和滑点 (双边 0.05% + 0.025%)
                net_pnl_pct = pnl_pct - (0.075 * 2)

                profit_str = f"🟩 盈利: {net_pnl_pct:+.2f}%" if net_pnl_pct > 0 else f"🟥 亏损: {net_pnl_pct:+.2f}%"
                print(f"[{current_time}] 平仓 {direction:2s} | 离场价: {current_price:>8.2f} | {profit_str}")

            # 2. 开立新仓位
            if current_position != 0.0:
                direction = "多头" if current_position > 0 else "空头"
                action_str = "🚀 开多" if current_position > 0 else "🩸 开空"
                print(f"[{current_time}] {action_str} {direction:2s} | 进场价: {current_price:>8.2f}")
                entry_price = current_price
                entry_time = current_time

            last_position = current_position

        done = dones[0]

    # ── 6. 打印期末成绩单 ─────────────────────────────────────────────
    final_info = infos[0]
    log.info("\n" + "="*80)
    log.info(" 🏆 阅兵成绩单 (Final Performance)")
    log.info("="*80)
    log.info(f"初始资金: 1.000000")
    log.info(f"最终净值: {final_info['net_worth']:.6f}")
    log.info(f"净收益率: {(final_info['net_worth'] - 1.0) * 100:.2f}%")
    log.info(f"最大回撤: {(1.0 - final_info['net_worth'] / final_info['peak_marked']) * 100:.2f}%")
    log.info(f"交易次数: {final_info['n_trades']} 次")
    log.info(f"交易胜率: {final_info['win_rate'] * 100:.2f}%")
    log.info(f"总手续费: {final_info['total_commission']:.6f}")
    log.info("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ETH-USDT")
    parser.add_argument("--run-dir", default="runs")
    args = parser.parse_args()
    main(args)