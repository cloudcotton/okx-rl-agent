"""
RL Trading Sandbox — Gymnasium-compatible backtesting environment.

Quick start
-----------
    from sandbox.data_loader import load_dataset
    from sandbox.trading_env import TradingEnv

    df  = load_dataset(["BTC-USDT"])
    env = TradingEnv(df)

    obs, info = env.reset(seed=42)
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
"""

from .trading_env import TradingEnv
from .data_loader import load_dataset, FEATURE_COLS

__all__ = ["TradingEnv", "load_dataset", "FEATURE_COLS"]
