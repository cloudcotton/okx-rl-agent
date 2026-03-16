"""
Smoke test: verify the full training pipeline runs end-to-end
on synthetic data without errors.

Run with:  python -m pytest sandbox/tests/test_train_pipeline.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sandbox.data_loader import FEATURE_COLS
from sandbox.trading_env import TradingEnv


def _make_df(n_rows: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_rows, freq="5min")
    close = 30_000.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n_rows)))
    noise = rng.uniform(0.998, 1.002, n_rows)
    open_ = np.roll(close, 1) * noise
    open_[0] = close[0] * 0.999
    high = np.maximum(open_, close) * rng.uniform(1.000, 1.005, n_rows)
    low  = np.minimum(open_, close) * rng.uniform(0.995, 1.000, n_rows)
    vol  = rng.uniform(1.0, 100.0, n_rows)
    df = pd.DataFrame({
        "datetime": dates,
        "open":  open_.astype(np.float64),
        "high":  high.astype(np.float64),
        "low":   low.astype(np.float64),
        "close": close.astype(np.float64),
        "vol":   vol.astype(np.float64),
    })
    for feat in FEATURE_COLS:
        df[feat] = rng.uniform(-1.0, 1.0, n_rows).astype(np.float32)
    return df


class TestTrainPipeline:

    def test_linear_schedule(self):
        from train import linear_schedule
        sched = linear_schedule(3e-4)
        assert sched(1.0) == pytest.approx(3e-4)
        assert sched(0.5) == pytest.approx(1.5e-4)
        assert sched(0.0) == pytest.approx(0.0)

    def test_train_eval_split_is_chronological(self):
        from train import split_train_eval
        df = _make_df()
        train_df, eval_df = split_train_eval(df, 0.8)
        # No temporal overlap
        assert train_df["datetime"].max() < eval_df["datetime"].min()
        # Sizes
        assert len(train_df) + len(eval_df) == len(df)
        assert abs(len(train_df) / len(df) - 0.8) < 0.01

    def test_make_env_returns_callable(self):
        from train import make_env
        df = _make_df()
        factory = make_env(df, {}, rank=0, seed=0)
        assert callable(factory)
        env = factory()
        assert isinstance(env, TradingEnv)
        obs, info = env.reset(seed=0)
        assert obs.shape == (16,)

    def test_ppo_short_training_run(self):
        """End-to-end: create VecEnv → PPO → learn 2048 steps → no crash."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
        from train import make_env, split_train_eval, linear_schedule, TradingMetricsCallback

        # 15000 rows → eval_df = 3000 rows > max_steps(2016) + 2
        df = _make_df(n_rows=15000)
        train_df, eval_df = split_train_eval(df, 0.8)

        # Use DummyVecEnv (single process) to avoid multiprocessing in tests
        n_envs = 2
        train_vec = DummyVecEnv([make_env(train_df, {}, rank=i, seed=42) for i in range(n_envs)])
        train_vec = VecMonitor(train_vec)
        train_vec = VecNormalize(train_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

        eval_vec = DummyVecEnv([make_env(eval_df, {}, rank=0, seed=9999)])
        eval_vec = VecMonitor(eval_vec)
        eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False,
                                clip_obs=10.0, training=False)

        model = PPO(
            policy="MlpPolicy",
            env=train_vec,
            n_steps=256,       # small for speed
            batch_size=64,
            n_epochs=2,
            gamma=0.99,
            learning_rate=linear_schedule(3e-4),
            verbose=0,
            seed=42,
        )

        # Train for exactly one rollout (n_steps * n_envs = 512 total steps)
        model.learn(total_timesteps=512, callback=TradingMetricsCallback())

        # Verify model can generate actions
        obs = train_vec.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (n_envs,)
        assert all(a in {0, 1, 2} for a in action)

    def test_sync_norm_copies_stats_before_eval(self):
        """
        SyncNormAndEvalCallback must deep-copy train obs_rms into eval obs_rms
        exactly at the eval boundary, so the agent sees the same input
        distribution during evaluation as during training.
        """
        from copy import deepcopy
        import numpy as np
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
        from train import make_env, split_train_eval, linear_schedule, SyncNormAndEvalCallback

        df = _make_df(n_rows=15000)
        train_df, eval_df = split_train_eval(df, 0.8)

        n_envs = 2
        train_vec = DummyVecEnv([make_env(train_df, {}, rank=i, seed=0) for i in range(n_envs)])
        train_vec = VecMonitor(train_vec)
        train_vec = VecNormalize(train_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

        eval_vec = DummyVecEnv([make_env(eval_df, {}, rank=0, seed=999)])
        eval_vec = VecMonitor(eval_vec)
        # Start eval with deliberately wrong stats (large mean) to confirm sync overwrites them
        eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False,
                                clip_obs=10.0, training=False)
        eval_vec.obs_rms.mean[:] = 999.0   # poison the initial stats

        # Warm up train_vec normaliser with a few random steps so obs_rms
        # has non-trivial mean/var to distinguish from the poisoned 999.0
        obs = train_vec.reset()
        for _ in range(50):
            obs, _, _, _ = train_vec.step(
                [train_vec.action_space.sample() for _ in range(n_envs)]
            )

        # Build callback and call _sync_stats() directly — this bypasses the
        # EvalCallback machinery (which needs a fully initialised model.logger)
        # and tests only the stat-copy logic we own.
        cb = SyncNormAndEvalCallback(
            train_env=train_vec,
            eval_env=eval_vec,
            eval_freq=50_000,
            n_eval_episodes=1,
            verbose=0,
        )
        cb._sync_stats()

        # eval_vec stats must now match train_vec stats (not the poisoned 999.0)
        assert not np.allclose(eval_vec.obs_rms.mean, 999.0), \
            "Sync did not overwrite the poisoned eval stats"
        np.testing.assert_array_almost_equal(
            eval_vec.obs_rms.mean, train_vec.obs_rms.mean, decimal=6,
            err_msg="eval obs_rms.mean must equal train obs_rms.mean after sync"
        )
        np.testing.assert_array_almost_equal(
            eval_vec.obs_rms.var, train_vec.obs_rms.var, decimal=6,
            err_msg="eval obs_rms.var must equal train obs_rms.var after sync"
        )

    def test_trading_metrics_callback_logs(self):
        """Callback collects episode info without crashing."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
        from train import make_env, TradingMetricsCallback

        df = _make_df(n_rows=4000)
        vec = DummyVecEnv([make_env(df, {}, rank=0, seed=0)])
        vec = VecMonitor(vec)

        model = PPO("MlpPolicy", vec, n_steps=128, batch_size=32,
                    n_epochs=1, verbose=0, seed=0)
        cb = TradingMetricsCallback()
        model.learn(total_timesteps=256, callback=cb)
        # If at least one episode finished, rolling deques are populated
        # (may not finish in 256 steps with max_steps=2016 — that's OK)
        from collections import deque
        assert isinstance(cb._ep_returns, deque)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
