"""
Phase-1 Training Loop — PPO on TradingEnv
==========================================

Strategy: Discrete(3) PPO with Stable-Baselines3.
Goal: validate whether the agent can learn directional prediction
      before adding continuous position-sizing complexity.

Design choices
--------------
* Time-based train/eval split (never leak future data into training).
* SubprocVecEnv for parallel rollout collection.
* VecNormalize on observations only (rewards kept raw to preserve
  penalty magnitudes — drawdown -1.0 must stay interpretable).
* Linear learning-rate decay from 3e-4 → 0.
* EvalCallback saves best model by mean episode reward.
* TradingMetricsCallback logs domain-specific KPIs to TensorBoard.

Usage
-----
    python train.py --symbol BTC-USDT
    python train.py --symbol BTC-USDT --n-envs 4 --total-steps 5_000_000
    python train.py --symbol BTC-USDT --resume runs/ppo_BTC-USDT/best_model
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import deque
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sandbox.data_loader import load_dataset
from sandbox.trading_env import TradingEnv

import torch
# 主进程保留 2 个线程用于梯度更新（BLAS/MKL 并行）；
# 子进程 env worker 在 make_env._init 里单独设为 1，防止核心争抢。
torch.set_num_threads(2)

# 固定使用 CPU（无 GPU 配置）
_DEVICE = "cpu"
logging.getLogger(__name__).info("运行设备: CPU（16c 无 GPU 配置）")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_RATIO   = 0.80    # chronological split; last 20 % is eval
EVAL_EPISODES = 20      # episodes per EvalCallback call
EVAL_FREQ     = 50_000  # steps between evaluations (per env)
CKPT_FREQ     = 100_000 # steps between checkpoints

# PPO hyper-parameters (Phase 1 baseline)
PPO_KWARGS = dict(
    policy="MlpPolicy",
    policy_kwargs=dict(
        net_arch=[64, 64],     # two hidden layers — 16-dim input needs no wide net
        activation_fn=__import__("torch.nn", fromlist=["Tanh"]).Tanh,
    ),
    n_steps=2048,      # MLP 无 BPTT 开销，用大 rollout 让 GAE 估计更准
    batch_size=512,    # 需满足：batch_size ∣ (n_steps × n_envs)；2048×14/512 = 56 ✓
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.03,      # entropy bonus — tripled to prevent long-only policy collapse
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def linear_schedule(initial_lr: float) -> Callable[[float], float]:
    """Return a function that linearly decays LR from initial_lr to 0."""
    def schedule(progress_remaining: float) -> float:
        return initial_lr * progress_remaining
    return schedule


def make_env(df: pd.DataFrame, env_kwargs: dict, rank: int, seed: int = 0):
    """
    Factory for SubprocVecEnv.

    Each subprocess gets the same DataFrame (shared read-only via fork on
    Linux; copied on Windows — acceptable since DataFrames are read-only
    in TradingEnv after construction).
    """
    def _init() -> TradingEnv:
        # 每个 env 子进程限 1 线程，防止 16 个 worker 各自开多线程撑爆 CPU
        import torch as _torch
        import os
        _torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        env = TradingEnv(df, **env_kwargs)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed + rank)
    return _init


def split_train_eval(df: pd.DataFrame, train_ratio: float):
    """Chronological split — never shuffle time-series data."""
    split_idx = int(len(df) * train_ratio)
    train_df  = df.iloc[:split_idx].reset_index(drop=True)
    eval_df   = df.iloc[split_idx:].reset_index(drop=True)
    log.info(
        f"Split: train {len(train_df):,} rows "
        f"({train_df['datetime'].min()} → {train_df['datetime'].max()}) | "
        f"eval {len(eval_df):,} rows "
        f"({eval_df['datetime'].min()} → {eval_df['datetime'].max()})"
    )
    return train_df, eval_df


# ---------------------------------------------------------------------------
# Custom callback — domain-specific KPIs
# ---------------------------------------------------------------------------

class TradingMetricsCallback(BaseCallback):
    """
    Logs trading-specific episode metrics to TensorBoard whenever
    any vectorised environment finishes an episode.

    Metrics logged under the "trade/" prefix:
        n_trades       — number of round-trip trades
        win_rate       — fraction of profitable trades
        total_return   — episode net return (marked NW / initial NW − 1)
        peak_marked    — highest portfolio value reached in episode
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        # deque(maxlen=100): O(1) append and automatic eviction of oldest element
        self._ep_returns:  deque[float] = deque(maxlen=100)
        self._ep_trades:   deque[int]   = deque(maxlen=100)
        self._ep_winrates: deque[float] = deque(maxlen=100)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if not done:
                continue
            n_trades     = info.get("n_trades", 0)
            win_rate     = info.get("win_rate", 0.0)
            total_return = info.get("total_return", 0.0)
            peak_marked  = info.get("peak_marked", 1.0)

            self._ep_returns.append(total_return)
            self._ep_trades.append(n_trades)
            self._ep_winrates.append(win_rate)

            self.logger.record("trade/total_return",    total_return)
            self.logger.record("trade/n_trades",        n_trades)
            self.logger.record("trade/win_rate",        win_rate)
            self.logger.record("trade/peak_marked",     peak_marked)
            self.logger.record("trade/return_mean100",  float(np.mean(self._ep_returns)))
            self.logger.record("trade/trades_mean100",  float(np.mean(self._ep_trades)))
            self.logger.record("trade/winrate_mean100", float(np.mean(self._ep_winrates)))

        return True


# ---------------------------------------------------------------------------
# Eval callback with guaranteed normalisation sync
# ---------------------------------------------------------------------------

class SyncNormAndEvalCallback(EvalCallback):
    """
    EvalCallback with an explicit VecNormalize stats sync injected before
    every evaluation window.

    Problem solved
    --------------
    train_vec.VecNormalize continuously updates obs_rms (running mean/std)
    as training progresses.  eval_vec.VecNormalize is frozen (training=False),
    so its stats stay at the initialisation values (mean=0, std=1).
    When EvalCallback hands the agent eval observations, those observations
    are normalised with the *wrong* ruler — the agent sees a completely
    different input distribution from what it trained on.

    Fix
    ---
    Before each evaluation run, deep-copy obs_rms and ret_rms from
    train_env into eval_env.  The copy happens *before* super()._on_step()
    so EvalCallback always sees consistent statistics.
    """

    def __init__(self, train_env: VecNormalize, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._train_env = train_env

    def _sync_stats(self) -> None:
        """Deep-copy running normalisation stats from train_env into eval_env."""
        if isinstance(self._train_env, VecNormalize) and \
           isinstance(self.eval_env,  VecNormalize):
            self.eval_env.obs_rms = deepcopy(self._train_env.obs_rms)
            self.eval_env.ret_rms = deepcopy(self._train_env.ret_rms)
            log.debug(
                "[SyncNormAndEvalCallback] obs_rms synced | "
                f"mean[:3]={self._train_env.obs_rms.mean[:3]}"
            )

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._sync_stats()
        return super()._on_step()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_callbacks(
    train_env: VecNormalize,
    eval_env: VecNormalize,
    run_dir: Path,
    eval_freq_total: int,
) -> CallbackList:
    """Assemble all training callbacks."""
    eval_cb = SyncNormAndEvalCallback(
        train_env=train_env,
        eval_env=eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=eval_freq_total,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=CKPT_FREQ,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo",
        save_vecnormalize=True,
    )
    metrics_cb = TradingMetricsCallback()
    return CallbackList([eval_cb, ckpt_cb, metrics_cb])


def main(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir) / f"ppo_{args.symbol}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & split data ─────────────────────────────────────────────
    log.info(f"Loading dataset for {args.symbol} …")
    df = load_dataset(args.symbol)
    train_df, eval_df = split_train_eval(df, TRAIN_RATIO)

    # Env constructor kwargs — use TradingEnv defaults (max_steps, friction params).
    # _MAX_STEPS in trading_env.py is the single source of truth.
    env_kwargs = {}

    # ── 2. Training VecEnv ───────────────────────────────────────────────
    log.info(f"Creating {args.n_envs} parallel training environments …")
    train_vec = SubprocVecEnv(
        [make_env(train_df, {}, rank=i, seed=args.seed) for i in range(args.n_envs)]
    )
    train_vec = VecMonitor(train_vec)
    train_vec = VecNormalize(
        train_vec,
        norm_obs=True,
        norm_reward=False,  # keep raw reward scale (penalty magnitudes must survive)
        clip_obs=10.0,
        gamma=PPO_KWARGS["gamma"],
    )

    # ── 3. Eval VecEnv (single env, synced normalisation stats) ──────────
    eval_vec = SubprocVecEnv(
        [make_env(eval_df, {}, rank=0, seed=args.seed + 9999)]
    )
    eval_vec = VecMonitor(eval_vec)
    eval_vec = VecNormalize(
        eval_vec,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=PPO_KWARGS["gamma"],
        training=False,     # eval env never updates running stats
    )

    # ── 4. PPO (MLP) model ───────────────────────────────────────────────
    if args.resume:
        resume_path = Path(args.resume)
        log.info(f"Resuming from {resume_path} …")
        model = PPO.load(
            resume_path,
            env=train_vec,
            tensorboard_log=str(run_dir / "tb"),
            device=_DEVICE,
        )
        # Load matching VecNormalize stats if available
        vecnorm_path = resume_path.parent / "vecnormalize.pkl"
        if vecnorm_path.exists():
            train_vec = VecNormalize.load(str(vecnorm_path), train_vec.venv)
            eval_vec  = VecNormalize.load(str(vecnorm_path), eval_vec.venv)
            eval_vec.training = False
            log.info(f"VecNormalize stats loaded from {vecnorm_path}")
    else:
        model = PPO(
            env=train_vec,
            learning_rate=linear_schedule(args.lr),
            tensorboard_log=str(run_dir / "tb"),
            seed=args.seed,
            device=_DEVICE,
            **PPO_KWARGS,
        )

    log.info(f"Policy network: {model.policy}")
    log.info(f"Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    # ── 5. Callbacks ─────────────────────────────────────────────────────
    # eval_freq in EvalCallback is per-env; multiply by n_envs for wall-clock steps
    eval_freq_per_env = max(EVAL_FREQ // args.n_envs, 1)
    callbacks = build_callbacks(train_vec, eval_vec, run_dir, eval_freq_per_env)

    # ── 6. Train ─────────────────────────────────────────────────────────
    log.info(
        f"Training MLP | symbol={args.symbol} | n_envs={args.n_envs} | "
        f"total_steps={args.total_steps:,} | run_dir={run_dir}"
    )
    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        reset_num_timesteps=not args.resume,
        tb_log_name="mlp",
        progress_bar=True,
    )

    # ── 7. Save final artefacts ───────────────────────────────────────────
    model.save(str(run_dir / "final_model"))
    train_vec.save(str(run_dir / "vecnormalize.pkl"))
    log.info(f"Saved final model → {run_dir / 'final_model.zip'}")
    log.info(f"Saved VecNormalize → {run_dir / 'vecnormalize.pkl'}")
    log.info("Done. Launch TensorBoard with:")
    log.info(f"    tensorboard --logdir {run_dir / 'tb'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-1 PPO training for OKX RL trading agent."
    )
    p.add_argument("--symbol",       default="BTC-USDT",
                   help="Symbol to train on (must have feature parquet ready).")
    p.add_argument("--n-envs",       type=int,   default=14,
                   help="并行环境数，16c 服务器建议 14（主进程留 2 核做梯度更新）。")
    p.add_argument("--total-steps",  type=int,   default=3_000_000,
                   help="Total environment steps to train.")
    p.add_argument("--lr",           type=float, default=3e-4,
                   help="Initial learning rate (decays linearly to 0).")
    p.add_argument("--seed",         type=int,   default=42,
                   help="Global random seed.")
    p.add_argument("--run-dir",      default="runs",
                   help="Root directory for run artefacts.")
    p.add_argument("--resume",       default=None,
                   help="Path to a saved model zip to resume training from.")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
