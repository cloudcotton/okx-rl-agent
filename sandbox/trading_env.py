"""
Gymnasium TradingEnv — RL trading sandbox (v1.1 — 15m edition)

Design philosophy
-----------------
* Realistic friction: commission + slippage ensure only real-edge strategies survive.
* Zero look-ahead: the agent acts on bar-t features and executes at bar-(t+1) open.
* Simple 1× leverage net-worth tracker — no margin, no liquidation logic.

Observation space (n_features+2 dims, Box[-10, 10])
    [0:n]   — market features (configured by data_loader.FEATURE_COLS)
    [n]     — current_position  ∈ {-1.0, 0.0, +1.0}
    [n+1]   — unrealized_pnl_norm  ∈ [-10, 10]  (pnl_ratio × 10, clipped; 1 unit = 10%)

Action space (Discrete 3)
    0 → target position = -1  (short)
    1 → target position =  0  (flat)
    2 → target position = +1  (long)

Reward components
    R = R_base + R_hold + R_adhd [+ R_drawdown if triggered]
    R_base     = ln(marked_nw_t / marked_nw_{t-1})
                 Naturally penalises losing positions via mark-to-market.
    R_hold     = +0.0001 each step while position is profitable (float pnl > 0)
                  0.0    otherwise
    R_adhd     = ADHD_PENALTY     (-0.0002) on any position change
                 REVERSAL_PENALTY (-0.0001) extra for direct flip (long ↔ short)
    R_draw     = DRAWDOWN_PENALTY (-1.0) once, then episode terminates

    *** No per-step stop-loss penalty — R_base already conveys price-move pain.
    *** A hard per-step stop-loss (-0.05) creates unrecoverable reward traps at
        short timeframes where 1 % intrabar moves are normal; removed in v1.1.

Episode lifecycle
    reset()  → random start index, leaving room for MAX_STEPS + 1 bars ahead
    step()   → execute at next open, mark-to-market at new close, check termination
    truncation at MAX_STEPS (672 = 1 week of 15-min bars); forced liquidation at end.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from .data_loader import FEATURE_COLS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (all overridable via __init__ kwargs)
# ---------------------------------------------------------------------------
_COMMISSION_RATE  = 0.0005    # 0.05 %  one-way Taker fee
_SLIPPAGE_RATE    = 0.00025   # 0.025 % of execution open price
_HOLDING_PENALTY  =  0.0      # 0 = no fear of holding; profit-holding gets +0.0001 bonus below
_ADHD_PENALTY     = -0.0002   # any position change — ~0.4× one-way commission (reverted from -0.001 which caused long-only collapse)
_REVERSAL_PENALTY = -0.0001   # extra for direct long ↔ short flip
_DRAWDOWN_LIMIT   = 0.10      # 10 % peak-to-trough → circuit-breaker
_DRAWDOWN_PENALTY = -1.0      # one-time terminal penalty
_MAX_STEPS        = 672       # 1 week of 15-min bars  (7 × 24 × 4 = 672)
_INITIAL_NW       = 1.0       # normalised; PnL is expressed in multiples
_MIN_HOLD_BARS    = 4         # minimum bars to hold a position (4 × 15min = 1 hour)

# Action → target position mapping
_ACTION_TO_POS = {0: -1.0, 1: 0.0, 2: 1.0}


class TradingEnv(gym.Env):
    """Gymnasium trading environment with realistic friction costs."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        max_steps: int = _MAX_STEPS,
        commission_rate: float = _COMMISSION_RATE,
        slippage_rate: float = _SLIPPAGE_RATE,
        holding_penalty: float = _HOLDING_PENALTY,
        adhd_penalty: float = _ADHD_PENALTY,
        reversal_penalty: float = _REVERSAL_PENALTY,
        drawdown_limit: float = _DRAWDOWN_LIMIT,
        drawdown_penalty: float = _DRAWDOWN_PENALTY,
        min_hold_bars: int = _MIN_HOLD_BARS,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._validate_df(df)

        # Store data as numpy arrays for fast indexing (avoid pandas overhead in step)
        self._feat_arr  = df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)   # (N, 14)
        self._open_arr  = df["open"].to_numpy(dtype=np.float64, copy=True)         # (N,)
        self._close_arr = df["close"].to_numpy(dtype=np.float64, copy=True)        # (N,)
        self._n_rows    = len(df)
        self._n_features = self._feat_arr.shape[1]  # <--- 新增：动态获取特征数量

        # Hyper-parameters
        self.max_steps        = max_steps
        self.commission_rate  = commission_rate
        self.slippage_rate    = slippage_rate
        self.holding_penalty  = holding_penalty
        self.adhd_penalty     = adhd_penalty
        self.reversal_penalty = reversal_penalty
        self.drawdown_limit   = drawdown_limit
        self.drawdown_penalty = drawdown_penalty
        self.min_hold_bars    = min_hold_bars
        self.render_mode      = render_mode

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self._n_features + 2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # Episode state (properly initialised in reset)
        self._step_idx:    int   = 0
        self._steps_done:  int   = 0
        self._position:    float = 0.0
        self._entry_price: float = 0.0
        self._net_worth:   float = _INITIAL_NW
        self._peak_marked: float = _INITIAL_NW

        # Episode statistics
        self._n_trades:          int   = 0
        self._n_winning_trades:  int   = 0
        self._total_commission:  float = 0.0
        self._steps_in_position: int   = 0  # bars held in current position

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Random start: leave room for max_steps bars + 1 (for the final next-open)
        max_start = self._n_rows - self.max_steps - 2
        if max_start < 0:
            raise ValueError(
                f"DataFrame has only {self._n_rows} rows but "
                f"max_steps={self.max_steps} requires at least {self.max_steps + 2}."
            )
        self._step_idx   = int(self.np_random.integers(0, max_start + 1))
        self._steps_done = 0

        # Account state — small random NW noise prevents the model from over-fitting
        # to a fixed starting value; ±10% jitter covers realistic drawdown scenarios.
        nw_jitter         = float(self.np_random.uniform(0.9, 1.1))
        self._position    = 0.0
        self._entry_price = 0.0
        self._net_worth   = _INITIAL_NW * nw_jitter
        self._peak_marked = _INITIAL_NW * nw_jitter

        # Episode statistics
        self._n_trades         = 0
        self._n_winning_trades = 0
        self._total_commission = 0.0
        self._steps_in_position = 0

        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        target_pos = _ACTION_TO_POS[action]
        prev_pos   = self._position

        # ── 0. Enforce minimum holding period ─────────────────────────────────
        # If in a position and trying to exit/reverse before min_hold_bars,
        # override to hold. No penalty — the model learns through r_base.
        if prev_pos != 0.0 and target_pos != prev_pos:
            if self._steps_in_position < self.min_hold_bars:
                target_pos = prev_pos  # force hold

        # ── 1. Mark-to-market BEFORE execution (at current bar's close) ──────
        prev_marked = self._mark_to_market(self._close_arr[self._step_idx])

        # ── 2. Execute trade at NEXT bar's open ───────────────────────────────
        exec_open      = self._open_arr[self._step_idx + 1]
        reward_shaping = 0.0

        if target_pos != prev_pos:
            # ADHD penalty for any position change
            reward_shaping += self.adhd_penalty
            # Extra reversal penalty for direct flip (bypassing flat)
            if prev_pos != 0.0 and target_pos != 0.0:
                reward_shaping += self.reversal_penalty

            # Explicit commission signal: makes fee cost a direct, undiluted reward
            # signal. The actual deduction already flows through r_base via net_worth,
            # but that signal is too weak in noisy bar-to-bar returns.
            if prev_pos != 0.0:
                reward_shaping -= self.commission_rate   # exit commission
            if target_pos != 0.0:
                reward_shaping -= self.commission_rate   # entry commission

            # Close existing position first (if any)
            if prev_pos != 0.0:
                self._close_position(exec_open)

            # Open new position (if not going flat)
            if target_pos != 0.0:
                self._open_position(target_pos, exec_open)

        # ── 3. Advance step ───────────────────────────────────────────────────
        self._step_idx  += 1
        self._steps_done += 1

        # ── 4. Mark-to-market AFTER execution (at new bar's close) ───────────
        curr_close  = self._close_arr[self._step_idx]
        curr_marked = self._mark_to_market(curr_close)

        # Update peak for drawdown tracking
        self._peak_marked = max(self._peak_marked, curr_marked)

        # ── 5. Compute reward ─────────────────────────────────────────────────
        r_base = _safe_log_return(curr_marked, prev_marked)

        # Holding reward: +0.0001 bonus when sitting on a profitable position,
        # 0 otherwise — eliminates the old "fear of holding" penalty.
        r_hold = 0.0
        if self._position != 0.0 and self._entry_price > 0.0:
            if self._position > 0.0:
                raw_pnl = curr_close / self._entry_price - 1.0
            else:
                raw_pnl = 1.0 - curr_close / self._entry_price

            if raw_pnl > 0.0:
                r_hold = 0.0001   # small bonus for holding a winner
        else:
            raw_pnl = 0.0

        reward = r_base + r_hold + reward_shaping

        # ── 6. Termination checks ─────────────────────────────────────────────
        terminated = False
        truncated  = False

        # Drawdown circuit-breaker
        if self._peak_marked > 0:
            drawdown = 1.0 - curr_marked / self._peak_marked
            if drawdown >= self.drawdown_limit:
                reward    += self.drawdown_penalty
                terminated = True

        # Bankruptcy guard — double penalty: account wiped out is worse than drawdown
        if curr_marked <= 0.0:
            reward    += self.drawdown_penalty * 2
            terminated = True

        # Max-steps truncation → force liquidation
        if self._steps_done >= self.max_steps and not terminated:
            if self._position != 0.0:
                self._close_position(curr_close)  # settle at current close
                curr_marked = self._net_worth
            truncated = True

        # ── 7. Update position hold counter ───────────────────────────────────
        if self._position == 0.0:
            self._steps_in_position = 0
        elif self._position != prev_pos:        # just entered a new position
            self._steps_in_position = 1
        else:                                   # continuing same position
            self._steps_in_position += 1

        # ── 8. Build output ───────────────────────────────────────────────────
        obs  = self._get_obs()
        info = self._get_info(curr_marked)

        if self.render_mode == "human":
            self._render_step(reward, curr_marked)

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        if self.render_mode == "human":
            self._render_step(reward=0.0, marked_nw=self._mark_to_market(
                self._close_arr[self._step_idx]
            ))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _close_position(self, exec_price: float) -> float:
        """
        Close the current open position at exec_price with adverse slippage.

        Closing long  → sell at  exec_price * (1 - slippage_rate)   [lower]
        Closing short → buy  at  exec_price * (1 + slippage_rate)   [higher]

        Returns the realised PnL ratio (positive = profit).
        """
        pos = self._position
        # Slippage is adverse: direction is opposite to the position being closed
        fill = exec_price * (1.0 - pos * self.slippage_rate)

        if pos > 0.0:
            pnl_ratio = fill / self._entry_price - 1.0
        else:
            pnl_ratio = 1.0 - fill / self._entry_price

        # Apply realised PnL
        self._net_worth *= (1.0 + pnl_ratio)

        # Deduct commission
        commission = self._net_worth * self.commission_rate
        self._net_worth      -= commission
        self._total_commission += commission

        # Statistics
        self._n_trades += 1
        if pnl_ratio > 0.0:
            self._n_winning_trades += 1

        self._position    = 0.0
        self._entry_price = 0.0
        return pnl_ratio

    def _open_position(self, target_pos: float, exec_price: float) -> None:
        """
        Open a new position at exec_price with adverse slippage.

        Going long  → buy  at exec_price * (1 + slippage_rate)   [higher]
        Going short → sell at exec_price * (1 - slippage_rate)   [lower]
        """
        # Slippage is adverse: direction matches the position being opened
        fill = exec_price * (1.0 + target_pos * self.slippage_rate)

        # Deduct commission at entry
        commission = self._net_worth * self.commission_rate
        self._net_worth      -= commission
        self._total_commission += commission

        self._entry_price = fill
        self._position    = target_pos

    def _mark_to_market(self, price: float) -> float:
        """
        Compute unrealised portfolio value at the given price.
        Flat position → returns _net_worth unchanged.
        """
        if self._position == 0.0 or self._entry_price <= 0.0:
            return self._net_worth

        if self._position > 0.0:
            pnl_ratio = price / self._entry_price - 1.0
        else:
            pnl_ratio = 1.0 - price / self._entry_price

        return self._net_worth * (1.0 + pnl_ratio)

    def _get_obs(self) -> np.ndarray:
        """Build the dynamic observation vector for the current step."""
        # 动态创建长度为 (特征数 + 2) 的空数组
        obs = np.empty(self._n_features + 2, dtype=np.float32)
        
        # 填充 K 线特征
        obs[:self._n_features] = self._feat_arr[self._step_idx]
        
        # 填充仓位状态 (倒数第2个位置)
        obs[self._n_features]  = np.float32(self._position)

        # 填充浮盈亏状态 (倒数第1个位置)
        if self._position != 0.0 and self._entry_price > 0.0:
            curr_close = self._close_arr[self._step_idx]
            if self._position > 0.0:
                raw_pnl = curr_close / self._entry_price - 1.0
            else:
                raw_pnl = 1.0 - curr_close / self._entry_price
            obs[self._n_features + 1] = np.float32(np.clip(raw_pnl * 10.0, -10.0, 10.0))
        else:
            obs[self._n_features + 1] = np.float32(0.0)

        return obs
    
    def _get_info(self, marked_nw: Optional[float] = None) -> Dict[str, Any]:
        if marked_nw is None:
            marked_nw = self._mark_to_market(self._close_arr[self._step_idx])
        win_rate = (
            self._n_winning_trades / self._n_trades if self._n_trades > 0 else 0.0
        )
        return {
            "step":              self._steps_done,
            "data_idx":          self._step_idx,
            "position":          self._position,
            "net_worth":         self._net_worth,          # realised (no open pnl)
            "marked_nw":         marked_nw,                # incl. unrealised
            "total_return":      marked_nw / _INITIAL_NW - 1.0,
            "n_trades":          self._n_trades,
            "win_rate":          win_rate,
            "total_commission":  self._total_commission,
            "peak_marked":       self._peak_marked,
        }

    def _render_step(self, reward: float, marked_nw: float) -> None:
        close   = self._close_arr[self._step_idx]
        pos_str = {-1.0: "SHORT", 0.0: " FLAT", 1.0: " LONG"}.get(
            self._position, "  ???"
        )
        dd = 1.0 - marked_nw / self._peak_marked if self._peak_marked > 0 else 0.0
        print(
            f"step {self._steps_done:4d} | "
            f"close {close:>12.4f} | "
            f"{pos_str} | "
            f"NW {marked_nw:.6f} | "
            f"DD {dd:.2%} | "
            f"R {reward:+.6f}"
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_df(df: pd.DataFrame) -> None:
        required = set(["open", "close"] + FEATURE_COLS)
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"TradingEnv: DataFrame is missing required columns: {sorted(missing)}"
            )
        if len(df) < 100:
            raise ValueError(
                f"TradingEnv: DataFrame has only {len(df)} rows — too short."
            )


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _safe_log_return(curr: float, prev: float, eps: float = 1e-8) -> float:
    """Log return with guard against zero / negative values."""
    if prev <= 0.0 or curr <= 0.0:
        return -1.0
    return float(np.log(curr / prev + eps))
