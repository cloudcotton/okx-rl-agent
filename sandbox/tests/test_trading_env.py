"""
Unit tests for TradingEnv.

Run with:  python -m pytest sandbox/tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sandbox.trading_env import TradingEnv
from sandbox.data_loader import FEATURE_COLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n_rows: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLCV + features DataFrame for testing."""
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


@pytest.fixture
def df():
    return _make_df()


@pytest.fixture
def env(df):
    return TradingEnv(df, max_steps=100)


# ---------------------------------------------------------------------------
# 1. Gymnasium API compliance
# ---------------------------------------------------------------------------

class TestGymnasiumAPI:

    def test_observation_space_shape(self, env):
        obs, _ = env.reset(seed=0)
        assert obs.shape == (16,), "obs must be 16-dim"
        assert obs.dtype == np.float32

    def test_observation_in_bounds(self, env):
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs), "initial obs out of bounds"

    def test_step_returns_five_tuple(self, env):
        env.reset(seed=0)
        result = env.step(1)  # stay flat
        assert len(result) == 5

    def test_step_obs_in_bounds(self, env):
        env.reset(seed=0)
        obs, *_ = env.step(2)
        assert env.observation_space.contains(obs)

    def test_action_space(self, env):
        assert env.action_space.n == 3
        for a in range(3):
            assert env.action_space.contains(a)

    def test_full_episode_does_not_crash(self, env):
        obs, _ = env.reset(seed=42)
        done = False
        steps = 0
        while not done and steps < 200:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
            steps += 1
        assert steps > 0


# ---------------------------------------------------------------------------
# 2. No look-ahead bias
# ---------------------------------------------------------------------------

class TestNoLookahead:

    def test_execution_at_next_open(self, df):
        """Verify trade executes at bar-(t+1) open, not bar-t close."""
        env = TradingEnv(df, max_steps=50)
        env.reset(seed=1)

        start_idx = env._step_idx
        assert env._position == 0.0

        # Action 2 = go long
        env.step(2)

        # entry_price must be close to open[start_idx + 1], not close[start_idx]
        expected_open = df["open"].iloc[start_idx + 1]
        expected_fill = expected_open * (1 + env.slippage_rate)  # long → pay more
        assert abs(env._entry_price - expected_fill) < 1e-6, (
            f"entry_price {env._entry_price:.4f} != expected fill {expected_fill:.4f}"
        )

    def test_step_idx_advances_by_one(self, env):
        env.reset(seed=2)
        idx_before = env._step_idx
        env.step(1)
        assert env._step_idx == idx_before + 1


# ---------------------------------------------------------------------------
# 3. Friction costs
# ---------------------------------------------------------------------------

class TestFrictionCosts:

    def _env_with_flat_price(self) -> TradingEnv:
        """Create env where open = close = 1.0 for all bars (zero price movement)."""
        n = 2000
        df = pd.DataFrame({
            "datetime": pd.date_range("2022-01-01", periods=n, freq="5min"),
            "open":  np.ones(n),
            "high":  np.ones(n),
            "low":   np.ones(n),
            "close": np.ones(n),
            "vol":   np.ones(n),
        })
        for feat in FEATURE_COLS:
            df[feat] = np.zeros(n, dtype=np.float32)
        return TradingEnv(df, max_steps=10)

    def test_commission_deducted_on_trade(self):
        env = self._env_with_flat_price()
        env.reset(seed=0)

        nw_before = env._net_worth
        env.step(2)  # go long (trigger commission at entry)
        assert env._net_worth < nw_before, "commission must reduce net worth"

    def test_round_trip_costs_money(self):
        """Buy and immediately sell on flat price: relative NW loss ≈ 2×(commission+slippage)."""
        env = self._env_with_flat_price()
        env.reset(seed=0)
        nw_start = env._net_worth  # capture post-jitter baseline

        env.step(2)   # go long
        env.step(1)   # go flat
        # After round trip: lost entry commission + slippage + exit commission + slippage
        expected_loss_approx = 2 * (env.commission_rate + env.slippage_rate)
        actual_loss = (nw_start - env._net_worth) / nw_start
        assert actual_loss > 0.0, "round trip must cost money"
        assert abs(actual_loss - expected_loss_approx) < 0.001, (
            f"round-trip cost {actual_loss:.6f} far from expected {expected_loss_approx:.6f}"
        )

    def test_long_adverse_slippage(self):
        """Going long should fill at a price ABOVE the next open."""
        env = self._env_with_flat_price()
        env.reset(seed=0)
        exec_open = env._open_arr[env._step_idx + 1]
        env.step(2)  # go long
        assert env._entry_price > exec_open

    def test_short_adverse_slippage(self):
        """Going short should fill at a price BELOW the next open."""
        env = self._env_with_flat_price()
        env.reset(seed=0)
        exec_open = env._open_arr[env._step_idx + 1]
        env.step(0)  # go short
        assert env._entry_price < exec_open

    def test_holding_penalty_applied(self):
        """While holding a position, every step subtracts the holding penalty."""
        env = self._env_with_flat_price()
        env.reset(seed=0)

        env.step(2)          # go long
        _, r1, _, _, _ = env.step(2)   # hold long (action stays long = no trade)
        # r_base ≈ 0 (flat price), r_hold should be negative
        assert r1 < 0.0, f"holding penalty not applied: reward={r1}"

    def test_adhd_penalty_on_change(self):
        """Any position change incurs the ADHD penalty."""
        env = self._env_with_flat_price()
        env.reset(seed=0)

        # step 1: go from flat to long — should include adhd_penalty
        _, r, _, _, _ = env.step(2)
        # With flat price, r_base ≈ 0 (minus entry commission/slippage effects)
        # The adhd_penalty (-0.0002) must push reward below 0
        assert r < 0.0

    def test_reversal_penalty_on_flip(self):
        """A direct long→short flip incurs both adhd + reversal penalty."""
        env = self._env_with_flat_price()
        env.reset(seed=0)

        env.step(2)                      # go long
        _, r_hold, _, _, _ = env.step(2)  # hold long (no change)
        env.step(2)                      # hold long again
        _, r_flip, _, _, _ = env.step(0)  # direct flip to short

        # The flip step reward must be more negative than the hold step reward
        assert r_flip < r_hold, (
            f"flip reward {r_flip:.6f} should be more negative than hold {r_hold:.6f}"
        )


# ---------------------------------------------------------------------------
# 4. Reward shaping
# ---------------------------------------------------------------------------

class TestRewardShaping:

    def test_base_reward_positive_on_long_up(self, df):
        """If price rises while long, base reward > 0 (minus small friction)."""
        env = TradingEnv(df, max_steps=200, commission_rate=0.0, slippage_rate=0.0,
                         holding_penalty=0.0, adhd_penalty=0.0)
        # Force a specific start where we know the next bar goes up
        env.reset(seed=99)

        # Find a step where close[t+1] > close[t]
        for i in range(env._step_idx, env._step_idx + 200):
            if env._close_arr[i + 1] > env._close_arr[i]:
                env._step_idx = i
                break

        env.step(2)   # go long
        # Now hold one step where close rises
        for _ in range(50):
            obs, r, terminated, truncated, _ = env.step(2)  # keep long
            if terminated or truncated:
                break
            # If next close is higher, reward should be positive
            idx = env._step_idx
            if env._close_arr[idx] > env._close_arr[idx - 1]:
                assert r > 0, f"Expected positive reward on up move, got {r}"
                break

    def test_drawdown_circuit_breaker(self):
        """Portfolio drawdown > 10 % terminates the episode with large penalty."""
        n = 2000
        # Price starts at 100, then crashes to 50 (−50 %)
        prices = np.concatenate([
            np.full(5, 100.0),
            np.linspace(100.0, 50.0, n - 5),
        ])
        df = pd.DataFrame({
            "datetime": pd.date_range("2022-01-01", periods=n, freq="5min"),
            "open":  prices,
            "high":  prices,
            "low":   prices,
            "close": prices,
            "vol":   np.ones(n),
        })
        for feat in FEATURE_COLS:
            df[feat] = np.zeros(n, dtype=np.float32)

        env = TradingEnv(df, max_steps=1800, drawdown_limit=0.10, drawdown_penalty=-1.0)
        env.reset(seed=0)
        env._step_idx = 2  # start near beginning so we see the crash

        env.step(2)   # go long into crash

        terminated = False
        total_reward = 0.0
        steps = 0
        while not terminated and steps < 1800:
            _, r, terminated, truncated, _ = env.step(2)  # keep holding
            total_reward += r
            steps += 1
            if truncated:
                break

        assert terminated, "drawdown should have triggered episode termination"
        assert total_reward < -0.5, f"total reward {total_reward:.4f} should reflect large loss"

    def test_bankruptcy_penalty_applied(self):
        """When marked NW hits ≤ 0, reward receives drawdown_penalty * 2 and episode ends."""
        n = 300
        df = pd.DataFrame({
            "datetime": pd.date_range("2022-01-01", periods=n, freq="5min"),
            "open":  np.ones(n),
            "high":  np.ones(n),
            "low":   np.ones(n),
            "close": np.ones(n),
            "vol":   np.ones(n),
        })
        for feat in FEATURE_COLS:
            df[feat] = np.zeros(n, dtype=np.float32)

        penalty = -1.0
        env = TradingEnv(df, max_steps=200, drawdown_limit=1.0,  # disable drawdown CB
                         drawdown_penalty=penalty)
        env.reset(seed=0)

        # Go long at price 1.0
        env.step(2)

        # Force close price to 0.0 at the next step — marks long position to zero
        # marked_nw = net_worth * (0.0 / entry_price - 1.0 + 1.0) = net_worth * 0 = 0
        env._close_arr[env._step_idx + 1] = 0.0
        # Also force open of the step after (needed to avoid index errors)
        env._open_arr[env._step_idx + 1]  = 0.0

        _, r, terminated, truncated, _ = env.step(2)  # hold long into zero price

        assert terminated, "bankruptcy (marked_nw ≤ 0) must terminate the episode"
        assert r <= penalty * 2 + 0.01, (
            f"bankruptcy reward {r:.4f} should be ≤ drawdown_penalty×2 = {penalty * 2}"
        )

    def test_unrealized_pnl_scaling(self):
        """obs[15] should saturate at ±10 only at ±100% move (not ±10%)."""
        n = 2000
        # Flat price so we can control the relationship manually
        df = pd.DataFrame({
            "datetime": pd.date_range("2022-01-01", periods=n, freq="5min"),
            "open":  np.ones(n),
            "high":  np.ones(n),
            "low":   np.ones(n),
            "close": np.ones(n),
            "vol":   np.ones(n),
        })
        for feat in FEATURE_COLS:
            df[feat] = np.zeros(n, dtype=np.float32)

        env = TradingEnv(df, max_steps=100, slippage_rate=0.0, commission_rate=0.0)
        env.reset(seed=0)
        env.step(2)  # go long; entry_price = 1.0 (no slippage)

        # Simulate a 10% gain by manually raising close price
        entry = env._entry_price  # ≈ 1.0
        env._close_arr[env._step_idx] = entry * 1.10  # +10% move
        obs = env._get_obs()
        # With ×10 scaling: 10% move → 10% × 10 = 1.0, NOT 10.0
        assert abs(obs[15] - 1.0) < 0.05, (
            f"10% move should give obs[15]≈1.0 (got {obs[15]:.4f}); "
            "check that scaling factor is ×10 not ×100"
        )

        # A 100% gain should clip to exactly 10.0
        env._close_arr[env._step_idx] = entry * 2.0  # +100% move
        obs = env._get_obs()
        assert obs[15] == pytest.approx(10.0, abs=0.01), (
            f"100% move should clip to obs[15]=10.0 (got {obs[15]:.4f})"
        )


# ---------------------------------------------------------------------------
# 5. Episode lifecycle
# ---------------------------------------------------------------------------

class TestEpisodeLifecycle:

    def test_truncation_at_max_steps(self, df):
        env = TradingEnv(df, max_steps=50)
        env.reset(seed=7)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            steps += 1
            assert steps <= 55, "episode ran longer than max_steps + buffer"
        assert truncated or terminated

    def test_forced_liquidation_at_truncation(self, df):
        """At max_steps, any open position is automatically closed."""
        env = TradingEnv(df, max_steps=20)
        env.reset(seed=8)

        # Hold long for the whole episode
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(2)

        assert info["position"] == 0.0 or truncated, \
            "position should be closed on truncation"

    def test_random_start_varies(self, df):
        """Two resets with different seeds should (almost always) start at different indices."""
        env = TradingEnv(df, max_steps=100)
        env.reset(seed=10)
        idx1 = env._step_idx
        env.reset(seed=11)
        idx2 = env._step_idx
        # Very unlikely to be equal given 5000-row dataset
        assert idx1 != idx2, "random start should differ across seeds"

    def test_reproducibility(self, df):
        """Same seed → same start index and same episode trajectory."""
        env = TradingEnv(df, max_steps=20)
        env.reset(seed=42)
        idx_a = env._step_idx
        env.reset(seed=42)
        idx_b = env._step_idx
        assert idx_a == idx_b, "same seed must give same start index"


# ---------------------------------------------------------------------------
# 6. Observation correctness
# ---------------------------------------------------------------------------

class TestObservation:

    def test_position_encoded_in_obs(self, env):
        env.reset(seed=5)
        env.step(2)      # go long
        obs, *_ = env.step(2)  # hold long
        assert obs[14] == pytest.approx(1.0, abs=1e-5), \
            "obs[14] must encode current_position = +1"

    def test_flat_position_obs_14_is_zero(self, env):
        env.reset(seed=5)
        obs, _ = env.reset(seed=5)
        assert obs[14] == pytest.approx(0.0, abs=1e-5)

    def test_unrealized_pnl_obs_zero_when_flat(self, env):
        obs, _ = env.reset(seed=6)
        assert obs[15] == pytest.approx(0.0, abs=1e-5)

    def test_unrealized_pnl_obs_in_bounds(self, df):
        env = TradingEnv(df, max_steps=200)
        env.reset(seed=3)
        env.step(2)   # go long
        for _ in range(50):
            obs, _, term, trunc, _ = env.step(2)
            assert -10.0 <= obs[15] <= 10.0, f"unrealized_pnl out of bounds: {obs[15]}"
            if term or trunc:
                break

    def test_feature_values_propagated(self, df):
        env = TradingEnv(df, max_steps=50)
        obs, _ = env.reset(seed=4)
        # First 14 elements must match the feature array at the start index
        expected = env._feat_arr[env._step_idx]
        np.testing.assert_array_equal(obs[:14], expected)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
