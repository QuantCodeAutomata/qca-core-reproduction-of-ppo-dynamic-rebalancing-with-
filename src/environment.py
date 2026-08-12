import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class PortfolioEnv(gym.Env):
    '''
    Custom gymnasium environment for portfolio rebalancing across K=3 strategy sleeves.
    
    Observation space (19-dim):
      [0:3]   Current strategy returns (strat_1, strat_2, strat_3)
      [3:6]   Current contextual returns/changes (hy_spread, vix_chg, move_proxy)
      [6:9]   40-day rolling mean of strategy returns
      [9:12]  60-day rolling std of strategy returns
      [12:15] 60-day rolling std of contextual series
      [15:18] Previous portfolio weights (w_{t-1})
      [18]    Current TC schedule scalar
    
    Action space (3-dim): unconstrained logits -> softmax -> portfolio weights
    
    Reward modes:
      'regret': -rho_bar @ (w_star - w_t)  [requires oracle]
      'diff_sharpe': Differential Sharpe reward (Moody & Saffell formula)
      'embedded_drawdown': Embedded drawdown reward with dynamic alpha
      'return_only': realized net daily return
    '''
    metadata = {'render_modes': []}

    def __init__(self,
                 features_df: pd.DataFrame,  # 15-column base features
                 strategy_returns_df: pd.DataFrame,  # 3 strategy returns
                 reward_type: str = 'regret',
                 oracle=None,  # OracleOptimizer instance
                 tc_schedule=None,  # TCScheduler instance
                 split_end_idx: int = None,
                 benchmark_mdd: float = -0.122,  # for embedded_drawdown
                 k_emb: float = 1.0,  # k param for embedded drawdown (assumed=1)
                 seed: int = 42):
        super().__init__()

        self.features_df = features_df.reset_index(drop=True)
        self.strategy_returns = strategy_returns_df.reset_index(drop=True)
        self.reward_type = reward_type
        self.oracle = oracle
        self.tc_schedule = tc_schedule
        self.split_end_idx = split_end_idx if split_end_idx is not None else len(features_df) - 1
        self.benchmark_mdd = benchmark_mdd
        self.k_emb = k_emb

        self.K = 3
        self.T = len(features_df)

        # Spaces
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(19,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-3.0, high=3.0, shape=(3,), dtype=np.float32)

        # Diff Sharpe state
        self.A_t = 0.0  # first moment estimator
        self.B_t = 0.0  # second moment estimator
        self.eta = 1/252

        # Running NAV for drawdown reward
        self.nav = 1.0
        self.peak_nav = 1.0
        self.mdd_running = 0.0

        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.t = 0
        self.prev_weights = np.array([1/3, 1/3, 1/3], dtype=np.float32)

        # Reset path-dependent reward state
        self.A_t = 0.0
        self.B_t = 0.0
        self.nav = 1.0
        self.peak_nav = 1.0
        self.mdd_running = 0.0

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        row = self.features_df.iloc[self.t]
        tc_scalar = self.tc_schedule.get_tc(self.t) if self.tc_schedule else 0.0

        obs = np.array([
            row['strat_1'], row['strat_2'], row['strat_3'],
            row['hy_spread'], row['vix_chg'], row['move_proxy'],
            row['mean40_s1'], row['mean40_s2'], row['mean40_s3'],
            row['std60_s1'], row['std60_s2'], row['std60_s3'],
            row['std60_ctx1'], row['std60_ctx2'], row['std60_ctx3'],
            self.prev_weights[0], self.prev_weights[1], self.prev_weights[2],
            tc_scalar
        ], dtype=np.float32)

        # Clip and handle NaN
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = np.clip(obs, -10.0, 10.0)
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Softmax to get weights
        action = np.clip(action, -3.0, 3.0)
        exp_a = np.exp(action - action.max())
        w_t = exp_a / exp_a.sum()

        # Get current strategy returns (for next step t+1)
        if self.t + 1 < self.T:
            next_ret = self.strategy_returns.iloc[self.t + 1].values
        else:
            next_ret = np.zeros(self.K)

        # TC for evaluation (inside env we track for reward accounting)
        tc_eval = 0.0025
        turnover = np.sum(np.abs(w_t - self.prev_weights))
        gross_ret = float(w_t @ next_ret)
        net_ret = gross_ret - tc_eval * turnover

        # Compute reward based on mode
        reward = self._compute_reward(w_t, net_ret, next_ret)

        # Update state
        self.prev_weights = w_t.copy()
        self.t += 1
        terminated = self.t >= self.T - 1

        # Update NAV for drawdown tracking
        self.nav *= (1 + net_ret)
        self.peak_nav = max(self.peak_nav, self.nav)
        self.mdd_running = min(self.mdd_running, (self.nav - self.peak_nav) / (self.peak_nav + 1e-12))

        obs = self._get_obs() if not terminated else self._get_obs()
        info = {'net_ret': net_ret, 'gross_ret': gross_ret, 'turnover': turnover, 'weights': w_t.copy()}
        return obs, float(reward), terminated, False, info

    def _compute_reward(self, w_t: np.ndarray, net_ret: float, next_ret: np.ndarray) -> float:
        if self.reward_type == 'regret':
            if self.oracle is None:
                return net_ret
            tc_train = self.tc_schedule.get_tc(self.t) if self.tc_schedule else 0.0
            w_star = self.oracle.compute_oracle_weights(
                self.strategy_returns, self.t, self.prev_weights, tc_train, self.split_end_idx)
            reward = self.oracle.compute_regret_reward(
                self.strategy_returns, self.t, w_t, w_star)
            return reward

        elif self.reward_type == 'diff_sharpe':
            # Moody & Saffell Differential Sharpe
            R_t = net_ret
            dA = R_t - self.A_t
            dB = R_t**2 - self.B_t
            denom = (self.B_t - self.A_t**2)
            if abs(denom) < 1e-10:
                D_t = 0.0
            else:
                D_t = (self.B_t * dA - 0.5 * self.A_t * dB) / (denom ** 1.5 + 1e-10)
            self.A_t += self.eta * dA
            self.B_t += self.eta * dB
            return float(D_t)

        elif self.reward_type == 'embedded_drawdown':
            # Embedded drawdown reward: [k / (1 + exp(-r))] * (-exp(MDD_t) + exp(alpha))
            # alpha = benchmark_mdd (dynamic, set from 60/40 benchmark MDD)
            mdd = self.mdd_running  # negative value
            r = net_ret
            alpha = self.benchmark_mdd
            sigmoid_r = 1.0 / (1.0 + np.exp(-r))
            reward = (self.k_emb / (1.0 + np.exp(-r))) * (-np.exp(mdd) + np.exp(alpha))
            return float(reward)

        elif self.reward_type == 'return_only':
            tc_train = self.tc_schedule.get_tc(self.t) if self.tc_schedule else 0.0
            turnover = np.sum(np.abs(w_t - self.prev_weights))
            return float(w_t @ next_ret - tc_train * turnover)

        else:
            raise ValueError(f'Unknown reward_type: {self.reward_type}')
