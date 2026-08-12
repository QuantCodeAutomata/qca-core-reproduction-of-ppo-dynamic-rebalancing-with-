"""Oracle optimizer for computing look-ahead optimal weights and regret rewards.

Speed design
------------
* ``precompute()`` builds rho_bar[t] and Sigma[t] lookup tables for all
  timesteps in a training window in one pass — O(T) with vectorised numpy
  operations.  The per-step oracle call then uses these cached arrays and runs
  a fully vectorised simplex grid search (n_grid=50, ~1326 candidates for K=3)
  in ~0.05 ms, down from ~3.5 ms with scipy.minimize + per-step cov estimation.
* When the environment's data changes (e.g. bootstrap regeneration), the caller
  must invoke ``precompute()`` again on the new data.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class OracleOptimizer:
    """Compute oracle (look-ahead) portfolio weights and regret-based rewards.

    Args:
        n: Look-ahead horizon in business days (default 14).
        epsilon: Covariance regularisation constant (default 1e-6).
        n_grid: Simplex grid resolution for vectorised search (default 50).
    """

    # Class-level simplex grid cache: grid_cache[n_grid] = W array
    _grid_cache: dict = {}

    def __init__(self, n: int = 14, epsilon: float = 1e-6, n_grid: int = 50) -> None:
        self.n = n
        self.epsilon = epsilon
        self.n_grid = n_grid

        # Pre-computed lookup tables (populated by precompute())
        self._rho_table: np.ndarray | None = None    # (T, 3)
        self._sigma_table: np.ndarray | None = None  # (T, 3, 3)
        self._split_end: int | None = None

        # Build simplex grid once
        self._W = self._build_simplex_grid(n_grid)

    # ------------------------------------------------------------------
    # Simplex grid construction (class-level cache)
    # ------------------------------------------------------------------

    @classmethod
    def _build_simplex_grid(cls, n_grid: int) -> np.ndarray:
        """Return (N, 3) array of all grid points on the K=3 simplex."""
        if n_grid in cls._grid_cache:
            return cls._grid_cache[n_grid]
        i_vals = np.arange(n_grid + 1, dtype=np.float64)
        ii, jj = np.meshgrid(i_vals, i_vals)
        kk = n_grid - ii - jj
        mask = kk >= 0
        W = np.column_stack([ii[mask], jj[mask], kk[mask]]) / float(n_grid)
        cls._grid_cache[n_grid] = W
        return W

    # ------------------------------------------------------------------
    # Pre-computation API
    # ------------------------------------------------------------------

    def precompute(
        self,
        strategy_returns: pd.DataFrame,
        split_end_idx: int,
    ) -> "OracleOptimizer":
        """Pre-compute rho_bar and Sigma lookup tables for all timesteps.

        Call this once before training on a given data window.  After
        bootstrap regeneration, call it again with the new data.

        Args:
            strategy_returns: DataFrame (T, 3) of daily strategy returns.
            split_end_idx: Row index of the last allowed training observation
                (training / validation split boundary).

        Returns:
            self, for chaining.
        """
        ret = strategy_returns.values.astype(np.float64)
        T = len(ret)
        n = self.n
        eps = self.epsilon
        end = min(split_end_idx, T - 1)

        rho_table = np.zeros((T, 3), dtype=np.float64)
        sigma_table = np.zeros((T, 3, 3), dtype=np.float64)

        for t in range(T):
            # Forward-looking mean (clipped to split boundary)
            t_fwd = min(t + n, end, T - 1)
            if t_fwd > t:
                rho_table[t] = ret[t:t_fwd].mean(axis=0)

            # Covariance over [t-3n, t+3n] clipped to split boundary
            t_cov_s = max(0, t - 3 * n)
            t_cov_e = min(t + 3 * n, end, T - 1) + 1
            window = ret[t_cov_s:t_cov_e]
            if len(window) >= 5:
                sigma_table[t] = np.cov(window.T) + np.eye(3) * eps
            else:
                sigma_table[t] = np.eye(3) * eps

        self._rho_table = rho_table
        self._sigma_table = sigma_table
        self._split_end = split_end_idx
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_oracle_weights(
        self,
        strategy_returns: pd.DataFrame,
        t: int,
        prev_weights: np.ndarray,
        tc_train: float,
        split_end_idx: int,
    ) -> np.ndarray:
        """Compute oracle weights w* by solving a Sharpe-minus-TC problem.

        Uses pre-computed lookup tables when available (fast path, ~0.05 ms).
        Falls back to per-step scipy.minimize if tables have not been built
        (slow path, ~3.5 ms — only for correctness testing).

        Objective (maximised):
            w' * rho_bar / sqrt(w' Sigma w) − tc_train * ||w − w_prev||_1
        subject to: w ≥ 0, sum(w) = 1.

        Args:
            strategy_returns: DataFrame (T, 3) — used only if tables missing.
            t: Current time index.
            prev_weights: Previous portfolio weights (length 3).
            tc_train: Training transaction-cost coefficient.
            split_end_idx: Train/val split boundary row index.

        Returns:
            Optimal weight vector of length 3 (non-negative, sums to 1).
        """
        if self._rho_table is not None and t < len(self._rho_table):
            rho_bar = self._rho_table[t]
            Sigma = self._sigma_table[t]
        else:
            # Slow fallback: compute from raw returns
            rho_bar, Sigma = self._compute_rho_sigma(
                strategy_returns.values, t, split_end_idx
            )

        w_star = self._vectorised_grid_search(rho_bar, Sigma, prev_weights, tc_train)
        w_star = np.clip(w_star, 0.0, 1.0)
        total = w_star.sum()
        if total < 1e-12:
            return np.array([1 / 3, 1 / 3, 1 / 3])
        return w_star / total

    def compute_regret_reward(
        self,
        strategy_returns: pd.DataFrame,
        t: int,
        w_t: np.ndarray,
        w_star: np.ndarray,
    ) -> float:
        """Compute the regret-based reward at time t.

        Reward_t = −rho_bar_{t→t+n} · (w* − w_t)

        A positive reward means the agent's weights performed at least as well
        as the oracle (regret is negative or zero).

        Args:
            strategy_returns: DataFrame (T, 3).
            t: Current time index.
            w_t: Agent's chosen weight vector at time t.
            w_star: Oracle's optimal weight vector at time t.

        Returns:
            Scalar regret reward (float).
        """
        if self._rho_table is not None and t < len(self._rho_table):
            rho_bar = self._rho_table[t]
        else:
            ret = strategy_returns.values
            T = len(ret)
            t_fwd = min(t + self.n, T)
            if t_fwd <= t:
                return 0.0
            rho_bar = ret[t:t_fwd].mean(axis=0)
        return -float(rho_bar @ (w_star - w_t))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _vectorised_grid_search(
        self,
        rho_bar: np.ndarray,
        Sigma: np.ndarray,
        prev_weights: np.ndarray,
        tc_train: float,
    ) -> np.ndarray:
        """Fully vectorised grid search over the pre-built simplex grid.

        All N~1326 candidate weight vectors are evaluated simultaneously,
        yielding ~0.05 ms per call for K=3 with n_grid=50.

        Args:
            rho_bar: Expected return vector (3,).
            Sigma: Covariance matrix (3, 3).
            prev_weights: Previous weight vector (3,).
            tc_train: Transaction-cost coefficient.

        Returns:
            Best weight vector found on the grid.
        """
        W = self._W  # (N, 3)
        port_ret = W @ rho_bar                                      # (N,)
        port_var = np.einsum("ij,jk,ik->i", W, Sigma, W)           # (N,)
        port_std = np.sqrt(np.maximum(port_var, self.epsilon))
        sharpe = port_ret / port_std
        turnover = np.abs(W - prev_weights).sum(axis=1)             # (N,)
        obj = sharpe - tc_train * turnover
        return W[np.argmax(obj)].copy()

    def _compute_rho_sigma(
        self,
        ret: np.ndarray,
        t: int,
        split_end_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute rho_bar and Sigma from raw returns (slow fallback)."""
        T = len(ret)
        t_fwd = min(t + self.n, split_end_idx, T - 1)
        if t_fwd > t:
            rho_bar = ret[t:t_fwd].mean(axis=0)
        else:
            rho_bar = np.zeros(3)

        t_cov_s = max(0, t - 3 * self.n)
        t_cov_e = min(t + 3 * self.n, split_end_idx, T - 1) + 1
        window = ret[t_cov_s:t_cov_e]
        if len(window) >= 5:
            Sigma = np.cov(window.T) + np.eye(3) * self.epsilon
        else:
            Sigma = np.eye(3) * self.epsilon
        return rho_bar, Sigma

    def _scipy_optimize(
        self,
        rho_bar: np.ndarray,
        Sigma: np.ndarray,
        prev_weights: np.ndarray,
        tc_train: float,
    ) -> np.ndarray:
        """scipy.minimize SLSQP fallback (kept for validation purposes)."""
        def neg_obj(w: np.ndarray) -> float:
            port_std = np.sqrt(max(float(w @ Sigma @ w), self.epsilon))
            sharpe = float(w @ rho_bar) / port_std
            turnover = float(np.sum(np.abs(w - prev_weights)))
            return -(sharpe - tc_train * turnover)

        result = minimize(
            neg_obj,
            np.array([1 / 3, 1 / 3, 1 / 3]),
            method="SLSQP",
            bounds=[(0, 1)] * 3,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
            options={"ftol": 1e-9, "maxiter": 500},
        )
        if result.success:
            return result.x
        return self._vectorised_grid_search(rho_bar, Sigma, prev_weights, tc_train)
