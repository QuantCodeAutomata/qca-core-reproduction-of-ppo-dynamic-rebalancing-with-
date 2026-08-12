"""Feature engineering for the PPO portfolio rebalancing pipeline."""

import os
import numpy as np
import pandas as pd

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw_prices.csv")

# Strategy blend weights for strat_2
_W2 = {"dev_eq": 0.55, "em_eq": 0.05, "credit": 0.20, "govies": 0.20}

# Observation feature column names (15 base features, indices 0–14)
OBS_FEATURE_COLS = [
    "strat_1",
    "strat_2",
    "strat_3",
    "hy_spread",
    "vix_chg",
    "move_proxy",
    "mean40_s1",
    "mean40_s2",
    "mean40_s3",
    "std60_s1",
    "std60_s2",
    "std60_s3",
    "std60_ctx1",
    "std60_ctx2",
    "std60_ctx3",
]


class FeatureBuilder:
    """Build observation features for the PPO rebalancing environment.

    The 19-dimensional observation vector is structured as:
        [0-2]   strat_1, strat_2, strat_3 daily returns
        [3-5]   hy_spread, vix_chg, move_proxy contextual changes
        [6-8]   mean40_s1, mean40_s2, mean40_s3 (40-day rolling mean)
        [9-11]  std60_s1, std60_s2, std60_s3 (60-day rolling std)
        [12-14] std60_ctx1, std60_ctx2, std60_ctx3 (60-day rolling std of context)
        [15-17] prev_w1, prev_w2, prev_w3 (filled with NaN here; set at runtime)
        [18]    tc_scalar (filled with NaN here; set at runtime)

    Args:
        price_path: Path to raw_prices.csv produced by data/download.py.
    """

    def __init__(self, price_path: str = _DATA_PATH) -> None:
        self._prices: pd.DataFrame = pd.read_csv(
            price_path, index_col="Date", parse_dates=True
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_returns(self) -> pd.DataFrame:
        """Compute daily log-style simple returns for each underlying series."""
        p = self._prices
        rets = pd.DataFrame(index=p.index)
        rets["dev_eq"] = p["EFA"].pct_change()
        rets["em_eq"] = p["EEM"].pct_change()
        rets["credit"] = p["LQD"].pct_change()
        rets["govies"] = p["TLT"].pct_change()
        rets["hyg_ret"] = p["HYG"].pct_change()
        # Absolute daily change (not percentage return) for index series
        rets["vix_abs"] = p["^VIX"].diff()
        rets["tnx_abs"] = p["^TNX"].diff()
        return rets

    def _build_features(self, rets: pd.DataFrame) -> pd.DataFrame:
        """Construct all 15 base feature columns from underlying returns."""
        feat = pd.DataFrame(index=rets.index)

        # Strategy returns
        feat["strat_1"] = rets["dev_eq"]
        feat["strat_2"] = (
            _W2["dev_eq"] * rets["dev_eq"]
            + _W2["em_eq"] * rets["em_eq"]
            + _W2["credit"] * rets["credit"]
            + _W2["govies"] * rets["govies"]
        )
        feat["strat_3"] = rets["govies"]

        # Contextual series
        feat["hy_spread"] = rets["hyg_ret"]
        feat["vix_chg"] = rets["vix_abs"]
        feat["move_proxy"] = rets["tnx_abs"]

        # Rolling mean (40-day) of strategy returns
        feat["mean40_s1"] = feat["strat_1"].rolling(40).mean()
        feat["mean40_s2"] = feat["strat_2"].rolling(40).mean()
        feat["mean40_s3"] = feat["strat_3"].rolling(40).mean()

        # Rolling std (60-day) of strategy returns
        feat["std60_s1"] = feat["strat_1"].rolling(60).std()
        feat["std60_s2"] = feat["strat_2"].rolling(60).std()
        feat["std60_s3"] = feat["strat_3"].rolling(60).std()

        # Rolling std (60-day) of contextual series
        feat["std60_ctx1"] = feat["hy_spread"].rolling(60).std()
        feat["std60_ctx2"] = feat["vix_chg"].rolling(60).std()
        feat["std60_ctx3"] = feat["move_proxy"].rolling(60).std()

        return feat

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        """Build the full feature DataFrame over the requested date range.

        All NaN rows from the rolling warm-up period are dropped before
        the date filter is applied so the returned DataFrame is clean.

        Args:
            start_date: Inclusive start date (YYYY-MM-DD). None → full history.
            end_date:   Inclusive end date (YYYY-MM-DD). None → full history.

        Returns:
            DataFrame with columns defined in OBS_FEATURE_COLS (15 columns),
            where strat_1/strat_2/strat_3 are also the raw strategy returns.
            NaN rows from the warm-up period are removed.
        """
        rets = self._compute_returns()
        feat = self._build_features(rets)
        feat = feat[OBS_FEATURE_COLS].dropna()

        if start_date is not None:
            feat = feat[feat.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            feat = feat[feat.index <= pd.Timestamp(end_date)]

        return feat

    def get_strategy_returns(
        self, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """Return daily strategy returns for the three sleeves.

        Args:
            start: Inclusive start date (YYYY-MM-DD).
            end:   Inclusive end date (YYYY-MM-DD).

        Returns:
            DataFrame with columns ['strat_1', 'strat_2', 'strat_3'].
        """
        feat = self.build(start, end)
        return feat[["strat_1", "strat_2", "strat_3"]].copy()

    def get_obs_features(
        self, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """Return the 15 base observation features (no weights or tc_scalar).

        Args:
            start: Inclusive start date (YYYY-MM-DD).
            end:   Inclusive end date (YYYY-MM-DD).

        Returns:
            DataFrame with exactly 15 columns (OBS_FEATURE_COLS), no NaN.
        """
        return self.build(start, end)
