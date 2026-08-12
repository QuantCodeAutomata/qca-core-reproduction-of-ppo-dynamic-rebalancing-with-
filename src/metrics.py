"""Performance metrics for portfolio evaluation."""

import numpy as np
import pandas as pd


def compute_annual_return(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    """Compound annual growth rate (CAGR) from a daily returns series.

    Args:
        returns: Daily simple return series.
        periods_per_year: Number of trading days per year (default 252).

    Returns:
        CAGR as a float (e.g. 0.08 for 8 %).  Returns 0.0 for empty series.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0
    n = len(returns)
    total_growth = (1.0 + returns).prod()
    if total_growth <= 0:
        return -1.0
    cagr = total_growth ** (periods_per_year / n) - 1.0
    return float(cagr)


def compute_max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from peak of the cumulative NAV.

    Args:
        returns: Daily simple return series.

    Returns:
        Maximum drawdown as a non-positive float (e.g. -0.30 for −30 %).
        Returns 0.0 for an empty or monotonically non-decreasing series.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0
    nav = (1.0 + returns).cumprod()
    peak = nav.cummax()
    drawdown = (nav - peak) / peak
    return float(drawdown.min())


def compute_sharpe_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252
) -> float:
    """Annualised Sharpe ratio.

    Args:
        returns: Daily simple return series.
        risk_free: Daily risk-free rate (default 0.0).
        periods_per_year: Trading days per year (default 252).

    Returns:
        Annualised Sharpe ratio.  Returns 0.0 for zero-std or empty series.
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free
    std = excess.std()
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float((excess.mean() / std) * np.sqrt(periods_per_year))


def compute_calmar_ratio(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    """Calmar ratio: CAGR divided by absolute maximum drawdown.

    Args:
        returns: Daily simple return series.
        periods_per_year: Trading days per year (default 252).

    Returns:
        Calmar ratio.  Returns 0.0 when max drawdown is zero (no loss ever).
    """
    cagr = compute_annual_return(returns, periods_per_year)
    mdd = compute_max_drawdown(returns)
    if mdd == 0.0:
        return 0.0
    return float(cagr / abs(mdd))


def compute_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Omega ratio: probability-weighted gain above threshold over loss below.

    Omega = sum(max(r - L, 0)) / sum(max(L - r, 0))

    Args:
        returns: Daily simple return series.
        threshold: Minimum acceptable return (MAR) threshold (default 0.0).

    Returns:
        Omega ratio.  Returns np.inf if there are no losses, 0.0 if empty.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0
    gains = np.sum(np.maximum(returns - threshold, 0.0))
    losses = np.sum(np.maximum(threshold - returns, 0.0))
    if losses == 0.0:
        return np.inf
    return float(gains / losses)


def compute_sortino_ratio(
    returns: pd.Series, mar: float = 0.0, periods_per_year: int = 252
) -> float:
    """Annualised Sortino ratio using downside deviation.

    Args:
        returns: Daily simple return series.
        mar: Minimum acceptable return threshold (default 0.0).
        periods_per_year: Trading days per year (default 252).

    Returns:
        Annualised Sortino ratio.  Returns 0.0 for empty or zero-downside series.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0
    excess = returns - mar
    downside = np.minimum(excess, 0.0)
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std == 0.0 or np.isnan(downside_std):
        return 0.0
    mean_excess = excess.mean()
    return float((mean_excess / downside_std) * np.sqrt(periods_per_year))


def compute_all_metrics(returns: pd.Series, label: str = "") -> dict:
    """Compute a full suite of risk-adjusted performance metrics.

    Args:
        returns: Daily simple return series.
        label: Optional label stored under key ``'label'`` in the result dict.

    Returns:
        Dictionary with keys:
            - ``label`` (str)
            - ``annual_return`` (float)
            - ``max_drawdown`` (float, ≤ 0)
            - ``sharpe`` (float)
            - ``calmar`` (float)
            - ``omega`` (float)
            - ``sortino`` (float)
    """
    return {
        "label": label,
        "annual_return": compute_annual_return(returns),
        "max_drawdown": compute_max_drawdown(returns),
        "sharpe": compute_sharpe_ratio(returns),
        "calmar": compute_calmar_ratio(returns),
        "omega": compute_omega_ratio(returns),
        "sortino": compute_sortino_ratio(returns),
    }


def compute_portfolio_returns(
    weights_df: pd.DataFrame,
    strategy_returns_df: pd.DataFrame,
    tc_eval: float = 0.0025,
) -> pd.Series:
    """Compute net daily portfolio returns given weight decisions and strategy returns.

    The weight vector decided at the end of day *t* earns the strategy return
    realised on day *t+1*.  Transaction costs are deducted when weights change:

        R_net_{t+1} = w_t · mu_{t+1} − TC_eval · ||w_t − w_{t−1}||_1

    For the first observation the previous weight is treated as **w_{−1} = 0**
    (entering from cash), so the full initial position counts as turnover.

    Args:
        weights_df: DataFrame of shape (T, 3) with portfolio weights, indexed
            by date.  Weights must sum to 1 each row.
        strategy_returns_df: DataFrame of shape (T, 3) with strategy returns,
            indexed by the same dates as *weights_df*.
        tc_eval: Evaluation-period transaction-cost rate (default 0.0025).

    Returns:
        ``pd.Series`` of net daily returns indexed by dates d_1 … d_{T−1},
        aligned to the day on which each return is realised.
    """
    w = weights_df.values.astype(float)      # shape (T, 3)
    r = strategy_returns_df.values.astype(float)  # shape (T, 3)
    T = len(w)

    if T < 2:
        return pd.Series(dtype=float)

    # w[t-1] earns r[t]; iterate t from 1 to T-1
    w_prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])  # shape (T, 3)

    gross = (w_prev * r).sum(axis=1)                          # shape (T,)
    turnover = np.abs(w - w_prev).sum(axis=1)                 # ||w_t - w_{t-1}||_1
    net = gross - tc_eval * turnover                           # shape (T,)

    # Return is realised starting from row index 1 (day t+1 for t=0)
    dates = strategy_returns_df.index
    return pd.Series(net[1:], index=dates[1:])
