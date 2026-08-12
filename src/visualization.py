"""Visualization utilities for PPO dynamic rebalancing experiments.

All functions produce self-contained Matplotlib/Seaborn figures and accept an
optional *save_path* argument to persist them to disk.  Figures are closed
automatically after saving to avoid memory leaks in batch runs.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# ── Aesthetic defaults ────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.1)
PALETTE = sns.color_palette("tab10")

STRAT_COLORS = {
    "s1": PALETTE[0],   # Strategy 1 – Developed Equities
    "s2": PALETTE[1],   # Strategy 2 – 60/40 blend
    "s3": PALETTE[2],   # Strategy 3 – Govies
}
METHOD_COLORS = {
    "Regret (BB+TC)":     PALETTE[0],
    "60/40 Benchmark":    PALETTE[3],
    "Diff. Sharpe":       PALETTE[1],
    "Emb. Drawdown":      PALETTE[2],
    "BB+Regret":          PALETTE[4],
    "TC+Regret":          PALETTE[5],
    "BB+TC+Return":       PALETTE[6],
    "Regret Only":        PALETTE[7],
    "BB+TC+Regret":       PALETTE[0],
}


# ── NAV plot ──────────────────────────────────────────────────────────────────

def plot_nav_comparison(
    nav_series: dict[str, pd.Series],
    title: str = "Cumulative NAV",
    *,
    phase_label: str = "",
    split: str = "test",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot multiple cumulative NAV curves on a single axes.

    Args:
        nav_series: Mapping from method name → daily NAV Series (starting at 1.0).
        title: Figure title.
        phase_label: Phase tag appended to title (e.g. 'Phase 1').
        split: Split name shown in subtitle (e.g. 'test').
        save_path: If provided, save figure to this path and close it.

    Returns:
        The Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, nav in nav_series.items():
        color = METHOD_COLORS.get(name, None)
        linestyle = "--" if "60/40" in name or "Benchmark" in name else "-"
        ax.plot(nav.values, label=name, linewidth=1.8, linestyle=linestyle, color=color)

    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
    suffix = f" — {phase_label} ({split})" if phase_label else f" ({split})"
    ax.set_title(title + suffix, fontsize=14, fontweight="bold")
    ax.set_xlabel("Trading Days", fontsize=12)
    ax.set_ylabel("NAV (starting = 1.0)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ── Allocation trajectory plot ────────────────────────────────────────────────

def plot_allocation_trajectory(
    weights_df: pd.DataFrame,
    title: str = "Portfolio Allocation",
    *,
    phase_label: str = "",
    split: str = "test",
    strat_names: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Stacked-area chart of portfolio allocation over time.

    Args:
        weights_df: DataFrame of shape (T, 3) with columns for each strategy;
            typically with strategy names as column names.
        title: Figure title.
        phase_label: Phase tag (e.g. 'Phase 1').
        split: Split name for subtitle.
        strat_names: Legend labels for each column (defaults to column names).
        save_path: If provided, save figure to this path and close it.

    Returns:
        The Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    cols = list(weights_df.columns)
    labels = strat_names if strat_names else cols
    colors = [STRAT_COLORS.get(c, PALETTE[i]) for i, c in enumerate(cols)]

    ax.stackplot(
        range(len(weights_df)),
        [weights_df[c].values for c in cols],
        labels=labels,
        colors=colors,
        alpha=0.8,
    )
    suffix = f" — {phase_label} ({split})" if phase_label else f" ({split})"
    ax.set_title(title + suffix, fontsize=14, fontweight="bold")
    ax.set_xlabel("Trading Days", fontsize=12)
    ax.set_ylabel("Allocation Weight", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ── Metrics bar chart ─────────────────────────────────────────────────────────

def plot_metrics_bar(
    metrics_table: pd.DataFrame,
    metric: str = "annual_return",
    title: Optional[str] = None,
    *,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Grouped bar chart comparing one metric across methods and phases.

    Args:
        metrics_table: DataFrame with MultiIndex (phase, method) or columns
            representing methods and rows representing phases.  Alternatively
            a flat DataFrame with columns ['phase', 'method', metric].
        metric: Metric column name (e.g. 'annual_return', 'max_drawdown').
        title: Chart title (auto-generated if None).
        save_path: If provided, save figure to this path and close it.

    Returns:
        The Matplotlib Figure object.
    """
    if title is None:
        title = f"{metric.replace('_', ' ').title()} by Phase and Method"

    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_table[metric].unstack().plot(kind="bar", ax=ax, colormap="tab10", width=0.75)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Phase", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.legend(title="Method", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.xticks(rotation=0)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ── Return distribution plot ──────────────────────────────────────────────────

def plot_return_distributions(
    returns_dict: dict[str, pd.Series],
    title: str = "Daily Return Distributions",
    *,
    phase_label: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """KDE plot of daily return distributions across multiple methods.

    Args:
        returns_dict: Mapping from method name → daily returns Series.
        title: Figure title.
        phase_label: Phase tag appended to title.
        save_path: If provided, save figure to this path and close it.

    Returns:
        The Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (name, rets) in enumerate(returns_dict.items()):
        rets = rets.dropna()
        color = METHOD_COLORS.get(name, PALETTE[i % len(PALETTE)])
        linestyle = "--" if "60/40" in name or "Benchmark" in name else "-"
        rets.plot.kde(ax=ax, label=name, linewidth=1.8, color=color, linestyle=linestyle)

    ax.axvline(0, color="grey", linewidth=0.8, linestyle=":")
    suffix = f" — {phase_label}" if phase_label else ""
    ax.set_title(title + suffix, fontsize=14, fontweight="bold")
    ax.set_xlabel("Daily Return", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ── Multi-panel phase comparison ──────────────────────────────────────────────

def plot_phase_comparison(
    phase_navs: dict[int, dict[str, pd.Series]],
    title: str = "Phase-by-Phase NAV Comparison",
    *,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Three-panel NAV plot — one panel per phase.

    Args:
        phase_navs: Mapping from phase (1, 2, 3) → {method_name: nav_series}.
        title: Overall figure title.
        save_path: If provided, save figure to this path and close it.

    Returns:
        The Matplotlib Figure object.
    """
    n_phases = len(phase_navs)
    fig, axes = plt.subplots(1, n_phases, figsize=(5 * n_phases, 5), sharey=False)
    if n_phases == 1:
        axes = [axes]

    for ax, (phase, navs) in zip(axes, sorted(phase_navs.items())):
        for name, nav in navs.items():
            color = METHOD_COLORS.get(name, None)
            linestyle = "--" if "60/40" in name or "Benchmark" in name else "-"
            ax.plot(nav.values, label=name, linewidth=1.8, linestyle=linestyle, color=color)
        ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
        ax.set_title(f"Phase {phase} (test)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Trading Days", fontsize=10)
        ax.set_ylabel("NAV" if phase == sorted(phase_navs)[0] else "", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ── Ablation NAV plot ─────────────────────────────────────────────────────────

def plot_ablation_nav(
    ablation_navs: dict[str, pd.Series],
    title: str = "Ablation Study — Cumulative NAV",
    *,
    phase_label: str = "",
    split: str = "test",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """NAV curves for all ablation arms on a single axes.

    Args:
        ablation_navs: Mapping from arm label → NAV Series.
        title: Figure title.
        phase_label: Phase tag.
        split: Split name.
        save_path: If provided, save figure.

    Returns:
        The Matplotlib Figure object.
    """
    return plot_nav_comparison(
        ablation_navs,
        title=title,
        phase_label=phase_label,
        split=split,
        save_path=save_path,
    )


# ── Sensitivity heatmap ───────────────────────────────────────────────────────

def plot_sensitivity_heatmap(
    results: dict[str, dict],
    metric: str = "annual_return",
    title: Optional[str] = None,
    *,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of a metric across sensitivity axes.

    Args:
        results: Nested dict: {axis_name: {param_value: metric_value}}.
        metric: Metric name (used in title).
        title: Chart title (auto-generated if None).
        save_path: If provided, save figure.

    Returns:
        The Matplotlib Figure object.
    """
    rows, cols, vals = [], [], []
    for axis, params in results.items():
        for pval, mval in params.items():
            rows.append(axis)
            cols.append(str(pval))
            vals.append(mval if np.isfinite(float(mval or 0)) else np.nan)

    if not rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    df = pd.DataFrame({"axis": rows, "param": cols, "value": vals})
    pivot = df.pivot(index="axis", columns="param", values="value")

    if title is None:
        title = f"Sensitivity — {metric.replace('_', ' ').title()}"

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.8), max(4, len(pivot) * 1.2)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": metric.replace("_", " ").title()},
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Parameter Value", fontsize=12)
    ax.set_ylabel("Sensitivity Axis", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
