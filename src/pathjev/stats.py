"""Cluster (group) bootstrap for metrics and paired differences (research_plan §5).

Resampling unit = group (patient, else slide, else sample), so CIs respect the
dependence between patches of the same patient. Paired comparisons resample
the same groups for both models, which requires both to score the same items.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

DEFAULT_RESAMPLES = 10_000
SEED = 20261016


def _group_index(groups: np.ndarray) -> list[np.ndarray]:
    _, inverse = np.unique(groups, return_inverse=True)
    return [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]


def bootstrap_ci(metric: Callable[[np.ndarray], float], n: int, groups, *, resamples: int = DEFAULT_RESAMPLES,
                 alpha: float = 0.05, seed: int = SEED) -> tuple[float, float, float]:
    """(point, low, high) for `metric(indices)` under a group bootstrap over `n` items."""
    members = _group_index(np.asarray(groups))
    rng = np.random.default_rng(seed)
    point = metric(np.arange(n))
    stats = np.empty(resamples)
    for b in range(resamples):
        chosen = rng.integers(0, len(members), len(members))
        stats[b] = metric(np.concatenate([members[g] for g in chosen]))
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(point), float(low), float(high)


def paired_difference_ci(metric_a: Callable[[np.ndarray], float], metric_b: Callable[[np.ndarray], float],
                         n: int, groups, **kwargs) -> tuple[float, float, float]:
    """CI of metric_a - metric_b with the same resampled groups for both."""
    return bootstrap_ci(lambda idx: metric_a(idx) - metric_b(idx), n, groups, **kwargs)


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm step-down: {name: rejected}."""
    order = sorted(p_values, key=p_values.get)
    rejected, m = {}, len(order)
    still = True
    for rank, name in enumerate(order):
        still = still and p_values[name] <= alpha / (m - rank)
        rejected[name] = still
    return rejected


def bootstrap_p_value(diff_metric: Callable[[np.ndarray], float], n: int, groups, *, direction: str = "less",
                      resamples: int = DEFAULT_RESAMPLES, seed: int = SEED) -> float:
    """One-sided bootstrap p-value for H1: diff < 0 ("less") or diff > 0 ("greater")."""
    members = _group_index(np.asarray(groups))
    rng = np.random.default_rng(seed)
    stats = np.empty(resamples)
    for b in range(resamples):
        chosen = rng.integers(0, len(members), len(members))
        stats[b] = diff_metric(np.concatenate([members[g] for g in chosen]))
    tail = np.mean(stats >= 0) if direction == "less" else np.mean(stats <= 0)
    return float(min(1.0, (tail * resamples + 1) / (resamples + 1)))
