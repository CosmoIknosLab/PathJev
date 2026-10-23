"""Accuracy, calibration and selective-prediction metrics (CLAUDE.md §0.7).

Conventions (fixed before any PathJev result, research_plan.md §1):
  * ECE: top-label, 15 equal-width bins, bin index = min(floor(conf * B), B-1)
    — identical to upstream OmniMed-Jev, so numbers are comparable.
  * Adaptive ECE: top-label, 15 equal-mass bins (sorted confidences split into
    near-equal groups).
  * Brier: multiclass, mean over samples of sum_k (p_k - y_k)^2 ("brier");
    top-label binary Brier is reported separately ("brier_top").
  * AURC: mean selective risk over all coverage levels when answering the most
    confident items first (upstream definition).
Inputs are probability matrices `probs` (n × K) and integer gold indices `y`.
"""
from __future__ import annotations

import numpy as np

DEFAULT_BINS = 15


def _top(probs: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    return probs.max(axis=1), probs.argmax(axis=1) == y


def ece(probs, y, bins: int = DEFAULT_BINS) -> float:
    conf, correct = _top(probs, y)
    slot = np.minimum((conf * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = slot == b
        if mask.any():
            total += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(total)


def adaptive_ece(probs, y, bins: int = DEFAULT_BINS) -> float:
    conf, correct = _top(probs, y)
    order = np.argsort(conf, kind="stable")
    total = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            total += len(chunk) / len(conf) * abs(conf[chunk].mean() - correct[chunk].mean())
    return float(total)


def brier(probs, y) -> float:
    probs = np.asarray(probs, dtype=float)
    onehot = np.eye(probs.shape[1])[np.asarray(y, dtype=int)]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def brier_top(probs, y) -> float:
    conf, correct = _top(probs, y)
    return float(((conf - correct) ** 2).mean())


def nll(probs, y, eps: float = 1e-12) -> float:
    probs = np.asarray(probs, dtype=float)
    return float(-np.log(np.clip(probs[np.arange(len(probs)), np.asarray(y, dtype=int)], eps, 1.0)).mean())


def aurc(probs, y) -> float:
    conf, correct = _top(probs, y)
    order = np.argsort(-conf, kind="stable")
    errors = np.cumsum(~correct[order])
    return float((errors / np.arange(1, len(order) + 1)).mean())


def reliability_curve(probs, y, bins: int = DEFAULT_BINS) -> dict[str, list[float]]:
    conf, correct = _top(probs, y)
    slot = np.minimum((conf * bins).astype(int), bins - 1)
    out = {"bin_lower": [], "count": [], "confidence": [], "accuracy": []}
    for b in range(bins):
        mask = slot == b
        out["bin_lower"].append(b / bins)
        out["count"].append(int(mask.sum()))
        out["confidence"].append(float(conf[mask].mean()) if mask.any() else float("nan"))
        out["accuracy"].append(float(correct[mask].mean()) if mask.any() else float("nan"))
    return out


def balanced_accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean([(y_pred[y_true == c] == c).mean() for c in np.unique(y_true)]))


def macro_f1(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for c in np.union1d(y_true, y_pred):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def summary(probs, y) -> dict[str, float]:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    pred = probs.argmax(axis=1)
    return {"n": int(len(y)), "accuracy": float((pred == y).mean()),
            "balanced_accuracy": balanced_accuracy(y, pred), "macro_f1": macro_f1(y, pred),
            "ece": ece(probs, y), "adaptive_ece": adaptive_ece(probs, y), "brier": brier(probs, y),
            "brier_top": brier_top(probs, y), "nll": nll(probs, y), "aurc": aurc(probs, y),
            "mean_confidence": float(probs.max(axis=1).mean())}
