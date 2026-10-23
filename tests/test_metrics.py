"""Calibration metrics vs reference implementations on synthetic data (Phase 2 test)."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, log_loss

from pathjev import metrics


def _synthetic(n=2000, k=5, temperature=1.0, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, k)) * 2
    y = np.array([rng.choice(k, p=np.exp(l) / np.exp(l).sum()) for l in logits])
    probs = np.exp(logits / temperature)
    return probs / probs.sum(axis=1, keepdims=True), y


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_ece_and_aurc_match_upstream(temperature):
    mm = pytest.importorskip("omnimedjev.engine.medjev_metrics")
    probs, y = _synthetic(temperature=temperature)
    conf = probs.max(axis=1).tolist()
    correct = (probs.argmax(axis=1) == y).tolist()
    assert metrics.ece(probs, y) == pytest.approx(mm.expected_calibration_error(conf, correct), abs=1e-12)
    assert metrics.aurc(probs, y) == pytest.approx(mm.risk_coverage(conf, correct)["aurc"], abs=1e-12)


def test_calibrated_model_has_small_ece_and_sharpened_model_large():
    probs, y = _synthetic(n=20000, temperature=1.0)
    sharp, _ = _synthetic(n=20000, temperature=0.4)
    assert metrics.ece(probs, y) < 0.02
    assert metrics.ece(sharp, y) > 0.1


def test_adaptive_ece_matches_bruteforce():
    probs, y = _synthetic(n=999)
    conf, correct = probs.max(axis=1), probs.argmax(axis=1) == y
    order = np.argsort(conf, kind="stable")
    ref = sum(len(c) / len(conf) * abs(conf[c].mean() - correct[c].mean())
              for c in np.array_split(order, 15))
    assert metrics.adaptive_ece(probs, y) == pytest.approx(ref)


def test_brier_nll_accuracy_f1_match_sklearn():
    probs, y = _synthetic()
    pred = probs.argmax(axis=1)
    onehot = np.eye(probs.shape[1])[y]
    assert metrics.brier(probs, y) == pytest.approx(((probs - onehot) ** 2).sum(1).mean())
    assert metrics.brier_top(probs, y) == pytest.approx(brier_score_loss(pred == y, probs.max(1)))
    assert metrics.nll(probs, y) == pytest.approx(log_loss(y, probs, labels=list(range(probs.shape[1]))))
    assert metrics.balanced_accuracy(y, pred) == pytest.approx(balanced_accuracy_score(y, pred))
    assert metrics.macro_f1(y, pred) == pytest.approx(f1_score(y, pred, average="macro"))


def test_perfect_ranking_gives_minimal_aurc():
    y = np.zeros(10, dtype=int)
    good = np.array([[0.9, 0.1]] * 5 + [[0.4, 0.6]] * 5)    # correct answers are the confident ones
    bad = np.array([[0.55, 0.45]] * 5 + [[0.05, 0.95]] * 5)  # wrong answers are the confident ones
    assert metrics.aurc(good, y) < metrics.aurc(bad, y)
    assert metrics.ece(good, y) < metrics.ece(bad, y)
