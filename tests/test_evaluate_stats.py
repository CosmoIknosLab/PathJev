"""Evaluator and group-bootstrap sanity checks on toy states."""
from __future__ import annotations

import json

import numpy as np
import pytest

from pathjev import evaluate, metrics, stats


def _toy(tmp_path, n=60, seed=0):
    rng = np.random.default_rng(seed)
    states, decisions = [], []
    for i in range(n):
        gold = int(rng.integers(3))
        target = {c: float(j == gold) for j, c in enumerate("abc")}
        states.append({"state_id": f"s{i}", "meta": {"dataset": "toy", "group": f"P{i // 3}"},
                       "decisions": [{"qid": "q", "type": "choice", "candidates": [{"id": c, "text": c} for c in "abc"],
                                      "target": target, "target_kind": "one_hot"}]})
        p = rng.dirichlet([1, 1, 1]); p[gold] += 1; p /= p.sum()
        decisions.append({"state_id": f"s{i}", "distributions": {"q": dict(zip("abc", p.tolist()))}})
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps(s) for s in states))
    (tmp_path / "d.jsonl").write_text("\n".join(json.dumps(d) for d in decisions))
    return tmp_path / "s.jsonl", tmp_path / "d.jsonl"


def test_evaluator_matches_direct_metrics(tmp_path):
    s, d = _toy(tmp_path)
    table = evaluate.collect([s], [d])
    entry = table[("toy", "q")]
    report = evaluate.score_entry(entry, resamples=200)
    probs, y = np.stack(entry["probs"]), np.stack(entry["target"]).argmax(1)
    assert report["ece"] == pytest.approx(metrics.ece(probs, y))
    assert report["n"] == 60 and report["n_groups"] == 20
    low, high = report["ece_ci95"]
    assert low <= report["ece"] <= high


def test_group_bootstrap_ci_contains_point_and_holm_orders():
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    groups = np.repeat(np.arange(100), 3)
    point, low, high = stats.bootstrap_ci(lambda idx: x[idx].mean(), 300, groups, resamples=500)
    assert low < point < high
    # thresholds 0.05/3, 0.05/2, 0.05/1; step-down stops at the first failure
    assert stats.holm({"a": 0.001, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert stats.holm({"a": 0.001, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}


def test_generative_render_and_sft_row():
    from pathjev import generative
    decision = {"qid": "q", "type": "choice", "instructions": "Which?",
                "candidates": [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}, {"id": "other", "text": "other"}],
                "target": {"a": 0.25, "b": 0.75, "other": 0.0}, "target_kind": "soft"}
    prompt, letters = generative.render(decision)
    assert letters == ["A", "B", "C"] and "(B) beta" in prompt
    row = generative.sft_row({"state_id": "s", "images": ["x.png"]}, decision)
    assert row["answer"] == "beta" and row["soft_target"] and "text of the single best option" in row["prompt"]


def test_sft_scoring_prompt_matches_training_prompt():
    import random
    train_mod = pytest.importorskip("omnimedjev.engine.train")
    from pathjev import generative

    class Identity(random.Random):
        def shuffle(self, x):      # evaluation keeps the state's candidate order
            pass

    decision = {"qid": "q", "type": "choice", "instructions": "Which?",
                "candidates": [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}], "target": {"a": 1.0, "b": 0.0}}
    row = {"images": ["x"], "task_type": "disease_diagnosis_classification", "prompt": "Question: Which?",
           "choices": ["alpha", "beta"], "answer": "alpha"}
    assert train_mod.build_prompt(row, Identity()) == generative.sft_prompt(decision)
