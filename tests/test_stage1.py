"""Stage 1 pipeline pieces: checkpoint selection rule (D-020), shared eval subsets, probe fallback."""
from __future__ import annotations

import json

import numpy as np
import pytest

from pathjev import baselines, evalsets, evaluate, select_checkpoint
from tests.test_baselines import _write


def _report(nll_a, nll_b, extra=None):
    rep = {"ds/tissue": {"type": "choice", "nll": nll_a}, "ds/grade": {"type": "choice", "nll": nll_b},
           "ds/count": {"type": "score", "choice_nll": 0.1}}
    rep.update(extra or {})
    return rep


def test_selection_uses_mean_choice_nll_only():
    out = select_checkpoint.select({"ckpt-100": _report(1.0, 1.0), "ckpt-200": _report(0.4, 1.4),
                                    "ckpt-300": _report(0.9, 0.95)})
    assert out["best"] == "ckpt-200"          # means 1.0, 0.9, 0.925; score decisions ignored
    assert out["scores"]["ckpt-200"] == pytest.approx(0.9)


def test_selection_refuses_mismatched_entries():
    with pytest.raises(ValueError):
        select_checkpoint.select({"a": _report(1, 1), "b": _report(1, 1, {"x/new": {"type": "choice", "nll": 0}})})


def test_evalsets_are_deterministic_and_stratified(tmp_path, monkeypatch):
    monkeypatch.setenv("PATHJEV_WORK", str(tmp_path))
    rows = [{"state_id": f"s{i}", "decisions": [{"target": {"a": float(i % 10 == 0), "b": float(i % 10 != 0)}}]}
            for i in range(500)]
    (tmp_path / "t.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg = tmp_path / "c.yaml"
    cfg.write_text("name: e\nseed: 1\nsets:\n  - {name: t, states: t.jsonl, cap: 60}\n")
    first = evalsets.build(cfg)
    second = evalsets.build(cfg)
    assert first["sets"]["t"]["sha256"] == second["sets"]["t"]["sha256"]
    assert first["sets"]["t"]["classes"] == {"a": 30, "b": 30}   # rare class kept by round-robin


def test_probe_falls_back_to_crossfit_for_questions_absent_from_training(tmp_path):
    centers = np.random.default_rng(0).normal(scale=3, size=(3, 8))
    tr_s, tr_f = _write(tmp_path, "tr", 120, 1, centers)
    te_s, te_f = _write(tmp_path, "te", 60, 2, centers)
    rows = [json.loads(l) for l in tr_s.read_text().splitlines()]
    for r in rows:
        r["decisions"][0]["qid"] = "other_question"
    tr_s.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "dec.jsonl"
    baselines.probe(tr_s, te_s, tr_f, te_f, out)
    report = evaluate.score_entry(evaluate.collect([te_s], [out])[("toy", "tissue")], resamples=0)
    assert report["n"] == 60 and report["accuracy"] > 0.9


def _state(gold, dataset="toy"):
    ids = ["a", "b", "other"]
    return {"state_id": f"s{gold}", "meta": {"dataset": dataset},
            "decisions": [{"qid": "tissue", "type": "choice", "candidates": [{"id": i, "text": i} for i in ids],
                           "target": {i: float(i == gold) for i in ids}},
                          {"qid": "present:x", "type": "noul", "candidates": [{"id": "no"}, {"id": "yes"}],
                           "target": {"no": 1.0, "yes": 0.0}}]}


def test_variant_drop_other_and_fixed_permutation():
    from pathjev import variants
    s = variants.drop_other(_state("a"))
    assert [c["id"] for c in s["decisions"][0]["candidates"]] == ["a", "b"]
    assert s["decisions"][0]["target"] == {"a": 1.0, "b": 0.0}
    assert len(variants.drop_other(_state("other"))["decisions"]) == 1        # gold "other": decision dropped
    p1, p2 = variants.permute(_state("a")), variants.permute(_state("b"))
    order = [c["id"] for c in p1["decisions"][0]["candidates"]]
    assert order != ["a", "b", "other"] and sorted(order) == ["a", "b", "other"]
    assert order == [c["id"] for c in p2["decisions"][0]["candidates"]]       # same order for every state
    assert p1["decisions"][1]["candidates"] == _state("a")["decisions"][1]["candidates"]


def test_evalsets_drop_unreadable(tmp_path, monkeypatch):
    monkeypatch.setenv("PATHJEV_WORK", str(tmp_path))
    good, bad = tmp_path / "g.png", tmp_path / "b.png"
    good.write_bytes(b"x"); bad.write_bytes(b"")
    rows = [{"state_id": "ok", "images": [str(good)], "decisions": [{"target": {"a": 1.0}}]},
            {"state_id": "empty", "images": [str(bad)], "decisions": [{"target": {"a": 1.0}}]}]
    (tmp_path / "t.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg = tmp_path / "c.yaml"
    cfg.write_text("name: e\nseed: 1\ndrop_unreadable: true\nsets:\n  - {name: t, states: t.jsonl, cap: 0}\n")
    info = evalsets.build(cfg)["sets"]["t"]
    assert info["used"] == 1 and info["dropped_unreadable"] == ["empty"]


def test_probe_keys_training_labels_by_candidate_list(tmp_path):
    """Same qid with different candidate lists in training (SPIDER `tissue`, D-032): each test set
    uses only the training rows with its own list, never mixed label indices."""
    centers = np.random.default_rng(0).normal(scale=3, size=(3, 8))
    tr_s, tr_f = _write(tmp_path, "tr", 120, 1, centers)
    te_s, te_f = _write(tmp_path, "te", 60, 2, centers)
    rows = [json.loads(l) for l in tr_s.read_text().splitlines()]
    feats = dict(np.load(tr_f))
    extra = []
    for i, r in enumerate(rows[:40]):    # a second source: same qid, 5 candidates, gold at index 3-4
        r2 = json.loads(json.dumps(r)); r2["state_id"] = f"x{i}"
        ids = [f"z{k}" for k in range(5)]
        r2["decisions"][0]["candidates"] = [{"id": k, "text": k} for k in ids]
        r2["decisions"][0]["target"] = {k: float(k == ids[3 + i % 2]) for k in ids}
        extra.append(r2)
    tr_s.write_text("\n".join(json.dumps(r) for r in rows + extra))     # old code: IndexError (spider-thorax)
    np.savez(tr_f, ids=np.concatenate([feats["ids"], [f"x{i}" for i in range(40)]]),
             features=np.concatenate([feats["features"], feats["features"][:40]]))
    out = tmp_path / "dec.jsonl"
    baselines.probe(tr_s, te_s, tr_f, te_f, out)
    report = evaluate.score_entry(evaluate.collect([te_s], [out])[("toy", "tissue")], resamples=0)
    assert report["n"] == 60 and report["accuracy"] > 0.9


def test_stage1_analysis_fast_metrics_match_reference():
    import importlib.util
    from pathlib import Path
    from pathjev import metrics
    spec = importlib.util.spec_from_file_location("s1a", Path(__file__).parents[1] / "analysis" / "stage1_analysis.py")
    s1a = importlib.util.module_from_spec(spec); spec.loader.exec_module(s1a)
    rng = np.random.default_rng(3)
    probs = rng.dirichlet(np.ones(5) * 0.4, size=400)
    y = rng.integers(0, 5, 400)
    item = s1a.Item(probs, y, np.array([str(i // 4) for i in range(400)]))
    idx = rng.integers(0, 400, 400)                       # a resample with repeats
    assert item.ece(idx) == pytest.approx(metrics.ece(probs[idx], y[idx]))
    assert item.bacc(idx) == pytest.approx(metrics.balanced_accuracy(y[idx], probs[idx].argmax(1)))
    yc = np.array([rng.choice(5, p=row) for row in probs])          # labels drawn from the probs: calibrated
    assert s1a.fit_temperature({5: (np.log(probs + 1e-12), yc)}) == pytest.approx(1.0, abs=0.15)
    sharp = probs ** 3 / (probs ** 3).sum(1, keepdims=True)          # over-confident version → T ≈ 3
    assert s1a.fit_temperature({5: (np.log(sharp + 1e-12), yc)}) == pytest.approx(3.0, rel=0.15)
