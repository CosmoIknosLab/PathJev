"""Linear-probe baseline on synthetic features: outputs valid distributions, learns separable data."""
from __future__ import annotations

import json

import numpy as np

from pathjev import baselines, evaluate


def _write(tmp_path, name, n, seed, centers):
    rng = np.random.default_rng(seed)
    states, ids, feats = [], [], []
    for i in range(n):
        gold = i % 3
        sid = f"{name}{i}"
        target = {c: float(j == gold) for j, c in enumerate("abc")}
        states.append({"state_id": sid, "images": ["x"], "meta": {"dataset": "toy", "group": f"G{i // 2}"},
                       "decisions": [{"qid": "tissue", "type": "choice", "candidates": [{"id": c, "text": c} for c in "abc"],
                                      "target": target, "target_kind": "one_hot"}]})
        ids.append(sid)
        feats.append(centers[gold] + rng.normal(scale=0.5, size=8))
    (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(s) for s in states))
    np.savez(tmp_path / f"{name}.npz", ids=np.array(ids), features=np.stack(feats))
    return tmp_path / f"{name}.jsonl", tmp_path / f"{name}.npz"


def test_probe_train_test_and_crossfit(tmp_path):
    centers = np.random.default_rng(0).normal(scale=3, size=(3, 8))
    tr_s, tr_f = _write(tmp_path, "tr", 120, 1, centers)
    te_s, te_f = _write(tmp_path, "te", 60, 2, centers)
    for train in (True, False):
        out = tmp_path / f"dec_{train}.jsonl"
        baselines.probe(tr_s if train else None, te_s, tr_f if train else None, te_f, out)
        report = evaluate.score_entry(evaluate.collect([te_s], [out])[("toy", "tissue")], resamples=0)
        assert report["n"] == 60 and report["accuracy"] > 0.9
