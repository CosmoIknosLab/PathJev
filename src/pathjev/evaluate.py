"""Score a model's decision distributions against PathJev states.

    python -m pathjev.evaluate --states <split>_patch.jsonl [...] --decisions <run>/<name>_decisions.jsonl \
        --out <run>/eval.json

`--decisions` is the output of upstream `scripts/infer_medjev.py` (one row per
state: {"state_id", "distributions": {qid: {candidate_id: prob}}}) or any
baseline written in the same format. For every (dataset, qid) the evaluator
reports:
  choice / noul:  metrics.summary (accuracy, balanced accuracy, macro-F1, ECE,
                  adaptive ECE, Brier, NLL, AURC) on one-hot targets, plus a
                  group-bootstrap CI for ECE and balanced accuracy;
  score:          expected-level MAE and ranked probability score (RPS).
Soft-target decisions (mixed mosaics) are scored for the homogeneity Noul only;
their composition targets are reported as total-variation distance.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pathjev import metrics, stats
from pathjev.groups import state_group


def _read(paths):
    for path in paths:
        for line in Path(path).open(encoding="utf-8"):
            if line.strip():
                yield json.loads(line)


def collect(state_paths, decision_paths) -> dict[tuple[str, str], dict]:
    """Group aligned (probs, target, group) per (dataset, qid)."""
    preds = {row["state_id"]: row["distributions"] for row in _read(decision_paths)}
    table: dict[tuple[str, str], dict] = defaultdict(lambda: {"probs": [], "target": [], "groups": [],
                                                              "type": None, "kinds": set(), "ids": None,
                                                              "missing": 0})
    for state in _read(state_paths):
        dist = preds.get(state["state_id"])
        meta = state.get("meta", {})
        for d in state["decisions"]:
            key = (meta.get("dataset", "?"), d["qid"])
            entry = table[key]
            ids = [c["id"] for c in d["candidates"]]
            if dist is None or d["qid"] not in dist:
                entry["missing"] += 1
                continue
            entry["type"] = d["type"]
            entry["kinds"].add(d.get("target_kind", "one_hot"))
            entry["ids"] = entry["ids"] or ids
            if ids != entry["ids"]:
                raise ValueError(f"{key}: candidate order differs between states; align before scoring")
            p = np.array([dist[d["qid"]].get(i, 0.0) for i in ids], dtype=float)
            entry["probs"].append(p / p.sum() if p.sum() > 0 else p)
            entry["target"].append(np.array([d["target"][i] for i in ids], dtype=float))
            entry["groups"].append(state_group(state))
    return table


def score_entry(entry: dict, resamples: int) -> dict:
    probs, target = np.stack(entry["probs"]), np.stack(entry["target"])
    groups = np.array(entry["groups"])
    out = {"type": entry["type"], "n": int(len(probs)), "missing": entry["missing"],
           "n_groups": int(len(set(entry["groups"]))), "candidates": entry["ids"]}
    soft = ~np.isclose(target.max(axis=1), 1.0)
    if soft.any():
        out["n_soft"] = int(soft.sum())
        out["soft_total_variation"] = float(0.5 * np.abs(probs[soft] - target[soft]).sum(axis=1).mean())
    hard = ~soft
    if not hard.any():
        return out
    p, y, g = probs[hard], target[hard].argmax(axis=1), groups[hard]
    if entry["type"] == "score":
        levels = np.arange(p.shape[1])
        expected = (p * levels).sum(axis=1)
        cdf_p, cdf_t = np.cumsum(p, axis=1), np.cumsum(np.eye(p.shape[1])[y], axis=1)
        out.update({"mae_expected_level": float(np.abs(expected - y).mean()),
                    "rps": float(((cdf_p - cdf_t) ** 2).sum(axis=1).mean() / (p.shape[1] - 1)),
                    **{f"choice_{k}": v for k, v in metrics.summary(p, y).items()}})
        return out
    out.update(metrics.summary(p, y))
    if resamples:
        out["ece_ci95"] = list(stats.bootstrap_ci(lambda idx: metrics.ece(p[idx], y[idx]), len(y), g,
                                                  resamples=resamples)[1:])
        out["balanced_accuracy_ci95"] = list(stats.bootstrap_ci(
            lambda idx: metrics.balanced_accuracy(y[idx], p[idx].argmax(axis=1)), len(y), g,
            resamples=resamples)[1:])
    out["reliability"] = metrics.reliability_curve(p, y)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--decisions", nargs="+", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--resamples", type=int, default=2000)
    args = ap.parse_args()
    table = collect(args.states, args.decisions)
    report = {f"{dataset}/{qid}": score_entry(entry, args.resamples)
              for (dataset, qid), entry in sorted(table.items()) if entry["probs"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    for key, row in report.items():
        print(key, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
