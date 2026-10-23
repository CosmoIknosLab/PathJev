"""Phase 0: score a released-checkpoint inference run against OmniMed-Jev's
reported classification numbers.

Balanced accuracy uses upstream's own evaluator helpers on the folded
predictions; ECE comes from upstream's eval_medjev_metrics.py output. The
verdict uses the tolerances fixed in configs/phase0_repro.yaml.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from omnimedjev.engine.evaluate import (accuracy_score, balanced_accuracy_score, ground_truth_value,
                                        normalize_text, prediction_value)

TASK = "disease_diagnosis_classification"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--upstream", required=True, type=Path)
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    gold_rows = [r for r in read_jsonl(args.upstream / cfg["ground_truth"])
                 if normalize_text(r.get("task_type")) == normalize_text(TASK)]
    predictions = {r["uid"]: r for r in read_jsonl(args.run_dir / "testing_predictions.jsonl")}
    missing = [r["uid"] for r in gold_rows if r["uid"] not in predictions]

    gold = [normalize_text(ground_truth_value(r)) for r in gold_rows]
    pred = [normalize_text(prediction_value(predictions.get(r["uid"]))) for r in gold_rows]
    metrics = json.loads((args.run_dir / "metrics.json").read_text())
    ece = metrics["families"][TASK]["ece"]

    ref, tol = cfg["reference"], cfg["tolerance"]
    bal = balanced_accuracy_score(gold, pred)
    result = {
        "n_rows": len(gold_rows),
        "n_missing_predictions": len(missing),
        "n_classes": len(set(gold)),
        "hits": sum(g == p for g, p in zip(gold, pred)),
        "balanced_accuracy": bal,
        "accuracy": accuracy_score(gold, pred),
        "ece": ece,
        "reference": ref,
        "delta_balanced_accuracy": bal - ref["classification_balanced_accuracy"],
        "delta_ece": ece - ref["classification_ece"],
    }
    result["within_tolerance"] = (
        not missing and len(gold_rows) == ref["classification_n"]
        and abs(result["delta_balanced_accuracy"]) <= tol["balanced_accuracy_abs"]
        and abs(result["delta_ece"]) <= tol["ece_abs"])
    out = args.run_dir / "repro_verdict.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
