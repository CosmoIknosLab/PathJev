"""Checkpoint selection rule fixed in D-020: lowest mean validation NLL over Choice decisions.

    python -m pathjev.select_checkpoint <ckpt_dir>=<eval.json> [...] --out selection.json

Each eval.json is a `pathjev.evaluate` report on the full validation mix. The mean is
unweighted over (dataset, qid) entries of type "choice" with a hard-label NLL; the
same entries must be present for every checkpoint, otherwise selection is refused.
Test data are never read here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def choice_nll(report: dict) -> dict[str, float]:
    return {key: row["nll"] for key, row in report.items() if row.get("type") == "choice" and "nll" in row}


def select(reports: dict[str, dict]) -> dict:
    per_ckpt = {ckpt: choice_nll(rep) for ckpt, rep in reports.items()}
    keys = {frozenset(v) for v in per_ckpt.values()}
    if len(keys) != 1 or not next(iter(keys)):
        raise ValueError("checkpoints were not scored on the same non-empty set of Choice decisions")
    scores = {ckpt: sum(v.values()) / len(v) for ckpt, v in per_ckpt.items()}
    best = min(scores, key=lambda c: (scores[c], c))
    return {"rule": "D-020: min mean val NLL over Choice decisions", "best": best, "scores": scores,
            "entries": sorted(next(iter(keys)))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+", help="<checkpoint>=<eval.json>")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    reports = {}
    for pair in args.pairs:
        ckpt, path = pair.split("=", 1)
        reports[ckpt] = json.loads(Path(path).read_text())
    result = select(reports)
    args.out.write_text(json.dumps(result, indent=1))
    print(result["best"])


if __name__ == "__main__":
    main()
