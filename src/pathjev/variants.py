"""Candidate-set variants of an evaluation set, for baseline (a) diagnostics (stage1_analysis_plan §6.4).

    python -m pathjev.variants --src $PATHJEV_WORK/eval_sets/pilot-stage1

Writes two sibling directories with the same states:
  <src>-no-other : the "other" candidate removed from every Choice decision (decisions whose
                   gold is "other" are dropped; targets renormalised otherwise unchanged);
  <src>-permuted : candidates of every Choice decision in one fixed, seeded permutation per
                   (dataset, qid) — identical for all states, so scoring stays aligned.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

OTHER = "other"


def drop_other(state: dict) -> dict:
    out = dict(state, decisions=[])
    for d in state["decisions"]:
        ids = [c["id"] for c in d["candidates"]]
        if d["type"] != "choice" or OTHER not in ids:
            out["decisions"].append(d)
            continue
        if d["target"].get(OTHER, 0) > 0:
            continue
        cands = [c for c in d["candidates"] if c["id"] != OTHER]
        out["decisions"].append(dict(d, candidates=cands, target={c["id"]: d["target"][c["id"]] for c in cands}))
    return out


def permute(state: dict, seed: str = "pathjev-perm-v1") -> dict:
    out = dict(state, decisions=[])
    for d in state["decisions"]:
        if d["type"] != "choice":
            out["decisions"].append(d)
            continue
        cands = list(d["candidates"])
        rng = random.Random(f"{seed}:{state['meta'].get('dataset')}:{d['qid']}:{len(cands)}")
        while len(cands) > 1 and [c["id"] for c in cands] == [c["id"] for c in d["candidates"]]:
            rng.shuffle(cands)
        out["decisions"].append(dict(d, candidates=cands))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    args = ap.parse_args()
    for suffix, fn in (("no-other", drop_other), ("permuted", permute)):
        dst = args.src.parent / f"{args.src.name}-{suffix}"
        dst.mkdir(parents=True, exist_ok=True)
        for path in sorted(args.src.glob("*.jsonl")):
            rows = [fn(json.loads(l)) for l in path.open() if l.strip()]
            (dst / path.name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(dst)


if __name__ == "__main__":
    main()
