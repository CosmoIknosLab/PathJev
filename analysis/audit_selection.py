"""Audit (second review, point 6): checkpoint/config selection with and without the organ question.

    python analysis/audit_selection.py --work $PATHJEV_WORK

Reads every eval_val-s1-<arm>-*_checkpoint-* /val_eval.json (pathjev.evaluate reports on the validation mix) and
recomputes the D-020 criterion (a) as used: mean NLL over (dataset, qid) Choice entries incl. organ, and (b) the
same over the main Choice entries only (organ excluded). Within each run the best checkpoint; across the seed-1 grid
the best run. Prints JSON.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, type=Path)
    args = ap.parse_args()
    runs = defaultdict(dict)
    for d in sorted((args.work / "runs").glob("eval_val-s1-*_checkpoint-*")):
        m = re.match(r"eval_val-(s1-(jev|sft)-lr\S+-e\d-s\d)_checkpoint-(\d+)$", d.name)
        if not m or not (d / "val_eval.json").exists():
            continue
        rep = json.loads((d / "val_eval.json").read_text())
        ch = {k: v["nll"] for k, v in rep.items() if v.get("type") == "choice" and "nll" in v}
        main_ = {k: v for k, v in ch.items() if not k.endswith("/organ")}
        runs[m.group(1)][int(m.group(3))] = {"a": sum(ch.values()) / len(ch), "b": sum(main_.values()) / len(main_),
                                             "n_a": len(ch), "n_b": len(main_)}
    out = {}
    for run, ck in runs.items():
        best_a = min(ck, key=lambda c: (ck[c]["a"], c))
        best_b = min(ck, key=lambda c: (ck[c]["b"], c))
        out[run] = {"best_a": best_a, "best_b": best_b, "same": best_a == best_b,
                    "scores": {c: {k: round(v, 4) for k, v in s.items()} for c, s in sorted(ck.items())}}
    for arm in ("jev", "sft"):
        grid = {r: v for r, v in out.items() if f"-{arm}-" in r and r.endswith("-s1")}
        for crit in ("a", "b"):
            best = min(grid, key=lambda r: grid[r]["scores"][grid[r][f"best_{crit}"]][crit])
            out[f"grid_{arm}_{crit}"] = {"run": best, "checkpoint": grid[best][f"best_{crit}"]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
