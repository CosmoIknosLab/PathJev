"""Stage 1 configuration selection across the hyper-parameter grid (D-029), validation only.

    python analysis/select_config.py --work $PATHJEV_WORK --arm jev|sft --runs <run name> ... --out <json>

For each grid run, its checkpoint was already selected by the D-020 rule (min mean val
Choice NLL; `selection/selection.json`). The configuration chosen for an arm is the run
whose selected checkpoint has the lowest such score (ties → lexicographically first run).
Reads only run folders; never test outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--arm", required=True, choices=("jev", "sft"))
    ap.add_argument("--runs", required=True, nargs="+")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    rows = {}
    for run in args.runs:
        sel = json.loads((args.work / "runs" / f"train_{run}" / "selection" / "selection.json").read_text())
        rows[run] = {"checkpoint": sel["best"], "val_choice_nll": sel["scores"][sel["best"]],
                     "entries": sel["entries"]}
    entries = {tuple(r["entries"]) for r in rows.values()}
    if len(entries) != 1:
        raise ValueError("grid runs were not scored on the same validation decisions")
    best = min(rows, key=lambda r: (rows[r]["val_choice_nll"], r))
    out = {"rule": "D-029: min val Choice NLL of the D-020-selected checkpoint across the grid",
           "arm": args.arm, "best_run": best, "runs": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(best)


if __name__ == "__main__":
    main()
