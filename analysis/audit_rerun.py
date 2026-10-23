"""Audit (second review, points 5 and 8.2): A3.4 Gen-SFT rerun (L40) vs the stored primary test runs (H100).

    python analysis/audit_rerun.py --work $PATHJEV_WORK

Per seed and clean set, main Choice: |Δ| of bAcc/ECE/NLL, max |Δp|, argmax agreement between the stored
eval_test-s1-sft-...-s<k> decisions and the eval_a3-sftvar-s<k> rerun; seed-mean macro differences; and from the
rerun's variants files the distribution of log p(<end_of_turn>) and max |p_primary - p_eos|. Prints JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stage1_analysis import IN_DOMAIN, MAIN_Q, UNSEEN, Item, arm_dirs, load_main, macro_value

CLEAN = UNSEEN + IN_DOMAIN


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, type=Path)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    test = args.work / "eval_sets" / "stage1-test"
    stored = [t for t, _ in arm_dirs(args.work, repo, "sft", interim=False)]
    out, items = {"per_seed": {}}, {"stored": [], "rerun": []}
    eos_vals, d_eos = [], 0.0
    for k, sdir in enumerate(stored, start=1):
        rdir = args.work / "runs" / f"eval_a3-sftvar-s{k}"
        row, it_s, it_r = {}, {}, {}
        for s in CLEAN:
            a = load_main(test / f"{s}.jsonl", sdir / f"{s}_decisions.jsonl", MAIN_Q[s])
            b = load_main(test / f"{s}.jsonl", rdir / f"{s}_decisions.jsonl", MAIN_Q[s])
            assert (a["y"] == b["y"]).all() and a["missing"] == b["missing"] == 0
            ia, ib = Item(a["probs"], a["y"], a["groups"]), Item(b["probs"], b["y"], b["groups"])
            it_s[s], it_r[s] = ia, ib
            idx = np.arange(len(a["y"]))
            row[s] = {"n": len(idx), "max_abs_dp": float(np.abs(a["probs"] - b["probs"]).max()),
                      "argmax_agree": float((a["probs"].argmax(1) == b["probs"].argmax(1)).mean()),
                      **{f"d_{m}": float(getattr(ib, m)(idx) - getattr(ia, m)(idx)) for m in ("bacc", "ece", "nll")}}
            for line in (rdir / f"{s}_variants.jsonl").open():
                r = json.loads(line)
                v = r["variants"][MAIN_Q[s]]
                eos_vals += [c["eos"] for c in v["raw"].values()]
        out["per_seed"][f"s{k}"] = {"stored_dir": sdir.name, "rerun_dir": rdir.name, "sets": row}
        items["stored"].append(it_s)
        items["rerun"].append(it_r)
    for grp, sets in (("clean", CLEAN), ("unseen", UNSEEN), ("in_domain", IN_DOMAIN)):
        out[grp] = {m: {"stored": macro_value(items["stored"], sets, m), "rerun": macro_value(items["rerun"], sets, m)}
                    for m in ("balanced_accuracy", "ece", "nll")}
    # EOS: compare primary probabilities (decisions file) with the eos variant, per state
    for k in range(1, len(stored) + 1):
        rdir = args.work / "runs" / f"eval_a3-sftvar-s{k}"
        for s in CLEAN:
            prim = {json.loads(l)["state_id"]: json.loads(l)["distributions"][MAIN_Q[s]]
                    for l in (rdir / f"{s}_decisions.jsonl").open() if l.strip()}
            for line in (rdir / f"{s}_variants.jsonl").open():
                r = json.loads(line)
                v = r["variants"][MAIN_Q[s]]["eos"]
                d_eos = max(d_eos, max(abs(v[c] - prim[r["state_id"]][c]) for c in v))
    e = np.array(eos_vals, dtype=float)
    out["eos_logp"] = {"n": int(e.size), "max_abs": float(np.abs(e).max()), "mean": float(e.mean()),
                       "n_nonzero": int((e != 0).sum())}
    out["max_abs_p_primary_minus_eos"] = d_eos
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
