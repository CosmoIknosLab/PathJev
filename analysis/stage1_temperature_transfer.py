"""Post-hoc cross-domain temperature analysis, docs/stage1_analysis_plan.md amendment A2 (D-036).

    python analysis/stage1_temperature_transfer.py --work $PATHJEV_WORK --out results/runs/stage1-temperature [--interim]

Descriptive only (no tests, no effect on the GATE rule). For every arm and seed, on the unseen-domain sets:
  raw       no temperature;
  T-val     §7 temperature (all validation Choice decisions);
  T-in      temperature from the BRACS + SICAPv2 validation Choice decisions only;
  T-cross   temperature fitted on the main question of the other unseen set (leave-one-domain-out);
  T-oracle  temperature fitted on the same set (upper bound; not deployable).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage1_analysis as s1  # noqa: E402

IN_SOURCES = {"bracs", "sicapv2"}
VARIANTS = ("raw", "T-val", "T-in", "T-cross", "T-oracle")


def _by_k(d: dict) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {d["probs"].shape[1]: (np.log(d["probs"] + s1.EPS), d["y"])}


def _metrics(d: dict, t: float | None) -> dict[str, float]:
    probs = s1.scaled(d["probs"], t) if t else d["probs"]
    it = s1.Item(probs, d["y"], d["groups"])
    idx = np.arange(len(d["y"]))
    return {m: fn(it, idx) for m, fn in s1.DESCRIPTIVE.items()} | ({"T": t} if t else {})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--interim", action="store_true")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    test_dir = args.work / "eval_sets" / "stage1-test"
    val_states = args.work / "mixes" / "stage1-p" / "val.jsonl"
    a, b = s1.UNSEEN
    out = {"interim": args.interim, "post_hoc": "amendment A2, D-036", "arms": {}}
    for arm in ["pathjev", "sft", "omnimedjev", "conch", "letter", "probe"]:
        seeds = []
        for tdir, vfile in s1.arm_dirs(args.work, repo, arm, args.interim):
            if not all((tdir / f"{s}_decisions.jsonl").exists() for s in s1.UNSEEN) or not vfile.exists():
                if args.interim:
                    continue
                raise FileNotFoundError(tdir)
            d = {s: s1.load_main(test_dir / f"{s}.jsonl", tdir / f"{s}_decisions.jsonl", s1.MAIN_Q[s])
                 for s in s1.UNSEEN}
            t_val = s1.fit_temperature(s1.load_val_choice(val_states, vfile))
            t_in = s1.fit_temperature(s1.load_val_choice(val_states, vfile, IN_SOURCES))
            t_own = {s: s1.fit_temperature(_by_k(d[s])) for s in s1.UNSEEN}
            temps = {"raw": {s: None for s in s1.UNSEEN}, "T-val": {s: t_val for s in s1.UNSEEN},
                     "T-in": {s: t_in for s in s1.UNSEEN}, "T-cross": {a: t_own[b], b: t_own[a]},
                     "T-oracle": t_own}
            res = {v: {s: _metrics(d[s], temps[v][s]) for s in s1.UNSEEN} for v in VARIANTS}
            for v in VARIANTS:
                res[v]["unseen"] = {m: float(np.mean([res[v][s][m] for s in s1.UNSEEN])) for m in s1.DESCRIPTIVE}
            seeds.append({"test_dir": tdir.name, "results": res})
        if seeds:
            mean = {v: {m: float(np.mean([x["results"][v]["unseen"][m] for x in seeds])) for m in s1.DESCRIPTIVE}
                    for v in VARIANTS}
            out["arms"][arm] = {"seeds": seeds, "seed_mean_unseen": mean}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(out, indent=1))
    lines = ["# Post-hoc cross-domain temperature (A2)" + (" — INTERIM" if args.interim else ""), "",
             "Unseen-domain macro (HepatoBench, HiESD), seed mean. ECE / NLL / Brier.", "",
             "| Arm | seeds | " + " | ".join(VARIANTS) + " |", "|---|---|" + "---|" * len(VARIANTS)]
    for arm, r in out["arms"].items():
        cells = [f"{r['seed_mean_unseen'][v]['ece']:.3f} / {r['seed_mean_unseen'][v]['nll']:.2f} / "
                 f"{r['seed_mean_unseen'][v]['brier']:.3f}" for v in VARIANTS]
        lines.append(f"| {s1.LABEL[arm]} | {len(r['seeds'])} | " + " | ".join(cells) + " |")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
