"""Secondary metrics of the main Choice question (plan §3; paper/generated/tab_secondary.tex), from run folders only.

    python analysis/secondary_metrics.py --work $PATHJEV_WORK

Recomputes pathjev.metrics.summary from the decision files with the analysis loader (same runs, states, groups
and exclusions as analysis/stage1_analysis.py), and reports accuracy, macro-F1, adaptive ECE,
Brier score, NLL and AURC per clean test set, macro-averaged over the clean sets and over the seen-source (SPIDER)
sets; trained arms: seed mean ± SD (SD of the per-seed values). Raw probabilities, no temperature, no CIs.
As a consistency check, balanced accuracy and ECE read here must equal the final analysis summary
(results/runs/stage1-analysis/summary.json) per set, seed mean, within 1e-6.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stage1_analysis import CLEAN, MAIN_Q, SEEN, arm_dirs, load_main

ARMS = [("pathjev", "PathJev-P"), ("sft", "Gen-SFT"), ("omnimedjev", "OmniMed-Jev"), ("conch", "CONCH"),
        ("letter", "Inner-Jev"), ("probe", "Probe$^\\dagger$")]
HIGHER = {"accuracy", "macro_f1"}
METRICS = [("accuracy", "Acc$\\uparrow$"), ("macro_f1", "Macro-F1$\\uparrow$"), ("adaptive_ece", "aECE$\\downarrow$"),
           ("brier", "Brier$\\downarrow$"), ("nll", "NLL$\\downarrow$"), ("aurc", "AURC$\\downarrow$")]
GROUPS = [("hepatobench", "HepatoBench", ["hepatobench"]), ("hiesd", "HiESD", ["hiesd"]), ("bracs", "BRACS", ["bracs"]),
          ("sicapv2", "SICAPv2", ["sicapv2"]), ("clean", "Clean average", CLEAN), ("seen", "Seen source", SEEN)]


def per_seed(work: Path, repo: Path, arm: str) -> list[dict[str, dict]]:
    """[{set: main-question metrics}] per seed, recomputed from the decision files with the analysis loader
    (same states, groups and exclusions as stage1_analysis; pathjev.metrics.summary as in the *_eval.json files)."""
    from pathjev.metrics import summary
    test_dir = work / "eval_sets" / "stage1-test"
    out = []
    for tdir, _ in arm_dirs(work, repo, arm, interim=False):
        row = {}
        for s, q in MAIN_Q.items():
            d = load_main(test_dir / f"{s}.jsonl", tdir / f"{s}_decisions.jsonl", q)
            row[s] = summary(d["probs"], d["y"])
        out.append(row)
    return out


def check(runs: dict[str, list[dict]], summary: Path) -> int:
    table = json.loads(summary.read_text())["table"]
    n = 0
    for arm, seeds in runs.items():
        for s in MAIN_Q:
            for m in ("balanced_accuracy", "ece"):
                mine, ref = float(np.mean([x[s][m] for x in seeds])), table[arm][s][m]
                if abs(mine - ref) > 1e-6:
                    raise ValueError(f"{arm} {s} {m}: recomputed {mine:.6f} vs summary {ref:.6f}")
                n += 1
    return n


def cell(seeds: list[dict], sets: list[str], m: str) -> tuple[float, str]:
    vals = [float(np.mean([x[s][m] for s in sets])) for x in seeds]
    if len(vals) < 2:
        return vals[0], f"{vals[0]:.3f}"
    return float(np.mean(vals)), f"{np.mean(vals):.3f}$_{{\\pm{np.std(vals, ddof=1):.3f}}}$".replace("\\pm0.", "\\pm.")


def table(runs: dict[str, list[dict]]) -> str:
    """Rows: set (or set group) x method; best language-model-based value per block and column in bold."""
    lines = [r"\begin{tabular}{@{}ll" + "c" * len(METRICS) + "@{}}", r"\toprule",
             "Set & Method & " + " & ".join(n for _, n in METRICS) + r" \\", r"\midrule"]
    for gi, (_, gname, sets) in enumerate(GROUPS):
        cells = {arm: {m: cell(runs[arm], sets, m) for m, _ in METRICS} for arm, _ in ARMS}
        for m, _ in METRICS:
            lm = [a for a, _ in ARMS if a != "probe"]
            pick = max if m in HIGHER else min
            best = cells[pick(lm, key=lambda a: cells[a][m][0])][m][1].split("$")[0]
            for a in lm:
                v, txt = cells[a][m]
                if txt.split("$")[0] == best:
                    cells[a][m] = (v, f"\\textbf{{{txt}}}")
        for i, (arm, aname) in enumerate(ARMS):
            if arm == "probe":
                lines.append(r"\cmidrule(l){2-" + str(2 + len(METRICS)) + "}")
            lines.append((gname if i == 0 else "") + f" & {aname} & "
                         + " & ".join(cells[arm][m][1] for m, _ in METRICS) + r" \\")
        lines.append(r"\midrule" if gi < len(GROUPS) - 1 else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, type=Path)
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--summary", type=Path, default=repo / "results" / "runs" / "stage1-analysis" / "summary.json")
    ap.add_argument("--out", type=Path, default=repo / "paper" / "generated" / "tab_secondary.tex")
    args = ap.parse_args()
    runs = {arm: per_seed(args.work, repo, arm) for arm, _ in ARMS}
    print(f"consistency check passed: {check(runs, args.summary)} (arm, set, metric) values match summary.json")
    for arm, _ in ARMS:
        print(arm, {g: {m: round(float(np.mean([np.mean([x[s][m] for s in sets]) for x in runs[arm]])), 3)
                        for m, _ in METRICS} for g, _, sets in GROUPS if g in ("clean", "seen")})
    args.out.write_text("% generated by analysis/secondary_metrics.py from run folders — do not edit\n" + table(runs))
    print(args.out)


if __name__ == "__main__":
    main()
