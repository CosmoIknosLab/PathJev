"""Phase 3 pilot report (Stage 1): reads only run folders, writes markdown + JSON.

    python analysis/pilot_report.py --work $PATHJEV_WORK --out results/runs/pilot-stage1

Per stage1_analysis_plan.md §3/§6: main Choice question per test set; macro over clean sets,
unseen-domain and in-domain breakdowns; sanity checks; timing. Pilot = 1 seed, subsets:
engineering and sanity only, no hypothesis test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MAIN_Q = {"bracs": "lesion", "sicapv2": "gleason_pattern", "hepatobench": "tissue", "hiesd": "tissue",
          "spider-colorectal": "tissue", "spider-thorax": "tissue", "spider-breast": "tissue", "spider-skin": "tissue"}
DATASET = {"spider-colorectal": "spider-colorectal", "spider-thorax": "spider-thorax",
           "spider-breast": "spider-breast", "spider-skin": "spider-skin"}
UNSEEN = ["hepatobench", "hiesd"]
IN_DOMAIN = ["bracs", "sicapv2"]
SEEN = [s for s in MAIN_Q if s.startswith("spider")]
ARMS = {  # label -> run-dir glob (relative to runs/)
    "PathJev-P": "eval_test-pilot-p-s1_*",
    "(a) OmniMed-Jev": "eval_pilot-omnimedjev_OmniMed-Jev-4B",
    "(a) no 'other'": "eval_pilot-omnimedjev-no-other_OmniMed-Jev-4B",
    "(a) permuted": "eval_pilot-omnimedjev-permuted_OmniMed-Jev-4B",
    "(b) Gen-SFT": "eval_test-pilot-sft-p-s1_*",
    "(c) CONCH": "eval_pilot-conch",
    "(d) probe": "eval_pilot-probe",
    "(e) Inner-Jev": "eval_pilot-letter",
}
METRICS = ("balanced_accuracy", "accuracy", "ece", "adaptive_ece", "nll", "brier", "aurc")


def main_row(report: dict, set_name: str) -> dict | None:
    for key, row in report.items():
        if key.endswith("/" + MAIN_Q[set_name]) and "balanced_accuracy" in row:
            return row
    return None


def macro(rows: dict, sets: list[str], metric: str):
    vals = [rows[s][metric] for s in sets if s in rows and rows[s] and metric in rows[s]]
    return sum(vals) / len(vals) if len(vals) == len(sets) else None


def load_arm(run_dir: Path) -> dict:
    out = {}
    for s in MAIN_Q:
        path = run_dir / f"{s}_eval.json"
        if path.exists():
            out[s] = main_row(json.loads(path.read_text()), s)
            out[s]["_missing"] = sum(v.get("missing", 0) for v in json.loads(path.read_text()).values())
    return out


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    runs = args.work / "runs"
    table, sources = {}, {}
    for arm, pattern in ARMS.items():
        dirs = sorted(runs.glob(pattern))
        if dirs:
            table[arm], sources[arm] = load_arm(dirs[-1]), str(dirs[-1])
    summary = {arm: {"clean": {m: macro(r, UNSEEN + IN_DOMAIN, m) for m in METRICS},
                     "unseen": {m: macro(r, UNSEEN, m) for m in METRICS},
                     "in_domain": {m: macro(r, IN_DOMAIN, m) for m in METRICS},
                     "seen_source": {m: macro(r, SEEN, m) for m in METRICS}} for arm, r in table.items()}
    lines = ["# Phase 3 pilot — Stage 1 (1 seed, 1 epoch, evaluation subsets)", "",
             "Engineering and sanity check only (stage1_analysis_plan §6): **no hypothesis is tested here and no number "
             "goes in the paper.** Main Choice question per set; macro = unweighted mean over sets.", ""]
    for group, label in (("clean", "Clean (unseen + in-domain)"), ("unseen", "Unseen domain (HepatoBench, HiESD)"),
                         ("in_domain", "In-domain clean (BRACS, SICAPv2)"), ("seen_source", "Seen-source (SPIDER ×4)")):
        lines += [f"## {label}", "", "| Arm | Bal. acc | Acc | ECE | aECE | NLL | Brier | AURC |", "|---|---|---|---|---|---|---|---|"]
        for arm in table:
            m = summary[arm][group]
            lines.append(f"| {arm} | " + " | ".join(fmt(m[k]) for k in METRICS) + " |")
        lines.append("")
    lines += ["## Per set (bal. acc / ECE / n / missing)", "", "| Arm | " + " | ".join(MAIN_Q) + " |",
              "|---|" + "---|" * len(MAIN_Q)]
    for arm, r in table.items():
        cells = [f"{fmt(r[s]['balanced_accuracy'])} / {fmt(r[s]['ece'])} / {r[s]['n']} / {r[s]['_missing']}"
                 if r.get(s) else "—" for s in MAIN_Q]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += ["", "## Sources", ""] + [f"- {a}: `{p}`" for a, p in sources.items()]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    (args.out / "summary.json").write_text(json.dumps({"summary": summary, "per_set": table, "sources": sources},
                                                      indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
