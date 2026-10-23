"""Write paper/generated/numbers.tex: every number the paper text uses, from run folders and configs only.

    python analysis/export_macros.py --work $PATHJEV_WORK --out paper/generated/numbers.tex

Sections: data (training mix, evaluation sets, analysis-plan constants). Result macros are added
by the Phase 4 analysis once all seeds are in. Never edit numbers.tex by hand (CLAUDE.md §9).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathjev.groups import excluded_state_ids  # noqa: E402

MAIN_Q = {"bracs": "lesion", "sicapv2": "gleason_pattern", "hepatobench": "tissue", "hiesd": "tissue"}


def _count(path: Path) -> int:
    with path.open() as fh:
        return sum(1 for line in fh if line.strip())


def _candidates(path: Path, qid: str) -> int:
    with path.open() as fh:
        for line in fh:
            for d in json.loads(line)["decisions"]:
                if d["qid"] == qid:
                    return len(d["candidates"])
    raise ValueError(f"{qid} not in {path}")


DIGITS = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))


def latex_name(name: str) -> str:
    """LaTeX command names may contain letters only: spell digits out (sicapv2 -> SicapvTwo)."""
    return "".join(DIGITS.get(c, c) for c in name)


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def data_macros(work: Path, repo: Path) -> dict[str, str]:
    mix = work / "mixes" / "stage1-p"
    test = work / "eval_sets" / "stage1-test"
    info = json.loads((test / "info.json").read_text())
    cfg = yaml.safe_load((repo / "configs" / "eval" / "stage1-test.yaml").read_text())
    caps = {s["cap"] for s in cfg["sets"] if s["cap"]}
    if len(caps) != 1:
        raise ValueError(f"expected one test cap, found {caps}")
    m = {
        "NTrainStates": fmt_int(_count(mix / "train.jsonl")),
        "NValStates": fmt_int(_count(mix / "val.jsonl")),
        "NOrganCandidates": str(_candidates(mix / "train.jsonl", "organ")),
        "NTestCap": fmt_int(caps.pop()),
        "NClassesHepato": str(len(info["sets"]["hepatobench"]["classes"])),
        "NClassesHiesd": str(len(info["sets"]["hiesd"]["classes"])),
        "NSeeds": "3",   # stage1_analysis_plan.md §4 (pre-registered)
    }
    excluded = excluded_state_ids()
    for name, s in info["sets"].items():
        key = latex_name("".join(p.capitalize() for p in name.split("-")))
        with (test / f"{name}.jsonl").open() as fh:   # states analysed: capped draw minus exclusions
            n = sum(1 for line in fh if line.strip() and json.loads(line)["state_id"] not in excluded)
        m[f"NTest{key}"] = fmt_int(n)
    return m


ARM_KEY = {"pathjev": "PJ", "sft": "SFT", "omnimedjev": "OMJ", "conch": "CONCH", "letter": "IJ", "probe": "Probe"}
AGG_KEY = {"clean": "Clean", "unseen": "Unseen", "in_domain": "InDom", "seen_source": "Seen"}
MET_KEY = {"balanced_accuracy": "Bacc", "ece": "Ece", "nll": "Nll", "brier": "Brier"}


def f3(x: float) -> str:
    return f"{x:.3f}"


def result_macros(analysis: Path, temperature: Path | None) -> dict[str, str]:
    """Stage 1 results (analysis/stage1_analysis.py summary.json; A2 summary.json). Refuses interim files."""
    r = json.loads(analysis.read_text())
    if r["interim"]:
        raise ValueError(f"{analysis} is an INTERIM analysis; the paper takes the final run only")
    m = {"NResamples": fmt_int(r["resamples"])}
    for arm, row in r["table"].items():
        a = ARM_KEY[arm]
        for agg, g in AGG_KEY.items():
            if agg not in row:
                continue
            for met, mk in MET_KEY.items():
                m[f"Res{a}{g}{mk}"] = f3(row[agg][met])
                if met in row[agg].get("per_seed", {}) and len(row[agg]["per_seed"][met]) > 1:
                    m[f"Res{a}{g}{mk}SD"] = f3(float(np.std(row[agg]["per_seed"][met], ddof=1)))
                if met in row[agg].get("temperature_scaled", {}):
                    m[f"Res{a}{g}{mk}T"] = f3(row[agg]["temperature_scaled"][met])
    h = r["hypotheses"]
    for hyp, key in (("H1a", "HOneA"), ("H1b", "HOneB")):
        m[f"{key}Supported"] = "supported" if h[f"{hyp}_supported"] else "not supported"
        for other, v in h["seed_mean"][hyp].items():
            o = ARM_KEY[other]
            m[f"{key}Diff{o}"] = f"{v['diff']:+.3f}"
            m[f"{key}Lo{o}"], m[f"{key}Hi{o}"] = f"{v['ci95'][0]:+.3f}", f"{v['ci95'][1]:+.3f}"
    for met, mk in (("balanced_accuracy", "Bacc"), ("ece", "Ece")):
        v = h["H1c"][met]
        m[f"HOneCDiff{mk}"], m[f"HOneCLo{mk}"], m[f"HOneCHi{mk}"] = (f"{v['diff']:+.3f}", f"{v['ci95'][0]:+.3f}",
                                                                   f"{v['ci95'][1]:+.3f}")
    m["GateRecommendation"] = r["gate"]["recommendation"]
    # secondary (A1): largest |capped - full| clean-macro change over the arms scored on the full files
    full, diffs = r["full_test_secondary"], {"balanced_accuracy": [], "ece": []}
    for arm, row in r["table"].items():
        runs = [v for k, v in full.items() if k == arm or k.startswith(f"{arm}-s")]
        if runs and "clean" in row:
            for met in diffs:
                diffs[met].append(abs(row["clean"][met] - float(np.mean([x["clean"][met] for x in runs]))))
    m["FullMaxDiffBacc"], m["FullMaxDiffEce"] = f3(max(diffs["balanced_accuracy"])), f3(max(diffs["ece"]))
    if temperature is not None:
        t = json.loads(temperature.read_text())
        if t["interim"]:
            raise ValueError(f"{temperature} is INTERIM")
        for arm, row in t["arms"].items():
            for var, vk in (("raw", "Raw"), ("T-val", "Val"), ("T-in", "In"), ("T-cross", "Cross"), ("T-oracle", "Oracle")):
                for met, mk in MET_KEY.items():
                    m[f"Tmp{ARM_KEY[arm]}{vk}{mk}"] = f3(row["seed_mean_unseen"][var][met])
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--analysis", type=Path, help="final stage1_analysis summary.json")
    ap.add_argument("--temperature", type=Path, help="final stage1_temperature_transfer summary.json")
    ap.add_argument("--placeholders", nargs=2, type=Path, metavar=("ANALYSIS", "TEMPERATURE"),
                    help="interim summaries used ONLY for macro names; every result value becomes a red ??")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    macros = data_macros(args.work, repo)
    if args.analysis:
        macros |= result_macros(args.analysis, args.temperature)
    elif args.placeholders:
        tmp = {}
        for src in args.placeholders:
            d = json.loads(src.read_text())
            d["interim"] = False
            tmp[src] = Path("/tmp") / f"pj_placeholder_{src.parent.name}.json"
            tmp[src].write_text(json.dumps(d))
        names = result_macros(tmp[args.placeholders[0]], tmp[args.placeholders[1]])
        macros |= {k: "\\textcolor{red}{??}" for k in names}
    lines = ["% Generated by analysis/export_macros.py; do not edit."]
    bad = [k for k in macros if not k.isalpha()]
    if bad:
        raise ValueError(f"macro names must be letters only: {bad}")
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"{len(macros)} macros -> {args.out}")


if __name__ == "__main__":
    main()
