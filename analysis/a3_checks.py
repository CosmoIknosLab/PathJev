"""Plan amendment A3.2–A3.5 (docs/stage1_analysis_plan.md; D-040), descriptive, post-hoc.

    PYTHONPATH=src:analysis python analysis/a3_checks.py --work $PATHJEV_WORK \
        --out results/runs/stage1-a3 --tables paper/generated

Reads only run folders: eval_a3-* (slurm/campaigns/a3_eval.sbatch) and, for the original candidate order,
the primary Stage 1 runs (analysis/stage1_analysis.py arm_dirs). Main Choice question of each set only.
  A3.2 blank image   : per arm and set, max total-variation distance of any state's output from the first
                       state's output, and balanced accuracy.
  A3.3 OmniMed-Jev 896: Table-main metrics (bAcc, ECE, NLL, Brier; per set and group macros) next to 448.
  A3.4 Gen-SFT read-outs: primary / +<end_of_turn> / length-normalised, per seed and seed mean, per set and
                       macro (clean, unseen, in-domain); also the max |Δp| of the re-run primary vs the stored
                       primary run and their argmax agreement (D-028 found the read-out bit-identical
                       run-to-run on one node; across nodes bf16 kernels may differ).
  A3.5 candidate order: clean-macro bAcc/ECE per permutation (mean, SD, range over permutations; trained
                       arms: seed mean per permutation), and on the main Choice the mean total-variation
                       distance to the original-order output, the argmax flip rate and the "other"-argmax rate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stage1_analysis import (CLEAN, DESCRIPTIVE, IN_DOMAIN, MAIN_Q, SEEN, UNSEEN, Item, _rows, arm_dirs,
                             macro_value)

GROUPS = {"clean": CLEAN, "unseen": UNSEEN, "in_domain": IN_DOMAIN, "seen_source": SEEN}
N_PERM = {"jev": 10, "omnimedjev": 10, "sft": 5}
OTHER = "other"


def load_aligned(states: Path, decisions: Path, qid: str) -> dict:
    """Main-question arrays with candidates in sorted-id order (so permuted and original runs align)."""
    preds = {r["state_id"]: r["distributions"] for r in _rows(decisions)}
    ids, probs, y, groups, sids = None, [], [], [], []
    for s in _rows(states):
        d = next((d for d in s["decisions"] if d["qid"] == qid), None)
        if d is None:
            continue
        cids = sorted(c["id"] for c in d["candidates"])
        ids = ids or cids
        if cids != ids:
            raise ValueError(f"{states}: candidate sets differ between states")
        dist = preds[s["state_id"]][qid]                       # missing prediction = invalid run here
        p = np.array([dist[i] for i in ids], dtype=float)
        probs.append(p / p.sum())
        y.append(ids.index(max(d["target"], key=d["target"].get)))
        groups.append(str(s.get("meta", {}).get("group", s["state_id"])))
        sids.append(s["state_id"])
    return {"probs": np.stack(probs), "y": np.array(y), "groups": np.array(groups), "ids": ids, "sids": sids}


def item(d: dict) -> Item:
    return Item(d["probs"], d["y"], d["groups"])


def metrics(per_seed: list[dict[str, Item]], sets: list[str]) -> dict:
    return {m: macro_value(per_seed, sets, m) for m in DESCRIPTIVE}


def tv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(a - b).sum(axis=1)


def run_dir(work: Path, tag: str) -> Path:
    return work / "runs" / f"eval_a3-{tag}"


def original_dirs(work: Path, repo: Path, arm: str) -> list[Path]:
    key = {"jev": "pathjev", "sft": "sft", "omnimedjev": "omnimedjev"}[arm]
    return [t for t, _ in arm_dirs(work, repo, key, interim=False)]


# ---------------------------------------------------------------- A3.2
def blank(work: Path) -> dict:
    sdir = work / "eval_sets" / "a3-blank"
    out = {}
    for arm, tag in (("jev", "blank-jev-s1"), ("sft", "blank-sft-s1"), ("omnimedjev", "blank-omnimedjev")):
        out[arm] = {}
        for s in CLEAN:
            d = load_aligned(sdir / f"{s}.jsonl", run_dir(work, tag) / f"{s}_decisions.jsonl", MAIN_Q[s])
            it = item(d)
            out[arm][s] = {"n": len(d["y"]), "classes": int(len(set(d["y"].tolist()))),
                           "max_tv_from_first": float(tv(d["probs"], d["probs"][:1]).max()),
                           "balanced_accuracy": it.bacc(np.arange(len(d["y"]))),
                           "chance": 1.0 / len(set(d["y"].tolist()))}
    return out


# ---------------------------------------------------------------- A3.3
def omj896(work: Path, repo: Path) -> dict:
    test = work / "eval_sets" / "stage1-test"
    res = {}
    for name, dirs in (("448", original_dirs(work, repo, "omnimedjev")), ("896", [run_dir(work, "omnimedjev-896")])):
        items = [{s: item(load_aligned(test / f"{s}.jsonl", d / f"{s}_decisions.jsonl", MAIN_Q[s])) for s in MAIN_Q}
                 for d in dirs]
        res[name] = {"per_set": {s: metrics(items, [s]) for s in MAIN_Q},
                     **{g: metrics(items, sets) for g, sets in GROUPS.items()}}
    return res


# ---------------------------------------------------------------- A3.4
def sft_variants(work: Path, repo: Path) -> dict:
    test = work / "eval_sets" / "stage1-test"
    orig = original_dirs(work, repo, "sft")
    per_variant = {"primary": [], "eos": [], "lennorm": []}
    max_diff, agree = [], []
    for seed in (1, 2, 3):
        rd = run_dir(work, f"sftvar-s{seed}")
        seed_items = {v: {} for v in per_variant}
        for s in CLEAN:
            q = MAIN_Q[s]
            prim = load_aligned(test / f"{s}.jsonl", rd / f"{s}_decisions.jsonl", q)
            ref = load_aligned(test / f"{s}.jsonl", orig[seed - 1] / f"{s}_decisions.jsonl", q)
            max_diff.append(float(np.abs(prim["probs"] - ref["probs"]).max()))
            agree.append((prim["probs"].argmax(1) == ref["probs"].argmax(1)).mean())
            seed_items["primary"][s] = item(prim)
            var = {r["state_id"]: r["variants"][q] for r in _rows(rd / f"{s}_variants.jsonl")}
            for v in ("eos", "lennorm"):
                p = np.stack([[var[sid][v][i] for i in prim["ids"]] for sid in prim["sids"]])
                seed_items[v][s] = Item(p / p.sum(axis=1, keepdims=True), prim["y"], prim["groups"])
        for v in per_variant:
            per_variant[v].append(seed_items[v])
    groups = {g: sets for g, sets in GROUPS.items() if g != "seen_source"}
    out = {"max_abs_diff_primary_rerun_vs_stored": max(max_diff),
           "argmax_agreement_primary_rerun_vs_stored": float(np.mean(agree))}
    for v, items in per_variant.items():
        out[v] = {"per_set": {s: metrics(items, [s]) for s in CLEAN},
                  **{g: metrics(items, sets) for g, sets in groups.items()},
                  "per_seed_clean": {m: [macro_value([x], CLEAN, m) for x in items] for m in DESCRIPTIVE}}
    return out


# ---------------------------------------------------------------- A3.5
def permutations(work: Path, repo: Path) -> dict:
    test = work / "eval_sets" / "stage1-test"
    out = {}
    for arm, seeds in (("jev", (1, 2, 3)), ("omnimedjev", (None,)), ("sft", (1,))):
        orig = original_dirs(work, repo, arm)
        base = {}
        for seed in seeds:
            od = orig[(seed or 1) - 1]
            base[seed] = {s: load_aligned(test / f"{s}.jsonl", od / f"{s}_decisions.jsonl", MAIN_Q[s]) for s in CLEAN}
        per_perm, tvd, flip, other = [], [], [], []
        for k in range(1, N_PERM[arm] + 1):
            pdir = work / "eval_sets" / f"a3-perm-{k}"
            seed_items, t_k, f_k, o_k = [], [], [], []
            for seed in seeds:
                tag = f"perm{k}-{arm}" + (f"-s{seed}" if seed else "")
                ds = {s: load_aligned(pdir / f"{s}.jsonl", run_dir(work, tag) / f"{s}_decisions.jsonl", MAIN_Q[s])
                      for s in CLEAN}
                seed_items.append({s: item(d) for s, d in ds.items()})
                for s, d in ds.items():
                    b = base[seed][s]
                    assert d["sids"] == b["sids"] and d["ids"] == b["ids"]
                    t_k.append(tv(d["probs"], b["probs"]).mean())
                    f_k.append((d["probs"].argmax(1) != b["probs"].argmax(1)).mean())
                    if OTHER in d["ids"]:
                        o_k.append((d["probs"].argmax(1) == d["ids"].index(OTHER)).mean())
            per_perm.append({m: macro_value(seed_items, CLEAN, m) for m in ("balanced_accuracy", "ece")})
            tvd.append(float(np.mean(t_k)))
            flip.append(float(np.mean(f_k)))
            other.append(float(np.mean(o_k)))
        orig_items = [{s: item(base[seed][s]) for s in CLEAN} for seed in seeds]
        orig_other = np.mean([(base[seed][s]["probs"].argmax(1) == base[seed][s]["ids"].index(OTHER)).mean()
                              for seed in seeds for s in CLEAN if OTHER in base[seed][s]["ids"]])
        summ = {m: {"original": macro_value(orig_items, CLEAN, m),
                    "mean": float(np.mean([p[m] for p in per_perm])), "sd": float(np.std([p[m] for p in per_perm], ddof=1)),
                    "min": float(min(p[m] for p in per_perm)), "max": float(max(p[m] for p in per_perm))}
                for m in ("balanced_accuracy", "ece")}
        out[arm] = {"seeds": [s for s in seeds if s] or ["zero-shot"], "permutations": N_PERM[arm],
                    "per_permutation": per_perm, **summ,
                    "mean_tv_to_original": float(np.mean(tvd)), "argmax_flip_rate": float(np.mean(flip)),
                    "other_argmax_rate": float(np.mean(other)), "other_argmax_rate_original": float(orig_other)}
    return out


# ---------------------------------------------------------------- tables
NAME = {"jev": "PathJev-P", "sft": "Gen-SFT", "omnimedjev": "OmniMed-Jev"}
SET = {"bracs": "BRACS", "sicapv2": "SICAPv2", "hepatobench": "HepatoBench", "hiesd": "HiESD"}
f3 = lambda x: f"{x:.3f}"


GROUP_NAME = {"clean": "Clean average", "unseen": "Unseen organs", "in_domain": "In domain", "seen_source": "Seen source"}


def _sci(x: float) -> str:
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def _bold(vals: dict, higher: bool, fmt=f3) -> dict:
    """Formatted values with the best one (as printed) in bold."""
    out = {k: fmt(v) for k, v in vals.items()}
    best = out[(max if higher else min)(vals, key=vals.get)]
    return {k: (f"\\textbf{{{t}}}" if t == best else t) for k, t in out.items()}


def tables(r: dict, dst: Path) -> None:
    hdr = "% generated by analysis/a3_checks.py — do not edit\n"
    b = r["A3.2"]
    lines = [r"\begin{tabular}{@{}l" + "cc" * len(CLEAN) + "@{}}", r"\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SET[s]}}}" for s in CLEAN) + r" \\",
             " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(CLEAN))),
             "Method & " + " & ".join(["max TV & bAcc"] * len(CLEAN)) + r" \\", r"\midrule"]
    for arm in ("jev", "sft", "omnimedjev"):
        lines.append(NAME[arm] + " & " + " & ".join(f"{_sci(b[arm][s]['max_tv_from_first'])} & {f3(b[arm][s]['balanced_accuracy'])}"
                                                    for s in CLEAN) + r" \\")
    lines += [r"\midrule", r"\textit{Chance} & " + " & ".join(f"-- & {f3(b['jev'][s]['chance'])}" for s in CLEAN) + r" \\"]
    (dst / "tab_a3_blank.tex").write_text(hdr + "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")

    o = r["A3.3"]
    lines = [r"\begin{tabular}{@{}l" + "cccc" * 1 + "cccc@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{Pre-resize 448 (primary)} & \multicolumn{4}{c}{Pre-resize 896} \\",
             r"\cmidrule(lr){2-5} \cmidrule(l){6-9}",
             r"Group & bAcc$\uparrow$ & ECE$\downarrow$ & NLL$\downarrow$ & Brier$\downarrow$ & bAcc$\uparrow$ & ECE$\downarrow$"
             r" & NLL$\downarrow$ & Brier$\downarrow$ \\", r"\midrule"]
    for g in ("clean", "unseen", "in_domain", "seen_source"):
        lines.append(GROUP_NAME[g] + " & " + " & ".join(f3(o[v][g][m]) for v in ("448", "896")
                                                         for m in ("balanced_accuracy", "ece", "nll", "brier")) + r" \\")
    (dst / "tab_a3_omj896.tex").write_text(hdr + "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")

    v = dict(r["A3.4"])
    rows = [("primary", "Gen-SFT, sum (primary)"),
            ("eos", r"Gen-SFT, sum + end-of-answer"),
            ("lennorm", "Gen-SFT, length-normalised")]
    summ = Path(__file__).resolve().parents[1] / "results" / "runs" / "stage1-analysis" / "summary.json"
    if summ.exists():                                    # reference row, from the primary analysis (no recomputation)
        v["pathjev"] = json.loads(summ.read_text())["table"]["pathjev"]
        rows = [("pathjev", "PathJev-P (reference)")] + rows
    gm = [(g, m) for g in ("clean", "unseen", "in_domain") for m in ("balanced_accuracy", "ece", "nll")]
    cols = [_bold({k: v[k][g][m] for k, _ in rows}, m == "balanced_accuracy") for g, m in gm]
    lines = [r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{Clean average} & \multicolumn{3}{c}{Unseen organs} & \multicolumn{3}{c}{In domain} \\",
             r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(l){8-10}",
             "Read-out & " + " & ".join([r"bAcc$\uparrow$ & ECE$\downarrow$ & NLL$\downarrow$"] * 3) + r" \\", r"\midrule"]
    for i, (key, name) in enumerate(rows):
        if key == "primary" and rows[0][0] == "pathjev":
            lines.append(r"\midrule")
        lines.append(name + " & " + " & ".join(c[key] for c in cols) + r" \\")
    (dst / "tab_a3_sftvar.tex").write_text(hdr + "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")

    p = r["A3.5"]
    arms = ("jev", "sft", "omnimedjev")
    pct = lambda x: f"{100 * x:.1f}"
    mean_b = _bold({a: p[a]["balanced_accuracy"]["mean"] for a in arms}, True)
    mean_e = _bold({a: p[a]["ece"]["mean"] for a in arms}, False)
    tvs = _bold({a: p[a]["mean_tv_to_original"] for a in arms}, False)
    flips = _bold({a: p[a]["argmax_flip_rate"] for a in arms}, False, pct)
    lines = [r"\setlength{\tabcolsep}{3.2pt}", r"\begin{tabular}{@{}lccccccccccc@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{Clean bAcc$\uparrow$} & \multicolumn{3}{c}{Clean ECE$\downarrow$} & & &"
             r" \multicolumn{2}{c}{``other'' (\%)} \\",
             r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(l){11-12}",
             r"Method & Perm. & orig. & mean & range & orig. & mean & range & TV$\downarrow$ & flips (\%)$\downarrow$"
             r" & orig. & perm. \\", r"\midrule"]
    for a in arms:
        q = p[a]
        rng = lambda m: f"{f3(q[m]['min'])}--{f3(q[m]['max'])}"
        lines.append(f"{NAME[a]} & {q['permutations']} & {f3(q['balanced_accuracy']['original'])} & {mean_b[a]} & "
                     f"{rng('balanced_accuracy')} & {f3(q['ece']['original'])} & {mean_e[a]} & {rng('ece')} & {tvs[a]} & "
                     f"{flips[a]} & {pct(q['other_argmax_rate_original'])} & {pct(q['other_argmax_rate'])} \\\\")
    (dst / "tab_a3_perm.tex").write_text(hdr + "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def report(r: dict) -> str:
    out = ["# Plan A3.2–A3.5 (post-hoc, descriptive; D-040)", ""]
    b = r["A3.2"]
    out.append("## A3.2 blank image (main Choice; max total-variation distance from the first state's output; bAcc)")
    for arm in b:
        out.append(f"- {NAME[arm]}: " + "; ".join(f"{s} TV {b[arm][s]['max_tv_from_first']:.2e}, bAcc {b[arm][s]['balanced_accuracy']:.3f} "
                                                   f"(chance {b[arm][s]['chance']:.3f}, {b[arm][s]['classes']} classes)" for s in CLEAN))
    o = r["A3.3"]
    out += ["", "## A3.3 OmniMed-Jev pre-resize 448 vs 896 (bAcc / ECE / NLL)"]
    for g in GROUPS:
        out.append(f"- {g}: 448 {o['448'][g]['balanced_accuracy']:.3f} / {o['448'][g]['ece']:.3f} / {o['448'][g]['nll']:.3f}; "
                   f"896 {o['896'][g]['balanced_accuracy']:.3f} / {o['896'][g]['ece']:.3f} / {o['896'][g]['nll']:.3f}")
    v = r["A3.4"]
    out += ["", f"## A3.4 Gen-SFT read-outs (seed mean; re-run primary vs stored: max |Δp| = {v['max_abs_diff_primary_rerun_vs_stored']:.2e}, "
            f"argmax agreement {v['argmax_agreement_primary_rerun_vs_stored']:.4f})"]
    for key in ("primary", "eos", "lennorm"):
        out.append(f"- {key}: " + "; ".join(f"{g} bAcc {v[key][g]['balanced_accuracy']:.3f} ECE {v[key][g]['ece']:.3f} "
                                            f"NLL {v[key][g]['nll']:.3f}" for g in ("clean", "unseen", "in_domain")))
    p = r["A3.5"]
    out += ["", "## A3.5 candidate order (clean macro; main Choice instability vs original order)"]
    for arm, a in p.items():
        out.append(f"- {NAME[arm]} ({a['permutations']} permutations, seeds {a['seeds']}): bAcc original "
                   f"{a['balanced_accuracy']['original']:.3f}, permuted mean {a['balanced_accuracy']['mean']:.3f} "
                   f"SD {a['balanced_accuracy']['sd']:.3f} range {a['balanced_accuracy']['min']:.3f}–{a['balanced_accuracy']['max']:.3f}; "
                   f"ECE original {a['ece']['original']:.3f}, mean {a['ece']['mean']:.3f} SD {a['ece']['sd']:.3f} "
                   f"range {a['ece']['min']:.3f}–{a['ece']['max']:.3f}; mean TV {a['mean_tv_to_original']:.3f}; "
                   f"argmax flips {a['argmax_flip_rate']:.3f}; 'other' argmax {a['other_argmax_rate_original']:.3f} → "
                   f"{a['other_argmax_rate']:.3f}")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, help="required unless --tables-only")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tables", type=Path)
    ap.add_argument("--only", nargs="*", default=["A3.2", "A3.3", "A3.4", "A3.5"])
    ap.add_argument("--tables-only", action="store_true", help="rewrite the tables from <out>/summary.json, no recomputation")
    args = ap.parse_args()
    if not args.tables_only and args.work is None:
        ap.error("--work is required unless --tables-only")
    if args.tables_only:
        tables(json.loads((args.out / "summary.json").read_text()), args.tables)
        return
    repo = Path(__file__).resolve().parents[1]
    fns = {"A3.2": lambda: blank(args.work), "A3.3": lambda: omj896(args.work, repo),
           "A3.4": lambda: sft_variants(args.work, repo), "A3.5": lambda: permutations(args.work, repo)}
    r = {k: fns[k]() for k in args.only}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / ("summary.json" if len(r) == 4 else "partial_" + "_".join(r) + ".json")).write_text(json.dumps(r, indent=1))
    if len(r) == 4:
        (args.out / "report.md").write_text(report(r))
        if args.tables:
            tables(r, args.tables)
    print(json.dumps({k: (v if k == "A3.2" else "...") for k, v in r.items()}, indent=1)[:3000])


if __name__ == "__main__":
    main()
