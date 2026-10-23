"""Amendment A3.1 (post-hoc, docs/stage1_analysis_plan.md): HepatoBench grouping sensitivity.

    python analysis/a3_hepato.py --work $PATHJEV_WORK --out results/runs/stage1-a3-hepato [--resamples 10000]

HepatoBench releases no slide IDs and consecutive image indices come from the same tissue, so the primary
analysis (one group per image) understates uncertainty. Here the HepatoBench groups are blocks of B consecutive
indices within each class (file stem NN_CLS_<i> -> block floor(i/B)), B = 50 and 200; nothing else changes.
  (a) H1a/H1b re-tested exactly as in plan §4 (same comparisons, paired cluster bootstrap with the plan seed,
      one-sided p, Holm within family, seed mean and per seed, rule "seed mean and >= 2 of 3 seeds").
      Variant "image" uses the original groups and must reproduce the primary summary.json exactly.
  (b) Endpoints and H1a/H1b without HepatoBench (clean = HiESD, BRACS, SICAPv2; unseen = HiESD).
  (c) The probe's nested cross-fit on HepatoBench re-run with block groups (cached H-optimus-0 features);
      the image-group re-run must reproduce the primary probe decisions.
Writes summary.json and report.md to --out, and paper/generated/{numbers_a3hepato.tex,tab_a3_hepato.tex}.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from stage1_analysis import (CLEAN, H1A, H1B, IN_DOMAIN, MAIN_Q, METRIC, UNSEEN, Item, ResamplePlan, arm_dirs,
                             compare, holm, load_main, macro_value)

ARMS = ["pathjev", "sft", "omnimedjev", "conch", "letter", "probe"]
NO_HEP = [s for s in CLEAN if s != "hepatobench"]
VARIANTS = {"image": (CLEAN, None), "block50": (CLEAN, 50), "block200": (CLEAN, 200), "nohep": (NO_HEP, None)}
REPO = Path(__file__).resolve().parents[1]
_DATA: dict = {}


def block_groups(groups: np.ndarray, size: int) -> np.ndarray:
    """'hepatobench:06_REA_1423' -> '06_REA:28' for size 50 (blocks of consecutive indices within a class)."""
    out = []
    for g in groups:
        stem = str(g).split(":", 1)[1]
        prefix, idx = stem.rsplit("_", 1)
        out.append(f"{prefix}:{int(idx) // size}")
    return np.array(out)


def load(work: Path) -> dict:
    data = {}
    for arm in ARMS:
        data[arm] = []
        for tdir, _ in arm_dirs(work, REPO, arm, interim=False):
            data[arm].append({s: load_main(work / "eval_sets" / "stage1-test" / f"{s}.jsonl",
                                           tdir / f"{s}_decisions.jsonl", MAIN_Q[s]) for s in CLEAN})
    return data


def items(arm: str, sets: list[str], block: int | None) -> list[dict[str, Item]]:
    out = []
    for per_set in _DATA[arm]:
        row = {}
        for s in sets:
            d = per_set[s]
            g = block_groups(d["groups"], block) if (s == "hepatobench" and block) else d["groups"]
            row[s] = Item(d["probs"], d["y"], g)
        out.append(row)
    return out


def run_variant(args: tuple[str, int]) -> tuple[str, dict]:
    name, resamples = args
    sets, block = VARIANTS[name]
    base = {a: items(a, sets, block) for a in ARMS}
    plan = ResamplePlan(base["pathjev"][0], resamples)

    def run_h(pj, others):   # identical to stage1_analysis.main
        ha = {o: compare(pj, others[o], sets, "ece", plan, resamples, 0.0, True) for o in H1A}
        hb = {o: compare(pj, others[o], sets, "balanced_accuracy", plan, resamples, H1B[o], False) for o in H1B}
        ra, rb = holm({o: v["p_one_sided"] for o, v in ha.items()}), holm({o: v["p_one_sided"] for o, v in hb.items()})
        for o in ha:
            ha[o]["holm_rejected"] = ra[o]
        for o in hb:
            hb[o]["holm_rejected"] = rb[o]
        return {"H1a": ha, "H1b": hb, "H1a_all": all(ra.values()), "H1b_all": all(rb.values())}

    hyp = {"seed_mean": run_h(base["pathjev"], base)}
    for k in range(len(base["pathjev"])):
        others = dict(base)
        others["sft"] = [base["sft"][min(k, len(base["sft"]) - 1)]]
        hyp[f"seed{k + 1}"] = run_h([base["pathjev"][k]], others)
    seeds = [hyp[f"seed{k + 1}"] for k in range(len(base["pathjev"]))]
    for h in ("H1a", "H1b"):
        hyp[f"{h}_supported"] = bool(hyp["seed_mean"][f"{h}_all"] and sum(s[f"{h}_all"] for s in seeds) >= 2)
    groups = {s: len(base["pathjev"][0][s].members) for s in sets}
    endpoints = {}
    for a in ARMS:
        endpoints[a] = {}
        for agg, agg_sets in (("clean", sets), ("unseen", [s for s in UNSEEN if s in sets]), ("in_domain", IN_DOMAIN)):
            endpoints[a][agg] = {m: macro_value(base[a], agg_sets, m) for m in METRIC}
    return name, {"sets": sets, "hepatobench_block": block, "groups": groups, "hypotheses": hyp, "endpoints": endpoints}


def probe_hepato(work: Path) -> dict:
    """Re-run baselines.probe's cross-fitting branch on HepatoBench with different groups (same code path)."""
    from sklearn.model_selection import GroupKFold

    from pathjev.baselines import _choice_table, _fit, _states

    states = _states(work / "eval_sets" / "stage1-test" / "hepatobench.jsonl")
    z = np.load(work / "runs" / "eval_s1-features" / "hepatobench.npz")
    feats = dict(zip(z["ids"].tolist(), z["features"]))
    (qid, _), te = next((k, v) for k, v in _choice_table(states, feats).items() if k[0] == MAIN_Q["hepatobench"])
    X, y = np.stack(te["X"]), np.array(te["y"])
    primary = {json.loads(l)["state_id"]: json.loads(l)["distributions"][qid]
               for l in (work / "runs" / "eval_s1-probe" / "hepatobench_decisions.jsonl").open()}
    out = {}
    for name, block in (("image", None), ("block50", 50), ("block200", 200)):
        g = np.array(te["g"]) if block is None else block_groups(np.array(te["g"]), block)
        logits = np.full((len(X), len(te["ids"])), -1e9)
        for tr_idx, te_idx in GroupKFold(n_splits=5).split(X, y, g):
            scaler, model, temp = _fit(X[tr_idx], y[tr_idx], [g[i] for i in tr_idx])
            logits[np.ix_(te_idx, model.classes_)] = model.decision_function(
                scaler.transform(X[te_idx])).reshape(len(te_idx), -1) / temp
        logits -= logits.max(1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        it = Item(probs, y, g)
        idx = np.arange(len(y))
        out[name] = {"balanced_accuracy": it.bacc(idx), "ece": it.ece(idx), "nll": it.nll(idx),
                     "n_groups": int(len(set(g.tolist())))}
        if block is None:
            ref = np.array([[primary[s][c] for c in te["ids"]] for s in te["sid"]])
            out[name]["max_abs_diff_vs_primary"] = float(np.abs(ref - probs).max())
    return out


# ---------------------------------------------------------------- outputs
KEY = {"sft": "SFT", "conch": "CONCH", "omnimedjev": "OMJ", "letter": "IJ"}
VKEY = {"block50": "BlockFifty", "block200": "BlockTwoHundred", "nohep": "NoHep", "image": "Image"}


def f3(x: float) -> str:
    return f"{x:.3f}"


def macros(r: dict) -> dict[str, str]:
    m = {}
    for v in ("block50", "block200", "nohep"):
        h = r["variants"][v]["hypotheses"]
        for hyp, hk in (("H1a", "HOneA"), ("H1b", "HOneB")):
            m[f"AThreeHep{VKEY[v]}{hk}Supported"] = "supported" if h[f"{hyp}_supported"] else "not supported"
            for o in ("sft", "conch"):
                c = h["seed_mean"][hyp][o]
                m[f"AThreeHep{VKEY[v]}{hk}Diff{KEY[o]}"] = f"{c['diff']:+.3f}"
                m[f"AThreeHep{VKEY[v]}{hk}Lo{KEY[o]}"] = f"{c['ci95'][0]:+.3f}"
                m[f"AThreeHep{VKEY[v]}{hk}Hi{KEY[o]}"] = f"{c['ci95'][1]:+.3f}"
    ep = r["variants"]["nohep"]["endpoints"]
    for a, ak in (("pathjev", "PJ"), ("sft", "SFT"), ("conch", "CONCH")):
        for agg, gk in (("clean", "Clean"), ("unseen", "Unseen")):
            m[f"AThreeHepNoHep{ak}{gk}Bacc"] = f3(ep[a][agg]["balanced_accuracy"])
            m[f"AThreeHepNoHep{ak}{gk}Ece"] = f3(ep[a][agg]["ece"])
    for v in ("image", "block50", "block200"):
        p = r["probe_hepatobench"][v]
        m[f"AThreeHepProbe{VKEY[v]}Bacc"] = f3(p["balanced_accuracy"])
        m[f"AThreeHepProbe{VKEY[v]}Ece"] = f3(p["ece"])
    bad = [k for k in m if not k.isalpha()]
    if bad:
        raise ValueError(f"macro names must be letters only: {bad}")
    return m


def table(r: dict) -> str:
    """Seed-mean difference (PathJev-P minus comparator) per grouping; the 95% interval on a second row."""
    neg = lambda x: x.replace("-", "$-$")
    comps = [("H1a", "sft"), ("H1a", "conch"), ("H1b", "sft"), ("H1b", "conch")]
    rows = [("image", "Image (primary)"), ("block50", "Blocks of 50"), ("block200", "Blocks of 200"),
            ("nohep", "Without HepatoBench")]
    lines = ["\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
             "& \\multicolumn{2}{c}{H1a: $\\Delta$ECE} & \\multicolumn{2}{c}{H1b: $\\Delta$bAcc}"
             " & \\multicolumn{2}{c}{Supported} \\\\",
             "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(l){6-7}",
             "HepatoBench groups & vs Gen-SFT & vs CONCH & vs Gen-SFT & vs CONCH & H1a & H1b \\\\", "\\midrule"]
    for v, name in rows:
        h = r["variants"][v]["hypotheses"]
        c = [h["seed_mean"][hyp][o] for hyp, o in comps]
        sup = " & ".join("yes" if h[f"{x}_supported"] else "no" for x in ("H1a", "H1b"))
        lines.append(f"{name} & " + " & ".join(neg(f"{x['diff']:+.3f}") for x in c) + f" & {sup} \\\\")
        lines.append(" & " + " & ".join(neg(f"{{\\tiny[{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]}}") for x in c)
                     + " & & \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def report_md(r: dict) -> str:
    out = ["# A3.1 — HepatoBench grouping sensitivity (post-hoc)", "",
           f"Resamples: {r['resamples']}. Verification (image groups vs primary summary.json): "
           f"{r['verification']}", ""]
    for v, d in r["variants"].items():
        h = d["hypotheses"]
        out.append(f"## {v} (sets {', '.join(d['sets'])}; groups {d['groups']})")
        out.append(f"H1a supported: {h['H1a_supported']} · H1b supported: {h['H1b_supported']}")
        for hyp in ("H1a", "H1b"):
            for o, c in h["seed_mean"][hyp].items():
                out.append(f"- {hyp} vs {o}: {c['diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] "
                           f"p={c['p_one_sided']:.4f} holm={c['holm_rejected']}")
        out.append("seeds: " + "; ".join(f"seed{k}: H1a {h[f'seed{k}']['H1a_all']} H1b {h[f'seed{k}']['H1b_all']}"
                                          for k in (1, 2, 3)))
        ep = d["endpoints"]
        out.append("endpoints (clean / unseen bAcc, ECE): " + "; ".join(
            f"{a} {ep[a]['clean']['balanced_accuracy']:.3f}/{ep[a]['unseen']['balanced_accuracy']:.3f}, "
            f"{ep[a]['clean']['ece']:.3f}/{ep[a]['unseen']['ece']:.3f}" for a in ARMS))
        out.append("")
    out.append("## Probe on HepatoBench (nested cross-fit, cached H-optimus-0 features)")
    for v, p in r["probe_hepatobench"].items():
        out.append(f"- {v}: " + ", ".join(f"{k}={x:.4g}" if isinstance(x, float) else f"{k}={x}" for k, x in p.items()))
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, help="required unless --table-only")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--table-only", action="store_true", help="rewrite tab_a3_hepato.tex from <out>/summary.json")
    args = ap.parse_args()
    if not args.table_only and args.work is None:
        ap.error("--work is required unless --table-only")
    if args.table_only:
        r = json.loads((args.out / "summary.json").read_text())
        (REPO / "paper" / "generated" / "tab_a3_hepato.tex").write_text(
            "% generated by analysis/a3_hepato.py — do not edit\n" + table(r))
        return
    _DATA.update(load(args.work))   # inherited by the forked workers
    with Pool(args.procs) as pool:
        variants = dict(pool.map(run_variant, [(v, args.resamples) for v in VARIANTS]))
    r = {"amendment": "A3.1", "post_hoc": True, "resamples": args.resamples, "variants": variants,
         "probe_hepatobench": probe_hepato(args.work)}
    # verification: image groups must reproduce the primary analysis (same code, same plan seed)
    prim = json.loads((REPO / "results" / "runs" / "stage1-analysis" / "summary.json").read_text())["hypotheses"]
    diffs = [abs(a - b) for hyp in ("H1a", "H1b") for o in prim["seed_mean"][hyp]
             for a, b in zip([prim["seed_mean"][hyp][o]["diff"], *prim["seed_mean"][hyp][o]["ci95"]],
                             [variants["image"]["hypotheses"]["seed_mean"][hyp][o]["diff"],
                              *variants["image"]["hypotheses"]["seed_mean"][hyp][o]["ci95"]])]
    r["verification"] = {"max_abs_diff_seed_mean_diff_and_ci": float(max(diffs)),
                         "probe_max_abs_prob_diff": r["probe_hepatobench"]["image"]["max_abs_diff_vs_primary"]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(r, indent=1))
    (args.out / "report.md").write_text(report_md(r))
    gen = REPO / "paper" / "generated"
    (gen / "numbers_a3hepato.tex").write_text("% Generated by analysis/a3_hepato.py (A3.1, post-hoc); do not edit.\n" +
                                             "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros(r).items())))
    (gen / "tab_a3_hepato.tex").write_text("% generated by analysis/a3_hepato.py — do not edit\n" + table(r))
    print(report_md(r))


if __name__ == "__main__":
    main()
