"""Generate the paper's result tables (paper/generated/*.tex) from the final analysis outputs only.

    python analysis/paper_tables.py --analysis results/runs/stage1-analysis/summary.json \
        --temperature results/runs/stage1-temperature/summary.json \
        --wsi $PATHJEV_WORK/wsi_demo/demo-hiesd --out paper/generated

Never edit the generated .tex by hand (CLAUDE.md §9). Interim analyses are refused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROWS = [("pathjev", "PathJev-P (ours)"), ("sft", "(b) Gen-SFT"), ("omnimedjev", "(a) OmniMed-Jev"),
        ("conch", "(c) CONCH"), ("letter", "(e) Inner-Jev"), ("probe", "(d) Linear probe$^\\dagger$")]
VARIANT_ROWS = [("omnimedjev-no-other", "\\quad (a) without ``other''"), ("omnimedjev-permuted", "\\quad (a) permuted")]
AGGS = [("clean", "Clean average"), ("unseen", "Unseen organs"), ("in_domain", "In domain"), ("seen_source", "Seen source")]


def cell(row: dict, agg: str, met: str) -> str:
    if agg not in row:
        return "--"
    v = row[agg][met]
    seeds = row[agg].get("per_seed", {}).get(met, [])
    if len(seeds) < 2:
        return f"{v:.3f}"
    return f"{v:.3f}$_{{\\pm{np.std(seeds, ddof=1):.3f}}}$".replace("\\pm0.", "\\pm.")


GROUPS = [("Trained on our data", ["pathjev", "sft"]), ("Zero-shot", ["omnimedjev", "conch", "letter"]),
          ("Supervised reference", ["probe"])]
LABEL = dict(ROWS)


def _best(r: dict, agg: str, met: str) -> float | None:
    """Best value among the language-model-based methods (the probe is a reference and is not ranked)."""
    vals = [r["table"][k][agg][met] for k in ("pathjev", "sft", "omnimedjev", "conch", "letter") if agg in r["table"][k]]
    if not vals:
        return None
    return max(vals) if met == "balanced_accuracy" else min(vals)


def main_table(r: dict) -> str:
    """Rows grouped by kind of method; best language-model-based value per column in bold."""
    ncol = 1 + 2 * len(AGGS)
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{name}}}" for _, name in AGGS)
    sub = " & ".join(["bAcc$\\uparrow$ & ECE$\\downarrow$"] * len(AGGS))
    lines = ["\\begin{tabular}{@{}l" + "cc" * len(AGGS) + "@{}}", "\\toprule", f" & {head} \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(AGGS))), f"Method & {sub} \\\\"]
    best = {(a, m): _best(r, a, m) for a, _ in AGGS for m in ("balanced_accuracy", "ece")}
    for g, (title, keys) in enumerate(GROUPS):
        lines += ["\\midrule" if g == 0 else "\\addlinespace[3pt]",
                  f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\"]
        for key in keys:
            row = r["table"][key]
            cells = []
            for a, _ in AGGS:
                for m in ("balanced_accuracy", "ece"):
                    c = cell(row, a, m)
                    if key != "probe" and a in row and best[(a, m)] is not None and abs(row[a][m] - best[(a, m)]) < 5e-4:
                        c = f"\\textbf{{{c}}}"
                    cells.append(c)
            lines.append(f"\\hspace{{0.6em}}{LABEL[key]} & " + " & ".join(cells) + " \\\\")
            if key == "omnimedjev":
                for vkey, vname in VARIANT_ROWS:
                    c = r["variants"].get(vkey, {}).get("clean")
                    lines.append("\\hspace{1.4em}" + vname.replace("\\quad ", "").replace("(a) ", "")
                                 + (f" & {c['balanced_accuracy']:.3f} & {c['ece']:.3f}" if c else " & -- & --")
                                 + " & & " * (len(AGGS) - 1) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


LM_KEYS = ("pathjev", "sft", "omnimedjev", "conch", "letter")
HIGHER_BETTER = {"balanced_accuracy", "accuracy", "macro_f1", "auroc"}


def bold_best(vals: dict[str, float], metric: str, fmt: str = "{:.3f}") -> dict[str, str]:
    """Formatted cells; the best language-model-based value (as printed) in bold. The probe is never bold."""
    out = {k: fmt.format(v) for k, v in vals.items()}
    lm = [k for k in vals if k in LM_KEYS]
    if lm:
        pick = max if metric in HIGHER_BETTER else min
        best = out[pick(lm, key=lambda k: vals[k])]
        out.update({k: f"\\textbf{{{out[k]}}}" for k in lm if out[k] == best})
    return out


def full_table(r: dict) -> str:
    """Secondary (A1): the same macro metrics on the full test files; every arm except Gen-SFT."""
    aggs = AGGS[:3]
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{name}}}" for _, name in aggs)
    lines = ["\\begin{tabular}{@{}l" + "cc" * len(aggs) + "@{}}", "\\toprule", f" & {head} \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(aggs))),
             "Method & " + " & ".join(["bAcc$\\uparrow$ & ECE$\\downarrow$"] * len(aggs)) + " \\\\", "\\midrule"]
    full = r["full_test_secondary"]
    means = {}
    for key, _ in ROWS:
        runs = [v for k, v in full.items() if k == key or k.startswith(f"{key}-s")]
        if runs:
            means[key] = {(a, m): float(np.mean([x[a][m] for x in runs])) for a, _ in aggs
                          for m in ("balanced_accuracy", "ece")}
    cols = {c: bold_best({k: v[c] for k, v in means.items()}, c[1]) for c in next(iter(means.values()))}
    for key, name in ROWS:
        if key not in means:
            continue
        if key == "probe":
            lines.append("\\midrule")
        lines.append(name + " & " + " & ".join(cols[c][key] for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


SHORT = {"pathjev": "PathJev-P", "sft": "Gen-SFT", "omnimedjev": "OmniMed-Jev", "conch": "CONCH",
         "letter": "Inner-Jev", "probe": "Probe$^\\dagger$"}
SET_NAMES = [("hepatobench", "HepatoBench"), ("hiesd", "HiESD"), ("bracs", "BRACS"), ("sicapv2", "SICAPv2"),
             ("spider-colorectal", "SPIDER colorectal"), ("spider-thorax", "SPIDER thorax"),
             ("spider-breast", "SPIDER breast"), ("spider-skin", "SPIDER skin")]
SET_GROUP = {"hepatobench": "Unseen organs", "hiesd": "Unseen organs", "bracs": "In domain", "sicapv2": "In domain"}


def perset_table(r: dict) -> str:
    """Per test set (no macro averaging): balanced accuracy / ECE / NLL per method; seed mean for trained arms.
    Best language-model-based value per row in bold; the probe column is set apart."""
    keys = [k for k, _ in ROWS]
    lines = ["\\begin{tabular}{@{}ll" + "c" * len(keys) + "@{}}", "\\toprule",
             "Set & Metric & " + " & ".join(SHORT[k] for k in keys) + " \\\\"]
    group = None
    for s, name in SET_NAMES:
        g = SET_GROUP.get(s, "Seen source")
        if g != group:
            lines += ["\\midrule", f"\\multicolumn{{{len(keys) + 2}}}{{@{{}}l}}{{\\textit{{{g}}}}} \\\\"]
            group = g
        else:
            lines.append("\\addlinespace[2pt]")
        for i, (met, mn) in enumerate((("balanced_accuracy", "bAcc$\\uparrow$"), ("ece", "ECE$\\downarrow$"),
                                       ("nll", "NLL$\\downarrow$"))):
            vals = {k: r["table"][k][s][met] for k in keys if s in r["table"][k]}
            cells = bold_best(vals, met)
            lines.append((name if i == 0 else "") + f" & {mn} & " + " & ".join(cells.get(k, "--") for k in keys) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def scaled_table(r: dict) -> str:
    """Per clean test set and method: ECE / Brier / NLL raw and after the validation temperature (plan §7),
    seed mean for trained arms. Best language-model-based value per set and column in bold."""
    lines = ["\\begin{tabular}{@{}ll" + "c" * 6 + "@{}}", "\\toprule",
             " & & \\multicolumn{3}{c}{Raw} & \\multicolumn{3}{c}{Validation temperature} \\\\",
             "\\cmidrule(lr){3-5} \\cmidrule(l){6-8}",
             "Set & Method & ECE$\\downarrow$ & Brier$\\downarrow$ & NLL$\\downarrow$ & ECE$\\downarrow$ & Brier$\\downarrow$"
             " & NLL$\\downarrow$ \\\\", "\\midrule"]
    clean = [x for x in SET_NAMES if not x[0].startswith("spider")]
    for si, (s, name) in enumerate(clean):
        keys = [k for k, _ in ROWS if s in r["table"][k]]
        cols = []
        for scaled in (False, True):
            for m in ("ece", "brier", "nll"):
                vals = {}
                for k in keys:
                    row = r["table"][k][s]
                    src = row.get("temperature_scaled") if scaled else row
                    if src is not None:
                        vals[k] = src[m]
                cols.append(bold_best(vals, m))
        for i, k in enumerate(keys):
            if k == "probe":
                lines.append("\\cmidrule(l){2-8}")
            lines.append((name if i == 0 else "") + f" & {SHORT[k]} & " + " & ".join(c.get(k, "--") for c in cols) + " \\\\")
        lines.append("\\midrule" if si < len(clean) - 1 else "\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def temperature_table(t: dict, metric: str = "nll") -> str:
    """Unseen-organ macro of one metric after one temperature per method, fitted on different data.
    Best language-model-based value per column in bold; the probe is listed apart."""
    variants = [("raw", "none"), ("T-val", "val.\\ mix"), ("T-in", "in-domain val."), ("T-cross", "other unseen organ"),
                ("T-oracle", "same set (oracle)")]
    lm = [k for k, _ in ROWS if k in t["arms"] and k != "probe"]
    best = {v: min(t["arms"][k]["seed_mean_unseen"][v][metric] for k in lm) for v, _ in variants}
    lines = ["\\begin{tabular}{@{}l" + "c" * len(variants) + "@{}}", "\\toprule",
             f"& \\multicolumn{{{len(variants)}}}{{c}}{{Temperature fitted on}} \\\\",
             f"\\cmidrule(l){{2-{len(variants) + 1}}}",
             "Method & " + " & ".join(n for _, n in variants) + " \\\\", "\\midrule"]
    for key, name in ROWS:
        if key not in t["arms"]:
            continue
        if key == "probe":
            lines.append("\\midrule")
        m = t["arms"][key]["seed_mean_unseen"]
        cells = []
        for v, _ in variants:
            c = f"{m[v][metric]:.2f}" if metric == "nll" else f"{m[v][metric]:.3f}"
            if key != "probe" and abs(m[v][metric] - best[v]) < (5e-3 if metric == "nll" else 5e-4):
                c = f"\\textbf{{{c}}}"
            cells.append(c)
        lines.append(name + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def wsi_table(wsi_dir: Path) -> str:
    """Per slide: patient (official HiESD map), tiles, tiles with a pure annotation label, clusters, tiles run through
    PathJev-P, and for the adenocarcinoma presence question AUROC, sensitivity and specificity at P(yes) > 0.5, each
    for the cluster-propagated probabilities and for PathJev-P on every tile."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from pathjev.groups import group_maps
    patient = group_maps()["hiesd"]
    lines = ["\\begin{tabular}{@{}llrrrrcccccc@{}}", "\\toprule",
             "& & & & & PathJev-P & \\multicolumn{2}{c}{AUROC} & \\multicolumn{2}{c}{Sensitivity}"
             " & \\multicolumn{2}{c}{Specificity} \\\\",
             "\\cmidrule(lr){7-8} \\cmidrule(lr){9-10} \\cmidrule(l){11-12}",
             "Slide & Patient & Tiles & Labelled & Clusters & tiles & clusters & all tiles & clusters & all tiles"
             " & clusters & all tiles \\\\", "\\midrule"]
    for f in sorted(wsi_dir.glob("*/demo_stats.json")):
        s = json.loads(f.read_text())
        auc, rates = s.get("carcinoma_auroc", {}), s["carcinoma_rates_at_0.5"]
        get = lambda d, k: (f"{d[k]:.3f}" if d.get(k) is not None else "--")
        cells = [get(auc, "cluster_propagated"), get(auc, "dense")]
        for k in ("sensitivity", "specificity"):
            cells += [get(rates.get("cluster_propagated", {}), k), get(rates.get("dense", {}), k)]
        pid = patient.get(s["slide"], "?").replace("Patient", "P")
        lines.append(f"{s['slide'][:8]} & {pid} & {s['tiles']:,} & {s['tiles_with_mask_label']:,} & {s['k']} & "
                     f"{s['pathjev_calls_cluster']:,} & " .replace(",", "{,}") + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", required=True, type=Path)
    ap.add_argument("--temperature", type=Path)
    ap.add_argument("--wsi", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--allow-interim", action="store_true", help="draft tables only; never for the paper")
    args = ap.parse_args()
    r = json.loads(args.analysis.read_text())
    if r["interim"] and not args.allow_interim:
        raise ValueError("interim analysis: tables are generated from the final run only")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tab_main.tex").write_text(main_table(r))
    (args.out / "tab_full.tex").write_text(full_table(r))
    (args.out / "tab_perset.tex").write_text(perset_table(r))
    (args.out / "tab_scaled.tex").write_text(scaled_table(r))
    if args.temperature:
        tt = json.loads(args.temperature.read_text())
        (args.out / "tab_temperature.tex").write_text(temperature_table(tt, "nll"))
        (args.out / "tab_temperature_brier.tex").write_text(temperature_table(tt, "brier"))
    if args.wsi:
        (args.out / "tab_wsi.tex").write_text(wsi_table(args.wsi))
    print("tables ->", args.out)


if __name__ == "__main__":
    main()
