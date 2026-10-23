"""Exploratory (not pre-registered) results for the Noul and Score decisions on the clean test sets
(paper/generated/tab_noul_score.tex), from run folders only.

    PYTHONPATH=src python analysis/noul_score.py --work $PATHJEV_WORK

Arms that answer every decision of a state: PathJev-P and Gen-SFT (seed mean over the runs analysed by
analysis/stage1_analysis.py), the released OmniMed-Jev and the Inner-Jev read-out. CONCH and the probe score only
Choice questions. The organ question is excluded (its answer is named in the question text). States are aligned
with src/pathjev/evaluate.collect, so hard targets only, exactly as in the per-run *_eval.json files (checked).
Noul: AUROC of P(yes), top-label ECE (15 bins), NLL, binary Brier (P(yes) - y)^2. Score: MAE of the expected level
(levels 0..K-1), ranked probability score, NLL. Raw probabilities, no temperature, no CIs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from pathjev import metrics
from pathjev.evaluate import collect
from stage1_analysis import CLEAN, IN_DOMAIN, UNSEEN, arm_dirs

ARMS = [("pathjev", "PathJev-P"), ("sft", "Gen-SFT"), ("omnimedjev", "OmniMed-Jev"), ("letter", "Inner-Jev")]
HIGHER = {"auroc"}
SET_NAME = {"hepatobench": "HepatoBench", "hiesd": "HiESD", "bracs": "BRACS", "sicapv2": "SICAPv2"}
NOUL = [("auroc", "AUROC$\\uparrow$"), ("ece", "ECE$\\downarrow$"), ("nll", "NLL$\\downarrow$"), ("brier", "Brier$\\downarrow$")]
SCORE = [("mae", "MAE (levels)$\\downarrow$"), ("rps", "RPS$\\downarrow$"), ("nll", "NLL$\\downarrow$")]


def score_question(entry: dict) -> tuple[str, dict]:
    p, t = np.stack(entry["probs"]), np.stack(entry["target"])
    hard = np.isclose(t.max(axis=1), 1.0)
    p, y = p[hard], t[hard].argmax(axis=1)
    if entry["type"] == "noul":
        yes = entry["ids"].index("yes")
        pyes, ybin = p[:, yes], (y == yes).astype(float)
        return "noul", {"auroc": float(roc_auc_score(ybin, pyes)), "ece": metrics.ece(p, y), "nll": metrics.nll(p, y),
                        "brier": float(((pyes - ybin) ** 2).mean())}
    levels = np.arange(p.shape[1])
    cdf_p, cdf_t = np.cumsum(p, axis=1), np.cumsum(np.eye(p.shape[1])[y], axis=1)
    return "score", {"mae": float(np.abs((p * levels).sum(axis=1) - y).mean()),
                     "rps": float(((cdf_p - cdf_t) ** 2).sum(axis=1).mean() / (p.shape[1] - 1)), "nll": metrics.nll(p, y)}


def arm_results(work: Path, repo: Path, arm: str) -> list[dict]:
    """[{(set, qid): (type, metrics)}] per seed; checked against the run's *_eval.json."""
    out = []
    test_dir = work / "eval_sets" / "stage1-test"
    for tdir, _ in arm_dirs(work, repo, arm, interim=False):
        res = {}
        for s in CLEAN:
            table = collect([test_dir / f"{s}.jsonl"], [tdir / f"{s}_decisions.jsonl"])
            ref = json.loads((tdir / f"{s}_eval.json").read_text())
            for (_, qid), entry in table.items():
                if entry["type"] not in ("noul", "score"):
                    continue
                if entry["missing"]:
                    raise ValueError(f"{tdir.name} {s}/{qid}: {entry['missing']} missing predictions")
                kind, m = score_question(entry)
                r = ref[f"{s}/{qid}"]
                pairs = [(m["nll"], r["nll"]), (m["ece"], r["ece"])] if kind == "noul" else \
                        [(m["mae"], r["mae_expected_level"]), (m["rps"], r["rps"]), (m["nll"], r["choice_nll"])]
                if any(abs(a - b) > 1e-6 for a, b in pairs):
                    raise ValueError(f"{tdir.name} {s}/{qid}: recomputed metrics differ from {s}_eval.json")
                res[(s, qid)] = (kind, m)
        out.append(res)
    return out


def _fmt(vals: list[float]) -> str:
    if len(vals) < 2:
        return f"{vals[0]:.3f}"
    return f"{np.mean(vals):.3f}$_{{\\pm{np.std(vals, ddof=1):.3f}}}$".replace("\\pm0.", "\\pm.")


def _label(s: str, qid: str) -> str:
    return f"{SET_NAME[s]}: " + qid.replace("present:", "").replace("_", " ")


def table(res: dict[str, list[dict]], kind: str, cols: list) -> str:
    keys = [k for k, (t, _) in res["pathjev"][0].items() if t == kind]
    keys.sort(key=lambda k: (CLEAN.index(k[0]), k[1]))
    lines = [r"\begin{tabular}{@{}ll" + "c" * len(cols) + "@{}}", r"\toprule",
             "Question & Method & " + " & ".join(n for _, n in cols) + r" \\", r"\midrule"]
    groups = [(f"{SET_NAME[s]}: {q}", [(s, q)]) for s, q in keys]
    for gname, sets in (("Unseen organs (average)", UNSEEN), ("In domain (average)", IN_DOMAIN)):
        sub = [k for k in keys if k[0] in sets]
        if len(sub) > 1:                                  # a one-question macro would repeat that question
            groups.append((gname, sub))
    for gi, (gname, sub) in enumerate(groups):
        label = _label(*sub[0]) if len(sub) == 1 else gname
        vals = {arm: {m: [float(np.mean([seed[k][1][m] for k in sub])) for seed in res[arm]] for m, _ in cols}
                for arm, _ in ARMS}
        txt = {arm: {m: _fmt(vals[arm][m]) for m, _ in cols} for arm, _ in ARMS}
        for m, _ in cols:                                  # best value per block and column in bold
            pick = max if m in HIGHER else min
            best = txt[pick(vals, key=lambda a: np.mean(vals[a][m]))][m].split("$")[0]
            for arm, _ in ARMS:
                if txt[arm][m].split("$")[0] == best:
                    txt[arm][m] = f"\\textbf{{{txt[arm][m]}}}"
        for i, (arm, aname) in enumerate(ARMS):
            lines.append((label if i == 0 else "") + f" & {aname} & " + " & ".join(txt[arm][m] for m, _ in cols) + r" \\")
        lines.append(r"\midrule" if gi < len(groups) - 1 else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, type=Path)
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--out", type=Path, default=repo / "paper" / "generated" / "tab_noul_score.tex")
    args = ap.parse_args()
    res = {arm: arm_results(args.work, repo, arm) for arm, _ in ARMS}
    for kind, cols in (("noul", NOUL), ("score", SCORE)):
        for gname, sets in (("unseen", UNSEEN), ("in_domain", IN_DOMAIN)):
            print(kind, gname, {arm: {m: round(float(np.mean([np.mean([v[1][m] for k, v in seed.items()
                                                                        if v[0] == kind and k[0] in sets])
                                                               for seed in res[arm]])), 3) for m, _ in cols}
                                 for arm, _ in ARMS})
    body = table(res, "noul", NOUL) + "\n\\medskip\n\n" + table(res, "score", SCORE)
    args.out.write_text("% generated by analysis/noul_score.py from run folders — do not edit\n" + body)
    print(args.out)


if __name__ == "__main__":
    main()
