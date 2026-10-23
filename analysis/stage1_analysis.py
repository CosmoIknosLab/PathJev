"""Stage 1 Phase 4 analysis, exactly as pre-registered in docs/stage1_analysis_plan.md (§3–§5, §7, A1).

    python analysis/stage1_analysis.py --work $PATHJEV_WORK --out results/runs/stage1-analysis [--resamples 10000]
    python analysis/stage1_analysis.py ... --interim        # whatever seeds exist; output marked INTERIM

Reads only run folders and the configuration-selection files (results/runs/stage1-config/*.json);
never chooses anything from test data. Per arm and seed, for the main Choice question of each
test set: balanced accuracy and ECE (15 bins, top label), macro-averaged over the clean sets and
the pre-specified breakdowns. H1a/H1b: paired cluster bootstrap (groups from the state files,
resampled within each test set, macro recomputed), one-sided p-values, Holm over the 4
comparisons of each hypothesis; a hypothesis is supported only if it holds for the seed-mean
metric and in >= 2 of 3 seeds. §5 GATE rule evaluated as the agent's recommendation.
Groups follow plan §3 (patient, else slide, else sample) through `pathjev.groups.state_group`: HiESD slides
are mapped to patients with the official map (data/manifests/hiesd_patient_slide_mapping.csv). Test states in
data/manifests/stage1_test_exclusions.csv (duplicates of training images) are dropped for every arm.
§7 secondary: one temperature per arm (and seed), fitted on the validation mix (all one-hot
Choice decisions), applied to the test distributions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathjev.groups import excluded_state_ids, state_group  # noqa: E402

MAIN_Q = {"bracs": "lesion", "sicapv2": "gleason_pattern", "hepatobench": "tissue", "hiesd": "tissue",
          "spider-colorectal": "tissue", "spider-thorax": "tissue", "spider-breast": "tissue", "spider-skin": "tissue"}
UNSEEN, IN_DOMAIN = ["hepatobench", "hiesd"], ["bracs", "sicapv2"]
CLEAN = UNSEEN + IN_DOMAIN
SEEN = [s for s in MAIN_Q if s.startswith("spider")]
BINS, EPS, SEED = 15, 1e-12, 20261016
ZERO_SHOT = {"omnimedjev": ("eval_s1-omnimedjev_OmniMed-Jev-4B", "eval_val-s1-omnimedjev_OmniMed-Jev-4B"),
             "conch": ("eval_s1-conch", "eval_val-s1-conch"),
             "letter": ("eval_s1-letter", "eval_val-s1-letter"),
             "probe": ("eval_s1-probe", "eval_val-s1-probe")}
FULL = {"omnimedjev": "eval_s1-omnimedjev-full_OmniMed-Jev-4B", "conch": "eval_s1-conch-full",
        "letter": "eval_s1-letter-full", "probe": "eval_s1-probe-full"}
VARIANTS = {"omnimedjev-no-other": ("eval_s1-omnimedjev-no-other_OmniMed-Jev-4B", "stage1-test-no-other"),
            "omnimedjev-permuted": ("eval_s1-omnimedjev-permuted_OmniMed-Jev-4B", "stage1-test-permuted")}
LABEL = {"pathjev": "PathJev-P", "sft": "(b) Gen-SFT", "omnimedjev": "(a) OmniMed-Jev", "conch": "(c) CONCH",
         "letter": "(e) Inner-Jev", "probe": "(d) probe", "omnimedjev-no-other": "(a) no 'other'",
         "omnimedjev-permuted": "(a) permuted"}
H1A = ["omnimedjev", "sft", "conch", "letter"]          # ECE: PathJev lower than each
H1B = {"sft": -0.02, "omnimedjev": 0.0, "conch": 0.0, "letter": 0.0}   # bal. acc.: diff > margin


# ---------------------------------------------------------------- loading
def _rows(path: Path):
    with path.open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_main(states: Path, decisions: Path, qid: str) -> dict:
    """Aligned arrays for one (set, main question). Missing predictions: uniform (plan §3)."""
    preds = {r["state_id"]: r["distributions"] for r in _rows(decisions)}
    probs, y, groups, ids, missing = [], [], [], None, 0
    excluded = excluded_state_ids()
    for s in _rows(states):
        if s["state_id"] in excluded:
            continue
        d = next((d for d in s["decisions"] if d["qid"] == qid), None)
        if d is None:
            continue
        cids = [c["id"] for c in d["candidates"]]
        ids = ids or cids
        if cids != ids:
            raise ValueError(f"{states}: candidate order differs between states")
        dist = preds.get(s["state_id"], {}).get(qid)
        if dist is None:
            missing += 1
            p = np.full(len(ids), 1.0 / len(ids))
        else:
            p = np.array([dist.get(i, 0.0) for i in ids], dtype=float)
            p = p / p.sum() if p.sum() > 0 else np.full(len(ids), 1.0 / len(ids))
        probs.append(p)
        y.append(cids.index(max(d["target"], key=d["target"].get)))
        groups.append(state_group(s))
    if not probs:
        raise ValueError(f"{qid} not in {states}")
    return {"probs": np.stack(probs), "y": np.array(y), "groups": np.array(groups), "missing": missing}


def load_val_choice(states: Path, decisions: Path, datasets=None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """One-hot Choice decisions of the validation mix (optionally only `datasets`), {K: (log-probs (n, K), labels)}."""
    preds = {r["state_id"]: r["distributions"] for r in _rows(decisions)}
    by_k: dict[int, tuple[list, list]] = {}
    for s in _rows(states):
        if datasets is not None and s.get("meta", {}).get("dataset") not in datasets:
            continue
        for d in s["decisions"]:
            if d["type"] != "choice" or d.get("target_kind", "one_hot") != "one_hot":
                continue
            dist = preds.get(s["state_id"], {}).get(d["qid"])
            if dist is None:
                continue
            ids = [c["id"] for c in d["candidates"]]
            p = np.array([dist.get(i, 0.0) for i in ids], dtype=float)
            lp, ys = by_k.setdefault(len(ids), ([], []))
            lp.append(np.log(p / p.sum() + EPS))
            ys.append(ids.index(max(d["target"], key=d["target"].get)))
    return {k: (np.stack(lp), np.array(ys)) for k, (lp, ys) in by_k.items()}


# ---------------------------------------------------------------- metrics on index arrays
class Item:
    """Pre-computed per-item quantities so that bootstrap metrics are bincounts."""

    def __init__(self, probs: np.ndarray, y: np.ndarray, groups: np.ndarray):
        self.probs, self.y, self.k = probs, y, probs.shape[1]
        self.conf = probs.max(axis=1)
        self.correct = (probs.argmax(axis=1) == y).astype(float)
        self.slot = np.minimum((self.conf * BINS).astype(int), BINS - 1)
        _, inv = np.unique(groups, return_inverse=True)
        self.members = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]

    def ece(self, idx: np.ndarray) -> float:
        s = self.slot[idx]
        gap = np.bincount(s, self.conf[idx], BINS) - np.bincount(s, self.correct[idx], BINS)
        return float(np.abs(gap).sum() / len(idx))

    def nll(self, idx: np.ndarray) -> float:
        return float(-np.log(self.probs[idx, self.y[idx]] + EPS).mean())

    def brier(self, idx: np.ndarray) -> float:
        p = self.probs[idx]
        onehot = np.eye(self.k)[self.y[idx]]
        return float(((p - onehot) ** 2).sum(axis=1).mean())

    def bacc(self, idx: np.ndarray) -> float:
        cnt = np.bincount(self.y[idx], minlength=self.k)
        hit = np.bincount(self.y[idx], self.correct[idx], self.k)
        present = cnt > 0
        return float((hit[present] / cnt[present]).mean())


METRIC = {"ece": Item.ece, "balanced_accuracy": Item.bacc}
DESCRIPTIVE = {**METRIC, "nll": Item.nll, "brier": Item.brier}   # secondary metrics (plan §3), no tests


class ResamplePlan:
    """Cluster bootstrap within each test set; regenerated from a fixed seed, so every comparison and
    every arm sees the same resampled indices (paired) without storing them."""

    def __init__(self, ref: dict[str, Item], resamples: int):
        self.members = {s: it.members for s, it in ref.items()}
        self.resamples = resamples

    def __iter__(self):
        rng = np.random.default_rng(SEED)
        for _ in range(self.resamples):
            yield {s: np.concatenate([m[g] for g in rng.integers(0, len(m), len(m))])
                   for s, m in self.members.items()}


def macro_value(arm_items: list[dict[str, Item]], sets: list[str], metric: str, idx_by_set=None) -> float:
    """Seed-mean of the macro over `sets`; arm_items = one {set: Item} per seed."""
    fn = DESCRIPTIVE[metric]
    vals = []
    for items in arm_items:
        per = [fn(items[s], idx_by_set[s] if idx_by_set else np.arange(len(items[s].y))) for s in sets]
        vals.append(np.mean(per))
    return float(np.mean(vals))


def compare(a: list[dict[str, Item]], b: list[dict[str, Item]], sets, metric, plan, resamples, margin, lower_better):
    """Point diff (a − b), 95% CI and one-sided p-value for H: diff < −margin... framed per metric.

    ECE (lower_better): H1 is diff < 0. Balanced accuracy: H1 is diff > margin.
    """
    point = macro_value(a, sets, metric) - macro_value(b, sets, metric)
    boot = np.empty(resamples)
    for r, idx in enumerate(plan):
        boot[r] = macro_value(a, sets, metric, idx) - macro_value(b, sets, metric, idx)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    tail = np.mean(boot >= 0) if lower_better else np.mean(boot <= margin)
    p = float(min(1.0, (tail * resamples + 1) / (resamples + 1)))
    return {"diff": point, "ci95": [float(lo), float(hi)], "p_one_sided": p, "margin": margin}


def holm(p: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    order, rejected, still = sorted(p, key=p.get), {}, True
    for rank, name in enumerate(order):
        still = still and p[name] <= alpha / (len(order) - rank)
        rejected[name] = still
    return rejected


# ---------------------------------------------------------------- temperature (§7)
def fit_temperature(by_k: dict[int, tuple[np.ndarray, np.ndarray]]) -> float:
    """One temperature minimising the summed NLL over all validation Choice decisions (grid search)."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(200), 600))
    nll = np.zeros(len(grid))
    for lp, y in by_k.values():
        for i, t in enumerate(grid):
            z = lp / t
            z = z - z.max(axis=1, keepdims=True)
            nll[i] -= (z[np.arange(len(y)), y] - np.log(np.exp(z).sum(axis=1))).sum()
    return float(grid[int(np.argmin(nll))])


def scaled(probs: np.ndarray, t: float) -> np.ndarray:
    z = np.log(probs + EPS) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- arms
def trained_runs(repo: Path, arm: str) -> list[str]:
    best = json.loads((repo / "results" / "runs" / "stage1-config" / f"{'jev' if arm == 'pathjev' else 'sft'}.json")
                      .read_text())["best_run"]
    return [best[:-3] + f"-s{s}" for s in (1, 2, 3)]


def arm_dirs(work: Path, repo: Path, arm: str, interim: bool) -> list[tuple[Path, Path | None]]:
    """[(test dir, val decisions file)] per seed."""
    runs = work / "runs"
    if arm in ZERO_SHOT:
        t, v = ZERO_SHOT[arm]
        return [(runs / t, runs / v / "val_decisions.jsonl")]
    out = []
    for run in trained_runs(repo, arm):
        best = work / "runs" / f"train_{run}" / "selection" / "best.txt"
        if not best.exists():
            if interim:
                continue
            raise FileNotFoundError(best)
        ck = Path(best.read_text().strip()).name
        out.append((runs / f"eval_test-{run}_{ck}", runs / f"eval_val-{run}_{ck}" / "val_decisions.jsonl"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--interim", action="store_true", help="use whatever seeds/sets exist; output marked INTERIM")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    test_dir = args.work / "eval_sets" / "stage1-test"
    val_states = args.work / "mixes" / "stage1-p" / "val.jsonl"
    arms = ["pathjev", "sft", "omnimedjev", "conch", "letter", "probe"]

    data, temps, notes = {}, {}, []
    for arm in arms:
        data[arm], temps[arm] = [], []
        for tdir, vfile in arm_dirs(args.work, repo, arm, args.interim):
            per_set = {}
            for s, q in MAIN_Q.items():
                dec = tdir / f"{s}_decisions.jsonl"
                if not dec.exists():
                    if args.interim:
                        continue
                    raise FileNotFoundError(dec)
                d = load_main(test_dir / f"{s}.jsonl", dec, q)
                if d["missing"] > 0.01 * len(d["y"]):
                    raise ValueError(f"{dec}: {d['missing']} missing (> 1 %): run invalid (plan §3)")
                if d["missing"]:
                    notes.append(f"{arm} {tdir.name} {s}: {d['missing']} missing → uniform")
                per_set[s] = d
            t = None
            if vfile is not None and vfile.exists():
                t = fit_temperature(load_val_choice(val_states, vfile))
            elif not args.interim:
                raise FileNotFoundError(vfile)
            data[arm].append(per_set)
            temps[arm].append(t)

    def items(arm, sets, temperature=False):
        out = []
        for per_set, t in zip(data[arm], temps[arm]):
            if not all(s in per_set for s in sets):
                return None
            out.append({s: Item(scaled(per_set[s]["probs"], t) if temperature and t else per_set[s]["probs"],
                                per_set[s]["y"], per_set[s]["groups"]) for s in sets})
        return out or None

    report = {"interim": args.interim, "resamples": args.resamples,
              "seeds": {a: len(data[a]) for a in arms}, "temperatures": temps, "notes": notes,
              "groups": {"hiesd": "patient (data/manifests/hiesd_patient_slide_mapping.csv)",
                         "other sets": "meta.group (patient for BRACS/SICAPv2, slide for SPIDER, image for HepatoBench)"},
              "excluded_states": sorted(excluded_state_ids())}

    # descriptive table: per arm, per seed-mean, per set and macro
    table = {}
    for arm in arms:
        row = {}
        for s in MAIN_Q:
            it = items(arm, [s])
            if it:
                row[s] = {m: macro_value(it, [s], m) for m in DESCRIPTIVE}
                its = items(arm, [s], temperature=True)
                if its and all(t for t in temps[arm]):
                    row[s]["temperature_scaled"] = {m: macro_value(its, [s], m) for m in DESCRIPTIVE}
        for name, sets in (("clean", CLEAN), ("unseen", UNSEEN), ("in_domain", IN_DOMAIN), ("seen_source", SEEN)):
            it = items(arm, sets)
            if it:
                row[name] = {m: macro_value(it, sets, m) for m in DESCRIPTIVE}
                row[name]["per_seed"] = {m: [macro_value([x], sets, m) for x in it] for m in METRIC}
                its = items(arm, sets, temperature=True)
                if its and all(t for t in temps[arm]):
                    row[name]["temperature_scaled"] = {m: macro_value(its, sets, m) for m in DESCRIPTIVE}
        table[arm] = row
    report["table"] = table

    # hypotheses on the clean macro (primary) — needs every arm on every clean set
    base = {a: items(a, CLEAN) for a in arms}
    if all(base[a] for a in ["pathjev"] + H1A):
        ref = base["pathjev"][0]
        for a in arms:   # paired bootstrap needs identical items and groups in every arm
            for x in base[a] or []:
                for s in CLEAN:
                    if not (np.array_equal(x[s].y, ref[s].y) and len(x[s].members) == len(ref[s].members)):
                        raise ValueError(f"{a}/{s}: items differ from PathJev-P; cannot pair")
        plan = ResamplePlan(ref, args.resamples)
        n_seed = len(base["pathjev"])

        def run_h(pj, others):
            ha = {o: compare(pj, others[o], CLEAN, "ece", plan, args.resamples, 0.0, True) for o in H1A}
            hb = {o: compare(pj, others[o], CLEAN, "balanced_accuracy", plan, args.resamples, H1B[o], False)
                  for o in H1B}
            ra, rb = holm({o: v["p_one_sided"] for o, v in ha.items()}), holm({o: v["p_one_sided"] for o, v in hb.items()})
            for o in ha:
                ha[o]["holm_rejected"] = ra[o]
            for o in hb:
                hb[o]["holm_rejected"] = rb[o]
            return {"H1a": ha, "H1b": hb, "H1a_all": all(ra.values()), "H1b_all": all(rb.values())}

        hyp = {"seed_mean": run_h(base["pathjev"], base)}
        for k in range(n_seed):   # seed k of PathJev vs seed k of Gen-SFT (if present), zero-shot arms unchanged
            others = dict(base)
            others["sft"] = [base["sft"][min(k, len(base["sft"]) - 1)]]
            hyp[f"seed{k + 1}"] = run_h([base["pathjev"][k]], others)
        seeds = [hyp[f"seed{k + 1}"] for k in range(n_seed)]
        for h in ("H1a", "H1b"):
            hyp[f"{h}_supported"] = bool(hyp["seed_mean"][f"{h}_all"] and sum(s[f"{h}_all"] for s in seeds) >= 2)
        # H1c: gap to the probe (reference), reported only
        if base["probe"]:
            hyp["H1c"] = {m: compare(base["pathjev"], base["probe"], CLEAN, m, plan, args.resamples, 0.0, m == "ece")
                          for m in METRIC}
        report["hypotheses"] = hyp

        # §5 GATE rule (recommendation)
        bacc_c = compare(base["pathjev"], base["conch"], CLEAN, "balanced_accuracy", plan, args.resamples, 0.0, False)
        h1a = hyp["seed_mean"]["H1a"]
        cond1 = (h1a["conch"]["holm_rejected"] and h1a["sft"]["holm_rejected"]) or bacc_c["ci95"][0] > 0
        unseen_ece = {a: table[a]["unseen"]["ece"] for a in ["omnimedjev", "conch", "letter"]}
        best_zs = min(unseen_ece, key=unseen_ece.get)
        cond2 = table["pathjev"]["unseen"]["ece"] <= unseen_ece[best_zs] + 0.02
        in_only = (table["pathjev"]["unseen"]["balanced_accuracy"] <= max(
            table[a]["unseen"]["balanced_accuracy"] for a in ["omnimedjev", "conch", "letter"]))
        report["gate"] = {"pathjev_vs_conch_bacc": bacc_c, "cond_effect": bool(cond1),
                          "unseen_ece_pathjev": table["pathjev"]["unseen"]["ece"],
                          "best_zero_shot_unseen": [best_zs, unseen_ece[best_zs]], "cond_unseen_ece": bool(cond2),
                          "gains_only_in_domain": bool(in_only),
                          "recommendation": ("go to Stage 2" if cond1 and cond2 and not in_only else
                                             "incremental only" if cond1 and in_only else
                                             "stop / redirect" if not cond1 else "discuss")}

    # descriptive: (a) variants (their own state files) and full test files (A1 secondary)
    var = {}
    for name, (rdir, sdir) in VARIANTS.items():
        row = {}
        for s in CLEAN:
            dec = args.work / "runs" / rdir / f"{s}_decisions.jsonl"
            if dec.exists():
                d = load_main(args.work / "eval_sets" / sdir / f"{s}.jsonl", dec, MAIN_Q[s])
                it = Item(d["probs"], d["y"], d["groups"])
                row[s] = {m: METRIC[m](it, np.arange(len(d["y"]))) for m in METRIC}
        if len(row) == len(CLEAN):
            row["clean"] = {m: float(np.mean([row[s][m] for s in CLEAN])) for m in METRIC}
        var[name] = row
    report["variants"] = var
    full = {}
    full_dirs = dict(FULL)
    for k, run in enumerate(trained_runs(repo, "pathjev")):
        best = args.work / "runs" / f"train_{run}" / "selection" / "best.txt"
        if best.exists():
            full_dirs[f"pathjev-s{k + 1}"] = f"eval_testfull-{run}_{Path(best.read_text().strip()).name}"
    for name, rdir in full_dirs.items():
        row = {}
        for s, q in MAIN_Q.items():
            dec = args.work / "runs" / rdir / f"{s}_decisions.jsonl"
            if dec.exists():
                d = load_main(args.work / "eval_sets" / "stage1-test-full" / f"{s}.jsonl", dec, q)
                it = Item(d["probs"], d["y"], d["groups"])
                row[s] = {m: METRIC[m](it, np.arange(len(d["y"]))) for m in METRIC} | {"n": int(len(d["y"]))}
        for agg, sets in (("clean", CLEAN), ("unseen", UNSEEN), ("in_domain", IN_DOMAIN)):
            if all(s in row for s in sets):
                row[agg] = {m: float(np.mean([row[s][m] for s in sets])) for m in METRIC}
        full[name] = row
    report["full_test_secondary"] = full

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(report, indent=1))
    (args.out / "report.md").write_text(render(report))
    print((args.out / "report.md").read_text())


def render(r: dict) -> str:
    f = lambda x: "—" if x is None else f"{x:.3f}"
    lines = ["# Stage 1 analysis" + (" — INTERIM (not for the paper)" if r["interim"] else ""), "",
             f"Seeds: {r['seeds']}. Resamples: {r['resamples']}. Temperatures (val): "
             + ", ".join(f"{a}={[None if t is None else round(t, 3) for t in ts]}" for a, ts in r["temperatures"].items()), "",
             f"Groups: {r.get('groups')}. Excluded test states (duplicates of training images, all arms): "
             f"{r.get('excluded_states')}", ""]
    lines += ["| Arm | clean bAcc | clean ECE | unseen bAcc | unseen ECE | in-dom bAcc | in-dom ECE | seen bAcc | seen ECE | clean ECE (T-scaled) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, row in r["table"].items():
        g = lambda k, m: row.get(k, {}).get(m)
        lines.append(f"| {LABEL[arm]} | {f(g('clean', 'balanced_accuracy'))} | {f(g('clean', 'ece'))} | "
                     f"{f(g('unseen', 'balanced_accuracy'))} | {f(g('unseen', 'ece'))} | "
                     f"{f(g('in_domain', 'balanced_accuracy'))} | {f(g('in_domain', 'ece'))} | "
                     f"{f(g('seen_source', 'balanced_accuracy'))} | {f(g('seen_source', 'ece'))} | "
                     f"{f(row.get('clean', {}).get('temperature_scaled', {}).get('ece'))} |")
    lines += ["", "## Secondary: temperature fitted on validation (plan §7), clean / unseen macro", "",
              "| Arm | T | ECE raw→T | NLL raw→T | Brier raw→T | unseen ECE raw→T | unseen NLL raw→T |",
              "|---|---|---|---|---|---|---|"]
    for arm, row in r["table"].items():
        c, u = row.get("clean", {}), row.get("unseen", {})
        cs, us = c.get("temperature_scaled", {}), u.get("temperature_scaled", {})
        ts = [round(x, 2) for x in r["temperatures"][arm] if x]
        pair = lambda d, s, m: f"{f(d.get(m))}→{f(s.get(m))}"
        lines.append(f"| {LABEL[arm]} | {ts or '—'} | {pair(c, cs, 'ece')} | {pair(c, cs, 'nll')} | "
                     f"{pair(c, cs, 'brier')} | {pair(u, us, 'ece')} | {pair(u, us, 'nll')} |")
    if "hypotheses" in r:
        h = r["hypotheses"]
        lines += ["", "## Hypotheses (clean macro; PathJev-P minus comparator)", ""]
        for key in [k for k in h if k.startswith("seed")]:
            lines.append(f"**{key}** — H1a all: {h[key]['H1a_all']}, H1b all: {h[key]['H1b_all']}")
            for hy in ("H1a", "H1b"):
                for o, v in h[key][hy].items():
                    lines.append(f"- {hy} vs {LABEL[o]}: diff {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}], "
                                 f"p={v['p_one_sided']:.4f}, Holm-rejected={v['holm_rejected']}")
        lines += ["", f"**H1a supported: {h['H1a_supported']} · H1b supported: {h['H1b_supported']}**"]
    if "gate" in r:
        g = r["gate"]
        lines += ["", "## §5 GATE rule (recommendation, user decides)", "",
                  f"- effect condition: {g['cond_effect']} (bAcc vs CONCH CI {g['pathjev_vs_conch_bacc']['ci95']})",
                  f"- unseen ECE PathJev {g['unseen_ece_pathjev']:.3f} vs best zero-shot {g['best_zero_shot_unseen']}: "
                  f"within 0.02 = {g['cond_unseen_ece']}", f"- gains only in-domain: {g['gains_only_in_domain']}",
                  f"- **recommendation: {g['recommendation']}**"]
    if r["notes"]:
        lines += ["", "Notes: " + "; ".join(r["notes"])]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
