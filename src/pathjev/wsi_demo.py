"""Qualitative WSI demo: unsupervised tissue clusters of a slide labelled by PathJev-P (D-037).

    python -m pathjev.wsi_demo select   --config configs/wsi/demo-hiesd.yaml
    python -m pathjev.wsi_demo tile     --config ... --slide <stem>
    python -m pathjev.wsi_demo cluster  --config ... --slide <stem>
    python -m pathjev.wsi_demo assemble --config ... --slide <stem> [--figdir paper/figures]

Pipeline (CLAUDE.md §0.1, research_plan §6.4): tissue tiles (224 px at 20×) → H-optimus-0 features
(`pathjev.baselines features`, run by the sbatch script) → PCA → GMM with soft membership (k by BIC)
→ PathJev-P on the most typical tiles of each cluster → cluster class distribution → per-tile
probability = Σ_k membership(tile, k) · P(class | k), with abstention on uncertain clusters.
A dense reference (PathJev-P on every tile) shows what the cluster shortcut costs. Masks are
used only for slide eligibility and for the ground-truth panel / descriptive agreement.
Outputs: $PATHJEV_WORK/wsi_demo/<name>/<slide>/.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from pathjev.paths import work_root
from pathjev.sources import hiesd

MASK_TO_CANDIDATE = {"others": "other"}       # HiESD residual tissue = the "other" candidate


def _cfg(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _out(cfg: dict, slide: str | None = None) -> Path:
    base = work_root() / "wsi_demo" / cfg["name"]
    return base / slide if slide else base


def _masks() -> dict[str, bytes]:
    with zipfile.ZipFile(hiesd.ROOT / hiesd.MASK_ZIP) as archive:
        return {Path(n).stem: archive.read(n) for n in archive.namelist() if n.endswith(".png")}


def _class_map(mask_bytes: bytes) -> np.ndarray:
    return hiesd._class_map(np.array(Image.open(io.BytesIO(mask_bytes)).convert("RGB")))


# ------------------------------------------------------------------------------------- select
def select(cfg: dict) -> list[str]:
    c = cfg["slides"]
    eligible = []
    for stem, data in sorted(_masks().items()):
        if not (hiesd.ROOT / f"{stem}.svs").exists():
            continue
        cmap = _class_map(data)
        _, counts = np.unique(cmap[cmap >= 0], return_counts=True)
        if (counts >= c["min_class_px"]).sum() >= c["min_classes"]:
            eligible.append(stem)
    chosen = sorted(random.Random(c["seed"]).sample(eligible, c["n"]))
    out = _out(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "slides.json").write_text(json.dumps({"eligible": len(eligible), "chosen": chosen}, indent=1))
    return chosen


# --------------------------------------------------------------------------------------- tile
def _decision_templates(qids: list[str]) -> list[dict]:
    """Question text and candidates copied from the HiESD test states (identical to the evaluation)."""
    first = json.loads((work_root() / "eval_sets" / "stage1-test" / "hiesd.jsonl").open().readline())
    by_q = {d["qid"]: d for d in first["decisions"]}
    out = []
    for q in qids:
        d = dict(by_q[q])
        ids = [c["id"] for c in d["candidates"]]
        d["target"] = {i: float(k == 0) for k, i in enumerate(ids)}   # placeholder: never scored
        d["target_kind"] = "one_hot"
        out.append(d)
    return out


def tile(cfg: dict, slide: str) -> Path:
    import openslide

    t = cfg["tiling"]
    out = _out(cfg, slide)
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    cmap = _class_map(_masks()[slide])
    step = t["tile_l0"] // hiesd.DOWNSAMPLE
    osl = openslide.OpenSlide(str(hiesd.ROOT / f"{slide}.svs"))
    w, h = osl.dimensions
    thumb = np.asarray(osl.get_thumbnail((w // hiesd.DOWNSAMPLE, h // hiesd.DOWNSAMPLE)).convert("RGB"))
    thumb_img = Image.fromarray(thumb)
    thumb_img.save(out / "thumbnail.png")
    classes = list(hiesd.PALETTE.values())
    cells = []
    for gy in range(0, h // t["tile_l0"]):
        for gx in range(0, w // t["tile_l0"]):
            win = thumb[gy * step:(gy + 1) * step, gx * step:(gx + 1) * step]
            if win.size and hiesd.tissue_fraction(Image.fromarray(win)) >= t["min_tissue"] * 0.8:
                cells.append((gx, gy))     # cheap pre-filter on the thumbnail; exact rule below

    def read(cell):
        gx, gy = cell
        img = osl.read_region((gx * t["tile_l0"], gy * t["tile_l0"]), 0, (t["tile_l0"],) * 2).convert("RGB")
        img = img.resize((t["tile_px"],) * 2, Image.Resampling.BICUBIC)
        if hiesd.tissue_fraction(img) < t["min_tissue"]:
            return None
        path = out / "tiles" / f"{gx}_{gy}.png"
        img.save(path, compress_level=1)
        win = cmap[gy * step:(gy + 1) * step, gx * step:(gx + 1) * step].ravel()
        labelled = win[win >= 0]
        gt = None
        if len(labelled) >= cfg["reference"]["gt_purity"] * win.size:
            v, n = np.unique(labelled, return_counts=True)
            if n.max() >= cfg["reference"]["gt_purity"] * win.size:
                gt = classes[v[n.argmax()]]
                gt = MASK_TO_CANDIDATE.get(gt, gt)
        return {"gx": gx, "gy": gy, "image": str(path), "gt": gt}

    with ThreadPoolExecutor(max_workers=8) as pool:   # openslide reads release the GIL
        tiles = [r for r in pool.map(read, cells) if r]
    decisions = _decision_templates(cfg["labelling"]["questions"])
    with (out / "tiles.jsonl").open("w") as fh:
        for r in tiles:
            fh.write(json.dumps({"state_id": f"{slide}:{r['gx']}:{r['gy']}", "split": "demo", "source": "",
                                 "modality": "H&E histopathology", "images": [r["image"]], "image_size": None,
                                 "decisions": decisions,
                                 "meta": {"dataset": "hiesd-wsi-demo", "group": slide, "gx": r["gx"], "gy": r["gy"],
                                          "gt": r["gt"]}}) + "\n")
    (out / "tile_info.json").write_text(json.dumps({"slide": slide, "dimensions": [w, h], "tiles": len(tiles),
                                                    "grid": [w // t["tile_l0"], h // t["tile_l0"]]}))
    return out / "tiles.jsonl"


# ------------------------------------------------------------------------------------ cluster
def cluster(cfg: dict, slide: str) -> Path:
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture

    c, out = cfg["clustering"], _out(cfg, slide)
    data = np.load(out / "tiles.npz")
    ids, X = data["ids"], data["features"]
    Z = PCA(n_components=c["pca_components"], random_state=c["seed"]).fit_transform(
        (X - X.mean(0)) / (X.std(0) + 1e-6))
    bic, models = {}, {}
    for k in range(c["k_range"][0], c["k_range"][1] + 1, c["k_step"]):
        g = GaussianMixture(n_components=k, covariance_type="diag", random_state=c["seed"], max_iter=500).fit(Z)
        bic[k], models[k] = float(g.bic(Z)), g
    k = min(bic, key=bic.get)
    resp = models[k].predict_proba(Z)
    np.savez(out / "clusters.npz", ids=ids, resp=resp)
    states = {json.loads(l)["state_id"]: l for l in (out / "tiles.jsonl").open()}
    n_rep = cfg["labelling"]["representatives"]
    reps = {}
    with (out / "reps.jsonl").open("w") as fh:
        for j in range(k):
            members = np.flatnonzero(resp.argmax(1) == j)
            if len(members) == 0:
                continue
            top = members[np.argsort(-resp[members, j])[:n_rep]]
            reps[j] = [str(ids[i]) for i in top]
            for i in top:
                fh.write(states[str(ids[i])])
    (out / "clusters.json").write_text(json.dumps({"k": k, "bic": bic, "representatives": reps,
                                                   "sizes": np.bincount(resp.argmax(1), minlength=k).tolist()}))
    return out / "reps.jsonl"


# ----------------------------------------------------------------------------------- assemble
def _dists(path: Path, qid: str) -> dict[str, dict[str, float]]:
    return {r["state_id"]: r["distributions"][qid] for r in map(json.loads, path.open()) if qid in r["distributions"]}


def assemble(cfg: dict, slide: str, figdir: Path | None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    out, lab = _out(cfg, slide), cfg["labelling"]
    info = json.loads((out / "tile_info.json").read_text())
    cl = json.loads((out / "clusters.json").read_text())
    z = np.load(out / "clusters.npz")
    ids, resp = [str(i) for i in z["ids"]], z["resp"]
    tiles = {json.loads(l)["state_id"]: json.loads(l)["meta"] for l in (out / "tiles.jsonl").open()}
    cand = [c["id"] for c in _decision_templates(["tissue"])[0]["candidates"]]
    rep_t, rep_c = _dists(out / "reps_decisions.jsonl", "tissue"), _dists(out / "reps_decisions.jsonl", "present:adenocarcinoma")
    k = resp.shape[1]
    P = np.zeros((k, len(cand)))
    Pc = np.zeros(k)
    for j in range(k):
        r = cl["representatives"].get(str(j), [])
        if r:
            P[j] = np.mean([[rep_t[s][c] for c in cand] for s in r], axis=0)
            Pc[j] = np.mean([rep_c[s]["yes"] for s in r])
    tile_p = resp @ P                                   # Σ_k membership · P(class | k)
    tile_c = resp @ Pc
    abstain = P.max(1) < lab["abstain_below"]
    hard = resp.argmax(1)
    dense = _dists(out / "dense_decisions.jsonl", "tissue") if (out / "dense_decisions.jsonl").exists() else {}

    gw, gh = info["grid"]
    colours = {MASK_TO_CANDIDATE.get(n, n): np.array(rgb) / 255 for rgb, n in hiesd.PALETTE.items()}
    blank = np.ones((gh, gw, 3))

    def canvas(values):
        img = blank.copy()
        for (s, v) in values:
            m = tiles[s]
            img[m["gy"], m["gx"]] = v
        return img

    gt_img = canvas([(s, colours[m["gt"]]) for s, m in tiles.items() if m["gt"]])
    cmap20 = plt.get_cmap("tab20")
    cl_img = canvas([(s, cmap20(hard[i] % 20)[:3]) for i, s in enumerate(ids)])
    lab_img = canvas([(s, np.array([0.6, 0.6, 0.6]) if abstain[hard[i]] else colours[cand[tile_p[i].argmax()]])
                      for i, s in enumerate(ids)])
    den_img = canvas([(s, colours[cand[int(np.argmax([dense[s][c] for c in cand]))]]) for s in ids if s in dense])
    carc = np.full((gh, gw), np.nan)
    for i, s in enumerate(ids):
        carc[tiles[s]["gy"], tiles[s]["gx"]] = tile_c[i]

    # descriptive agreement with the mask (tiles with a pure mask label only)
    lab_ok = [(cand[tile_p[i].argmax()] == tiles[s]["gt"]) for i, s in enumerate(ids)
              if tiles[s]["gt"] and not abstain[hard[i]]]
    den_ok = [(cand[int(np.argmax([dense[s][c] for c in cand]))] == tiles[s]["gt"]) for s in ids
              if tiles[s]["gt"] and s in dense]
    n_gt = sum(1 for s in ids if tiles[s]["gt"])
    # carcinoma presence (Noul) vs mask: added after viewing slide 1's figure (D-037), descriptive
    from sklearn.metrics import roc_auc_score
    carc_classes = {"tub1", "tub2", "pap"}
    lab_idx = [i for i, s in enumerate(ids) if tiles[s]["gt"]]
    yc = [tiles[ids[i]]["gt"] in carc_classes for i in lab_idx]
    dense_c = _dists(out / "dense_decisions.jsonl", "present:adenocarcinoma") if dense else {}
    auc = {}
    if 0 < sum(yc) < len(yc):
        auc["cluster_propagated"] = float(roc_auc_score(yc, [tile_c[i] for i in lab_idx]))
        if dense_c:
            auc["dense"] = float(roc_auc_score(yc, [dense_c[ids[i]]["yes"] for i in lab_idx]))
    # sensitivity / specificity at P(yes) > 0.5 (added after the review of 2026-10-22; descriptive, no tuning)
    ycn = np.array(yc, dtype=bool)
    rates = {}
    for name, score in (("cluster_propagated", [tile_c[i] for i in lab_idx]),
                        ("dense", [dense_c[ids[i]]["yes"] for i in lab_idx] if dense_c else None)):
        if score is None or not len(ycn):
            continue
        pos = np.array(score) > 0.5
        rates[name] = {"sensitivity": float(pos[ycn].mean()) if ycn.any() else None,
                       "specificity": float((~pos[~ycn]).mean()) if (~ycn).any() else None}
    stats = {"slide": slide, "tiles": len(ids), "k": k, "pathjev_calls_cluster": sum(len(v) for v in cl["representatives"].values()),
             "pathjev_calls_dense": len(dense), "tiles_with_mask_label": n_gt,
             "cluster_labelled_agreement": float(np.mean(lab_ok)) if lab_ok else None,
             "cluster_coverage": float(np.mean([not abstain[hard[i]] for i, s in enumerate(ids) if tiles[s]["gt"]])) if n_gt else None,
             "dense_agreement": float(np.mean(den_ok)) if den_ok else None,
             "abstaining_clusters": int(abstain.sum()), "carcinoma_tiles": int(sum(yc)), "carcinoma_auroc": auc,
             "carcinoma_rates_at_0.5": rates,
             "cluster_labels": [{"cluster": j, "size": cl["sizes"][j], "top": cand[int(P[j].argmax())],
                                 "p_top": float(P[j].max()), "p_adenocarcinoma": float(Pc[j])} for j in range(k)]}
    (out / "demo_stats.json").write_text(json.dumps(stats, indent=1))

    thumb = np.asarray(Image.open(out / "thumbnail.png").resize((gw * 7, gh * 7)))
    fig, ax = plt.subplots(2, 3, figsize=(15, 10 * gh / max(gw, 1) + 1.5))
    panels = [(thumb, "H&E (thumbnail)"), (gt_img, "Region annotation (pure tiles)"), (cl_img, f"GMM clusters (k={k})"),
              (lab_img, "Cluster labels by PathJev-P (grey = abstain)"), (den_img, "PathJev-P on every tile (reference)"),
              (carc, "P(adenocarcinoma), cluster-propagated")]
    for a, (img, title) in zip(ax.ravel(), panels):
        if img.ndim == 2:
            im = a.imshow(img, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            fig.colorbar(im, ax=a, fraction=0.04)
        else:
            a.imshow(img, interpolation="nearest")
        a.set_title(title, fontsize=10)
        a.axis("off")
    used = {cand[tile_p[i].argmax()] for i in range(len(ids))} | {m["gt"] for m in tiles.values() if m["gt"]}
    fig.legend(handles=[Patch(color=colours[c], label=c) for c in cand if c in used], loc="lower center",
               ncol=5, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for d in [out] + ([figdir] if figdir else []):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"wsi_demo_{slide[:8]}.png", dpi=150)
    plt.close(fig)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("select", "tile", "cluster", "assemble"))
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--slide")
    ap.add_argument("--figdir", type=Path)
    args = ap.parse_args()
    cfg = _cfg(args.config)
    if args.step == "select":
        print("\n".join(select(cfg)))
    elif args.step == "tile":
        print(tile(cfg, args.slide))
    elif args.step == "cluster":
        print(cluster(cfg, args.slide))
    else:
        print(json.dumps(assemble(cfg, args.slide, args.figdir), indent=1))


if __name__ == "__main__":
    main()
