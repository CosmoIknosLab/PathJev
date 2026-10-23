"""RQ3 WSI pipeline: tissue → grid → H-optimus-0 features → PCA → GMM (soft) → cluster mosaics
→ PathJev decisions → per-patch probability maps with abstention (research_plan §6.4).

Stages (each writes into <out>/<slide>/ and is skipped if its output exists):
  grid      tissue mask (Otsu on saturation of a low-res thumbnail) + 224-px tile grid at 20×-equivalent
  features  H-optimus-0 CLS features read straight from the WSI (no tile files written)        [GPU]
  cluster   PCA(n_pca) → GaussianMixture, k chosen by BIC in [k_min, k_max]; soft memberships
  states    per cluster, `mosaics_per_cluster` k×k mosaics of its most typical tiles → state JSONL
            (decisions from the template given; candidates as in training)
  maps      after inference: P(class | patch) = Σ_c membership(patch, c) · P(class | cluster c);
            a cluster abstains when its top probability < tau or P(homogeneous) < tau_hom
  score     against a region mask (HiESD format): cluster purity, patch accuracy on non-abstained
            tiles, coverage, per-class Dice of the arg-max map (oracle labeller = majority GT per cluster)
UMAP is not used (visualisation only, CLAUDE.md §0.1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _slide(path: str):
    import openslide
    return openslide.OpenSlide(path)


def tile_size_l0(slide, target_mpp: float = 0.5, tile_px: int = 224) -> int:
    mpp = float(slide.properties.get("openslide.mpp-x", 0.25))
    return int(round(tile_px * target_mpp / mpp))


def threshold_otsu(values: np.ndarray, bins: int = 256) -> float:
    """Otsu threshold of a 1-D sample in [0, 1] (between-class variance maximiser)."""
    hist, edges = np.histogram(values.ravel(), bins=bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    return float(centers[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def grid(wsi: str, out: Path, *, min_tissue: float = 0.5, thumb_px: int = 2048) -> Path:
    target = out / "grid.npz"
    if target.exists():
        return target
    slide = _slide(wsi)
    W, H = slide.dimensions
    scale = max(W, H) / thumb_px
    thumb = np.asarray(slide.get_thumbnail((int(W / scale), int(H / scale))).convert("RGB"), dtype=np.float32) / 255
    sat = (thumb.max(2) - thumb.min(2)) / np.maximum(thumb.max(2), 1e-6)
    mask = sat > max(threshold_otsu(sat), 0.05)
    t0 = tile_size_l0(slide)
    ts = t0 / scale
    coords = []
    for y in range(0, H - t0 + 1, t0):
        for x in range(0, W - t0 + 1, t0):
            y0, x0 = int(y / scale), int(x / scale)
            window = mask[y0:int(y0 + ts) + 1, x0:int(x0 + ts) + 1]
            if window.size and window.mean() >= min_tissue:
                coords.append((x, y))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(target, coords=np.array(coords, dtype=np.int64), tile_l0=t0, dims=np.array([W, H]))
    return target


def features(wsi: str, out: Path, *, device: str = "cuda:0", batch: int = 128) -> Path:
    import timm
    import torch
    from torchvision import transforms

    from pathjev.baselines import HOPT_MEAN, HOPT_STD

    target = out / "features.npz"
    if target.exists():
        return target
    g = np.load(out / "grid.npz")
    slide = _slide(wsi)
    model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5,
                              dynamic_img_size=False).to(device).eval()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(HOPT_MEAN, HOPT_STD)])
    t0 = int(g["tile_l0"])
    feats = []
    with torch.no_grad():
        for start in range(0, len(g["coords"]), batch):
            tiles = [tf(slide.read_region((int(x), int(y)), 0, (t0, t0)).convert("RGB")) for x, y in g["coords"][start:start + batch]]
            with torch.autocast("cuda", dtype=torch.float16):
                feats.append(model(torch.stack(tiles).to(device)).float().cpu().numpy())
    np.savez(target, features=np.concatenate(feats) if feats else np.zeros((0, 1536), np.float32))
    return target


def cluster(out: Path, *, n_pca: int = 50, k_min: int = 8, k_max: int = 30, seed: int = 0) -> Path:
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture

    target = out / "clusters.npz"
    if target.exists():
        return target
    X = np.load(out / "features.npz")["features"]
    Z = PCA(n_components=min(n_pca, X.shape[1], len(X) - 1), random_state=seed).fit_transform(X)
    best = None
    for k in range(k_min, min(k_max, max(k_min, len(X) // 20)) + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed, max_iter=300).fit(Z)
        bic = gmm.bic(Z)
        if best is None or bic < best[0]:
            best = (bic, k, gmm)
    _, k, gmm = best
    np.savez(target, membership=gmm.predict_proba(Z).astype(np.float32), k=k, bic=best[0])
    return target


def states(wsi: str, out: Path, template_path: Path, *, mosaic_k: int = 4, tile_px: int = 224,
           mosaics_per_cluster: int = 3, seed: int = 0) -> Path:
    """Mosaics from each cluster's most typical tiles (highest membership), sampled without replacement."""
    from pathjev import templates
    from pathjev.states import HOMOGENEITY, MODALITY

    target = out / "cluster_states.jsonl"
    if target.exists():
        return target
    template = templates.load(template_path)
    g, c = np.load(out / "grid.npz"), np.load(out / "clusters.npz")
    slide, t0, memb = _slide(wsi), int(g["tile_l0"]), c["membership"]
    hard = memb.argmax(1)
    rng = np.random.default_rng(seed)
    any_label = next(iter(template.spec["label_map"]))
    rows = []
    for k in range(memb.shape[1]):
        members = np.flatnonzero(hard == k)
        if len(members) < mosaic_k ** 2:
            continue
        typical = members[np.argsort(-memb[members, k])][: max(mosaic_k ** 2 * mosaics_per_cluster, 4 * mosaic_k ** 2)]
        for m in range(mosaics_per_cluster):
            pick = rng.choice(typical, size=mosaic_k ** 2, replace=False)
            canvas = Image.new("RGB", (mosaic_k * tile_px, mosaic_k * tile_px))
            for i, idx in enumerate(pick):
                x, y = g["coords"][idx]
                tile = slide.read_region((int(x), int(y)), 0, (t0, t0)).convert("RGB").resize((tile_px, tile_px))
                canvas.paste(tile, ((i % mosaic_k) * tile_px, (i // mosaic_k) * tile_px))
            path = out / "mosaics" / f"c{k:02d}_m{m}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(path, compress_level=1)
            # decisions carry the template's candidate sets; targets are placeholders (unknown at inference)
            decisions = [dict(d, target={cand["id"]: 0.0 for cand in d["candidates"]})
                         for d in template.decisions(any_label)]
            decisions.append({**HOMOGENEITY, "candidates": [{"id": "no", "text": "No"}, {"id": "yes", "text": "Yes"}],
                              "target": {"no": 0.0, "yes": 0.0}, "target_kind": "one_hot"})
            rows.append({"state_id": f"{out.name}:c{k:02d}:m{m}", "source": "", "modality": MODALITY,
                         "images": [str(path)], "image_size": None, "decisions": decisions,
                         "meta": {"dataset": "wsi", "slide": out.name, "cluster": k, "kind": "cluster_mosaic",
                                  "tiles": pick.tolist()}})
    target.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return target


def maps(out: Path, decisions_path: Path, *, qid: str = "tissue", tau: float = 0.5, tau_hom: float = 0.5) -> Path:
    """Aggregate cluster decisions (mean over a cluster's mosaics) into per-patch probabilities."""
    target = out / "maps.npz"
    c = np.load(out / "clusters.npz")
    memb = c["membership"]
    per_cluster: dict[int, list[dict]] = {}
    for line in Path(decisions_path).open():
        row = json.loads(line)
        if not row["state_id"].startswith(f"{out.name}:"):
            continue
        k = int(row["state_id"].split(":")[1][1:])
        per_cluster.setdefault(k, []).append(row["distributions"])
    candidates = sorted({cid for rows in per_cluster.values() for r in rows for cid in r[qid]})
    P = np.zeros((memb.shape[1], len(candidates)))
    abstain = np.ones(memb.shape[1], dtype=bool)
    for k, rows in per_cluster.items():
        P[k] = np.mean([[r[qid].get(cid, 0.0) for cid in candidates] for r in rows], axis=0)
        hom = np.mean([r.get("homogeneous", {}).get("yes", 1.0) for r in rows])
        abstain[k] = P[k].max() < tau or hom < tau_hom
    weights = memb * (~abstain)[None, :]                       # abstaining clusters contribute nothing
    covered = weights.sum(1)
    patch_probs = np.divide(weights @ P, covered[:, None], out=np.zeros((len(memb), len(candidates))),
                            where=covered[:, None] > 0)
    np.savez(target, candidates=np.array(candidates), cluster_probs=P, cluster_abstain=abstain,
             patch_probs=patch_probs.astype(np.float32), coverage=covered.astype(np.float32))
    return target


def score(out: Path, mask_png: bytes | str, palette: dict, label_to_candidate: dict, downsample: int = 64,
          purity_tiles: float = 0.9) -> dict:
    """Cluster purity, labelling accuracy and Dice against a region mask (HiESD layout)."""
    from pathjev.sources.hiesd import _class_map

    import io
    g, c, m = np.load(out / "grid.npz"), np.load(out / "clusters.npz"), np.load(out / "maps.npz", allow_pickle=False)
    raw = Image.open(io.BytesIO(mask_png) if isinstance(mask_png, bytes) else mask_png).convert("RGB")
    cmap = _class_map(np.asarray(raw))
    names = list(palette.values())
    t0 = int(g["tile_l0"])
    gt = []
    for x, y in g["coords"]:
        win = cmap[y // downsample:(y + t0) // downsample, x // downsample:(x + t0) // downsample].ravel()
        win = win[win >= 0]
        if len(win) and np.bincount(win).max() >= purity_tiles * len(win):
            gt.append(label_to_candidate.get(names[np.bincount(win).argmax()], "other"))
        else:
            gt.append(None)
    gt = np.array(gt, dtype=object)
    known = np.array([v is not None for v in gt])
    hard = c["membership"].argmax(1)
    cands = list(m["candidates"])
    pred = np.array([cands[i] for i in m["patch_probs"].argmax(1)], dtype=object)
    covered = m["coverage"] > 0
    purities, oracle = [], np.empty(len(gt), dtype=object)
    for k in np.unique(hard):
        idx = np.flatnonzero((hard == k) & known)
        if len(idx):
            values, counts = np.unique(gt[idx].astype(str), return_counts=True)
            purities.append((counts.max() / len(idx), len(idx)))
            oracle[hard == k] = values[counts.argmax()]
    sel = known & covered
    result = {"tiles": int(len(gt)), "tiles_with_gt": int(known.sum()), "coverage": float(covered[known].mean()) if known.any() else 0.0,
              "cluster_purity_weighted": float(sum(p * n for p, n in purities) / max(1, sum(n for _, n in purities))),
              "pathjev_accuracy_covered": float((pred[sel] == gt[sel]).mean()) if sel.any() else float("nan"),
              "oracle_accuracy": float((oracle[known] == gt[known]).mean()) if known.any() else float("nan"),
              "dice": {}}
    for cls in sorted({v for v in gt[known]}):
        a, b = (pred == cls) & sel, (gt == cls) & sel
        result["dice"][cls] = float(2 * (a & b).sum() / max(1, a.sum() + b.sum()))
    (out / "score.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("grid", "features", "cluster", "states", "maps"))
    ap.add_argument("--wsi", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--template", type=Path)
    ap.add_argument("--decisions", type=Path)
    args = ap.parse_args()
    for wsi in args.wsi:
        out = args.out / Path(wsi).stem
        out.mkdir(parents=True, exist_ok=True)
        if args.stage == "grid":
            grid(wsi, out)
        elif args.stage == "features":
            features(wsi, out)
        elif args.stage == "cluster":
            cluster(out)
        elif args.stage == "states":
            states(wsi, out, args.template)
        else:
            maps(out, args.decisions)
        print(f"[{args.stage}] {out.name}", flush=True)


if __name__ == "__main__":
    main()
