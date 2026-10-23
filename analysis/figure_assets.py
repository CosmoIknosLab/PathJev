"""Image assets for the overview figure (paper/figures/overview/), from run folders only.

    python analysis/figure_assets.py --work $PATHJEV_WORK --slide 46e7dccd-521a-4d0f-80bd-10e960cd5905 \
        --out paper/figures/overview

Patches come from CC BY 4.0 sources only (SICAPv2 training split, HiESD WSI demo); picked deterministically
(first state of each listed class in file order), never by looking at model outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

SICAP_CLASSES = ["benign", "gp3", "gp4"]


def first_patches(states: Path, dataset: str, qid: str, classes: list[str]) -> dict[str, str]:
    found = {}
    for line in states.open():
        s = json.loads(line)
        if s.get("meta", {}).get("dataset") != dataset:
            continue
        d = next(d for d in s["decisions"] if d["qid"] == qid)
        gold = max(d["target"], key=d["target"].get)
        if gold in classes and gold not in found:
            found[gold] = s["images"][0]
        if len(found) == len(classes):
            break
    return found


def save_square(src: str | Path, dst: Path, size: int = 256) -> None:
    Image.open(src).convert("RGB").resize((size, size), Image.Resampling.LANCZOS).save(dst)


def wsi_maps(demo: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    info = json.loads((demo / "tile_info.json").read_text())
    gw, gh = info["grid"]
    tiles = {json.loads(l)["state_id"]: json.loads(l)["meta"] for l in (demo / "tiles.jsonl").open()}
    z = np.load(demo / "clusters.npz")
    ids, resp = [str(i) for i in z["ids"]], z["resp"]
    cl = json.loads((demo / "clusters.json").read_text())
    rep = {json.loads(l)["state_id"]: json.loads(l)["distributions"]["present:adenocarcinoma"]["yes"]
           for l in (demo / "reps_decisions.jsonl").open()}
    pc = np.array([np.mean([rep[s] for s in cl["representatives"][str(j)]]) if str(j) in cl["representatives"] else 0.0
                   for j in range(resp.shape[1])])
    prob, hard = resp @ pc, resp.argmax(1)
    ys = [tiles[s]["gy"] for s in ids]
    xs = [tiles[s]["gx"] for s in ids]
    y0, y1, x0, x1 = min(ys), max(ys) + 1, min(xs), max(xs) + 1
    cmap = plt.get_cmap("tab20")
    cl_img = np.ones((gh, gw, 4))
    cl_img[..., 3] = 0
    pr_img = np.full((gh, gw), np.nan)
    for i, s in enumerate(ids):
        cl_img[tiles[s]["gy"], tiles[s]["gx"]] = (*cmap(hard[i] % 20)[:3], 1.0)
        pr_img[tiles[s]["gy"], tiles[s]["gx"]] = prob[i]
    for name, img, kw in (("wsi_clusters.png", cl_img[y0:y1, x0:x1], {}),
                          ("wsi_prob.png", pr_img[y0:y1, x0:x1], {"cmap": "YlOrRd", "vmin": 0, "vmax": 1})):
        fig = plt.figure(figsize=(4, 4 * (y1 - y0) / (x1 - x0)))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(img, interpolation="nearest", **kw)
        ax.axis("off")
        fig.savefig(out / name, dpi=200, transparent=True)
        plt.close(fig)
    thumb = Image.open(demo / "thumbnail.png")
    step = thumb.width / gw
    thumb.crop((int(x0 * step), int(y0 * step), int(x1 * step), int(y1 * step))).save(out / "wsi_thumb.png")
    # four most typical tiles of the cluster with the highest P(adenocarcinoma) and of the lowest
    for tag, j in (("hi", int(pc.argmax())), ("lo", int(pc.argmin()))):
        reps = cl["representatives"][str(j)][:4]
        grid = Image.new("RGB", (2 * 224, 2 * 224), "white")
        for k, s in enumerate(reps):
            m = tiles[s]
            grid.paste(Image.open(demo / "tiles" / f"{m['gx']}_{m['gy']}.png").convert("RGB"), ((k % 2) * 224, (k // 2) * 224))
        grid.save(out / f"wsi_reps_{tag}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--slide", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    found = first_patches(args.work / "mixes" / "stage1-p" / "train.jsonl", "sicapv2", "gleason_pattern", SICAP_CLASSES)
    for cls, path in found.items():
        save_square(path, args.out / f"patch_sicap_{cls}.png")
    wsi_maps(args.work / "wsi_demo" / "demo-hiesd" / args.slide, args.out)
    print(sorted(p.name for p in args.out.iterdir()))


if __name__ == "__main__":
    main()
