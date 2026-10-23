"""HiESD (gastric ESD; Wang et al., Sci Data 2025; figshare 10.6084/m9.figshare.28919840, CC BY 4.0).

104 WSIs (40×, ~0.246 µm/px, Xi'an) with 10-class region masks at a fixed 64×
downsampling, anchored at the top-left corner (mask width × 64 matches the
slide width; a few masks are slightly shorter than the slide → the remainder is
unlabelled). The slide→patient mapping is not released (44 patients), so the
whole dataset is one held-out clean test set; group = slide.

Patch extraction (run inside a Slurm job via `load()`): 224-px tiles at
20×-equivalent (448 level-0 px), non-overlapping grid; a tile is kept when
≥ `PURITY` of its mask pixels belong to one class (near-colour JPEG/resampling
artefacts are snapped to the nearest palette colour) and ≥ `MIN_TISSUE` of its
pixels are tissue (annotated regions include background glass); at most `CAP`
tiles per (slide, class), seeded. Mask/slide alignment was checked visually on
one slide (plausible, not pathologist-verified).
"""
from __future__ import annotations

import io
import json
import random
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from pathjev.paths import data_root
from pathjev.records import Record

ROOT = data_root() / "hiesd"
MASK_ZIP = "ESD_40X_annotation_downsample64.zip"
DOWNSAMPLE = 64
TILE_L0 = 448            # level-0 pixels per tile (40× → 20×-equivalent at 224 px)
TILE_PX = 224
PURITY = 0.90
CAP = 60
MIN_TISSUE = 0.50        # fraction of non-background pixels required in a tile
SEED = "pathjev-v1"
PALETTE = {
    (139, 0, 0): "tub1", (255, 0, 255): "tub2", (128, 0, 128): "pap", (75, 0, 130): "others",
    (138, 43, 226): "normal_gland", (0, 0, 255): "chronic_gastritis", (70, 130, 180): "atrophic_gastritis",
    (0, 255, 0): "lymphoid_follicle", (0, 128, 0): "complete_im", (255, 255, 0): "incomplete_im",
}


def _class_map(mask_rgb: np.ndarray) -> np.ndarray:
    """Per-pixel class index (-1 = unannotated/black or no palette colour within 8 per channel)."""
    colours = np.array(list(PALETTE), dtype=np.int16)
    flat = mask_rgb.reshape(-1, 3).astype(np.int16)
    dist = np.abs(flat[:, None, :] - colours[None, :, :]).max(axis=2)
    best = dist.argmin(axis=1)
    out = np.where(dist[np.arange(len(flat)), best] <= 8, best, -1)
    out[(flat.max(axis=1) <= 8)] = -1
    return out.reshape(mask_rgb.shape[:2])


def tissue_fraction(tile: Image.Image) -> float:
    """Share of pixels that are not near-white glass (max channel < 220 or saturation > 0.08)."""
    rgb = np.asarray(tile, dtype=np.float32) / 255.0
    sat = (rgb.max(axis=2) - rgb.min(axis=2)) / np.maximum(rgb.max(axis=2), 1e-6)
    return float(((rgb.max(axis=2) < 220 / 255) | (sat > 0.08)).mean())


def _extract_slide(args) -> list[dict]:
    import openslide

    svs, mask_bytes, out_dir = args
    classes = list(PALETTE.values())
    cmap = _class_map(np.array(Image.open(io.BytesIO(mask_bytes)).convert("RGB")))
    step = TILE_L0 // DOWNSAMPLE                       # 7 mask pixels per tile side
    candidates: dict[str, list[tuple[int, int]]] = {}
    for my in range(0, cmap.shape[0] - step + 1, step):
        for mx in range(0, cmap.shape[1] - step + 1, step):
            window = cmap[my:my + step, mx:mx + step].ravel()
            labelled = window[window >= 0]
            if len(labelled) < PURITY * window.size:
                continue
            values, counts = np.unique(labelled, return_counts=True)
            if counts.max() >= PURITY * window.size:
                candidates.setdefault(classes[values[counts.argmax()]], []).append((mx, my))
    slide = openslide.OpenSlide(str(svs))
    rng = random.Random(f"{SEED}:{svs.stem}")
    rows = []
    for label, cells in sorted(candidates.items()):
        rng.shuffle(cells)
        kept = 0
        for mx, my in cells:
            if kept == CAP:
                break
            x, y = mx * DOWNSAMPLE, my * DOWNSAMPLE
            if x + TILE_L0 > slide.dimensions[0] or y + TILE_L0 > slide.dimensions[1]:
                continue
            tile = slide.read_region((x, y), 0, (TILE_L0, TILE_L0)).convert("RGB").resize(
                (TILE_PX, TILE_PX), Image.Resampling.BICUBIC)
            if tissue_fraction(tile) < MIN_TISSUE:
                continue
            kept += 1
            path = out_dir / svs.stem / f"{label}_{x}_{y}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            tile.save(path, compress_level=1)
            rows.append({"slide": svs.stem, "label": label, "x": x, "y": y,
                         "image": str(path.relative_to(out_dir))})
    return rows


def extract(root: Path = ROOT, workers: int = 8) -> Path:
    index = root / "patches" / "index.jsonl"
    if index.exists():
        return index
    with zipfile.ZipFile(root / MASK_ZIP) as archive:
        masks = {Path(n).stem: archive.read(n) for n in archive.namelist() if n.endswith(".png")}
    jobs = [(root / f"{stem}.svs", data, root / "patches") for stem, data in sorted(masks.items())
            if (root / f"{stem}.svs").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for chunk in pool.map(_extract_slide, jobs) for row in chunk]
    tmp = index.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.rename(index)
    return index


def load(root: Path = ROOT) -> list[Record]:
    import os

    index = extract(root, workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    records = []
    for line in index.open():
        r = json.loads(line)
        records.append(Record(
            sample_id=f"hiesd:{r['slide']}:{r['label']}:{r['x']}:{r['y']}", source="hiesd", organ="stomach",
            # Path rebuilt from the record fields: indexes written on another machine hold absolute paths (D-021).
            image=str(index.parent / r["slide"] / f"{r['label']}_{r['x']}_{r['y']}.png"),
            label=r["label"], patient_id="", slide_id=r["slide"], official_split="test",
            tier="derived", extra=(("x", r["x"]), ("y", r["y"]), ("tile_l0", TILE_L0))))
    return records
