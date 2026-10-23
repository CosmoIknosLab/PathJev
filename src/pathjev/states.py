"""Build decision states (OmniMed-Jev JSONL format) from records and templates.

Two state kinds (research_plan.md §3.4, D-012):
  * patch:  one image, decisions from the template for its label;
  * mosaic: k×k patches tiled into ONE image (upstream supports one image per
            state; single-image mosaics cost the same as a patch, run 5004).
            Homogeneous mosaics draw all tiles from the same group (patient /
            slide) and label; mixed mosaics draw from two labels of the same
            group with a known composition, and carry soft targets plus a
            homogeneity Noul.

The prompt header never names the dataset (D-012): `source` is left empty and
`modality` is a fixed string; the dataset name lives in `meta`.
"""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # trusted local WSI-derived RoIs (BRACS RoIs reach ~1.5e8 px)

from pathjev.records import Record
from pathjev.templates import Template

MODALITY = "H&E histopathology"
HOMOGENEITY = {"qid": "homogeneous", "type": "noul",
               "instructions": "Does this region show a single tissue category?"}


def _meta(record_or_records, kind: str, split: str, template: Template, **extra) -> dict[str, Any]:
    records = record_or_records if isinstance(record_or_records, list) else [record_or_records]
    return {"dataset": records[0].source, "organ": records[0].organ, "kind": kind, "split": split,
            "samples": [r.sample_id for r in records], "group": records[0].group,
            "tier": min((r.tier for r in records), key=["gold", "derived", "silver"].index),
            "template": template.path.name, "template_sha256": template.sha256, **extra}


def patch_state(record: Record, template: Template, split: str) -> dict[str, Any]:
    return {"state_id": record.sample_id, "split": split, "source": "", "modality": MODALITY,
            "images": [record.image], "image_size": None,
            "decisions": template.decisions(record.label),
            "meta": _meta(record, "patch", split, template)}


def _tile(image_path: str, size: int, mode: str, rng: random.Random) -> Image.Image:
    """mode: "resize" (whole image → tile), "crop" (random native-resolution crop),
    or "crop:<f>" (random crop of size*f px resized to size, e.g. crop:2 turns a
    40× RoI at 0.25 µm/px into 20×-equivalent tiles)."""
    scale = float(mode.split(":", 1)[1]) if mode.startswith("crop:") else 1.0
    span = int(round(size * scale))
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if mode == "resize" or (image.width <= span and image.height <= span):
            return image.resize((size, size), Image.Resampling.BICUBIC)
        x = rng.randint(0, max(0, image.width - span))
        y = rng.randint(0, max(0, image.height - span))
        tile = image.crop((x, y, x + span, y + span))
        return tile if span == size else tile.resize((size, size), Image.Resampling.BICUBIC)


def write_mosaic(paths: list[str], k: int, tile_px: int, mode: str, out: Path, seed: str) -> str:
    rng = random.Random(seed)
    canvas = Image.new("RGB", (k * tile_px, k * tile_px))
    for index, path in enumerate(paths):
        row, col = divmod(index, k)
        canvas.paste(_tile(path, tile_px, mode, rng), (col * tile_px, row * tile_px))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, "PNG", compress_level=1)
    return str(out)


def _mix(targets: list[dict[str, float]], weights: list[float]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for target, weight in zip(targets, weights):
        for key, value in target.items():
            out[key] += weight * value
    return dict(out)


def mosaic_state(records: list[Record], template: Template, split: str, *, k: int, tile_px: int,
                 tile_mode: str, image_dir: Path) -> dict[str, Any]:
    """One mosaic state from k*k records (homogeneous if they share one label)."""
    assert len(records) == k * k, "a k×k mosaic needs exactly k*k records"
    ids = "|".join(r.sample_id for r in records)
    digest = hashlib.sha256(ids.encode()).hexdigest()[:16]
    state_id = f"mosaic:{records[0].source}:{k}x{k}:{digest}"
    image = write_mosaic([r.image for r in records], k, tile_px, tile_mode,
                         image_dir / records[0].source / f"{digest}.png", seed=digest)

    labels = [r.label for r in records]
    counts: dict[str, int] = defaultdict(int)
    for label in labels:
        counts[label] += 1
    homogeneous = len(counts) == 1
    per_label = {label: template.decisions(label) for label in counts}
    weights = {label: n / len(labels) for label, n in counts.items()}

    decisions = []
    for index, base in enumerate(per_label[labels[0]]):
        decision = dict(base)
        if base["type"] == "noul":
            # presence: yes if any tile shows a positive class
            yes = max(per_label[label][index]["target"]["yes"] for label in counts)
            decision["target"] = {"no": 1.0 - yes, "yes": yes}
        else:
            decision["target"] = _mix([per_label[label][index]["target"] for label in counts],
                                      [weights[label] for label in counts])
            decision["target_kind"] = "one_hot" if homogeneous else "soft"
        decisions.append(decision)
    decisions.append({**HOMOGENEITY, "candidates": [{"id": "no", "text": "No"}, {"id": "yes", "text": "Yes"}],
                      "target": {"no": float(not homogeneous), "yes": float(homogeneous)},
                      "target_kind": "one_hot"})
    return {"state_id": state_id, "split": split, "source": "", "modality": MODALITY,
            "images": [image], "image_size": None, "decisions": decisions,
            "meta": _meta(records, "mosaic", split, template, k=k, tile_px=tile_px,
                          composition={label: weights[label] for label in counts})}


def sample_mosaics(records: Iterable[Record], *, k: int, n_homogeneous: int, n_mixed: int,
                   mixed_fractions: tuple[float, ...] = (0.75, 0.5), seed: str = "pathjev-v1",
                   with_replacement: bool = False) -> list[list[Record]]:
    """Draw tile groups for mosaics, never mixing patients/slides (CLAUDE.md §0.5).

    Homogeneous: k*k records with the same (group, label). Mixed: same group,
    two labels, majority fraction from `mixed_fractions`. Sampling is seeded
    and without replacement inside one mosaic unless `with_replacement` (large
    RoIs: several distinct random crops of the same region).
    """
    rng = random.Random(seed)
    by_group_label: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for r in records:
        by_group_label[(r.group, r.label)].append(r)
    for pool in by_group_label.values():
        pool.sort(key=lambda r: r.sample_id)
    need = k * k

    def draw(pool: list[Record], n: int) -> list[Record]:
        return rng.choices(pool, k=n) if with_replacement else rng.sample(pool, n)

    full = sorted(key for key, pool in by_group_label.items() if len(pool) >= (1 if with_replacement else need))
    out: list[list[Record]] = []
    for _ in range(n_homogeneous if full else 0):
        key = rng.choice(full)
        out.append(draw(by_group_label[key], need))

    by_group: dict[str, list[str]] = defaultdict(list)
    for group, label in by_group_label:
        by_group[group].append(label)
    pairs = []
    for group, labels in by_group.items():
        for a in labels:
            for b in labels:
                if a != b:
                    pairs.append((group, a, b))
    pairs.sort()
    attempts = 0
    while pairs and len(out) < n_homogeneous + n_mixed and attempts < 20 * max(1, n_mixed):
        attempts += 1
        group, a, b = rng.choice(pairs)
        n_a = round(need * rng.choice(mixed_fractions))
        n_b = need - n_a
        if n_b == 0 or (not with_replacement and (len(by_group_label[(group, a)]) < n_a
                                                  or len(by_group_label[(group, b)]) < n_b)):
            continue
        tiles = draw(by_group_label[(group, a)], n_a) + draw(by_group_label[(group, b)], n_b)
        rng.shuffle(tiles)
        out.append(tiles)
    return out
