"""SPIDER (HistAI; CC BY-NC 4.0): colorectal, thorax, breast, skin.

Extracted layout: <root>/SPIDER-<organ>/extracted/<...>/metadata.json + images/.
Each metadata record: slide_id, image_name (labelled 224-px central patch, 20×),
context_info (24 unlabelled neighbours), class, split (train/test, slide-level).
Only the central patch is labelled (D-013). No patient IDs → group = slide.
SPIDER ⊂ HISTAI slides → training only (research_plan §3.2).
"""
from __future__ import annotations

import json
from pathlib import Path

from pathjev.paths import data_root
from pathjev.records import Record

ROOT = data_root() / "spider"
ORGAN = {"colorectal": "colorectal", "thorax": "lung", "breast": "breast", "skin": "skin"}


def _metadata(organ: str, root: Path) -> Path:
    hits = sorted((root / f"SPIDER-{organ}" / "extracted").rglob("metadata.json"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one metadata.json for SPIDER-{organ}, found {hits}")
    return hits[0]


def load_organ(organ: str, root: Path = ROOT) -> list[Record]:
    meta_path = _metadata(organ, root)
    images = meta_path.parent / "images"
    records = []
    for rec in json.loads(meta_path.read_text()):
        split = {"train": "train", "test": "test", "val": "val", "validation": "val"}[rec.get("split", "train")]
        records.append(Record(
            sample_id=f"spider-{organ}:{Path(rec['image_name']).stem}", source=f"spider-{organ}",
            organ=ORGAN[organ], image=str(images / rec["image_name"]), label=rec["class"],
            patient_id="", slide_id=str(rec["slide_id"]), official_split=split, tier="gold",
            extra=(("context_n", len(rec.get("context_info", {}))),)))
    return records


def make_loader(organ: str):
    return lambda root=ROOT: load_organ(organ, root)
