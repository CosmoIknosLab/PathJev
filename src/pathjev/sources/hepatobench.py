"""HepatoBench (liver; Xiao et al., arXiv 2604.22858; HF xtxx/HepatoBench).

Seven per-class zips of 150×150 px patches at 20× (01_TUM ... 07_STE). No slide
or patient IDs are released, so the whole dataset is one held-out test set
(research_plan §3.3) and bootstrap CIs are over patches (stated as such).
Licence conflict: paper CC BY-NC 4.0 vs card CC BY 4.0 → treated as CC BY-NC.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from pathjev.paths import data_root
from pathjev.records import Record

ROOT = data_root() / "hepatobench"
CLASSES = {"01_TUM": "TUM", "02_FIB": "FIB", "03_INF": "INF", "04_NEC": "NEC",
           "05_NOR": "NOR", "06_REA": "REA", "07_STE": "STE"}


def _extract(root: Path) -> Path:
    out = root / "extracted"
    done = out / ".done"
    if not done.exists():
        for name in CLASSES:
            with zipfile.ZipFile(root / f"{name}.zip") as archive:
                archive.extractall(out / name)
        done.touch()
    return out


def load(root: Path = ROOT) -> list[Record]:
    out = _extract(root)
    records = []
    for folder, label in CLASSES.items():
        for image in sorted((out / folder).rglob("*.png")):
            records.append(Record(sample_id=f"hepatobench:{image.stem}", source="hepatobench", organ="liver",
                                  image=str(image), label=label, patient_id="", slide_id="",
                                  official_split="test", tier="gold"))
    return records
