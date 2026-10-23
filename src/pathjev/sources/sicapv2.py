"""SICAPv2 (prostate; Silva-Rodríguez et al., CMPB 2020; Mendeley 10.17632/9xxm58dvs3.2, CC BY 4.0).

512-px patches at 10× (50% overlap) from 182 WSIs / 95 patients (Hospital
Clínico de Valencia). Official patient-based partition `partition/Test/{Train,Test}.xlsx`
(labels NC/G3/G4/G5 = majority of the pixel mask; G4C = cribriform). Patient IDs
from `wsi_labels.xlsx`. Used for training (train partition) and as an in-domain
test (test partition) after DiagSet became unavailable (D-016).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from pathjev.paths import data_root
from pathjev.records import Record

ROOT = data_root() / "sicapv2"
GRADES = ("NC", "G3", "G4", "G5")


def _extract(root: Path) -> Path:
    """Extract images and tables only (pixel masks are not used); resumable, skips files already present."""
    out = root / "extracted"
    if not (out / ".done").exists():
        with zipfile.ZipFile(root / "SICAPv2.zip") as archive:
            for info in archive.infolist():
                if info.is_dir() or "/masks/" in info.filename:
                    continue
                target = out / info.filename
                if target.exists() and target.stat().st_size == info.file_size:
                    continue
                archive.extract(info, out)
        (out / ".done").touch()
    return out / "SICAPv2"


def load(root: Path = ROOT) -> list[Record]:
    import pandas as pd

    base = _extract(root)
    labels = pd.read_excel(base / "wsi_labels.xlsx")
    patient_of = {str(s): str(p) for s, p in zip(labels["slide_id"], labels["patient_id"])}
    records = []
    for split, table in (("train", "Train.xlsx"), ("test", "Test.xlsx")):
        df = pd.read_excel(base / "partition" / "Test" / table)
        for row in df.itertuples(index=False):
            onehot = [int(getattr(row, g)) for g in GRADES]
            if sum(onehot) != 1:
                raise ValueError(f"{row.image_name}: expected exactly one grade, got {onehot}")
            slide = row.image_name.split("_")[0]
            records.append(Record(
                sample_id=f"sicapv2:{Path(row.image_name).stem}", source="sicapv2", organ="prostate",
                image=str(base / "images" / row.image_name), label=GRADES[onehot.index(1)],
                patient_id=patient_of[slide], slide_id=slide, official_split=split, tier="derived",
                extra=(("cribriform", int(row.G4C)), ("magnification", "10x"))))
    return records
