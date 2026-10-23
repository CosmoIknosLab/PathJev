"""BRACS RoIs (breast; Brancati et al., Database 2022). Local copy, see data/registry.csv.

Layout: <root>/BRACS_RoI/latest_version/{train,val,test}/<k>_<LABEL>/BRACS_<wsi>_<LABEL>_<n>.png
and <root>/BRACS.xlsx, sheet `WSI_Information` (WSI Filename, Patient Id, Set).
RoIs are at 40× (0.25 µm/px, Aperio AT2). The official RoI split is
patient-disjoint (D-014) and is kept as `official_split`.
"""
from __future__ import annotations

from pathlib import Path

from pathjev.paths import data_root
from pathjev.records import Record

ROOT = data_root() / "bracs"
MPP = 0.25


def load(root: Path = ROOT) -> list[Record]:
    import pandas as pd

    info = pd.read_excel(root / "BRACS.xlsx", sheet_name="WSI_Information")
    info.columns = [c.strip() for c in info.columns]
    patient_of = {str(w): str(p) for w, p in zip(info["WSI Filename"], info["Patient Id"])}

    records = []
    for split in ("train", "val", "test"):
        for class_dir in sorted((root / "BRACS_RoI" / "latest_version" / split).iterdir()):
            label = class_dir.name.split("_", 1)[1]
            for image in sorted(class_dir.glob("*.png")):
                parts = image.stem.split("_")          # BRACS, <wsi>, <LABEL>, <n>
                wsi = f"{parts[0]}_{parts[1]}"
                if parts[2] != label:
                    raise ValueError(f"{image}: file label {parts[2]} != folder label {label}")
                records.append(Record(
                    sample_id=f"bracs:{image.stem}", source="bracs", organ="breast", image=str(image),
                    label=label, patient_id=patient_of[wsi], slide_id=wsi, official_split=split,
                    tier="gold", extra=(("mpp", MPP),)))
    return records
