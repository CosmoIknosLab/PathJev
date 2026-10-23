"""Resampling / cross-fitting groups and test-state exclusions for the Stage 1 analysis.

Plan §3 (docs/stage1_analysis_plan.md): groups are patients, else slides, else samples. The state files
carry the finest group their source releases (`meta.group`). Where a patient map was obtained later it is
applied here, from a versioned manifest, so every consumer (bootstrap, probe cross-fit, tables) uses the same
groups:

- HiESD: `meta.group` is the slide id; `data/manifests/hiesd_patient_slide_mapping.csv` (official map,
  HF JSGe-AI/HiESD rev 89878d18) gives the patient. Every HiESD slide must be mapped (KeyError otherwise).

`data/manifests/stage1_test_exclusions.csv` lists test states removed for every method because they
duplicate a training/validation image (near-duplicate audit, job 7548).
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = REPO / "data" / "manifests"


@lru_cache(maxsize=None)
def group_maps() -> dict[str, dict[str, str]]:
    """{dataset: {meta.group: analysis group}} for datasets whose groups are refined by a manifest."""
    with (MANIFESTS / "hiesd_patient_slide_mapping.csv").open(encoding="utf-8-sig") as fh:
        hiesd = {r["slide_id"]: r["patient_id"] for r in csv.DictReader(fh)}
    return {"hiesd": hiesd}


def state_group(state: dict) -> str:
    """Analysis group of one state: patient where a map exists, else the state's own `meta.group`."""
    meta = state.get("meta", {})
    g = str(meta.get("group", state["state_id"]))
    m = group_maps().get(meta.get("dataset", ""))
    if m is None:
        return g
    if g not in m:
        raise KeyError(f"{state['state_id']}: group {g!r} missing from the {meta.get('dataset')} patient map")
    return m[g]


@lru_cache(maxsize=None)
def excluded_state_ids() -> frozenset[str]:
    path = MANIFESTS / "stage1_test_exclusions.csv"
    if not path.exists():
        return frozenset()
    with path.open() as fh:
        return frozenset(r["state_id"] for r in csv.DictReader(fh))
