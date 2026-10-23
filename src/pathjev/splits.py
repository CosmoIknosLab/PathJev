"""Patient-level (or slide-level) split manifests.

Rules (CLAUDE.md §0.4.4): a split never cuts through a group (patient, else
slide, else sample). Official splits are kept when the dataset defines them and
they pass the disjointness check; otherwise groups are assigned by a seeded
hash, so the assignment is stable across machines and Python versions.
Manifests are versioned in `data/manifests/`; every run logs their sha256.
"""
from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from pathjev.records import Record

SPLITS = ("train", "val", "test")


class SplitLeakError(ValueError):
    """A patient/slide group appears in more than one split."""


def _unit_interval(seed: str, group: str) -> float:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign(records: list[Record], *, mode: str, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
           seed: str = "pathjev-v1") -> dict[str, str]:
    """Return {sample_id: split}.

    mode:
      "official"  keep `record.official_split` (val carved from train by hash if absent);
      "hash"      assign each group by `ratios`;
      "all-test"  whole dataset is a held-out test set (no internal split);
      "all-train" whole dataset is training data.
    """
    if mode == "all-test":
        return {r.sample_id: "test" for r in records}
    if mode == "all-train":
        return {r.sample_id: "train" for r in records}

    if mode == "official":
        out = {}
        has_val = any(r.official_split == "val" for r in records)
        val_cut = ratios[1] / (ratios[0] + ratios[1])
        for r in records:
            split = r.official_split
            if split not in SPLITS:
                raise ValueError(f"{r.sample_id}: official split {split!r} not in {SPLITS}")
            if split == "train" and not has_val and _unit_interval(seed + ":val", r.group) < val_cut:
                split = "val"
            out[r.sample_id] = split
        check_disjoint(records, out)
        return out

    if mode == "hash":
        cut_train, cut_val = ratios[0], ratios[0] + ratios[1]
        out = {}
        for r in records:
            u = _unit_interval(seed, r.group)
            out[r.sample_id] = "train" if u < cut_train else ("val" if u < cut_val else "test")
        return out

    raise ValueError(f"unknown split mode {mode!r}")


def check_disjoint(records: list[Record], assignment: dict[str, str]) -> None:
    seen: dict[str, set[str]] = defaultdict(set)
    for r in records:
        seen[r.group].add(assignment[r.sample_id])
    leaks = {g: s for g, s in seen.items() if len(s) > 1}
    if leaks:
        preview = list(leaks.items())[:5]
        raise SplitLeakError(f"{len(leaks)} groups span several splits, e.g. {preview}")


def write_manifest(records: list[Record], assignment: dict[str, str], path: Path) -> str:
    """Write the manifest CSV and return its sha256 (first 16 hex chars)."""
    check_disjoint(records, assignment)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted((r.sample_id, r.source, r.group, assignment[r.sample_id]) for r in records)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["sample_id", "source", "group", "split"])
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def read_manifest(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        return {row["sample_id"]: row["split"] for row in csv.DictReader(handle)}
