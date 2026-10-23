"""Source-agnostic sample records.

Every dataset adapter in `pathjev.sources` turns its native layout into a list
of `Record`s. Everything downstream (templates, splits, states, de-duplication)
reads only these fields, so adding a dataset never touches the state builder.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class Record:
    sample_id: str            # globally unique: "<source>:<native id>"
    source: str               # registry short name, e.g. "bracs", "spider-colorectal"
    organ: str                # "breast", "colorectal", ...
    image: str                # absolute path to one image (patch or ROI)
    label: str                # native class label (template maps it to candidate ids)
    patient_id: str           # "" if unknown -> split falls back to slide_id
    slide_id: str             # "" if unknown -> split falls back to sample_id
    official_split: str = ""  # native split if the dataset defines one ("train"/"val"/"test")
    tier: str = "gold"        # gold | derived | silver (CLAUDE.md §0.4.7)
    extra: tuple = ()         # frozen key/value pairs for adapter-specific fields

    @property
    def group(self) -> str:
        """Unit that must never be split across train/val/test."""
        return self.patient_id or self.slide_id or self.sample_id

    def extras(self) -> dict[str, Any]:
        return dict(self.extra)


def write_records(records: list[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            row = dataclasses.asdict(record)
            row["extra"] = dict(record.extra)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_records(path: Path) -> list[Record]:
    records = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        row["extra"] = tuple(sorted(row.get("extra", {}).items()))
        records.append(Record(**row))
    return records
