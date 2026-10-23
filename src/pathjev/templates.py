"""Typed-decision templates (CLAUDE.md §0.5).

A template is a versioned YAML file in `configs/questions/` that turns one
native label into a list of decisions in the OmniMed-Jev state format:

    {"qid", "type": "choice"|"noul"|"score", "instructions",
     "candidates": [{"id", "text"}...], "target": {id: prob}, "target_kind"}

Template schema (see configs/questions/bracs.yaml for a worked example):

    template_version: int
    source: str                 # registry short name
    organ: str
    reads: [Record fields]      # documentation of what the template consumes
    candidates: {id: text}      # vocabulary of this template
    label_map: {native label: candidate id}
    decisions:
      - {qid, type: choice, instructions, candidates: [ids], add_other: bool,
         constant_gold: id}     # optional: same answer for every sample (e.g. organ)
      - {qid, type: noul, instructions, positive: [ids]}
      - {qid, type: score, instructions, levels: [{id, text}...], level_of: {candidate id: level id}}
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

OTHER = {"id": "other", "text": "other tissue / none of these"}
NOUL_CANDIDATES = [{"id": "no", "text": "No"}, {"id": "yes", "text": "Yes"}]


class TemplateError(ValueError):
    pass


@dataclass
class Template:
    spec: dict[str, Any]
    path: Path
    sha256: str = ""      # hash of the file content actually loaded (set by `load`)

    @property
    def source(self) -> str:
        return self.spec["source"]

    def candidate_id(self, native_label: str) -> str:
        try:
            return self.spec["label_map"][native_label]
        except KeyError:
            raise TemplateError(f"{self.path.name}: label {native_label!r} not in label_map") from None

    def decisions(self, native_label: str) -> list[dict[str, Any]]:
        """All decisions for one sample whose ground truth is `native_label`."""
        gold = self.candidate_id(native_label)
        texts = self.spec["candidates"]
        out = []
        for spec in self.spec["decisions"]:
            kind = spec["type"]
            if kind == "choice":
                ids = list(spec["candidates"])
                candidates = [{"id": i, "text": texts[i]} for i in ids]
                if spec.get("add_other"):
                    candidates.append(dict(OTHER))
                target_id = spec.get("constant_gold") or (gold if gold in ids else "other")
                if target_id == "other" and not spec.get("add_other"):
                    raise TemplateError(f"{spec['qid']}: gold {gold!r} not among candidates and no 'other'")
                target = {c["id"]: float(c["id"] == target_id) for c in candidates}
            elif kind == "noul":
                candidates = [dict(c) for c in NOUL_CANDIDATES]
                positive = gold in spec["positive"]
                target = {"no": float(not positive), "yes": float(positive)}
            elif kind == "score":
                candidates = [dict(level) for level in spec["levels"]]
                level = spec["level_of"][gold]
                target = {c["id"]: float(c["id"] == level) for c in candidates}
            else:
                raise TemplateError(f"unknown decision type {kind!r}")
            out.append({"qid": spec["qid"], "type": kind, "instructions": spec["instructions"],
                        "candidates": candidates, "target": target, "target_kind": "one_hot"})
        return out


def validate(spec: dict[str, Any], name: str = "template") -> None:
    """Static checks: every label maps, candidate sets are valid, ordinal maps are total."""
    for key in ("template_version", "source", "organ", "reads", "candidates", "label_map", "decisions"):
        if key not in spec:
            raise TemplateError(f"{name}: missing key {key!r}")
    vocab = spec["candidates"]
    if OTHER["id"] in vocab:
        raise TemplateError(f"{name}: candidate id 'other' is reserved")
    for label, cid in spec["label_map"].items():
        if cid not in vocab:
            raise TemplateError(f"{name}: label {label!r} maps to unknown candidate {cid!r}")
    qids = [d["qid"] for d in spec["decisions"]]
    if len(qids) != len(set(qids)):
        raise TemplateError(f"{name}: duplicate qids {qids}")
    mapped = set(spec["label_map"].values())
    for d in spec["decisions"]:
        q = f"{name}:{d['qid']}"
        if not str(d.get("instructions", "")).strip():
            raise TemplateError(f"{q}: empty instructions")
        if d["type"] == "choice":
            ids = d["candidates"]
            if len(ids) != len(set(ids)) or any(i not in vocab for i in ids):
                raise TemplateError(f"{q}: invalid or duplicate candidate ids {ids}")
            if len(ids) + bool(d.get("add_other")) < 2:
                raise TemplateError(f"{q}: needs at least 2 candidates")
            if "constant_gold" in d:
                if d["constant_gold"] not in ids:
                    raise TemplateError(f"{q}: constant_gold not among candidates")
            elif not d.get("add_other") and not mapped <= set(ids):
                raise TemplateError(f"{q}: labels {sorted(mapped - set(ids))} unreachable without add_other")
        elif d["type"] == "noul":
            if not set(d["positive"]) <= set(vocab):
                raise TemplateError(f"{q}: positive ids not in vocabulary")
        elif d["type"] == "score":
            level_ids = [lv["id"] for lv in d["levels"]]
            if len(level_ids) < 2 or len(level_ids) != len(set(level_ids)):
                raise TemplateError(f"{q}: invalid levels {level_ids}")
            missing = mapped - set(d["level_of"])
            if missing or not set(d["level_of"].values()) <= set(level_ids):
                raise TemplateError(f"{q}: level_of incomplete ({sorted(missing)}) or maps to unknown levels")
        else:
            raise TemplateError(f"{q}: unknown type {d['type']!r}")


def load(path: Path) -> Template:
    raw = path.read_bytes()
    spec = yaml.safe_load(raw.decode("utf-8"))
    validate(spec, path.name)
    return Template(spec, path, hashlib.sha256(raw).hexdigest()[:16])
