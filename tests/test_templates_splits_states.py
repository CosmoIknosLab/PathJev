"""Phase 2 tests: decision templates, patient-level splits, state construction."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from pathjev import splits, states, templates
from pathjev.records import Record, read_records, write_records

REPO = Path(__file__).resolve().parents[1]
TEMPLATE_FILES = sorted((REPO / "configs" / "questions").glob("*.yaml"))


# --- templates -----------------------------------------------------------------

@pytest.mark.parametrize("path", TEMPLATE_FILES, ids=lambda p: p.name)
def test_every_template_validates_and_yields_valid_candidate_sets(path):
    template = templates.load(path)
    for label in template.spec["label_map"]:
        decisions = template.decisions(label)
        assert decisions, label
        for d in decisions:
            ids = [c["id"] for c in d["candidates"]]
            assert len(ids) >= 2 and len(ids) == len(set(ids))
            assert set(d["target"]) == set(ids)
            assert sum(d["target"].values()) == pytest.approx(1.0)
            assert all(v in (0.0, 1.0) for v in d["target"].values())


def test_choice_with_other_includes_distractor_and_none_option():
    template = templates.load(REPO / "configs" / "questions" / "bracs.yaml")
    choice = next(d for d in template.decisions("IC") if d["type"] == "choice")
    ids = [c["id"] for c in choice["candidates"]]
    assert "other" in ids and len(ids) >= 3
    assert choice["target"]["invasive"] == 1.0


def test_validate_rejects_unreachable_label():
    spec = {"template_version": 1, "source": "toy", "organ": "x", "reads": ["label"],
            "candidates": {"a": "A", "b": "B"}, "label_map": {"A": "a", "B": "b"},
            "decisions": [{"qid": "q", "type": "choice", "instructions": "?", "candidates": ["a"]}]}
    with pytest.raises(templates.TemplateError):
        templates.validate(spec)


# --- splits ---------------------------------------------------------------------

def _records(n_patients=40, per_patient=5, official=False):
    out = []
    for p in range(n_patients):
        split = "test" if p % 5 == 0 else "train"
        for s in range(per_patient):
            out.append(Record(sample_id=f"toy:{p}:{s}", source="toy", organ="x", image="/dev/null",
                              label="A" if s % 2 else "B", patient_id=f"P{p}", slide_id=f"S{p}",
                              official_split=split if official else ""))
    return out


@pytest.mark.parametrize("mode", ["hash", "official"])
def test_splits_are_patient_disjoint_and_deterministic(mode):
    records = _records(official=(mode == "official"))
    a = splits.assign(records, mode=mode)
    b = splits.assign(records, mode=mode)
    assert a == b
    splits.check_disjoint(records, a)
    assert set(a.values()) == {"train", "val", "test"}


def test_leak_is_detected():
    records = _records()
    assignment = {r.sample_id: "train" for r in records}
    assignment[records[1].sample_id] = "test"          # same patient as records[0]
    with pytest.raises(splits.SplitLeakError):
        splits.check_disjoint(records, assignment)


def test_manifest_roundtrip_and_hash(tmp_path):
    records = _records()
    assignment = splits.assign(records, mode="hash")
    h1 = splits.write_manifest(records, assignment, tmp_path / "m.csv")
    h2 = splits.write_manifest(records, assignment, tmp_path / "m2.csv")
    assert h1 == h2 and splits.read_manifest(tmp_path / "m.csv") == assignment


def test_records_roundtrip(tmp_path):
    records = _records(3, 2)
    write_records(records, tmp_path / "r.jsonl")
    assert read_records(tmp_path / "r.jsonl") == records


# --- states ---------------------------------------------------------------------

def _image_records(tmp_path, labels=("N", "IC"), groups=2, per=9):
    out = []
    for g in range(groups):
        for label in labels:
            for i in range(per):
                path = tmp_path / f"{g}_{label}_{i}.png"
                Image.new("RGB", (64, 64), (40 * i % 255, 80, 120)).save(path)
                out.append(Record(sample_id=f"bracs:{g}:{label}:{i}", source="bracs", organ="breast",
                                  image=str(path), label=label, patient_id=f"P{g}", slide_id=f"S{g}"))
    return out


def test_patch_state_has_no_dataset_name_in_prompt(tmp_path):
    template = templates.load(REPO / "configs" / "questions" / "bracs.yaml")
    record = _image_records(tmp_path, per=1)[0]
    state = states.patch_state(record, template, "train")
    assert state["source"] == "" and state["modality"] == states.MODALITY
    assert state["meta"]["dataset"] == "bracs"


def test_mosaics_never_mix_groups_and_targets_are_distributions(tmp_path):
    template = templates.load(REPO / "configs" / "questions" / "bracs.yaml")
    records = _image_records(tmp_path)
    groups = states.sample_mosaics(records, k=2, n_homogeneous=4, n_mixed=4)
    assert len(groups) == 8
    for tiles in groups:
        assert len({r.group for r in tiles}) == 1
        state = states.mosaic_state(tiles, template, "train", k=2, tile_px=32, tile_mode="resize",
                                    image_dir=tmp_path / "mosaics")
        assert Image.open(state["images"][0]).size == (64, 64)
        homogeneous = len({r.label for r in tiles}) == 1
        for d in state["decisions"]:
            assert sum(d["target"].values()) == pytest.approx(1.0)
        hom = next(d for d in state["decisions"] if d["qid"] == "homogeneous")
        assert hom["target"]["yes"] == float(homogeneous)


def test_states_render_with_upstream_decision_block(tmp_path):
    medjev = pytest.importorskip("omnimedjev.engine.medjev")
    template = templates.load(REPO / "configs" / "questions" / "bracs.yaml")
    record = _image_records(tmp_path, per=1)[0]
    text = medjev.decision_block(states.patch_state(record, template, "test")["decisions"])
    n_candidates = sum(len(d["candidates"]) for d in template.decisions(record.label))
    assert text.count("⟩") == n_candidates and text.count("✦") == len(template.spec["decisions"])


def test_mosaics_with_replacement_from_few_large_rois(tmp_path):
    template = templates.load(REPO / "configs" / "questions" / "bracs.yaml")
    records = _image_records(tmp_path, per=1)            # one RoI per (group, label)
    assert states.sample_mosaics(records, k=3, n_homogeneous=3, n_mixed=0) == []
    groups = states.sample_mosaics(records, k=3, n_homogeneous=3, n_mixed=2, with_replacement=True)
    assert len(groups) == 5 and all(len({r.group for r in g}) == 1 for g in groups)
    state = states.mosaic_state(groups[0], template, "train", k=3, tile_px=16, tile_mode="crop:2",
                                image_dir=tmp_path / "m")
    assert Image.open(state["images"][0]).size == (48, 48)
