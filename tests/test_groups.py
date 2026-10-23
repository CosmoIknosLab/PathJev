"""Analysis groups (plan §3: patient, else slide, else sample) and test-state exclusions."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from pathjev.groups import excluded_state_ids, group_maps, state_group  # noqa: E402


def _state(sid, dataset, group):
    return {"state_id": sid, "meta": {"dataset": dataset, "group": group}}


def test_hiesd_map_is_complete_and_official():
    m = group_maps()["hiesd"]
    assert len(m) == 104 and len(set(m.values())) == 44
    # demo slides (reviewer check, 2026-10-23)
    by_prefix = {k[:8]: v for k, v in m.items()}
    assert by_prefix["1a7624bc"] == by_prefix["e492f29a"] == "Patient22"
    assert by_prefix["46e7dccd"] == "Patient27" and by_prefix["c6a80707"] == "Patient38"


def test_state_group_maps_hiesd_and_leaves_others():
    slide = next(iter(group_maps()["hiesd"]))
    assert state_group(_state("x", "hiesd", slide)) == group_maps()["hiesd"][slide]
    assert state_group(_state("y", "bracs", "23")) == "23"
    assert state_group({"state_id": "z", "meta": {"dataset": "spider-skin"}}) == "z"
    with pytest.raises(KeyError):
        state_group(_state("w", "hiesd", "not-a-slide"))


def test_load_main_drops_excluded_states(tmp_path):
    from stage1_analysis import load_main

    excluded = sorted(excluded_state_ids())
    assert "spider-skin:patch_0789779" in excluded and "spider-skin:patch_0968213" in excluded
    cands = [{"id": "a"}, {"id": "b"}]
    ids = [excluded[0], "spider-skin:keep_1", "spider-skin:keep_2"]
    states = [{"state_id": i, "meta": {"dataset": "spider-skin", "group": f"slide_{k}"},
               "decisions": [{"qid": "tissue", "candidates": cands, "target": {"a": 1.0, "b": 0.0}}]}
              for k, i in enumerate(ids)]
    sp, dp = tmp_path / "s.jsonl", tmp_path / "d.jsonl"
    sp.write_text("".join(json.dumps(s) + "\n" for s in states))
    dp.write_text("".join(json.dumps({"state_id": i, "distributions": {"tissue": {"a": 0.7, "b": 0.3}}}) + "\n"
                          for i in ids))
    d = load_main(sp, dp, "tissue")
    assert len(d["y"]) == 2 and list(d["groups"]) == ["slide_1", "slide_2"] and d["missing"] == 0
