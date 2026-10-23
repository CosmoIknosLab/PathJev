"""Plan A3: the Gen-SFT read-out variants leave the primary distribution bit-identical, and the
A3 evaluation-set builders (blank images, seeded permutations) behave as specified."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pathjev import generative
from pathjev.generative import Scorer

VOCAB, EOS = 50, 7


class FakeTokenizer:
    def convert_tokens_to_ids(self, tok):
        assert tok == "<end_of_turn>"
        return EOS


class FakeModel:
    """Deterministic logits that depend on the whole prefix (as a causal LM's do)."""

    def __call__(self, input_ids, **_):
        ids = input_ids[0].tolist()
        rows = []
        for t in range(len(ids)):
            g = torch.Generator().manual_seed(hash(tuple(ids[: t + 1])) % (2 ** 31))
            rows.append(torch.randn(VOCAB, generator=g))
        return SimpleNamespace(logits=torch.stack(rows)[None].to(torch.bfloat16))


def make_scorer():
    sc = Scorer.__new__(Scorer)
    sc.model, sc.device, sc.image_size = FakeModel(), "cpu", 8
    sc.processor = SimpleNamespace(tokenizer=FakeTokenizer())
    prompt_ids = [1, 2, 3, 4, 5]

    def _inputs(image_paths, prompt, answer=None, system=None):
        ans = [] if answer is None else [10 + (ord(ch) % 30) for ch in answer]
        return {"input_ids": torch.tensor([prompt_ids + ans])}

    sc._inputs = _inputs
    return sc


DECISION = {"qid": "tissue", "instructions": "Which tissue?", "type": "choice",
            "candidates": [{"id": "a", "text": "tumour"}, {"id": "b", "text": "normal tissue"},
                           {"id": "c", "text": "other / none of these"}]}


def test_variants_keep_primary_bit_identical(monkeypatch):
    monkeypatch.setattr(generative, "sft_prompt", lambda d: "prompt")
    sc = make_scorer()
    primary = sc.text_distribution([], DECISION)
    again, variants = sc.text_distribution_variants([], DECISION)
    assert primary == again                                   # exact equality, not approx
    for name in ("eos", "lennorm"):
        assert set(variants[name]) == {"a", "b", "c"}
        assert abs(sum(variants[name].values()) - 1) < 1e-9
    raw = variants["raw"]
    assert raw["b"]["n"] == len("normal tissue")
    # variant (iii) is the softmax of mean log-probabilities
    m = np.array([raw[i]["sum"] / raw[i]["n"] for i in "abc"])
    ref = np.exp(m - m.max()) / np.exp(m - m.max()).sum()
    assert np.allclose([variants["lennorm"][i] for i in "abc"], ref)


def test_permutation_seeds_differ_and_are_shared(tmp_path):
    from pathjev.a3_sets import permuted_rows

    state = {"state_id": "s1", "images": ["x.png"], "meta": {"dataset": "bracs"},
             "decisions": [dict(DECISION, candidates=[{"id": str(i), "text": str(i)} for i in range(8)]),
                           {"qid": "n", "type": "noul", "candidates": [{"id": "no"}, {"id": "yes"}]}]}
    other = dict(state, state_id="s2")
    p1 = permuted_rows([state, other], "perm-1")
    p2 = permuted_rows([state, other], "perm-2")
    order = lambda r: [c["id"] for c in r["decisions"][0]["candidates"]]
    assert order(p1[0]) == order(p1[1])                       # same permutation for every state
    assert order(p1[0]) != order(p2[0])                       # different seeds, different orders
    assert order(p1[0]) != [str(i) for i in range(8)]         # never the original order
    assert p1[0]["decisions"][1] == state["decisions"][1]     # non-Choice decisions untouched


def test_blank_rows_replace_images(tmp_path):
    from PIL import Image

    from pathjev.a3_sets import blank_rows

    src = tmp_path / "img.png"
    Image.new("RGB", (30, 20), (200, 10, 10)).save(src)
    rows = blank_rows([{"state_id": "s", "images": [str(src)], "meta": {}, "decisions": []}], tmp_path / "blank")
    im = Image.open(rows[0]["images"][0])
    assert im.size == (30, 20)
    assert np.all(np.asarray(im) == 128)
    assert json.dumps(rows[0])
