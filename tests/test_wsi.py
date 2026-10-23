"""WSI pipeline: clustering and probability maps with abstention on synthetic features."""
from __future__ import annotations

import json

import numpy as np

from pathjev import wsi


def test_cluster_and_maps(tmp_path):
    out = tmp_path / "slideA"
    out.mkdir()
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=6, size=(3, 16))
    X = np.concatenate([c + rng.normal(size=(200, 16)) for c in centers])
    np.savez(out / "features.npz", features=X.astype(np.float32))
    wsi.cluster(out, n_pca=8, k_min=2, k_max=4)
    c = np.load(out / "clusters.npz")
    assert c["membership"].shape == (600, int(c["k"])) and np.allclose(c["membership"].sum(1), 1, atol=1e-4)
    hard = c["membership"].argmax(1)
    # decisions: confident "a" for the clusters holding the first 200 tiles, ambiguous for the rest
    first = set(np.unique(hard[:200]).tolist())
    rows = []
    for k in range(int(c["k"])):
        dist = {"a": 0.9, "b": 0.1} if k in first else {"a": 0.5, "b": 0.5}
        rows.append({"state_id": f"slideA:c{k:02d}:m0", "distributions": {"tissue": dist, "homogeneous": {"no": 0.1, "yes": 0.9}}})
    (tmp_path / "dec.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    wsi.maps(out, tmp_path / "dec.jsonl", tau=0.6)
    m = np.load(out / "maps.npz")
    assert m["patch_probs"].shape == (600, 2)
    assert (m["coverage"][:200] > 0.5).mean() > 0.9           # confident clusters cover their tiles
    assert m["cluster_abstain"].sum() >= 1                    # ambiguous clusters abstain
    assert np.allclose(m["patch_probs"][m["coverage"] > 0].sum(1), 1, atol=1e-5)


def test_otsu_separates_two_modes():
    rng = np.random.default_rng(0)
    low, high = rng.normal(0.1, 0.03, 5000).clip(0, 1), rng.normal(0.6, 0.05, 5000).clip(0, 1)
    t = wsi.threshold_otsu(np.concatenate([low, high]))
    # any threshold in the empty gap is optimal (flat between-class variance); check the split itself
    assert (low < t).mean() > 0.99 and (high > t).mean() > 0.99
