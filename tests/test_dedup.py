"""Near-duplicate detection must catch re-encoded/resized copies and ignore distinct images."""
from __future__ import annotations

import numpy as np
from PIL import Image

from pathjev import dedup


def _texture(seed: int, size: int = 256) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size // 8, size // 8, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((size, size), Image.Resampling.BICUBIC)


def test_copies_are_found_and_distinct_images_are_not(tmp_path):
    originals, copies = [], []
    for seed in range(20):
        image = _texture(seed)
        a = tmp_path / f"orig_{seed}.png"
        image.save(a)
        originals.append(str(a))
        b = tmp_path / f"copy_{seed}.jpg"
        image.resize((224, 224)).save(b, quality=85)      # resized + JPEG re-encoded copy
        copies.append(str(b))
    ref = np.array([dedup.image_hashes(p) for p in originals], dtype=np.uint64)
    qry = np.array([dedup.image_hashes(p) for p in copies], dtype=np.uint64)
    pairs = dedup.near_duplicates(qry, ref, max_bits=3)
    found = {(i, j) for i, j, _, _ in pairs}
    assert {(i, i) for i in range(20)} <= found          # every copy matched to its original
    assert all(i == j for i, j in found)                 # and nothing else


def test_banding_is_exact_for_threshold_below_band_count():
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 2**63, size=(500, 2), dtype=np.uint64)
    qry = ref.copy()
    flips = np.uint64(0b1011)                            # 3 bits flipped in both hashes
    qry ^= flips
    pairs = dedup.near_duplicates(qry, ref, max_bits=3)
    assert {(i, j) for i, j, _, _ in pairs} >= {(i, i) for i in range(500)}
