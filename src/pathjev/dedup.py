"""Near-duplicate detection across train and test (CLAUDE.md §0.4.5).

Each image gets a 64-bit pHash and a 64-bit dHash (on a downscaled copy, so
crops of large RoIs and re-encoded/resized copies hash alike). A pair is a
near-duplicate when BOTH Hamming distances are <= `max_bits` (default 3).
Candidate pairs come from banding: the 64-bit pHash is cut into 4 bands of 16
bits; by the pigeonhole principle two hashes within 3 bits share at least one
identical band, so banding finds every pair the threshold accepts.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pathjev.paths import work_root
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
BANDS = 4
BAND_BITS = 64 // BANDS


def image_hashes(path: str) -> tuple[int, int]:
    import imagehash

    with Image.open(path) as image:
        image.draft("RGB", (512, 512))          # fast JPEG downscale when possible
        image = image.convert("RGB")
        image.thumbnail((512, 512))
        return int(str(imagehash.phash(image)), 16), int(str(imagehash.dhash(image)), 16)


def hash_many(paths: list[str], workers: int = 8) -> np.ndarray:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return np.array(list(pool.map(image_hashes, paths, chunksize=64)), dtype=np.uint64)


def hamming(a: int, b: int) -> int:
    return bin(int(a) ^ int(b)).count("1")


def near_duplicates(query: np.ndarray, reference: np.ndarray, max_bits: int = 3) -> list[tuple[int, int, int, int]]:
    """Return (query_index, reference_index, phash_bits, dhash_bits) for every near-duplicate pair."""
    if max_bits >= BANDS:
        raise ValueError(f"banding with {BANDS} bands is exact only for max_bits < {BANDS}")
    buckets: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(BANDS)]
    mask = (1 << BAND_BITS) - 1
    for j, (ph, _) in enumerate(reference):
        for band in range(BANDS):
            buckets[band][(int(ph) >> (band * BAND_BITS)) & mask].append(j)
    out = []
    for i, (ph, dh) in enumerate(query):
        candidates = set()
        for band in range(BANDS):
            candidates.update(buckets[band].get((int(ph) >> (band * BAND_BITS)) & mask, ()))
        for j in candidates:
            p_bits = hamming(ph, reference[j][0])
            if p_bits <= max_bits:
                d_bits = hamming(dh, reference[j][1])
                if d_bits <= max_bits:
                    out.append((i, j, p_bits, d_bits))
    return out


def write_report(pairs, query_ids: list[str], reference_ids: list[str], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["query_sample", "reference_sample", "phash_bits", "dhash_bits"])
        for i, j, p, d in pairs:
            writer.writerow([query_ids[i], reference_ids[j], p, d])
    return len(pairs)


def cached_hashes(records_path: Path, cache_dir: Path, workers: int = 8) -> tuple[list[str], np.ndarray]:
    """Hashes for every record in a records.jsonl, cached per file content."""
    import hashlib
    import json

    key = hashlib.sha256(records_path.read_bytes()).hexdigest()[:16]
    cache = cache_dir / f"{records_path.parent.name}_{key}.npz"
    ids, paths = [], []
    for line in records_path.open():
        row = json.loads(line)
        ids.append(row["sample_id"])
        paths.append(row["image"])
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        if list(data["ids"]) == ids:
            return ids, data["hashes"]
    hashes = hash_many(paths, workers)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=np.array(ids), hashes=hashes)
    return ids, hashes


def main() -> None:
    """Report near-duplicates between training records and test records.

    python -m pathjev.dedup --train <records.jsonl>... --test <records.jsonl>... --out <report.csv>
    """
    import argparse
    import os

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--train", nargs="+", type=Path, required=True)
    ap.add_argument("--test", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-bits", type=int, default=3)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()
    cache = work_root() / "hash_cache"
    ref_ids, ref = [], []
    for p in args.train:
        ids, h = cached_hashes(p, cache, args.workers)
        ref_ids += ids
        ref.append(h)
    qry_ids, qry = [], []
    for p in args.test:
        ids, h = cached_hashes(p, cache, args.workers)
        qry_ids += ids
        qry.append(h)
    pairs = near_duplicates(np.concatenate(qry), np.concatenate(ref), args.max_bits)
    n = write_report(pairs, qry_ids, ref_ids, args.out)
    print(f"{n} near-duplicate pairs between {len(qry_ids)} test and {len(ref_ids)} train images -> {args.out}")


if __name__ == "__main__":
    main()
