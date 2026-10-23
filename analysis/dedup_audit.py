"""Near-duplicate / provenance audit of the Stage 1 splits (CLAUDE.md §0.4.5).

    python analysis/dedup_audit.py --work $PATHJEV_WORK --out results/runs/stage1-dedup [--workers 32]

Images: Stage 1 training mix (train + val) and every test set, capped (stage1-test) and full
(stage1-test-full). Per unique image file: SHA-256 of the decoded RGB pixels (exact duplicates) and the
pHash/dHash of pathjev.dedup (near-duplicates: both distances <= 3 bits, as implemented there).
All-vs-all pairs are found by the same 4x16-bit pHash banding (exact for <= 3 bits).

Reports (i) train/val vs test hits per test set, (ii) within-test pairs that span different bootstrap
groups, (iii) cross-dataset pairs, and, for capped test sets with train/val hits, the main-Choice
bAcc/ECE of PathJev-P and Gen-SFT with the hit states removed (from saved decisions).
Hashes are cached in $PATHJEV_WORK/checks/dedup-stage1/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from pathjev import dedup
from stage1_analysis import MAIN_Q, Item, arm_dirs, load_main

Image.MAX_IMAGE_PIXELS = None
MAX_BITS = 3
POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def records(work: Path) -> list[dict]:
    """One record per (role, state): role in train, val, test (capped), test-full."""
    files = [("train", work / "mixes" / "stage1-p" / "train.jsonl"), ("val", work / "mixes" / "stage1-p" / "val.jsonl")]
    files += [("test", f) for f in sorted((work / "eval_sets" / "stage1-test").glob("*.jsonl"))]
    files += [("test-full", f) for f in sorted((work / "eval_sets" / "stage1-test-full").glob("*.jsonl"))]
    out = []
    for role, f in files:
        for line in f.open():
            if not line.strip():
                continue
            s = json.loads(line)
            for p in s["images"]:
                out.append({"role": role, "dataset": s["meta"]["dataset"], "state_id": s["state_id"],
                            "group": str(s["meta"].get("group", s["state_id"])), "path": p})
    return out


def one_image(path: str) -> tuple[int, int, str]:
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    sha = hashlib.sha256(np.array(rgb.shape, dtype=np.int64).tobytes() + rgb.tobytes()).hexdigest()
    ph, dh = dedup.image_hashes(path)
    return ph, dh, sha


def hashes(paths: list[str], cache: Path, workers: int) -> tuple[np.ndarray, list[str]]:
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["paths"]) == paths:
            return z["hashes"], list(z["sha"])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(one_image, paths, chunksize=256))
    h = np.array([(a, b) for a, b, _ in res], dtype=np.uint64)
    sha = [c for _, _, c in res]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, paths=np.array(paths), hashes=h, sha=np.array(sha))
    return h, sha


def popcount(x: np.ndarray) -> np.ndarray:
    return POP8[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1)


def all_pairs(h: np.ndarray, max_bits: int = MAX_BITS, chunk: int = 512) -> tuple[np.ndarray, dict]:
    """(i, j) with i < j and pHash and dHash distances both <= max_bits; banding as in pathjev.dedup."""
    ph, dh = h[:, 0], h[:, 1]
    mask = np.uint64((1 << dedup.BAND_BITS) - 1)
    found, largest = set(), 0
    for band in range(dedup.BANDS):
        key = (ph >> np.uint64(band * dedup.BAND_BITS)) & mask
        order = np.argsort(key, kind="stable")
        ks = key[order]
        cuts = np.flatnonzero(np.diff(ks)) + 1
        for g in np.split(order, cuts):
            if len(g) < 2:
                continue
            largest = max(largest, len(g))
            for a in range(0, len(g), chunk):
                rows = g[a:a + chunk]
                dp = popcount(ph[rows][:, None] ^ ph[g][None, :])
                dd = popcount(dh[rows][:, None] ^ dh[g][None, :])
                ii, jj = np.nonzero((dp <= max_bits) & (dd <= max_bits))
                for i, j in zip(rows[ii], g[jj]):
                    if i < j:
                        found.add((int(i), int(j)))
    return np.array(sorted(found), dtype=np.int64).reshape(-1, 2), {"largest_band_bucket": largest}


def effect(work: Path, repo: Path, test_set: str, drop: set[str]) -> dict:
    """Main-Choice bAcc/ECE per seed of PathJev-P and Gen-SFT on a capped test set, all vs hits removed."""
    states = work / "eval_sets" / "stage1-test" / f"{test_set}.jsonl"
    q = MAIN_Q[test_set]
    ids = [s["state_id"] for s in map(json.loads, filter(str.strip, states.open()))
           if any(d["qid"] == q for d in s["decisions"])]
    keep = np.array([i for i, s in enumerate(ids) if s not in drop])
    allidx = np.arange(len(ids))
    out = {"n": len(ids), "removed": len(ids) - len(keep)}
    for arm in ("pathjev", "sft"):
        rows = []
        for tdir, _ in arm_dirs(work, repo, arm, interim=False):
            d = load_main(states, tdir / f"{test_set}_decisions.jsonl", q)
            it = Item(d["probs"], d["y"], d["groups"])
            rows.append({"bacc_all": it.bacc(allidx), "bacc_kept": it.bacc(keep),
                         "ece_all": it.ece(allidx), "ece_kept": it.ece(keep)})
        out[arm] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        out[arm]["per_seed"] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    recs = records(args.work)
    paths = sorted({r["path"] for r in recs})
    pidx = {p: i for i, p in enumerate(paths)}
    by_img = defaultdict(list)
    for r in recs:
        by_img[pidx[r["path"]]].append(r)
    print(f"{len(recs)} records, {len(paths)} unique image files", flush=True)
    h, sha = hashes(paths, args.work / "checks" / "dedup-stage1" / "hashes.npz", args.workers)
    print("hashed", flush=True)

    # pairs of distinct files: near-duplicates (banding) + exact pixel duplicates
    near, info = all_pairs(h)
    exact = set()
    by_sha = defaultdict(list)
    for i, s in enumerate(sha):
        by_sha[s].append(i)
    for members in by_sha.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                exact.add((members[a], members[b]))
    pairs = {tuple(p): "near" for p in near.tolist()}
    for p in exact:
        pairs[p] = "exact"
    print(f"{len(near)} near-duplicate pairs, {len(exact)} exact pixel-duplicate pairs "
          f"(largest band bucket {info['largest_band_bucket']})", flush=True)

    TRAIN, TEST = {"train", "val"}, {"test", "test-full"}
    cross_role = defaultdict(lambda: {"test_states": set(), "train_datasets": defaultdict(int), "examples": []})
    train_val = {"pairs": 0, "examples": []}
    within = defaultdict(lambda: {"pairs": 0, "states": set(), "examples": []})
    cross_ds = defaultdict(lambda: {"pairs": 0, "examples": []})
    same_file_roles = defaultdict(int)
    for i, rs in by_img.items():          # the same file used under two roles
        roles = {r["role"] for r in rs}
        if roles & TRAIN and roles & TEST:
            same_file_roles[",".join(sorted({r["dataset"] for r in rs}))] += 1

    for (i, j), kind in pairs.items():
        for a in by_img[i]:
            for b in by_img[j]:
                for x, y in ((a, b), (b, a)):
                    if x["role"] in TEST and y["role"] in TRAIN:
                        c = cross_role[(x["role"], x["dataset"])]
                        c["test_states"].add(x["state_id"])
                        c["train_datasets"][f"{y['role']}:{y['dataset']}"] += 1
                        if len(c["examples"]) < 5:
                            c["examples"].append([x["state_id"], y["state_id"], kind])
                if a["role"] == "train" and b["role"] == "val" or a["role"] == "val" and b["role"] == "train":
                    train_val["pairs"] += 1
                    if len(train_val["examples"]) < 5:
                        train_val["examples"].append([a["state_id"], b["state_id"], kind])
                if (a["role"] == b["role"] and a["role"] in TEST and a["dataset"] == b["dataset"]
                        and a["group"] != b["group"]):
                    w = within[(a["role"], a["dataset"])]
                    w["pairs"] += 1
                    w["states"].update([a["state_id"], b["state_id"]])
                    if len(w["examples"]) < 5:
                        w["examples"].append([a["state_id"], b["state_id"], kind])
                if a["dataset"] != b["dataset"]:
                    key = tuple(sorted([f"{a['role']}:{a['dataset']}", f"{b['role']}:{b['dataset']}"]))
                    cross_ds[key]["pairs"] += 1
                    if len(cross_ds[key]["examples"]) < 3:
                        cross_ds[key]["examples"].append([a["state_id"], b["state_id"], kind])

    effects = {}
    for (role, ds), c in cross_role.items():
        if role == "test" and ds in MAIN_Q:
            effects[ds] = effect(args.work, repo, ds, c["test_states"])

    summary = {
        "records": len(recs), "unique_images": len(paths), "max_bits": MAX_BITS,
        "near_pairs": len(near), "exact_pairs": len(exact), **info,
        "same_file_in_train_and_test": dict(same_file_roles),
        "cross_role": {f"{r}:{d}": {"test_states_hit": len(c["test_states"]),
                                     "train_side": dict(c["train_datasets"]), "examples": c["examples"]}
                       for (r, d), c in sorted(cross_role.items())},
        "train_vs_val": train_val,
        "within_test_cross_group": {f"{r}:{d}": {"pairs": w["pairs"], "states": len(w["states"]),
                                                 "examples": w["examples"]}
                                    for (r, d), w in sorted(within.items())},
        "cross_dataset": {" | ".join(k): v for k, v in sorted(cross_ds.items())},
        "effect_capped_sets": effects,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "cross_dataset"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
