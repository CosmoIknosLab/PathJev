"""Evaluation sets for plan amendment A3 (docs/stage1_analysis_plan.md), built from the capped Stage 1 test sets.

    python -m pathjev.a3_sets --src $PATHJEV_WORK/eval_sets/stage1-test --out $PATHJEV_WORK/eval_sets

Writes, for the four clean sets only (BRACS, SICAPv2, HepatoBench, HiESD):
  a3-blank/<set>.jsonl     A3.2: 100 states of each set, taken round-robin over the classes of the main
                           question in file order (deterministic), every image replaced by a constant
                           mid-grey (128,128,128) image of the same size (written to a3-blank/images/);
  a3-perm-<k>/<set>.jsonl  A3.5: every state, Choice candidates in the fixed permutation of seed
                           "perm-<k>" (k = 1..10; `pathjev.variants.permute`, one permutation per
                           (dataset, qid), identical for every state and method; never the original order).
Questions, targets and every other field are unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .variants import permute

CLEAN = ["bracs", "sicapv2", "hepatobench", "hiesd"]
MAIN_Q = {"bracs": "lesion", "sicapv2": "gleason_pattern", "hepatobench": "tissue", "hiesd": "tissue"}
N_BLANK, GREY, N_PERM = 100, (128, 128, 128), 10


def blank_rows(rows: list[dict], img_dir: Path) -> list[dict]:
    img_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for r in rows:
        paths = []
        for i, p in enumerate(r["images"]):
            with Image.open(p) as im:
                size = im.size
            dst = img_dir / f"grey_{size[0]}x{size[1]}.png"   # one file per image size
            if not dst.exists():
                Image.new("RGB", size, GREY).save(dst)
            paths.append(str(dst))
        out.append(dict(r, images=paths, meta=dict(r.get("meta", {}), a3="blank")))
    return out


def balanced_head(rows: list[dict], qid: str, n: int) -> list[dict]:
    """First n states taken round-robin over the gold classes of `qid`, each class in file order."""
    by_class: dict[str, list[dict]] = {}
    for r in rows:
        d = next(d for d in r["decisions"] if d["qid"] == qid)
        by_class.setdefault(max(d["target"], key=d["target"].get), []).append(r)
    queues, out = [list(v) for v in by_class.values()], []
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop(0))
    return out


def permuted_rows(rows: list[dict], seed: str) -> list[dict]:
    return [permute(r, seed=seed) for r in rows]


def _write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    for s in CLEAN:
        rows = [json.loads(l) for l in (args.src / f"{s}.jsonl").open() if l.strip()]
        _write(blank_rows(balanced_head(rows, MAIN_Q[s], N_BLANK), args.out / "a3-blank" / "images"), args.out / "a3-blank" / f"{s}.jsonl")
        for k in range(1, N_PERM + 1):
            _write(permuted_rows(rows, f"perm-{k}"), args.out / f"a3-perm-{k}" / f"{s}.jsonl")
    print(sorted(p.name for p in args.out.glob("a3-*")))


if __name__ == "__main__":
    main()
