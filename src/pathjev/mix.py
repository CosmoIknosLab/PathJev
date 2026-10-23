"""Assemble a training/validation mix from per-source state files.

    python -m pathjev.mix --config configs/train/pathjev-p.yaml

Config:
    name: str
    seed: int
    train: [{states: <path relative to $PATHJEV_WORK>, cap: int}, ...]
    val:   [{states: ..., cap: int}, ...]
Sampling inside a file is stratified by the gold candidate of its first
decision (round-robin over classes), so a cap keeps rare classes. Output:
$PATHJEV_WORK/mixes/<name>/{train,val}.jsonl and mix_info.json (input sha256,
counts per source and class) — the mix hashes are logged by every training run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from pathjev.paths import work_root


def _gold(state: dict) -> str:
    target = state["decisions"][0]["target"]
    return max(target, key=target.get)


def stratified_sample(rows: list[dict], cap: int, rng: random.Random) -> list[dict]:
    if cap <= 0 or len(rows) <= cap:
        return list(rows)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[_gold(row)].append(row)
    pools = [rng.sample(v, len(v)) for _, v in sorted(by_class.items())]
    out: list[dict] = []
    while len(out) < cap:
        for pool in pools:
            if pool and len(out) < cap:
                out.append(pool.pop())
    return out


def build(config: Path) -> dict:
    cfg = yaml.safe_load(config.read_text())
    work = work_root()
    out = work / "mixes" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    info = {"config": str(config), "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()[:16],
            "inputs": {}, "counts": {}}
    for split in ("train", "val"):
        rng = random.Random(f"{cfg['seed']}:{split}")
        chosen: list[dict] = []
        for entry in cfg.get(split, []):
            path = work / entry["states"]
            raw = path.read_bytes()
            info["inputs"][entry["states"]] = hashlib.sha256(raw).hexdigest()[:16]
            rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
            picked = stratified_sample(rows, entry.get("cap", 0), rng)
            info["counts"][f"{split}:{entry['states']}"] = {"available": len(rows), "used": len(picked),
                                                            "classes": dict(Counter(map(_gold, picked)))}
            chosen.extend(picked)
        rng.shuffle(chosen)
        target = out / f"{split}.jsonl"
        target.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in chosen))
        info[f"{split}_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()[:16]
        info[f"{split}_states"] = len(chosen)
    (out / "mix_info.json").write_text(json.dumps(info, indent=1))
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    info = build(ap.parse_args().config.resolve())
    print(json.dumps({k: v for k, v in info.items() if k != "counts"}, indent=1))


if __name__ == "__main__":
    main()
