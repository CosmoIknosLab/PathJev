"""Fixed evaluation subsets shared by every arm (same states, same order).

    python -m pathjev.evalsets --config configs/eval/pilot-stage1.yaml

Config: name, seed, [drop_unreadable: true], sets: [{name, states: <path relative to $PATHJEV_WORK>, cap}].
drop_unreadable removes states whose image file is empty (listed in info.json).
Sampling is the class-stratified draw used for training mixes (pathjev.mix). Output:
$PATHJEV_WORK/eval_sets/<name>/<set>.jsonl and info.json (input/output sha256, counts).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import yaml

from pathjev.mix import _gold, stratified_sample
from pathjev.paths import work_root


def build(config: Path) -> dict:
    cfg = yaml.safe_load(config.read_text())
    work = work_root()
    out = work / "eval_sets" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    info = {"config": str(config), "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()[:16], "sets": {}}
    for entry in cfg["sets"]:
        raw = (work / entry["states"]).read_bytes()
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        dropped = []
        if cfg.get("drop_unreadable"):   # e.g. 0-byte images shipped in a source release (D-030)
            dropped = [r["state_id"] for r in rows if any(Path(p).stat().st_size == 0 for p in r["images"])]
            rows = [r for r in rows if r["state_id"] not in set(dropped)]
        picked = stratified_sample(rows, entry.get("cap", 0), random.Random(f"{cfg['seed']}:{entry['name']}"))
        target = out / f"{entry['name']}.jsonl"
        target.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in picked))
        info["sets"][entry["name"]] = {"input": entry["states"], "input_sha256": hashlib.sha256(raw).hexdigest()[:16],
                                       "available": len(rows), "used": len(picked), "dropped_unreadable": dropped,
                                       "sha256": hashlib.sha256(target.read_bytes()).hexdigest()[:16],
                                       "classes": dict(Counter(map(_gold, picked)))}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    print(json.dumps(build(ap.parse_args().config.resolve()), indent=1))


if __name__ == "__main__":
    main()
