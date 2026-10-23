"""Build records, the split manifest and decision states for one source.

    python -m pathjev.build --config configs/build/bracs.yaml

Config keys: source (adapter module in pathjev.sources), template (YAML in
configs/questions), split_mode (official|hash|all-test|all-train), seed,
mosaic: {k: [2, 3, 4], per_split: {train: n_hom, ...}, mixed_ratio, tile_px, tile_mode},
out (directory under $PATHJEV_WORK). Writes:
  data/manifests/<source>.csv           (versioned; sha256 printed and stored)
  <out>/records.jsonl, <out>/<split>_patch.jsonl, <out>/<split>_mosaic_k<k>.jsonl
  <out>/build_info.json                 (config hash, manifest hash, template hash, counts)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections import Counter
from pathlib import Path

import yaml

from pathjev.paths import work_root
from pathjev import splits, states, templates
from pathjev.records import write_records

REPO = Path(__file__).resolve().parents[2]


def _write_jsonl(rows, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def build(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    work = work_root()
    out = work / cfg["out"]
    source = importlib.import_module(f"pathjev.sources.{cfg['source']}")
    template = templates.load(REPO / "configs" / "questions" / cfg["template"])
    records = source.load()
    assignment = splits.assign(records, mode=cfg["split_mode"], seed=cfg.get("seed", "pathjev-v1"))
    manifest_hash = splits.write_manifest(records, assignment, REPO / "data" / "manifests" / f"{cfg['source']}.csv")
    write_records(records, out / "records.jsonl")

    counts: dict[str, int] = {}
    for split in splits.SPLITS:
        members = [r for r in records if assignment[r.sample_id] == split]
        if not members:
            continue
        counts[f"{split}_patch"] = _write_jsonl((states.patch_state(r, template, split) for r in members),
                                                out / f"{split}_patch.jsonl")
        mosaic = cfg.get("mosaic") or {}
        n_hom = (mosaic.get("per_split") or {}).get(split, 0)
        for k in mosaic.get("k", []) if n_hom else []:
            groups = states.sample_mosaics(members, k=k, n_homogeneous=n_hom,
                                           n_mixed=int(round(n_hom * mosaic.get("mixed_ratio", 0.5))),
                                           seed=f"{cfg.get('seed', 'pathjev-v1')}:{split}:{k}",
                                           with_replacement=mosaic.get("with_replacement", False))
            rows = (states.mosaic_state(g, template, split, k=k, tile_px=mosaic.get("tile_px", 224),
                                        tile_mode=mosaic.get("tile_mode", "resize"),
                                        image_dir=out / "mosaic_images") for g in groups)
            counts[f"{split}_mosaic_k{k}"] = _write_jsonl(rows, out / f"{split}_mosaic_k{k}.jsonl")

    info = {"config": str(config_path.relative_to(REPO)),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()[:16],
            "manifest": f"data/manifests/{cfg['source']}.csv", "manifest_sha256": manifest_hash,
            "template": template.path.name, "template_sha256": template.sha256,
            "records": len(records), "labels": dict(Counter(r.label for r in records)),
            "split_records": dict(Counter(assignment.values())), "states": counts}
    (out / "build_info.json").write_text(json.dumps(info, indent=2))
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    print(json.dumps(build(args.config.resolve()), indent=2))


if __name__ == "__main__":
    main()
