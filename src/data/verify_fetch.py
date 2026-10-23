"""Re-verify downloaded/copied sources against their *_fetch_manifest.json (size, md5/sha256).

    python -m data.verify_fetch --data $PATHJEV_DATA [--sources sicapv2 hiesd ...] --out report.json

Exit code 1 if any listed file is missing or does not match. Sources whose manifest
records only remote archives that were not kept (SPIDER tar parts) are reported as
"not_kept" and checked indirectly through the rebuilt split manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected(manifest: dict) -> dict[str, dict]:
    if "file" in manifest:                                   # single archive (SICAPv2)
        return {manifest["file"]: {"sha256": manifest.get("sha256")}}
    files = manifest.get("files", {})
    out = {}
    for name, meta in files.items():
        if isinstance(meta, dict):
            entry = {"size": meta.get("size")}
            if meta.get("md5"):
                entry["md5"] = meta["md5"]
            if meta.get("checksum"):
                algo, digest = meta["checksum"].split(":", 1)
                entry[algo] = digest
        else:
            entry = {"size": meta}
        out[name] = entry
    return out


def _targets(data: Path, source: str, manifest: dict):
    """(path, expected) pairs; SPIDER's manifest is nested per HF repo."""
    nested = {k: v for k, v in manifest.items() if isinstance(v, dict) and "files" in v and "/" in k}
    if nested:
        for repo, sub in nested.items():
            for name, exp in _expected(sub).items():
                yield name, data / source / repo.split("/")[1] / name, exp
    else:
        for name, exp in _expected(manifest).items():
            yield name, data / source / name, exp


def verify(data: Path, source: str) -> dict:
    manifest = json.loads((data / f"{source}_fetch_manifest.json").read_text())
    result = {"ok": 0, "bad": [], "missing": [], "not_kept": 0}
    for name, path, exp in _targets(data, source, manifest):
        if not path.exists():
            if source == "spider" and ".tar." in name:
                result["not_kept"] += 1
            else:
                result["missing"].append(name)
            continue
        if exp.get("size") is not None and path.stat().st_size != exp["size"]:
            result["bad"].append(f"{name}: size")
            continue
        for algo in ("sha256", "md5"):
            if exp.get(algo) and _digest(path, algo) != exp[algo]:
                result["bad"].append(f"{name}: {algo}")
                break
        else:
            result["ok"] += 1
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--sources", nargs="*")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    sources = args.sources or sorted(p.name.removesuffix("_fetch_manifest.json")
                                     for p in args.data.glob("*_fetch_manifest.json"))
    report = {s: verify(args.data, s) for s in sources}
    args.out.write_text(json.dumps(report, indent=1))
    failed = [s for s, r in report.items() if r["bad"] or r["missing"]]
    for s, r in report.items():
        print(s, {k: (len(v) if isinstance(v, list) else v) for k, v in r.items()})
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
