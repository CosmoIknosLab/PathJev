"""Extract only the labelled central patches of SPIDER from its split tar archives.

SPIDER ships ~5.3M patch files (central + 24 unlabelled context patches per
sample). PathJev uses only the labelled central patches (D-013), ~0.4M files.
Writing millions of small files over NFS ran at ~12 files/s (job 5052), so this
script (1) streams the concatenated tar parts once to find `metadata.json`,
(2) streams them again writing only the central patches, and is meant to run
close to the data to limit small-file NFS traffic.

    python -m data.spider_extract --root $PATHJEV_DATA/spider [--organs breast ...]
Output: <root>/SPIDER-<organ>/extracted/SPIDER-<organ>/{metadata.json, images/<central>.png, .done}
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from pathlib import Path


class Chain(io.RawIOBase):
    """Read-only stream over several files in order (the split tar parts)."""

    def __init__(self, paths: list[Path]):
        self.paths, self.index, self.handle = list(paths), 0, None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while self.index < len(self.paths):
            if self.handle is None:
                self.handle = self.paths[self.index].open("rb")
            n = self.handle.readinto(buffer)
            if n:
                return n
            self.handle.close()
            self.handle, self.index = None, self.index + 1
        return 0


def _stream(parts: list[Path]) -> tarfile.TarFile:
    return tarfile.open(fileobj=io.BufferedReader(Chain(parts), buffer_size=16 << 20), mode="r|")


def extract_organ(organ_dir: Path) -> None:
    out = organ_dir / "extracted" / organ_dir.name
    if (out / ".done").exists():
        print(f"[skip] {organ_dir.name}", flush=True)
        return
    parts = sorted(organ_dir.glob("spider-*.tar.*"))
    started = time.time()
    meta_path = out / "metadata.json"
    if not meta_path.exists():
        with _stream(parts) as tar:
            for member in tar:
                if member.isfile() and member.name.endswith("metadata.json"):
                    out.mkdir(parents=True, exist_ok=True)
                    meta_path.write_bytes(tar.extractfile(member).read())
                    break
        print(f"[meta] {organ_dir.name} {time.time() - started:.0f}s", flush=True)
    wanted = {rec["image_name"] for rec in json.loads(meta_path.read_text())}
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    written = 0
    with _stream(parts) as tar:
        for member in tar:
            name = Path(member.name).name
            if member.isfile() and name in wanted:
                target = images / name
                if not (target.exists() and target.stat().st_size == member.size):
                    target.write_bytes(tar.extractfile(member).read())
                written += 1
                if written % 20000 == 0:
                    print(f"[img] {organ_dir.name} {written}/{len(wanted)} {time.time() - started:.0f}s", flush=True)
    if written != len(wanted):
        raise RuntimeError(f"{organ_dir.name}: found {written} of {len(wanted)} central patches")
    (out / ".done").touch()
    print(f"[done] {organ_dir.name} {written} central patches in {time.time() - started:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--organs", nargs="*", default=["breast", "colorectal", "thorax", "skin"])
    args = ap.parse_args()
    for organ in args.organs:
        extract_organ(args.root / f"SPIDER-{organ}")


if __name__ == "__main__":
    main()
