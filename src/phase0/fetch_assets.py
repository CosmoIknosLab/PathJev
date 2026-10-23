"""Phase 0: fetch the weights and the MedMNIST+ 224 archives needed to reproduce
OmniMed-Jev's released test numbers, then materialise only the test images the
released states reference.

Usage (inside a CPU Slurm job, see slurm/campaigns/phase0_fetch.sbatch):
    python -m phase0.fetch_assets --work $PATHJEV_WORK --upstream external/OmniMed-Jev

Every download is verified: small archives by the Zenodo MD5, large ones by the
CRC32 of the test members fetched with HTTP ranges (see _fetch_test_members);
HF files by the hub client.
and the resolved revisions are written to <work>/phase0/assets.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

MODELS = {
    # repo_id -> pinned revision (resolved 2026-10-15)
    "google/medgemma-1.5-4b-it": "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b",
    "xmutly/OmniMed-Jev-4B": "3b933ab3fff893e2727c94b63e2ab4cd6e156ac3",
}
ZENODO_RECORD = "10519652"  # MedMNIST+ (CC BY 4.0)
USER_AGENT = "PathJev-research-fetch/0.1 (University of Naples Federico II)"
CONNECTIONS = 8
RANGE_CHUNK = 32 << 20  # 32 MiB per request


def fetch_models(models_dir: Path) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    resolved = {}
    for repo_id, revision in MODELS.items():
        target = models_dir / repo_id.split("/")[1]
        path = snapshot_download(repo_id, revision=revision, local_dir=target)
        resolved[repo_id] = revision
        print(f"[model] {repo_id}@{revision[:8]} -> {path}", flush=True)
    return resolved


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_range(url: str, path: Path, start: int, end: int, tries: int = 6) -> None:
    """Write bytes [start, end] of `url` at the same offset of `path`."""
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                       "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"expected 206 Partial Content, got {response.status}")
                payload = response.read()
            if len(payload) != end - start + 1:
                raise RuntimeError(f"short range {start}-{end}: {len(payload)} bytes")
            with path.open("r+b") as out:
                out.seek(start)
                out.write(payload)
            return
        except Exception as exc:  # network errors are retried; MD5 is the final check
            if attempt == tries:
                raise
            print(f"[range-retry] {path.name} {start}-{end}: {exc}", flush=True)
            time.sleep(5 * attempt)


def _ranged_download(url: str, path: Path, size: int) -> None:
    """Parallel ranged download: Zenodo serves ~2 MB/s per connection, so one
    stream would take hours per archive. 8 connections, as upstream's aria2c
    setup (more were measured upstream to corrupt files); the caller verifies MD5."""
    from concurrent.futures import ThreadPoolExecutor

    with path.open("wb") as out:
        out.truncate(size)
    chunk = RANGE_CHUNK
    ranges = [(s, min(s + chunk, size) - 1) for s in range(0, size, chunk)]
    with ThreadPoolExecutor(max_workers=CONNECTIONS) as pool:
        list(pool.map(lambda r: _fetch_range(url, path, *r), ranges))


FULL_DOWNLOAD_MAX = 300 * 10**6  # archives up to this size are fetched whole (MD5-checked)
TEST_MEMBERS = ("test_images.npy", "test_labels.npy")


def _fetch_test_members(url: str, size: int, dest: Path) -> dict[str, str]:
    """Fetch only the test members of a remote .npz (a zip) into a sparse local
    file at their original offsets, plus the central directory, then verify each
    member's CRC32 against the archive's own directory.

    Zenodo throttles to ~0.1-2.5 MB/s, and the 10 MedMNIST+ archives total ~30 GB
    of which the test split is ~15%; the per-member CRC32 stored by Zenodo's
    original zip replaces the whole-file MD5 as the integrity check.
    """
    with dest.open("wb") as out:
        out.truncate(size)
    tail = min(size, 1 << 20)
    _fetch_range(url, dest, size - tail, size - 1)
    with zipfile.ZipFile(dest) as archive:
        infos = sorted(archive.infolist(), key=lambda i: i.header_offset)
    offsets = [i.header_offset for i in infos] + [size - tail]
    crcs = {}
    for info, next_offset in zip(infos, offsets[1:]):
        if info.filename not in TEST_MEMBERS:
            continue
        start, end = info.header_offset, min(next_offset, size) - 1
        ranges = [(s, min(s + RANGE_CHUNK, end + 1) - 1) for s in range(start, end + 1, RANGE_CHUNK)]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=CONNECTIONS) as pool:
            list(pool.map(lambda r: _fetch_range(url, dest, *r), ranges))
        crcs[info.filename] = f"{info.CRC:08x}"
    with zipfile.ZipFile(dest) as archive:
        for name in TEST_MEMBERS:
            with archive.open(name) as member:  # raises BadZipFile on CRC mismatch
                while member.read(1 << 24):
                    pass
    return crcs


def fetch_medmnist(raw_dir: Path, subsets: list[str]) -> dict[str, dict]:
    request = urllib.request.Request(f"https://zenodo.org/api/records/{ZENODO_RECORD}",
                                     headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        record = json.load(response)
    files = {f["key"]: f for f in record["files"]}
    raw_dir.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for subset in subsets:
        meta = files[f"{subset}_224.npz"]
        expected = meta["checksum"].removeprefix("md5:")
        dest = raw_dir / meta["key"]
        testonly = raw_dir / f"{subset}_224.testonly.npz"
        started = time.time()
        if dest.exists() and dest.stat().st_size == meta["size"] and _md5(dest) == expected:
            print(f"[skip] {dest.name} already verified", flush=True)
            provenance[subset] = {"kind": "full", "md5": expected}
            continue
        if meta["size"] > FULL_DOWNLOAD_MAX:
            if testonly.exists() and testonly.stat().st_size == meta["size"]:
                try:  # resume after a requeue: keep a copy that still verifies
                    with zipfile.ZipFile(testonly) as archive:
                        crcs = {}
                        for name in TEST_MEMBERS:
                            with archive.open(name) as member:
                                while member.read(1 << 24):
                                    pass
                            crcs[name] = f"{archive.getinfo(name).CRC:08x}"
                    print(f"[skip] {testonly.name} already verified", flush=True)
                    provenance[subset] = {"kind": "test-members", "zenodo_md5_of_full_archive": expected,
                                          "member_crc32": crcs}
                    continue
                except (zipfile.BadZipFile, KeyError, OSError):
                    pass
            for attempt in range(1, 4):
                try:
                    crcs = _fetch_test_members(meta["links"]["self"], meta["size"], testonly)
                    break
                except zipfile.BadZipFile as exc:
                    print(f"[retry] {testonly.name}: {exc} (attempt {attempt})", flush=True)
            else:
                raise RuntimeError(f"could not fetch verified test members of {dest.name}")
            print(f"[ok]   {testonly.name} test members, CRC32 verified, {time.time() - started:.0f}s", flush=True)
            provenance[subset] = {"kind": "test-members", "zenodo_md5_of_full_archive": expected,
                                  "member_crc32": crcs}
            continue
        partial = dest.with_suffix(".part")
        for attempt in range(1, 4):
            _ranged_download(meta["links"]["self"], partial, meta["size"])
            if partial.stat().st_size == meta["size"] and _md5(partial) == expected:
                partial.rename(dest)
                print(f"[ok]   {dest.name} {meta['size'] / 1e9:.2f} GB in {time.time() - started:.0f}s", flush=True)
                break
            print(f"[retry] {dest.name}: size/md5 mismatch on attempt {attempt}", flush=True)
        else:
            raise RuntimeError(f"could not download a verified copy of {dest.name}")
        provenance[subset] = {"kind": "full", "md5": expected}
    return provenance


def load_member(npz: Path, name: str) -> np.ndarray:
    """Read one array from an .npz via zipfile (works on the sparse test-only
    copies, whose leading bytes are holes, so np.load cannot sniff them)."""
    with zipfile.ZipFile(npz) as archive, archive.open(f"{name}.npy") as member:
        return np.lib.format.read_array(member, allow_pickle=False)


def npz_for(raw_dir: Path, subset: str) -> Path | None:
    """The full archive if present, else the test-members-only sparse copy."""
    for name in (f"{subset}_224.npz", f"{subset}_224.testonly.npz"):
        if (raw_dir / name).exists():
            return raw_dir / name
    return None


def materialise_test_images(raw_dir: Path, upstream: Path, states_file: Path) -> dict[str, int]:
    """Write the PNGs referenced by the released states, exactly as upstream's
    convert_datasets.py names and encodes them (lossless PNG of the raw array)."""
    wanted: dict[str, list[tuple[int, Path]]] = {}
    for line in states_file.open(encoding="utf-8"):
        state = json.loads(line)
        for image in state["images"]:
            parts = Path(image).stem.rsplit("_", 2)  # e.g. organamnist_test_000074
            if len(parts) != 3 or not parts[2].isdigit():
                continue  # not a MedMNIST image (TXL-PBC, CAMUS, ChestX-Det)
            subset, split, index = parts
            if split != "test" or npz_for(raw_dir, subset) is None:
                continue
            wanted.setdefault(subset, []).append((int(index), upstream / image))
    written = {}
    for subset, items in sorted(wanted.items()):
        images = load_member(npz_for(raw_dir, subset), "test_images")
        for index, dest in items:
            dest.parent.mkdir(parents=True, exist_ok=True)
            image = Image.fromarray(np.asarray(images[index]))
            if image.mode not in ("L", "RGB"):
                image = image.convert("RGB")
            image.save(dest, "PNG", optimize=False, compress_level=1)
        written[subset] = len(items)
        print(f"[img]  {subset}: {len(items)} test images", flush=True)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--upstream", required=True, type=Path)
    ap.add_argument("--subsets", nargs="+", default=[
        "pathmnist", "chestmnist", "dermamnist", "octmnist", "pneumoniamnist", "retinamnist",
        "breastmnist", "bloodmnist", "tissuemnist", "organamnist", "organcmnist", "organsmnist"])
    ap.add_argument("--skip-models", action="store_true")
    args = ap.parse_args()

    out = {"models": {}, "medmnist": {}, "test_images": {}}
    if not args.skip_models:
        out["models"] = fetch_models(args.work / "models")
    raw_dir = args.work / "omnimedjev-datas" / "raw" / "medmnistplus"
    out["medmnist"] = fetch_medmnist(raw_dir, args.subsets)
    out["test_images"] = materialise_test_images(
        raw_dir, args.upstream, args.upstream / "data" / "splits" / "testing_states.jsonl")
    report = args.work / "phase0" / "assets.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(out, indent=2))
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
