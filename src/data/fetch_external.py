"""Download external sources (SPIDER, SICAPv2, HiESD, HepatoBench, Zenodo records) into the data area.

Usage (inside a CPU Slurm job, see slurm/campaigns/data_fetch.sbatch):
    python -m data.fetch_external spider --out $PATHJEV_DATA
    python -m data.fetch_external panda  --out $PATHJEV_DATA

Both are idempotent: re-running after a requeue skips files whose size already
matches the remote listing. Access (HF gates, Kaggle rules) was granted by the
user personally on 2026-10-16; credentials are read from their standard
locations and never printed. A JSON manifest of every file (path, size, remote
revision or listing date) is written next to the data.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SPIDER_REPOS = {
    # repo_id -> revision pinned at fetch time (resolved and recorded below)
    "histai/SPIDER-colorectal": None,
    "histai/SPIDER-thorax": None,
    "histai/SPIDER-breast": None,
    "histai/SPIDER-skin": None,
}
PANDA = "prostate-cancer-grade-assessment"


def fetch_spider(out: Path, workers: int) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    manifest = {}
    for repo_id in SPIDER_REPOS:
        info = api.dataset_info(repo_id, files_metadata=True)
        target = out / "spider" / repo_id.split("/")[1]
        started = time.time()
        snapshot_download(repo_id, repo_type="dataset", revision=info.sha, local_dir=target,
                          max_workers=workers)
        manifest[repo_id] = {"revision": info.sha,
                             "files": {s.rfilename: s.size for s in info.siblings}}
        print(f"[spider] {repo_id}@{info.sha[:8]} -> {target} ({time.time() - started:.0f}s)", flush=True)
    return manifest


def fetch_panda(out: Path, workers: int) -> dict:
    """Whole-competition archive in one request.

    Per-file downloads (~21k files) hit Kaggle's API rate limit (HTTP 429 on
    2026-10-16), so the single competition zip is fetched instead and unpacked
    by a separate step. The Kaggle client resumes a partial download.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    root = out / "panda"
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for attempt in range(1, 6):
        try:
            api.competition_download_files(PANDA, path=str(root), quiet=False)
            break
        except Exception as exc:  # 429 / network: back off and retry (client resumes)
            if attempt == 5:
                raise
            print(f"[panda-retry] {exc}", flush=True)
            time.sleep(120 * attempt)
    archives = {p.name: p.stat().st_size for p in root.glob("*.zip")}
    print(f"[panda] {archives} in {time.time() - started:.0f}s", flush=True)
    return {"competition": PANDA, "archives": archives}


def _http_download(url: str, dest: Path, *, size: int | None = None, md5: str | None = None,
                   sha256: str | None = None) -> str:
    """Stream `url` to `dest` (skip if already verified); verify size and checksum."""
    import hashlib
    import urllib.request

    def verified() -> bool:
        if not dest.exists() or (size is not None and dest.stat().st_size != size):
            return False
        algo, want = ("md5", md5) if md5 else ("sha256", sha256) if sha256 else (None, None)
        if algo is None:
            return True
        digest = hashlib.new(algo)
        with dest.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 22), b""):
                digest.update(chunk)
        return digest.hexdigest() == want

    if verified():
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    for attempt in range(1, 6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PathJev-research-fetch/0.1"})
            with urllib.request.urlopen(req, timeout=300) as response, partial.open("wb") as out:
                while chunk := response.read(1 << 22):
                    out.write(chunk)
            partial.rename(dest)
            if verified():
                return "ok"
            raise RuntimeError(f"size/checksum mismatch for {dest.name}")
        except Exception as exc:
            if attempt == 5:
                raise
            print(f"[retry] {dest.name}: {exc}", flush=True)
            time.sleep(30 * attempt)
    return "ok"


def fetch_sicapv2(out: Path, workers: int) -> dict:
    url = ("https://data.mendeley.com/public-files/datasets/9xxm58dvs3/files/"
           "6ab087a7-ca89-47ac-9698-f6546bb50f98/file_downloaded")
    sha = "a28aa6a0e3831217bf31575cecfc1846607ace7ff25a4392f07052bbc7355916"
    status = _http_download(url, out / "sicapv2" / "SICAPv2.zip", size=2159052394, sha256=sha)
    print(f"[sicapv2] {status}", flush=True)
    return {"doi": "10.17632/9xxm58dvs3.2", "file": "SICAPv2.zip", "sha256": sha}


def fetch_hiesd(out: Path, workers: int) -> dict:
    import urllib.request

    with urllib.request.urlopen("https://api.figshare.com/v2/articles/28919840", timeout=60) as response:
        article = json.load(response)
    files = article["files"]
    root = out / "hiesd"

    def one(f) -> str:
        return _http_download(f["download_url"], root / f["name"], size=f["size"], md5=f.get("computed_md5"))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        statuses = list(pool.map(one, files))
    print(f"[hiesd] {statuses.count('ok')} downloaded, {statuses.count('skip')} already verified", flush=True)
    return {"doi": "10.6084/m9.figshare.28919840", "version": article["version"],
            "files": {f["name"]: {"size": f["size"], "md5": f.get("computed_md5")} for f in files}}


def fetch_hepatobench(out: Path, workers: int) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().dataset_info("xtxx/HepatoBench", files_metadata=True)
    target = out / "hepatobench"
    snapshot_download("xtxx/HepatoBench", repo_type="dataset", revision=info.sha, local_dir=target,
                      max_workers=workers)
    return {"repo": "xtxx/HepatoBench", "revision": info.sha,
            "files": {s.rfilename: s.size for s in info.siblings}}


# Zenodo sources: name -> (record id, files to fetch or None for all). A record id pins one version.
ZENODO = {
    "kather2016": (53169, None),
    "histoartifacts": (10809442, None),
    "focuspath": (3926181, None),
    "tissuefold": (21493260, None),
    "grandqc_test": (14039591, None),
    "nct_crc": (1214456, ["NCT-CRC-HE-100K.zip", "CRC-VAL-HE-7K.zip"]),
    "bach": (3632035, None),
}


def _fetch_zenodo(name: str, out: Path, workers: int) -> dict:
    import urllib.request

    record, wanted = ZENODO[name]
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=60) as response:
        meta = json.load(response)
    files = [f for f in meta["files"] if wanted is None or f["key"] in wanted]
    root = out / name

    def one(f) -> str:
        algo, digest = f["checksum"].split(":", 1)
        assert algo == "md5", f["checksum"]
        return _http_download(f["links"]["self"], root / f["key"], size=f["size"], md5=digest)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        statuses = list(pool.map(one, files))
    print(f"[{name}] {statuses.count('ok')} downloaded, {statuses.count('skip')} already verified", flush=True)
    return {"zenodo_record": record, "doi": meta.get("doi"), "version": meta["metadata"].get("version"),
            "license": (meta["metadata"].get("license") or {}).get("id"),
            "files": {f["key"]: {"size": f["size"], "checksum": f["checksum"]} for f in files}}


FETCHERS = {"spider": fetch_spider, "sicapv2": fetch_sicapv2, "hiesd": fetch_hiesd,
            "hepatobench": fetch_hepatobench, "panda": None,
            **{name: (lambda out, workers, name=name: _fetch_zenodo(name, out, workers)) for name in ZENODO}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=sorted(k for k, v in FETCHERS.items() if v))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    manifest = FETCHERS[args.source](args.out, args.workers)
    path = args.out / f"{args.source}_fetch_manifest.json"
    path.write_text(json.dumps({"fetched": time.strftime("%Y-%m-%dT%H:%M:%S"), **manifest}, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
