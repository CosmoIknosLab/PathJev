"""Keep only the released states whose images are materialised on disk.

Phase 0 fetches MedMNIST+ only, so TXL-PBC, CAMUS and ChestX-Det states are
dropped; the classification family (all MedMNIST) stays complete.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream", required=True, type=Path)
    ap.add_argument("--states", required=True, type=Path, help="relative to --upstream")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    kept, dropped = collections.Counter(), collections.Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with (args.upstream / args.states).open(encoding="utf-8") as src, args.out.open("w") as dst:
        for line in src:
            state = json.loads(line)
            if all((args.upstream / image).is_file() for image in state["images"]):
                dst.write(line)
                kept[state["source"]] += 1
            else:
                dropped[state["source"]] += 1
    print(json.dumps({"kept": dict(kept), "dropped": dict(dropped),
                      "n_kept": sum(kept.values())}, indent=2))


if __name__ == "__main__":
    main()
