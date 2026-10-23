"""Cluster-specific storage roots, read from the environment (set in slurm/templates/common.sh).

Defaults are under ~/pathjev; set PATHJEV_DATA and PATHJEV_WORK to override. Nothing else in the code hardcodes paths.
"""

from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    return Path(os.environ.get("PATHJEV_DATA", str(Path.home() / "pathjev" / "data")))


def work_root() -> Path:
    return Path(os.environ.get("PATHJEV_WORK", str(Path.home() / "pathjev" / "work")))
