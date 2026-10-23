"""No-op replacements for the two Erbium symbols OmniMed-Jev imports (D-004)."""
from __future__ import annotations

from typing import Any


def get_all_gpu_info() -> list[dict[str, Any]]:
    """Return a minimal GPU description via torch instead of NVML."""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    return [{"index": i, "name": torch.cuda.get_device_name(i),
             "memory_total": torch.cuda.get_device_properties(i).total_memory}
            for i in range(torch.cuda.device_count())]


class ResourceMonitor:
    """Accepts the upstream constructor and lifecycle calls and records nothing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str):
        return lambda *args, **kwargs: None
