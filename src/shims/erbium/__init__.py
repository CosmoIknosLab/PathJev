"""Minimal stand-in for ProjectNeura/Erbium, which requires Python >= 3.12.

Upstream OmniMed-Jev imports Erbium only for a GPU report (`sanity.py`) and a
resource monitor used by `--smoke_test` (`__entry__.py`); neither touches the
model, the data or the metrics. See docs/DECISIONS.md D-004.
"""
