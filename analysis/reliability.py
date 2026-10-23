"""Reliability diagrams (plan §7, secondary) for the main Choice question, from run folders only.

    python analysis/reliability.py --work $PATHJEV_WORK --out paper/figures/reliability.pdf

Top-label confidence in 15 equal-width bins, one panel per clean test set (the endpoint is computed per
set and then macro-averaged, so sets are not pooled). Trained arms: one line per seed (seeds are not pooled).
Every non-empty bin is drawn; marker area is proportional to the share of states in the bin.
Raw probabilities, no temperature.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stage1_analysis import BINS, MAIN_Q, arm_dirs, load_main

# categorical slots 1-4 of the dataviz reference palette, fixed order; marker shape as second encoding
ARMS = [("sft", "Generative SFT", "#2a78d6", "s"), ("pathjev", "PathJev-P", "#eb6834", "o"),
        ("conch", "CONCH", "#1baf7a", "^"), ("omnimedjev", "OmniMed-Jev", "#eda100", "D")]
SETS = [("bracs", "BRACS (in domain)"), ("sicapv2", "SICAPv2 (in domain)"),
        ("hepatobench", "HepatoBench (unseen organ)"), ("hiesd", "HiESD (unseen organ)")]


def per_seed(work: Path, repo: Path, arm: str, s: str) -> list[tuple[np.ndarray, np.ndarray]]:
    test_dir = work / "eval_sets" / "stage1-test"
    out = []
    for tdir, _ in arm_dirs(work, repo, arm, interim=False):
        d = load_main(test_dir / f"{s}.jsonl", tdir / f"{s}_decisions.jsonl", MAIN_Q[s])
        out.append((d["probs"].max(1), d["probs"].argmax(1) == d["y"]))
    return out


def curve(conf: np.ndarray, correct: np.ndarray):
    edges = np.linspace(0, 1, BINS + 1)
    b = np.clip(np.digitize(conf, edges[1:-1]), 0, BINS - 1)
    out = []
    for k in range(BINS):
        m = b == k
        if m.any():
            out.append((conf[m].mean(), correct[m].mean(), m.mean()))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": "#8a8f98",
                         "axes.linewidth": 0.6, "xtick.color": "#555555", "ytick.color": "#555555"})
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.15), sharey=True)
    for ax, (s, title) in zip(axes, SETS):
        ax.plot([0, 1], [0, 1], color="#b0b3b8", lw=1, ls="--", zorder=1)
        for arm, label, color, marker in ARMS:
            for i, (conf, correct) in enumerate(per_seed(args.work, repo, arm, s)):
                c = curve(conf, correct)
                ax.plot(c[:, 0], c[:, 1], color=color, lw=1.2, zorder=3, label=label if i == 0 else None,
                        marker=marker, markersize=3, markeredgecolor="white", markeredgewidth=0.4)
                ax.scatter(c[:, 0], c[:, 1], s=2 + 120 * c[:, 2], color=color, marker=marker, zorder=4,
                           edgecolors="white", linewidths=0.5)
        ax.set(xlim=(0, 1), ylim=(0, 1), title=title, xlabel="Confidence")
        ax.title.set_fontsize(7)
        ax.grid(color="#e6e6e6", lw=0.5)
        ax.set_aspect("equal")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Accuracy")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
