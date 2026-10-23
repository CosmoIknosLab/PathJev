"""Training budget of the two trained arms as evaluated (paper/generated/tab_training.tex), from run folders only.

    python analysis/training_details.py --work $PATHJEV_WORK

Per arm and seed (runs chosen exactly as in analysis/stage1_analysis.py): selected configuration, selected
checkpoint, optimiser steps at that checkpoint (Stage A + Stage B for PathJev-P), planned steps, items per
optimiser step (states for PathJev-P, single-decision examples for Gen-SFT, from the trainer's epoch counter),
images and decision targets seen up to the selected checkpoint, trainable parameters and wall time of the
whole run. Seeds that agree are collapsed; otherwise the seed range is shown.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from stage1_analysis import trained_runs

ARMS = [("pathjev", "PathJev-P"), ("sft", "Gen-SFT")]


def _int(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", "{,}")


def _log_number(path: Path, pattern: str) -> float:
    text = path.read_text(errors="replace").replace("\r", "\n")
    hits = re.findall(pattern, text)
    if not hits:
        raise ValueError(f"{pattern!r} not found in {path}")
    return float(hits[-1].replace(",", ""))


def _state(ckpt: Path) -> dict:
    return json.loads((ckpt / "trainer_state.json").read_text())


def _train_counts(work: Path) -> tuple[int, int]:
    """States and decisions in the Stage 1 training mix."""
    states = decisions = 0
    for line in (work / "mixes" / "stage1-p" / "train.jsonl").open():
        if line.strip():
            states += 1
            decisions += len(json.loads(line)["decisions"])
    return states, decisions


def run_facts(work: Path, run: str, arm: str, n_states: int, n_decisions: int) -> dict:
    rdir = work / "runs" / f"train_{run}"
    ckpt = Path((rdir / "selection" / "best.txt").read_text().strip())
    lr, epochs = re.search(r"-lr([0-9.e-]+)-e(\d+)-s\d+$", run).groups()
    st = _state(ckpt)
    step_b, planned = st["global_step"], st["max_steps"]
    if arm == "pathjev":
        st_a = _state(rdir / "head" / "checkpoint-200")
        step_a = st_a["global_step"]
        per_step = round(st["epoch"] * n_states / step_b)
        per_step_a = round(st_a["epoch"] * n_states / step_a)
        images = step_a * per_step_a + step_b * per_step
        targets = images * n_decisions / n_states            # mean decisions per state × states seen
        params = _log_number(rdir / "stage_b.log", r"trainable ([0-9,]+) /")
        hours = (_log_number(rdir / "stage_a.log", r"'train_runtime': '([0-9.e+]+)'")
                 + _log_number(rdir / "stage_b.log", r"'train_runtime': '([0-9.e+]+)'")) / 3600
    else:
        step_a = 0
        per_step = round(st["epoch"] * n_decisions / step_b)
        images = targets = step_b * per_step                 # one image and one decision per example
        params = _log_number(rdir / "train.log", r"trainable params: ([0-9,]+)")
        hours = _log_number(rdir / "train.log", r"'train_runtime': '([0-9.e+]+)'") / 3600
    passes = st["epoch"]                                  # passes over states (PathJev-P) or examples (Gen-SFT)
    return {"lr": lr, "epochs": int(epochs), "ckpt": ckpt.name, "step_a": step_a, "step_b": step_b, "passes": passes,
            "per_pass": n_states if arm == "pathjev" else n_decisions,
            "planned": planned + step_a, "per_step": per_step, "images": images, "targets": targets,
            "params": params, "hours": hours}


def _cell(vals: list[str]) -> str:
    """One value if all seeds agree, else the per-seed values in seed order."""
    return vals[0] if len(set(vals)) == 1 else " / ".join(vals)


def table(facts: dict[str, list[dict]]) -> str:
    rows = [
        ("Selected configuration", lambda f: ("lr ${}\\times10^{{{}}}$, {} epochs".format(*f["lr"].split("e"), f["epochs"])
                                              if f["step_a"] else
                                              "lr ${}\\times10^{{{}}}$, budget of {} PathJev-P epochs".format(
                                                  *f["lr"].split("e"), f["epochs"]))),
        ("Training items per pass", lambda f: (f"{_int(f['per_pass'])} states" if f["step_a"]
                                               else f"{_int(f['per_pass'])} single-decision examples")),
        ("Optimiser steps at checkpoint", lambda f: (f"{_int(f['step_a'])} (A) + {_int(f['step_b'])} (B)"
                                                     if f["step_a"] else _int(f["step_b"]))),
        ("Planned steps (whole run)", lambda f: _int(f["planned"])),
        ("Items per optimiser step", lambda f: _int(f["per_step"])),
        ("Passes at checkpoint (Stage B)", lambda f: f"{f['passes']:.2f}"),
        ("Images seen up to checkpoint", lambda f: _int(f["images"])),
        ("Decision targets seen up to checkpoint", lambda f: ("$\\approx$" if f["step_a"] else "") + _int(f["targets"])),
        ("Trainable parameters", lambda f: _int(f["params"])),
    ]
    head = " & ".join(f"{name} ({len(facts[a])} seeds)" for a, name in ARMS)
    lines = [r"\begin{tabular}{@{}lll@{}}", r"\toprule", f"Quantity & {head} \\\\", r"\midrule"]
    for label, fn in rows:
        lines.append(f"{label} & " + " & ".join(_cell([fn(f) for f in facts[a]]) for a, _ in ARMS) + " \\\\")
    hours = [f"{min(f['hours'] for f in facts[a]):.1f}--{max(f['hours'] for f in facts[a]):.1f}" for a, _ in ARMS]
    lines.append("Wall time of the whole run (h, 1 GPU) & " + " & ".join(hours) + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "paper" / "generated" / "tab_training.tex")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    n_states, n_decisions = _train_counts(args.work)
    facts = {arm: [run_facts(args.work, run, arm, n_states, n_decisions) for run in trained_runs(repo, arm)]
             for arm, _ in ARMS}
    for arm, fs in facts.items():
        for f in fs:
            print(arm, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in f.items()})
    print(f"train mix: {n_states} states, {n_decisions} decisions")
    args.out.write_text("% generated by analysis/training_details.py from run folders — do not edit\n" + table(facts))
    print(args.out)


if __name__ == "__main__":
    main()
