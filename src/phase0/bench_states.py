"""Phase 0: memory and throughput of single-patch vs. cluster-level (mosaic) states.

Three state layouts, all built from PathMNIST test patches of one class:
  single      one 224 px patch;
  mosaic_k    k x k patches tiled into ONE image (k = 2, 3, 4);
  multi_n     n patches passed as n SEPARATE images in one state (n = 4, 9, 16).
Each state carries one Choice over the 9 PathMNIST classes plus 9 Noul
presence decisions, i.e. the shape of a cluster-labelling query.

For each layout and image size it measures (a) inference with the released
OmniMed-Jev checkpoint at batch 1 and 8, and (b) one Stage-B-like training step
(fresh LoRA r16 all-linear + head, gradient checkpointing). OOM is recorded,
not fatal. These are engineering numbers, not results: the PathMNIST images are
seen-source (CLAUDE.md §0.4) and no accuracy is reported.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from safetensors.torch import load_file
from rich.console import Console
from transformers import AutoProcessor

from omnimedjev.engine.medjev import assert_marker_ids
from omnimedjev.engine.medjev_model import MedJevCollator, MedJevModel, build_batch
from omnimedjev.engine.train import (choose_attention_backends, choose_dtype,
                                     load_model_with_attention_fallback, make_quant_config)

HEAD_FILE = "medjev_head.safetensors"


def pathmnist_classes(states_file: Path) -> list[dict]:
    for line in states_file.open(encoding="utf-8"):
        state = json.loads(line)
        if state["source"] == "pathmnist":
            return state["decisions"][0]["candidates"]
    raise RuntimeError("no pathmnist state in the released test states")


def build_states(npz: Path, classes: list[dict], out_dir: Path, seed: int) -> dict[str, list[dict]]:
    from phase0.fetch_assets import load_member
    images, labels = load_member(npz, "test_images"), load_member(npz, "test_labels").reshape(-1)
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(arr: np.ndarray, name: str) -> str:
        path = out_dir / f"{name}.png"
        Image.fromarray(arr).save(path)
        return str(path)

    def decisions(label: int) -> list[dict]:
        ids = [c["id"] for c in classes]
        choice = {"qid": "tissue", "type": "choice", "instructions": "Which tissue type is shown?",
                  "candidates": classes, "target": {i: float(k == label) for k, i in enumerate(ids)},
                  "target_kind": "one_hot"}
        nouls = [{"qid": f"has_{i}", "type": "noul", "instructions": f"Is {c['text']} present?",
                  "candidates": [{"id": "no", "text": "No"}, {"id": "yes", "text": "Yes"}],
                  "target": {"no": float(k != label), "yes": float(k == label)}, "target_kind": "one_hot"}
                 for k, (i, c) in enumerate(zip(ids, classes))]
        return [choice, *nouls]

    layouts: dict[str, list[dict]] = {}
    n_states = 8
    for name, n, tiled in [("single", 1, False), ("mosaic_2", 4, True), ("mosaic_3", 9, True),
                           ("mosaic_4", 16, True), ("multi_4", 4, False), ("multi_9", 9, False),
                           ("multi_16", 16, False)]:
        states = []
        for s in range(n_states):
            label = int(rng.integers(len(classes)))
            idx = rng.choice(np.flatnonzero(labels == label), size=n, replace=False)
            if tiled:
                k = int(round(n ** 0.5))
                rows = [np.concatenate(list(images[idx[r * k:(r + 1) * k]]), axis=1) for r in range(k)]
                paths = [save(np.concatenate(rows, axis=0), f"{name}_{s}")]
            else:
                paths = [save(images[i], f"{name}_{s}_{j}") for j, i in enumerate(idx)]
            states.append({"state_id": f"{name}_{s}", "source": "bench", "images": paths,
                           "decisions": decisions(label)})
        layouts[name] = states
    return layouts


def load_backbone(model_path: str, dtype, kbit_training: bool):
    quant = make_quant_config(True, dtype)
    backbone = load_model_with_attention_fallback(
        model_path, choose_attention_backends("auto"), Console(),
        torch_dtype=dtype, device_map=None, quantization_config=quant, trust_remote_code=True)
    backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=kbit_training)
    backbone.config.use_cache = False
    return backbone


def timed(fn, repeats: int) -> tuple[float, float]:
    """Median seconds over `repeats` runs after one warm-up; peak GB."""
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        began = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - began)
    return statistics.median(times), torch.cuda.max_memory_allocated() / 1e9


def measure(rows: list[dict], mode: str, layout: str, image_size: int, batch_size: int, fn,
            width: int, repeats: int) -> None:
    row = {"mode": mode, "layout": layout, "image_size": image_size, "batch_size": batch_size,
           "seq_len": width}
    try:
        seconds, peak = timed(fn, repeats)
        row.update(status="ok", seconds_per_batch=round(seconds, 4),
                   seconds_per_state=round(seconds / batch_size, 4), peak_allocated_gb=round(peak, 2))
    except torch.cuda.OutOfMemoryError:
        row.update(status="oom")
    except Exception as exc:  # e.g. upstream MedJevModel supports one image per state
        row.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:200]}")
    gc.collect()
    torch.cuda.empty_cache()
    rows.append(row)
    print(json.dumps(row), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pathmnist-npz", required=True, type=Path)
    ap.add_argument("--states", required=True, type=Path, help="released testing_states.jsonl")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--image-sizes", type=int, nargs="+", default=[448, 896])
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    dtype = choose_dtype("auto", None)
    layouts = build_states(args.pathmnist_npz, pathmnist_classes(args.states),
                           args.out_dir / "bench_images", args.seed)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    assert_marker_ids(processor.tokenizer)
    rows: list[dict] = []

    # (a) inference with the released checkpoint
    backbone = PeftModel.from_pretrained(load_backbone(args.model, dtype, False), args.checkpoint)
    model = MedJevModel(backbone, backbone.config.text_config.hidden_size)
    model.head.load_state_dict(load_file(str(Path(args.checkpoint) / HEAD_FILE)))
    model.to(device).eval()
    for image_size in args.image_sizes:
        for layout, states in layouts.items():
            collator = MedJevCollator(processor, image_size=image_size, max_images_per_sample=16,
                                      max_length=args.max_length, with_targets=False)
            for batch_size in (1, 8):
                batch = build_batch([collator.encode(s) for s in states[:batch_size]],
                                    collator.pad_token_id, device=device)
                batch.pop("meta")
                width = int(batch["input_ids"].shape[1])

                def infer(batch=batch):
                    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                        model(**batch)
                measure(rows, "inference", layout, image_size, batch_size, infer, width, args.repeats)
    del model, backbone
    gc.collect()
    torch.cuda.empty_cache()

    # (b) one Stage-B-like training step (fresh adapters; weights are irrelevant here)
    backbone = get_peft_model(load_backbone(args.model, dtype, True), LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none", target_modules="all-linear",
        task_type="CAUSAL_LM"))
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = MedJevModel(backbone, backbone.config.text_config.hidden_size).to(device).train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    for image_size in args.image_sizes:
        for layout, states in layouts.items():
            collator = MedJevCollator(processor, image_size=image_size, max_images_per_sample=16,
                                      max_length=args.max_length, with_targets=True)
            batch = collator(states)  # batch 8 = upstream per-device batch
            batch.pop("meta", None)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            width = int(batch["input_ids"].shape[1])

            def step(batch=batch):
                with torch.autocast("cuda", dtype=dtype):
                    loss = model(**batch)["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            measure(rows, "train_step", layout, image_size, len(states), step, width, max(2, args.repeats // 2))

    meta = {"gpu": torch.cuda.get_device_name(0),
            "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            "dtype": str(dtype), "max_length": args.max_length, "rows": rows}
    (args.out_dir / "bench_states.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out_dir / 'bench_states.json'}")


if __name__ == "__main__":
    main()
