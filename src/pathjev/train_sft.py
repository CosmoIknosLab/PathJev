"""Generative SFT baseline (b): same backbone, data, LoRA config and step budget as PathJev Stage B.

    python -m pathjev.train_sft --train mix/train.jsonl --eval mix/val.jsonl --out RUN --seed 1 [--epochs 3]

Each decision of each state becomes one SFT example (prompt = instructions,
choices = candidate texts, answer = gold candidate text; soft targets of mixed
mosaics → majority candidate). Training uses upstream `ImageSFTCollator`
(prompt tokens masked, options re-lettered and shuffled every draw — the
counterpart of PathJev's `--shuffle-candidates`). `max_steps` can be pinned to
the PathJev run's optimiser steps so both arms get the same budget.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

SFT_TASK = "disease_diagnosis_classification"


def sft_examples(states_path: Path) -> list[dict]:
    rows = []
    for line in states_path.open():
        state = json.loads(line)
        for d in state["decisions"]:
            gold = max(d["target"], key=d["target"].get)
            texts = [c["text"] for c in d["candidates"]]
            rows.append({"state_id": state["state_id"], "qid": d["qid"], "images": state["images"],
                         "task_type": SFT_TASK, "prompt": f"Question: {d['instructions'].strip()}",
                         "choices": texts, "answer": next(c["text"] for c in d["candidates"] if c["id"] == gold)})
    return rows


class _Rows(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def main() -> None:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, Trainer,
                              TrainingArguments)

    from omnimedjev.engine.train import ImageSFTCollator, resolve_warmup_steps

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--eval", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=896)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(args.model, quantization_config=quant,
                                                        torch_dtype=torch.bfloat16, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    collator = ImageSFTCollator(processor, image_size=args.image_size, shuffle_options=True, seed=args.seed,
                                max_length=3072)
    train = _Rows(sft_examples(args.train))
    evaluation = _Rows(sft_examples(args.eval)[:128]) if args.eval else None
    # transformers 5.x dropped `warmup_ratio`; same conversion as upstream train_medjev.py
    warmup_steps = resolve_warmup_steps(None, 0.03, len(train), args.batch, args.accum, args.epochs, args.max_steps)
    targs = TrainingArguments(
        output_dir=str(args.out), seed=args.seed, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, lr_scheduler_type="linear", warmup_steps=warmup_steps, weight_decay=0.0,
        max_grad_norm=0.3, optim="paged_adamw_8bit", bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=25,
        save_steps=args.save_steps, eval_strategy="steps" if evaluation else "no", eval_steps=args.save_steps,
        save_total_limit=None, remove_unused_columns=False, dataloader_num_workers=6, report_to=[])
    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=evaluation, data_collator=collator)
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(args.out))
    (args.out / ".done").touch()


if __name__ == "__main__":
    main()
