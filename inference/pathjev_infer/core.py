"""Minimal, self-contained PathJev-P inference.

Loads google/medgemma-1.5-4b-it, attaches the PathJev-P LoRA adapter and the decision head, and returns one
probability distribution per question over the candidates the caller supplies. One forward pass per image.

The prompt layout, marker tokens, head architecture and image preprocessing are copied from OmniMed-Jev
(github.com/lytang63/OmniMed-Jev, Apache-2.0; omnimedjev/engine/medjev.py, medjev_model.py, train.py) so that
the released weights see exactly the input they were trained and evaluated on:

    <image> [modality line]

    Q1: <question>
    - <candidate text> ⟩
    ...
    ✦

The hidden state at each ⟩ is the candidate key, at ✦ the question query. Nothing else is added to the prompt.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

MARKER_KEY, MARKER_QUERY = "⟩", "✦"
MARKER_KEY_ID, MARKER_QUERY_ID = 241287, 244193
MODALITY = "H&E histopathology"          # the only header line every PathJev-P training state carried
IMAGE_SIZE = 896                         # square resize used in training and evaluation
OTHER = {"id": "other", "text": "other tissue / none of these"}
HEAD_FILE = "medjev_head.safetensors"
BASE_MODEL = "google/medgemma-1.5-4b-it"


# --------------------------------------------------------------------------- decision head (OmniMed-Jev)
class MedJevHead(nn.Module):
    """Shared scorer: per-candidate scalar + question-candidate interaction + set attention over candidates."""

    def __init__(self, hidden_size: int, attn_dim: int = 128, attn_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.scalar = nn.Linear(hidden_size, 1)
        self.query_proj = nn.Linear(hidden_size, attn_dim)
        self.key_proj = nn.Linear(hidden_size, attn_dim)
        self.set_project = nn.Linear(hidden_size + 1, attn_dim)
        self.set_attention = nn.MultiheadAttention(attn_dim, attn_heads, dropout=0.0, batch_first=True)
        self.set_output = nn.Linear(attn_dim, 1)

    def forward(self, hidden, key_sample, key_pos, query_sample, query_pos, valid):
        h_key = self.norm(hidden[key_sample.unsqueeze(1), key_pos.clamp(min=0)])
        z = self.scalar(h_key).squeeze(-1).float()
        h_q = self.norm(hidden[query_sample, query_pos])
        q, k = self.query_proj(h_q).float(), self.key_proj(h_key).float()
        z = z + (q.unsqueeze(1) * k).sum(-1) / math.sqrt(k.shape[-1])
        count = valid.sum(-1).clamp(min=1).float()
        log_k = count.log()[:, None, None].expand(-1, key_pos.shape[1], 1)
        u = self.set_project(torch.cat([h_key, log_k.to(h_key.dtype)], dim=-1))
        mixed, _ = self.set_attention(u, u, u, key_padding_mask=~valid, need_weights=False)
        z = z + self.set_output(torch.tanh(u + mixed)).squeeze(-1).float()
        return z.masked_fill(~valid, float("-inf"))


# --------------------------------------------------------------------------- input construction
def decision_block(questions: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for index, q in enumerate(questions, start=1):
        if len(q["candidates"]) < 2:
            raise ValueError(f"question {index} needs at least 2 candidates")
        lines.append(f"Q{index}: {str(q['instructions']).strip()}")
        lines += [f"- {str(c['text']).strip()} {MARKER_KEY}" for c in q["candidates"]]
        lines.append(MARKER_QUERY)
    return "\n".join(lines)


def load_image(image, size: int = IMAGE_SIZE):
    """Path or PIL image -> RGB, EXIF-transposed, square bicubic resize (as in training)."""
    from PIL import Image, ImageOps

    if not isinstance(image, Image.Image):
        with Image.open(image) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
    else:
        image = ImageOps.exif_transpose(image).convert("RGB")
    return image.resize((size, size), Image.Resampling.BICUBIC)


def encode(processor, image, questions: Sequence[Mapping[str, Any]], modality: str = MODALITY) -> dict:
    body = decision_block(questions)
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"{modality}\n\n{body}\n"}]}],
        add_generation_prompt=False, tokenize=False)
    batch = processor(text=[text], images=[[image]], return_tensors="pt")
    ids = batch["input_ids"][0].tolist()
    query_pos = [i for i, v in enumerate(ids) if v == MARKER_QUERY_ID]
    key_pos = [i for i, v in enumerate(ids) if v == MARKER_KEY_ID]
    if len(query_pos) != len(questions) or len(key_pos) != sum(len(q["candidates"]) for q in questions):
        raise ValueError("marker count mismatch: a candidate or question text merged with a marker token")
    keys, cursor = [], 0
    for boundary in query_pos:
        group = [p for p in key_pos[cursor:] if p < boundary]
        cursor += len(group)
        keys.append(group)
    width = max(len(g) for g in keys)
    key_rows = torch.full((len(keys), width), -1, dtype=torch.long)
    valid = torch.zeros((len(keys), width), dtype=torch.bool)
    for row, group in enumerate(keys):
        key_rows[row, :len(group)] = torch.tensor(group)
        valid[row, :len(group)] = True
    return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
            "pixel_values": batch["pixel_values"], "key_pos": key_rows, "valid": valid,
            "query_pos": torch.tensor(query_pos), "n": len(keys)}


# --------------------------------------------------------------------------- model
def _vision_language_stack(model: nn.Module) -> nn.Module:
    """The inner Gemma3Model (language_model + vision_tower), whatever PEFT wrapping is present."""
    seen, frontier = set(), [model]
    while frontier:
        m = frontier.pop(0)
        if m is None or id(m) in seen:
            continue
        seen.add(id(m))
        if hasattr(m, "language_model") and hasattr(m, "vision_tower"):
            return m
        frontier += [getattr(m, a) for a in ("base_model", "model") if isinstance(getattr(m, a, None), nn.Module)]
    raise TypeError("Gemma3Model not found")


class PathJev:
    """`PathJev.load(adapter_dir)` then `.predict(image, questions)`."""

    def __init__(self, backbone, head, processor, device, dtype):
        self.backbone, self.head, self.processor, self.device, self.dtype = backbone, head, processor, device, dtype

    @classmethod
    def load(cls, adapter: str | Path, base: str = BASE_MODEL, device: str | None = None,
             four_bit: bool | None = None, token: str | bool | None = None) -> "PathJev":
        """adapter: folder (or HF repo id) with adapter_config.json, adapter_model.safetensors, medjev_head.safetensors.

        CUDA: 4-bit NF4 with double quantisation, bf16 compute (the configuration of every reported run).
        CPU: bf16 weights, no quantisation (needs ~10 GB RAM; slow).
        """
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        cuda = device.startswith("cuda")
        four_bit = cuda if four_bit is None else four_bit
        dtype = torch.bfloat16
        adapter = str(adapter)
        if not Path(adapter).exists():
            from huggingface_hub import snapshot_download
            adapter = snapshot_download(adapter, token=token)
        processor = AutoProcessor.from_pretrained(base, use_fast=True, token=token)
        processor.tokenizer.padding_side = "right"
        for text, expected in ((MARKER_KEY, MARKER_KEY_ID), (MARKER_QUERY, MARKER_QUERY_ID)):
            if expected not in processor.tokenizer.encode("a" + text, add_special_tokens=False):
                raise ValueError(f"marker {text!r} is not token {expected} in this tokenizer")
        kwargs = {"dtype": dtype, "token": token}
        if four_bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype)
            kwargs["device_map"] = {"": device}
        backbone = AutoModelForImageTextToText.from_pretrained(base, attn_implementation="sdpa", **kwargs)
        if four_bit:
            # as in the evaluation code: casts the non-quantised parameters (norms) to fp32
            from peft import prepare_model_for_kbit_training
            backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=False)
        backbone.config.use_cache = False
        backbone = PeftModel.from_pretrained(backbone, adapter)
        head = MedJevHead(backbone.config.text_config.hidden_size)
        head.load_state_dict(load_file(str(Path(adapter) / HEAD_FILE)))
        if not four_bit:
            backbone.to(device)
        head.to(device)
        backbone.eval()
        head.eval()
        return cls(backbone, head, processor, device, dtype)

    @torch.no_grad()
    def predict(self, image, questions: Sequence[Mapping[str, Any]], modality: str = MODALITY,
                add_other: bool = False) -> list[dict[str, float]]:
        """questions: [{"instructions": str, "candidates": [{"id": str, "text": str}, ...]}, ...].

        Returns one {candidate_id: probability} per question. `add_other=True` appends the
        "other tissue / none of these" candidate used by every PathJev-P training Choice.
        """
        questions = [dict(q, candidates=list(q["candidates"]) + ([OTHER] if add_other and
                     all(c["id"] != "other" for c in q["candidates"]) else [])) for q in questions]
        enc = encode(self.processor, load_image(image), questions, modality)
        stack = _vision_language_stack(self.backbone)
        cuda = self.device.startswith("cuda")
        zeros = torch.zeros(enc["n"], dtype=torch.long, device=self.device)
        # backbone and head under one autocast region, as in the evaluation code
        with torch.autocast("cuda" if cuda else "cpu", dtype=self.dtype):
            hidden = stack(input_ids=enc["input_ids"].to(self.device),
                           attention_mask=enc["attention_mask"].to(self.device),
                           pixel_values=enc["pixel_values"].to(self.device),
                           use_cache=False).last_hidden_state
            logits = self.head(hidden, zeros, enc["key_pos"].to(self.device), zeros,
                               enc["query_pos"].to(self.device), enc["valid"].to(self.device))
        probs = logits.float().softmax(-1).cpu()
        return [{c["id"]: float(probs[i, j]) for j, c in enumerate(q["candidates"])}
                for i, q in enumerate(questions)]
