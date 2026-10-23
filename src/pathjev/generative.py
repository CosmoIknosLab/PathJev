"""Distributions over candidate sets from a *generative* MedGemma (baselines b and e).

Two read-outs, both producing the same decision-file format as upstream
`infer_medjev.py` ({"state_id", "distributions": {qid: {candidate_id: prob}}}),
so `pathjev.evaluate` scores every arm identically:

  letter  (baseline e, "Inner Jev" read-out, D-008): the decision is rendered as a
          lettered multiple-choice question, the assistant turn is pre-filled with
          "The answer is (" and the distribution is the softmax of the next-token
          logits restricted to the option letters. Training-free.
  text    (baseline b, generative SFT; `text-slow` = reference implementation that re-runs
          the full sequence per candidate, `text` = same scores with a cached prefix, D-027):
          each candidate's text is scored as the
          assistant continuation (sum of token log-probabilities); scores are
          normalised over the candidate set — the reconstruction OmniMed-Jev
          used for its generative arm (REPRODUCE.md §3), a pseudo-distribution.

Candidate order is the state's order (no shuffling at evaluation).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# The assistant turn is pre-filled so that the next token can only be an option
# letter. Without it, "I" (pronoun) and "A" (article) are frequent first tokens
# and inflate options I and A — measured on BRACS (all 100 organ answers = "I").
LETTER_PREFILL = "The answer is ("
SYSTEM = "You are an expert pathologist."
SFT_SYSTEM = "You are an expert medical imaging assistant."   # upstream ImageSFTCollator default


def sft_prompt(decision: dict) -> str:
    """Prompt exactly as upstream build_prompt renders an SFT row (options lettered, state order)."""
    from omnimedjev.engine.train import TASK_INSTRUCTIONS

    options = "Options: " + " ".join(f"{LETTERS[i]}) {c['text']}" for i, c in enumerate(decision["candidates"]))
    return "\n\n".join([TASK_INSTRUCTIONS["disease_diagnosis_classification"],
                          f"Question: {decision['instructions'].strip()}", options])


def render(decision: dict) -> tuple[str, list[str]]:
    """Prompt text and the letters assigned to candidates, in order."""
    candidates = decision["candidates"]
    letters = list(LETTERS[: len(candidates)])
    options = "\n".join(f"({l}) {c['text']}" for l, c in zip(letters, candidates))
    prompt = (f"Question: {decision['instructions'].strip()}\nOptions:\n{options}\n"
              "Answer with the letter of the single best option.")
    return prompt, letters


def sft_row(state: dict, decision: dict) -> dict:
    """One generative-SFT training row (answer = gold candidate text; soft targets → majority)."""
    target = decision["target"]
    gold = max(target, key=target.get)
    text = next(c["text"] for c in decision["candidates"] if c["id"] == gold)
    prompt, _ = render(decision)
    return {"state_id": state["state_id"], "qid": decision["qid"], "images": state["images"],
            "prompt": prompt.replace("Answer with the letter of the single best option.",
                                     "Answer with the text of the single best option."),
            "answer": text, "soft_target": decision.get("target_kind") == "soft"}


class Scorer:
    def __init__(self, model_path: str, adapter: str | None, image_size: int, device: str = "cuda:0"):
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        self.processor = AutoProcessor.from_pretrained(model_path)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                   bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForImageTextToText.from_pretrained(model_path, quantization_config=quant,
                                                            torch_dtype=torch.bfloat16, device_map={"": device})
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.eval()
        self.device = device
        self.image_size = image_size
        tok = self.processor.tokenizer
        self.letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]

    def _inputs(self, image_paths: list[str], prompt: str, answer: str | None = None, system: str = SYSTEM):
        from PIL import Image

        images = [Image.open(p).convert("RGB").resize((self.image_size, self.image_size)) for p in image_paths]
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if answer is not None:
            text = text + answer
        return self.processor(text=[text], images=[images], return_tensors="pt").to(self.device)

    @torch.no_grad()
    def letter_distribution(self, image_paths, decision) -> dict[str, float]:
        prompt, letters = render(decision)
        logits = self.model(**self._inputs(image_paths, prompt, answer=LETTER_PREFILL)).logits[0, -1].float()
        sub = logits[self.letter_ids[: len(letters)]].softmax(-1).tolist()
        return {c["id"]: p for c, p in zip(decision["candidates"], sub)}

    @torch.no_grad()
    def text_distribution_cached(self, image_paths, decision) -> dict[str, float]:
        """Same scores as `text_distribution`, with the image + prompt prefix encoded once.

        The prefix is run once with a KV cache; each candidate's answer tokens are then
        scored as a continuation of a copy of that cache. Mathematically identical to
        re-running the full sequence per candidate (causal attention); only floating-point
        accumulation order differs. Equivalence is checked by `pathjev.check_readout` (D-027).
        """
        import copy

        prompt = sft_prompt(decision)
        base = self._inputs(image_paths, prompt, system=SFT_SYSTEM)
        n_prompt = base["input_ids"].shape[1]
        out = self.model(**base, use_cache=True)
        first = out.logits[0, -1].float().log_softmax(-1)
        prefix_text = self._chat_text(len(image_paths), prompt, SFT_SYSTEM)
        tok = self.processor.tokenizer
        prefix_ids = tok(prefix_text, add_special_tokens=False)["input_ids"]
        scores = []
        for c in decision["candidates"]:
            full_ids = tok(prefix_text + c["text"], add_special_tokens=False)["input_ids"]
            if full_ids[: len(prefix_ids)] != prefix_ids or len(full_ids) == len(prefix_ids):
                raise ValueError(f"answer tokenisation is not a clean continuation for {c['id']!r}")
            cont = torch.tensor([full_ids[len(prefix_ids):]], device=self.device)
            score = first[cont[0, 0]]
            if cont.shape[1] > 1:
                mask = torch.ones(1, n_prompt + cont.shape[1], dtype=base["attention_mask"].dtype, device=self.device)
                step = self.model(input_ids=cont, past_key_values=copy.deepcopy(out.past_key_values),
                                  attention_mask=mask, use_cache=True)
                logp = step.logits[0, :-1].float().log_softmax(-1)
                score = score + logp.gather(1, cont[0, 1:, None]).sum()
            scores.append(score)
        probs = torch.stack(scores).softmax(-1).tolist()
        return {c["id"]: p for c, p in zip(decision["candidates"], probs)}

    def _chat_text(self, n_images: int, prompt: str, system: str) -> str:
        content = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": prompt}]
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": content}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def text_distribution(self, image_paths, decision) -> dict[str, float]:
        prompt = sft_prompt(decision)
        base = self._inputs(image_paths, prompt, system=SFT_SYSTEM)
        n_prompt = base["input_ids"].shape[1]
        scores = []
        for c in decision["candidates"]:
            full = self._inputs(image_paths, prompt, answer=c["text"], system=SFT_SYSTEM)
            logp = self.model(**full).logits[0, :-1].float().log_softmax(-1)
            targets = full["input_ids"][0, 1:]
            scores.append(logp[n_prompt - 1:].gather(1, targets[n_prompt - 1:, None]).sum())
        probs = torch.stack(scores).softmax(-1).tolist()
        return {c["id"]: p for c, p in zip(decision["candidates"], probs)}


    @torch.no_grad()
    def text_distribution_variants(self, image_paths, decision) -> tuple[dict[str, float], dict]:
        """`text_distribution` plus two read-out variants from the same forward passes (plan A3.4).

        The primary distribution uses exactly the operations of `text_distribution`. Per candidate it also
        records the summed answer log-probability, the number of answer tokens and log p(<end_of_turn>)
        after the answer; variants: "eos" = sum + log p(<end_of_turn>), "lennorm" = sum / n_tokens.
        """
        prompt = sft_prompt(decision)
        base = self._inputs(image_paths, prompt, system=SFT_SYSTEM)
        n_prompt = base["input_ids"].shape[1]
        eos_id = self.processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        scores, raw = [], {}
        for c in decision["candidates"]:
            full = self._inputs(image_paths, prompt, answer=c["text"], system=SFT_SYSTEM)
            logits = self.model(**full).logits
            logp = logits[0, :-1].float().log_softmax(-1)
            targets = full["input_ids"][0, 1:]
            score = logp[n_prompt - 1:].gather(1, targets[n_prompt - 1:, None]).sum()
            scores.append(score)
            eos = logits[0, -1].float().log_softmax(-1)[eos_id]
            raw[c["id"]] = {"sum": float(score), "n": int(full["input_ids"].shape[1] - n_prompt), "eos": float(eos)}
        probs = torch.stack(scores).softmax(-1).tolist()
        primary = {c["id"]: p for c, p in zip(decision["candidates"], probs)}
        ids = [c["id"] for c in decision["candidates"]]
        variants = {"raw": raw}
        for name, fn in (("eos", lambda r: r["sum"] + r["eos"]), ("lennorm", lambda r: r["sum"] / r["n"])):
            v = torch.tensor([fn(raw[i]) for i in ids], dtype=torch.float64).softmax(-1).tolist()
            variants[name] = dict(zip(ids, v))
        return primary, variants

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--readout", choices=("letter", "text", "text-slow"), required=True,
                    help="text = cached-prefix candidate scoring (D-027); text-slow = full re-run per candidate")
    ap.add_argument("--image-size", type=int, default=896)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--qids", nargs="*", default=None, help="score only these decisions (default: all)")
    ap.add_argument("--variants-out", type=Path, default=None,
                    help="text-slow only: also write read-out variants (plan A3.4) to this file")
    args = ap.parse_args()
    if args.variants_out and args.readout != "text-slow":
        ap.error("--variants-out requires --readout text-slow")
    scorer = Scorer(args.model, args.adapter, args.image_size)
    states = [json.loads(l) for l in args.states.open() if l.strip()]
    if args.limit:
        states = states[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fn = {"letter": scorer.letter_distribution, "text": scorer.text_distribution_cached,
          "text-slow": scorer.text_distribution}[args.readout]
    keep = (lambda d: True) if args.qids is None else (lambda d: d["qid"] in args.qids)
    vh = args.variants_out.open("w") if args.variants_out else None
    with args.out.open("w") as handle:
        for state in states:
            if vh is None:
                dist = {d["qid"]: fn(state["images"], d) for d in state["decisions"] if keep(d)}
            else:
                both = {d["qid"]: scorer.text_distribution_variants(state["images"], d)
                        for d in state["decisions"] if keep(d)}
                dist = {q: v[0] for q, v in both.items()}
                vh.write(json.dumps({"state_id": state["state_id"], "variants": {q: v[1] for q, v in both.items()}}) + "\n")
                vh.flush()
            handle.write(json.dumps({"state_id": state["state_id"], "distributions": dist}) + "\n")
            handle.flush()   # NFS block size makes Python buffer ~1 MB; flush so progress is visible
    if vh:
        vh.close()


if __name__ == "__main__":
    main()
