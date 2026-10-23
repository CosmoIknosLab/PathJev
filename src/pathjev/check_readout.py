"""Equivalence and speed of the cached vs. full-sequence SFT read-out (D-027).

    python -m pathjev.check_readout --states <val.jsonl> --model <medgemma> --adapter <ckpt> --n 40 --out check.json

Acceptance (fixed in D-027 before running): max |Δp| ≤ 0.02 over all candidates and
argmax agreement ≥ 0.99 over all decisions. Also checks that the continuation token ids
equal the processor's ids of the full sequence for every candidate scored.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from pathjev.generative import SFT_SYSTEM, Scorer, sft_prompt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--image-size", type=int, default=896)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--slow-vs-slow", action="store_true",
                    help="noise floor: compare two runs of the full-sequence read-out instead")
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.states.open() if l.strip()]
    states = random.Random("check-readout").sample(rows, min(args.n, len(rows)))
    scorer = Scorer(args.model, args.adapter, args.image_size)
    tok = scorer.processor.tokenizer
    max_dp, agree, total, tok_ok, tok_n = 0.0, 0, 0, 0, 0
    t_slow = t_fast = 0.0
    for state in states:
        for d in state["decisions"]:
            torch.cuda.synchronize(); t0 = time.time()
            slow = scorer.text_distribution(state["images"], d)
            torch.cuda.synchronize(); t1 = time.time()
            fast = (scorer.text_distribution if args.slow_vs_slow else scorer.text_distribution_cached)(state["images"], d)
            torch.cuda.synchronize(); t2 = time.time()
            t_slow += t1 - t0; t_fast += t2 - t1
            max_dp = max(max_dp, max(abs(slow[k] - fast[k]) for k in slow))
            agree += max(slow, key=slow.get) == max(fast, key=fast.get); total += 1
            prompt = sft_prompt(d)
            base = scorer._inputs(state["images"], prompt, system=SFT_SYSTEM)
            prefix = tok(scorer._chat_text(len(state["images"]), prompt, SFT_SYSTEM), add_special_tokens=False)["input_ids"]
            for c in d["candidates"][:3]:
                full = scorer._inputs(state["images"], prompt, answer=c["text"], system=SFT_SYSTEM)["input_ids"][0]
                cont = tok(scorer._chat_text(len(state["images"]), prompt, SFT_SYSTEM) + c["text"],
                           add_special_tokens=False)["input_ids"][len(prefix):]
                tok_ok += full[base["input_ids"].shape[1]:].tolist() == cont; tok_n += 1
    result = {"states": len(states), "decisions": total, "max_abs_dp": max_dp, "argmax_agreement": agree / total,
              "token_continuation_match": tok_ok / tok_n, "seconds_slow": t_slow, "seconds_fast": t_fast,
              "speedup": t_slow / t_fast, "device": torch.cuda.get_device_name(0), "mode": "slow-vs-slow" if args.slow_vs_slow else "slow-vs-cached",
              "accepted": max_dp <= 0.02 and agree / total >= 0.99 and tok_ok == tok_n}
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
