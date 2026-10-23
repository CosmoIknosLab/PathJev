"""Non-MedGemma baselines (research_plan §2): (c) CONCH v1 zero-shot, (d) H-optimus-0 linear probe.

(c) conch-zeroshot — every Choice decision is scored by cosine similarity between
    the CONCH image embedding and an ensemble of text prompts per candidate
    (mean of normalised prompt embeddings), softmax with CONCH's learned logit
    scale. Noul/Score decisions are not scored (primary endpoint = Choice).
(d) features + probe — H-optimus-0 (Apache-2.0) CLS features; a multinomial
    logistic regression per (dataset, Choice qid), trained on the dataset's
    train split, or by grouped 5-fold cross-fitting when the dataset is
    test-only (HepatoBench, HiESD). C and temperature chosen on held-out
    folds of the training data only.
Outputs use the upstream decision-file format so `pathjev.evaluate` scores them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pathjev.groups import state_group

Image.MAX_IMAGE_PIXELS = None
PROMPTS = ["an H&E image of {}.", "a histopathology image showing {}.", "{}, H&E stain.",
           "a photomicrograph of {}.", "an image of {} tissue."]
HOPT_MEAN, HOPT_STD = (0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517)


def _states(path: Path, limit: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in path.open() if l.strip()]
    return rows[:limit] if limit else rows


# --- (c) CONCH zero-shot ---------------------------------------------------------------------

def conch_tokenize(tokenizer, texts):
    """conch.open_clip_custom.tokenize re-implemented with `tokenizer(...)`: transformers 5.x
    removed `batch_encode_plus`. Same arguments: 127 tokens + one <pad> placeholder for <cls>."""
    import torch.nn.functional as F

    tokens = tokenizer(texts, max_length=127, add_special_tokens=True, return_token_type_ids=False,
                       truncation=True, padding="max_length", return_tensors="pt")
    return F.pad(tokens["input_ids"], (0, 1), value=tokenizer.pad_token_id)

@torch.no_grad()
def conch_zeroshot(states_path: Path, out: Path, device: str = "cuda:0", limit: int = 0, batch: int = 64) -> None:
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch")
    model = model.to(device).eval()
    tokenizer = get_tokenizer()
    text_cache: dict[str, torch.Tensor] = {}

    def text_embedding(text: str) -> torch.Tensor:
        if text not in text_cache:
            tokens = conch_tokenize(tokenizer, [p.format(text) for p in PROMPTS]).to(device)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            mean = emb.mean(0)
            text_cache[text] = mean / mean.norm()
        return text_cache[text]

    states = _states(states_path, limit)
    scale = model.logit_scale.exp().item()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for start in range(0, len(states), batch):
            group = states[start:start + batch]
            images = torch.stack([preprocess(Image.open(s["images"][0]).convert("RGB")) for s in group]).to(device)
            feats = model.encode_image(images, proj_contrast=True, normalize=True)
            for s, f in zip(group, feats):
                dist = {}
                for d in s["decisions"]:
                    if d["type"] != "choice":
                        continue
                    text = torch.stack([text_embedding(c["text"]) for c in d["candidates"]])
                    probs = (scale * text @ f).softmax(-1).tolist()
                    dist[d["qid"]] = {c["id"]: p for c, p in zip(d["candidates"], probs)}
                handle.write(json.dumps({"state_id": s["state_id"], "distributions": dist}) + "\n")


# --- (d) H-optimus-0 features + linear probe --------------------------------------------------

@torch.no_grad()
def extract_features(states_path: Path, out: Path, device: str = "cuda:0", batch: int = 64) -> None:
    import timm
    from torchvision import transforms

    model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5,
                              dynamic_img_size=False).to(device).eval()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(HOPT_MEAN, HOPT_STD)])
    states = _states(states_path)
    ids, feats = [], []
    for start in range(0, len(states), batch):
        group = states[start:start + batch]
        x = torch.stack([tf(Image.open(s["images"][0]).convert("RGB")) for s in group]).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            feats.append(model(x).float().cpu().numpy())
        ids += [s["state_id"] for s in group]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=np.array(ids), features=np.concatenate(feats))


def _choice_table(states: list[dict], feats: dict[str, np.ndarray]):
    """{(qid, candidate_ids): (X, y, groups, candidate_ids, state_ids)} for one-hot Choice decisions.

    Keyed by the candidate list too: one qid can carry different candidate sets across sources
    (e.g. `tissue` for the four SPIDER organs), and labels are indices into that list (D-032)."""
    table: dict[tuple, dict] = {}
    for s in states:
        for d in s["decisions"]:
            if d["type"] != "choice" or d.get("target_kind", "one_hot") != "one_hot":
                continue
            ids = [c["id"] for c in d["candidates"]]
            entry = table.setdefault((d["qid"], tuple(ids)), {"X": [], "y": [], "g": [], "ids": ids, "sid": []})
            entry["X"].append(feats[s["state_id"]])
            entry["y"].append(ids.index(max(d["target"], key=d["target"].get)))
            entry["g"].append(state_group(s))   # patient where a map exists (pathjev.groups)
            entry["sid"].append(s["state_id"])
    return table


def _fit(X, y, groups, seed: int = 0):
    """Logistic regression; C by grouped 5-fold CV on the training data only, then temperature on OOF."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    folds = GroupKFold(n_splits=min(5, len(set(groups))))
    best = None
    for C in (0.001, 0.01, 0.1, 1.0):
        oof = np.zeros((len(y), int(max(y)) + 1))
        for tr, va in folds.split(Xs, y, groups):
            m = LogisticRegression(C=C, max_iter=2000).fit(Xs[tr], y[tr])
            oof[np.ix_(va, m.classes_)] = m.decision_function(Xs[va]) if m.coef_.shape[0] > 1 else \
                np.stack([-m.decision_function(Xs[va]), m.decision_function(Xs[va])], 1)
        nll = min((_nll(oof / t, y), t) for t in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0))
        if best is None or nll[0] < best[0]:
            best = (nll[0], C, nll[1])
    model = LogisticRegression(C=best[1], max_iter=2000).fit(Xs, y)
    return scaler, model, best[2]


def _nll(logits, y):
    logits = logits - logits.max(1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


def probe(train_states: Path | None, test_states: Path, train_feats: Path | None, test_feats: Path,
          out: Path) -> None:
    def load(states_path, feats_path):
        data = np.load(feats_path)
        return _states(states_path), dict(zip(data["ids"].tolist(), data["features"]))

    test, f_test = load(test_states, test_feats)
    tab_test = _choice_table(test, f_test)
    rows: dict[str, dict] = {s["state_id"]: {} for s in test}
    if train_states is not None:
        train, f_train = load(train_states, train_feats)
        tab_train = _choice_table(train, f_train)
    for (qid, cand), te in tab_test.items():
        if len(set(te["y"])) < 2:      # constant-answer questions (e.g. organ within one dataset)
            continue
        Xte = np.stack(te["X"])
        n_cls = len(te["ids"])
        tr = tab_train.get((qid, cand)) if train_states is not None else None   # None: no training labels
        if tr is not None:
            scaler, model, temp = _fit(np.stack(tr["X"]), np.array(tr["y"]), tr["g"])
            logits = np.full((len(Xte), n_cls), -1e9)
            logits[:, model.classes_] = model.decision_function(scaler.transform(Xte)).reshape(len(Xte), -1)
        else:   # no training labels for this question: grouped 5-fold cross-fitting inside the test set
            from sklearn.model_selection import GroupKFold
            y, g = np.array(te["y"]), te["g"]
            logits = np.full((len(Xte), n_cls), -1e9)
            for tr_idx, te_idx in GroupKFold(n_splits=5).split(Xte, y, g):
                scaler, model, temp = _fit(Xte[tr_idx], y[tr_idx], [g[i] for i in tr_idx])
                logits[np.ix_(te_idx, model.classes_)] = model.decision_function(
                    scaler.transform(Xte[te_idx])).reshape(len(te_idx), -1) / temp
            temp = 1.0
        z = logits / temp
        z = z - z.max(1, keepdims=True)
        probs = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        for sid, p in zip(te["sid"], probs):
            rows[sid][qid] = dict(zip(te["ids"], p.tolist()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps({"state_id": k, "distributions": v}) + "\n" for k, v in rows.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("conch"); a.add_argument("--states", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True); a.add_argument("--limit", type=int, default=0)
    b = sub.add_parser("features"); b.add_argument("--states", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("probe"); c.add_argument("--train-states", type=Path); c.add_argument("--train-feats", type=Path)
    c.add_argument("--test-states", type=Path, required=True); c.add_argument("--test-feats", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "conch":
        conch_zeroshot(args.states, args.out, limit=args.limit)
    elif args.cmd == "features":
        extract_features(args.states, args.out)
    else:
        probe(args.train_states, args.test_states, args.train_feats, args.test_feats, args.out)


if __name__ == "__main__":
    main()
