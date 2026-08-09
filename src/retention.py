"""Retention probes -- the axis the field optimises, and half of the I4 curve.

I4 compares two axes at matched recall:
  retention   -- what injection costs the model's prior capability  (what the field measures)
  integration -- whether injected facts compose                     (what it does not)

Retention here is held-out general-text perplexity. Chosen over a generative QA
benchmark because it is continuous rather than thresholded, needs no generation
(one forward pass per chunk), and is directly sensitive to weight drift -- which is
exactly the quantity isolation is supposed to limit. A thresholded accuracy metric
would be near-flat across most of the sweep and hide the trade we are trying to see.

A second probe (base-model factual QA) is included as a coarser, more interpretable
cross-check, since perplexity alone is easy to over-read.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

CACHE = Path("data/retention")

# Fallback corpus if the datasets download is unavailable. Deliberately generic:
# retention must be measured on capability UNRELATED to the injected biomedical
# facts, or it would confound damage with interference from the injection domain.
_FALLBACK = [
    "The Industrial Revolution was a period of global transition of human economy "
    "towards more efficient and stable manufacturing processes.",
    "In mathematics, a prime number is a natural number greater than 1 that is not "
    "a product of two smaller natural numbers.",
    "Photosynthesis is a system of biological processes by which photosynthetic "
    "organisms convert light energy into chemical energy.",
    "The city of Vienna is the capital and most populous city of Austria, situated "
    "on the Danube in the east of the country.",
    "A compiler is software that translates computer code written in one programming "
    "language into another target language.",
]


def load_heldout(n_chunks: int = 64, chunk_chars: int = 1200) -> list[str]:
    """Fixed held-out general text. Cached so every run scores identical material."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"heldout_{n_chunks}x{chunk_chars}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))

    # datasets 5.x dropped script-based loaders, so the canonical "wikitext" id fails.
    # Parquet mirrors work. Ordered by preference; first that loads wins.
    CANDIDATES = [
        ("Salesforce/wikitext", "wikitext-2-raw-v1", "test", "text"),
        ("Salesforce/wikitext", "wikitext-103-raw-v1", "test", "text"),
        ("NeelNanda/pile-10k", None, "train", "text"),
    ]

    texts: list[str] = []
    for repo, config, split, field in CANDIDATES:
        try:
            from datasets import load_dataset

            ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
            buf = ""
            for row in ds:
                buf += row[field]
                while len(buf) >= chunk_chars and len(texts) < n_chunks:
                    texts.append(buf[:chunk_chars])
                    buf = buf[chunk_chars:]
                if len(texts) >= n_chunks:
                    break
            if len(texts) >= n_chunks:
                print(f"[retention] held-out corpus from {repo}")
                break
        except Exception as exc:
            print(f"[retention] {repo} unavailable ({type(exc).__name__})")
            texts = []

    if len(texts) < n_chunks:
        # A degenerate corpus makes the retention axis meaningless, and retention is
        # half of the I4 curve. Fail loudly rather than silently score repeated text.
        raise RuntimeError(
            f"could not build a held-out corpus ({len(texts)}/{n_chunks} chunks). "
            "Retention would be measured on repeated fallback text, which is not "
            "sensitive to weight drift. Fix the data source before running the sweep."
        )

    cached.write_text(json.dumps(texts), encoding="utf-8")
    return texts


@torch.no_grad()
def perplexity(model, tokenizer, texts: list[str], *, max_len: int = 512) -> float:
    """Token-level perplexity over held-out text. Lower is better; rising = damage."""
    was_training = model.training
    model.eval()
    total_nll, total_tok = 0.0, 0
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len)
        ids = {k: v.to(model.device) for k, v in ids.items()}
        labels = ids["input_ids"].clone()
        out = model(**ids, labels=labels)
        n = int(labels.numel()) - 1
        total_nll += float(out.loss) * n
        total_tok += n
    if was_training:
        model.train()
    return float(torch.exp(torch.tensor(total_nll / max(total_tok, 1))))


@torch.no_grad()
def factual_probe(model, tokenizer, probes: list[dict], *, batch_size: int = 8) -> float:
    """Coarse cross-check: accuracy on facts the BASE model already answered correctly.

    Built once from the base model, so it measures degradation of prior capability
    rather than absolute knowledge. Perplexity is the primary; this guards against
    over-reading a metric with no task semantics.
    """
    from eval_runner import generate, match_strict

    if not probes:
        return float("nan")
    was_training = model.training
    model.eval()
    preds = generate(model, tokenizer, [p["question"] for p in probes],
                     batch_size=batch_size, max_new_tokens=24)
    if was_training:
        model.train()
    hits = sum(match_strict([p["answer"]], pr) for p, pr in zip(probes, preds))
    return hits / len(probes)


def measure(model, tokenizer, *, texts: list[str] | None = None,
            probes: list[dict] | None = None) -> dict:
    texts = texts if texts is not None else load_heldout()
    res = {"heldout_ppl": round(perplexity(model, tokenizer, texts), 4),
           "n_chunks": len(texts)}
    if probes:
        res["factual_probe_acc"] = round(factual_probe(model, tokenizer, probes), 4)
        res["n_probes"] = len(probes)
    return res
