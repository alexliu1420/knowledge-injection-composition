"""Feasibility probe for the anchored design (B'').

The redesign requires the FIRST hop to be knowledge the base model already has:

    E1 --r1--> E2   real PrimeKG edge, model already knows it   (anchor)
    E2 --r2--> E3   injected, novel                             (under test)

That is only possible if the base model actually knows some real PrimeKG edges.
Under design B' the base scored exactly 0.000 -- but those were FABRICATED pairings,
so zero was expected and says nothing about real ones.

This measures, per relation type, what fraction of real edges the base model can
recall. Relations with usable anchors become the templates for the new dataset;
relations at floor are unusable no matter how appealing the template reads.

Expectation is wide variance: "what drug is indicated for type 2 diabetes" is
plausible for a 1.5B model, "which gene is expressed in the agranular insular cortex"
is not. The point is to find out rather than assume.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_dataset import name_ok, single_hop_question
from eval_runner import distractor_control, generate, match_strict
from prime_graph import CHAINING_TEMPLATES, PrimeGraph


def sample_real_edges(g: PrimeGraph, head_t: str, rel: str, tail_t: str,
                      n: int, rng: random.Random, tokenizer) -> list[dict]:
    """Real (E1, r, E2) edges with clean names, one item per head entity."""
    adj = g.typed_adjacency(rel, head_t, tail_t)
    heads = [h for h in adj if name_ok(g.name(h), tokenizer)]
    rng.shuffle(heads)

    out = []
    for h in heads:
        tails = [t for t in adj[h] if name_ok(g.name(t), tokenizer)]
        if not tails:
            continue
        # a unique tail keeps the probe unambiguous; many-tailed heads are ambiguous
        # to score and would understate what the model knows
        if len(tails) != 1:
            continue
        out.append({
            "template_key": [head_t, rel, tail_t],
            "e1_id": h, "e2_id": tails[0],
            "e1_name": g.name(h), "e2_name": g.name(tails[0]),
            "anchor_prompt": single_hop_question(rel, head_t, g.name(h), tail_t),
            "anchor_answer": g.name(tails[0]),
        })
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--per-relation", type=int, default=60)
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/anchor_probe.json")
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    rng = random.Random(args.seed)
    g = PrimeGraph()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to("cuda").eval()

    # first hop of each chaining template, plus the second hop as a candidate anchor
    hops: dict[tuple[str, str, str], None] = {}
    for h1, r1, br, r2, tail in CHAINING_TEMPLATES:
        hops[(h1, r1, br)] = None
        hops[(br, r2, tail)] = None

    report: dict[str, dict] = {}
    print(f"{'relation (head -r-> tail)':<52} {'n':>4} {'acc':>7} {'discrim':>8}")
    print("-" * 76)

    for head_t, rel, tail_t in hops:
        items = sample_real_edges(g, head_t, rel, tail_t, args.per_relation, rng, tok)
        key = f"{head_t} -{rel}-> {tail_t}"
        if len(items) < 10:
            print(f"{key:<52} {len(items):>4} {'--':>7} {'--':>8}   (too few clean edges)")
            report[key] = {"n": len(items), "accuracy": None, "discrimination": None}
            continue

        preds = generate(model, tok, [i["anchor_prompt"] for i in items],
                         batch_size=args.batch_size, max_new_tokens=24)
        acc = sum(match_strict([i["anchor_answer"]], p)
                  for i, p in zip(items, preds)) / len(items)
        ctrl = distractor_control(model, tok, items, "anchor_prompt", "anchor_answer",
                                  n_distractors=4, batch_size=args.batch_size)
        report[key] = {
            "n": len(items),
            "accuracy": round(acc, 4),
            "discrimination": ctrl["discrimination"],
            "delta_logprob": ctrl["delta_logprob"],
        }
        print(f"{key:<52} {len(items):>4} {acc:>7.3f} {ctrl['discrimination']:>8.4f}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    usable = {k: v for k, v in report.items()
              if v.get("accuracy") is not None and v["accuracy"] >= 0.15}
    strong = {k: v for k, v in report.items()
              if v.get("discrimination") is not None and v["discrimination"] >= 0.70}

    print()
    print("=== anchor availability ===")
    print(f"  relations with accuracy >= 0.15    : {len(usable)}/{len(report)}")
    print(f"  relations with discrimination>=0.70: {len(strong)}/{len(report)}")
    for k, v in sorted(usable.items(), key=lambda kv: -kv[1]["accuracy"])[:8]:
        print(f"    {v['accuracy']:.3f} acc  {v['discrimination']:.3f} discrim   {k}")
    print()
    if not usable:
        print("  NO USABLE ANCHORS. The base model does not recall real PrimeKG edges")
        print("  at this scale, so the one-leg-pretrained design is not available here.")
        print("  Options: larger model, or an anchor domain outside biomedicine.")
    else:
        print("  Anchored design is feasible. Build the dataset from the relations above,")
        print("  filtering item-by-item on the anchor being answered correctly.")


if __name__ == "__main__":
    main()
