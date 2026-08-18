"""Build the anchored dataset (design B'').

    E1 --r1--> E2    REAL PrimeKG edge, base model verified to know it   (anchor)
    E2 --r2--> E3    INJECTED, novel, pure addition                     (under test)

Design B' made BOTH legs novel to avoid pretrained contamination. That was right for
attributing recall to injection and fatal for measuring composition: the literature
(EVIDENCE-LOG B2/B3) reports both-legs-injected composes at CHANCE, and we measured
exactly that -- controlled composition discrimination 0.5296 against a 0.4972 floor,
leaving nothing for any lead to move.

Here the first hop is knowledge the model already has, so composition has somewhere
to come from. This is also a direct test of the anchor hypothesis, which is our own
prediction rather than an inherited one.

Anchor verification is PER ITEM, not per relation. Discrimination of 0.83 for a
relation means most items are recoverable, not all; an item whose anchor the model
cannot rank above its distractors is not an anchored item and is dropped.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_dataset import SYSTEM_PROMPT, NeighborIndex, name_ok, single_hop_question
from eval_runner import continuous_scores
from prime_graph import CHAINING_TEMPLATES, PrimeGraph
from model_pin import revision_for

# First-hop anchor discrimination measured by src/probe_anchors.py. Only templates
# whose FIRST hop is recoverable can anchor; the rest are excluded by construction.
MIN_ANCHOR_DISCRIM = 0.55


@dataclass
class AnchoredItem:
    task_id: str
    template_key: list[str]
    e1_id: int
    e2_id: int
    e3_id: int
    e1_name: str
    e2_name: str
    e3_name: str
    anchor_prompt: str      # real, NOT injected -- verified known
    anchor_answer: str
    fact2_prompt: str       # injected
    fact2_answer: str
    chain_prompt: str
    chain_answer: str
    anchor_margin: float    # logprob(correct E2) - best distractor, base model


def candidate_items(g: PrimeGraph, tokenizer, per_template: int, seed: int,
                    templates: list[tuple[str, ...]]) -> list[AnchoredItem]:
    rng = random.Random(seed)
    nbr = NeighborIndex(g.edge_index)
    type_id = {v: k for k, v in g.node_type_dict.items()}
    nodes_of = {
        t: [n for n in (g.node_types == i).nonzero().flatten().tolist()
            if name_ok(g.name(n), tokenizer)]
        for t, i in type_id.items()
    }

    used: set[int] = set()
    out: list[AnchoredItem] = []

    for key in templates:
        h1_t, r1, br_t, r2, tail_t = key
        adj1 = g.typed_adjacency(r1, h1_t, br_t)   # REAL anchor edges
        adj2 = g.typed_adjacency(r2, br_t, tail_t)  # must be ABSENT for E2

        heads = [h for h in adj1 if name_ok(g.name(h), tokenizer)]
        rng.shuffle(heads)
        e3_pool = [n for n in nodes_of[tail_t]]
        rng.shuffle(e3_pool)

        made, i3 = 0, 0
        for e1 in heads:
            if made >= per_template * 3:  # oversample; verification will cull
                break
            if e1 in used:
                continue
            tails = [t for t in adj1[e1] if name_ok(g.name(t), tokenizer)]
            if len(tails) != 1:          # unique bridge keeps the chain unambiguous
                continue
            e2 = tails[0]
            if e2 in used or e2 in adj2:  # E2 must have NO existing r2 edge: pure addition
                continue

            e3 = None
            while i3 < len(e3_pool):
                cand = e3_pool[i3]; i3 += 1
                if cand in used or cand in (e1, e2):
                    continue
                if cand in nbr.neighbors(e1):   # no direct E1->E3 shortcut
                    continue
                e3 = cand
                break
            if e3 is None:
                break

            n1, n2, n3 = g.name(e1), g.name(e2), g.name(e3)
            out.append(AnchoredItem(
                task_id=f"{h1_t}|{r1}|{r2}|{tail_t}|{made:04d}",
                template_key=list(key),
                e1_id=e1, e2_id=e2, e3_id=e3,
                e1_name=n1, e2_name=n2, e3_name=n3,
                anchor_prompt=single_hop_question(r1, h1_t, n1, br_t),
                anchor_answer=n2,
                fact2_prompt=single_hop_question(r2, br_t, n2, tail_t),
                fact2_answer=n3,
                chain_prompt=CHAINING_TEMPLATES[key]["template"].format(entity_1=n1),
                chain_answer=n3,
                anchor_margin=float("nan"),
            ))
            used |= {e1, e2, e3}
            made += 1
    return out


@torch.no_grad()
def verify_anchors(model, tok, items: list[AnchoredItem], n_distractors: int,
                   batch_size: int, seed: int) -> tuple[list[AnchoredItem], list[AnchoredItem]]:
    """Split items by whether the BASE model ranks the anchor above every distractor.

    Per item, not per relation: a relation at 0.83 discrimination still contains
    items the model cannot recover, and those are not anchored items.

    Returns (anchored, control). The REJECTED items are not waste -- they are the
    matched control for the anchored arm: same templates, same entity types, same
    one-fact injection. The split is observational, so the two groups differ in whether
    hop 1 is recoverable and may differ in other item properties as well; building the
    control from rejections constrains those differences rather than eliminating them.
    Sampling a control separately would additionally confound anchoring with template
    identity and fact count.
    """
    rng = random.Random(seed)
    by_tmpl: dict[str, list[str]] = {}
    for it in items:
        by_tmpl.setdefault("|".join(it.template_key), []).append(it.anchor_answer)

    prompts, golds, owners = [], [], []
    for i, it in enumerate(items):
        pool = [a for a in by_tmpl["|".join(it.template_key)] if a != it.anchor_answer]
        if len(pool) < n_distractors:
            continue
        prompts.append(it.anchor_prompt); golds.append(it.anchor_answer); owners.append((i, True))
        for d in rng.sample(pool, n_distractors):
            prompts.append(it.anchor_prompt); golds.append(d); owners.append((i, False))

    sc = continuous_scores(model, tok, prompts, golds, batch_size=batch_size)
    correct: dict[int, float] = {}
    distr: dict[int, list[float]] = {}
    for (i, is_c), lp in zip(owners, sc["logprob_mean"]):
        (correct.__setitem__(i, lp) if is_c else distr.setdefault(i, []).append(lp))

    kept, rejected = [], []
    for i, lp_c in correct.items():
        ds = distr.get(i, [])
        if not ds:
            continue
        items[i].anchor_margin = round(lp_c - max(ds), 4)
        (kept if lp_c > max(ds) else rejected).append(items[i])
    return kept, rejected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--probe", default="results/anchor_probe.json")
    ap.add_argument("--per-template", type=int, default=40)
    ap.add_argument("--n-distractors", type=int, default=4)
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/tasks/anchored.json")
    args = ap.parse_args()

    probe = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    templates = []
    for key in CHAINING_TEMPLATES:
        h1, r1, br, _, _ = key
        d = probe.get(f"{h1} -{r1}-> {br}", {}).get("discrimination")
        if d is not None and d >= MIN_ANCHOR_DISCRIM:
            templates.append(key)
    print(f"{len(templates)} template(s) with first-hop discrimination >= {MIN_ANCHOR_DISCRIM}:")
    for k in templates:
        print(f"   {k[0]} -{k[1]}-> {k[2]}  (discrim "
              f"{probe[f'{k[0]} -{k[1]}-> {k[2]}']['discrimination']:.4f})")
    if not templates:
        raise SystemExit("no templates clear the anchor threshold")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision_for(args.model))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    g = PrimeGraph()

    cands = candidate_items(g, tok, args.per_template, args.seed, templates)
    print(f"\n{len(cands)} candidates before anchor verification")

    model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision_for(args.model), dtype=dtype).to("cuda").eval()
    kept, rejected = verify_anchors(model, tok, cands, args.n_distractors,
                                    args.batch_size, args.seed)
    print(f"{len(kept)} survive per-item anchor verification "
          f"({len(kept)/max(len(cands),1):.1%}); {len(rejected)} become the matched control")

    from collections import Counter
    per = Counter("|".join(i.template_key) for i in kept)
    for k, v in per.items():
        print(f"   {v:>4}  {k}")

    def emit(items: list[AnchoredItem]) -> list[dict]:
        """Serialise, aliasing anchor -> fact1.

        Every downstream evaluator addresses probes as fact1/fact2. The alias used to
        be applied by a separate post-processing step, which was silently lost when
        this builder was re-run to emit the control split -- costing two seed runs
        that trained fine and then died at the final evaluation with
        KeyError: 'fact1_prompt'.

        Doing it here makes the dataset self-contained: a rebuild cannot produce a
        file that trains but cannot be evaluated.
        """
        rows = []
        for i in items:
            r = asdict(i)
            r["fact1_prompt"] = r["anchor_prompt"]
            r["fact1_answer"] = r["anchor_answer"]
            rows.append(r)
        return rows

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "design": "B-double-prime (anchored)",
        "seed": args.seed,
        "system_prompt": SYSTEM_PROMPT,
        "constraints": [
            "hop 1 is a REAL PrimeKG edge, per-item verified recoverable by the base model",
            "hop 2 is injected: E2 has no existing r2 edge to the tail type (pure addition)",
            "E3 is not a graph neighbour of E1 (no direct shortcut)",
            "unique bridge; each entity used at most once",
        ],
        "note": "only fact2 is injected during training; the anchor is pretrained knowledge",
        "injected_facts": ["fact2"],
        "items": emit(kept),
    }, indent=2), encoding="utf-8")

    ctrl = Path(str(out).replace(".json", "_control.json"))
    ctrl.write_text(json.dumps({
        "design": "B-double-prime CONTROL (anchor present but NOT recoverable)",
        "seed": args.seed,
        "system_prompt": SYSTEM_PROMPT,
        "note": ("matched control: identical templates, entity types and one-fact "
                 "injection; hop 1 is a real edge the base model CANNOT recover"),
        "injected_facts": ["fact2"],
        "items": emit(rejected),
    }, indent=2), encoding="utf-8")

    print(f"\nwrote {out}   ({len(kept)} anchored)")
    print(f"wrote {ctrl}   ({len(rejected)} matched control)")
    if kept:
        it = kept[0]
        print("\n--- example ---")
        print("anchor (NOT injected):", it.anchor_prompt, "->", it.anchor_answer,
              f"[margin {it.anchor_margin:+.3f}]")
        print("fact2  (injected)    :", it.fact2_prompt, "->", it.fact2_answer)
        print("chain                :", it.chain_prompt, "->", it.chain_answer)


if __name__ == "__main__":
    main()
