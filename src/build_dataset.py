"""Build the fact-injection dataset (design B', decision D2).

Each item is a two-hop chain asserted over real PrimeKG entities:

    E1 --r1--> E2 --r2--> E3

with three constraints that make the chaining question well-posed and make a
correct answer attributable to the *injected* facts rather than to pretraining:

  1. E1 has NO existing edge of relation r1 to any node of the bridge type, and
     E2 has NO existing edge of r2 to any node of the tail type.
     => pure knowledge ADDITION, not editing. Asserting the fact overwrites no
        prior belief, so we are not accidentally running a knowledge-editing
        experiment (which has different mechanisms and its own literature).

  2. E3 is not a graph neighbour of E1 by any relation.
     => the model cannot shortcut E1->E3 directly; composition must route through
        the injected bridge.

  3. Each entity is used at most once across the dataset.
     => no cross-item interference, and the chaining question has one answer.

Question wording:
  - chaining      : verbatim from their CHAINING_TEMPLATES
  - single-hop    : deterministic canonical form from their generator spec,
                    "What <C_type> has <relation> relation with <A_type> <A>?"
                    Their instances used LLM paraphrases of this; a fixed template
                    removes generation variance as a confound (spec-gap T13).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from prime_graph import CHAINING_TEMPLATES, PrimeGraph
from model_pin import revision_for

SYSTEM_PROMPT = (
    "You are a biomedical assistant. "
    "Answer the question with the most appropriate entity name."
)

# Entity-name quality filter (spec-gap T14).
#
# PrimeKG mixes readable names with IUPAC/chemical strings -- 18% of drugs, 22% of
# molecular_functions and exposures. Unfiltered, an answer like
#   "(3R,4R)-4-(pyrrolidin-1-ylcarbonyl)-1-(quinoxalin-2-ylcarbonyl)pyrrolidin-3-amine"
# breaks the experiment two ways:
#   - under their bidirectional substring matcher, a prediction of "a" scores correct
#   - under a strict matcher, the item tests long-string memorisation, not composition
# Neither measures integration, so such names are excluded rather than scored.
IUPAC_RE = re.compile(r"^[\d(\[]|\d,\d|-yl\b|-yl[a-z]|amino|methyl|phenyl|\bacid\b|\d'", re.I)

MAX_NAME_CHARS = 40
MIN_NAME_CHARS = 4  # guards `pred in gold` false positives on very short strings
MAX_NAME_TOKENS = 10


def name_ok(name: str, tokenizer=None) -> bool:
    n = name.strip()
    if not (MIN_NAME_CHARS <= len(n) <= MAX_NAME_CHARS):
        return False
    if IUPAC_RE.search(n):
        return False
    if not re.search(r"[A-Za-z]", n):
        return False
    if tokenizer is not None and len(tokenizer.encode(n, add_special_tokens=False)) > MAX_NAME_TOKENS:
        return False
    return True


def single_hop_question(rel: str, head_type: str, head_name: str, tail_type: str) -> str:
    """Canonical form from their SINGLE_TASK_SYSTEM_TEMPLATE spec."""
    return f"What {tail_type} has {rel} relation with {head_type} {head_name}?"


@dataclass
class Item:
    task_id: str
    template_key: list[str]
    e1_id: int
    e2_id: int
    e3_id: int
    e1_name: str
    e2_name: str
    e3_name: str
    fact1_prompt: str
    fact1_answer: str
    fact2_prompt: str
    fact2_answer: str
    chain_prompt: str
    chain_answer: str


class NeighborIndex:
    """Undirected neighbour lookup over all 8.1M edges via sorted CSR + searchsorted.

    Built once. A dict-of-sets over 16.2M entries would cost several GB; this costs
    ~260 MB and answers membership in O(log n).
    """

    def __init__(self, edge_index: torch.Tensor):
        src = torch.cat([edge_index[0], edge_index[1]])
        dst = torch.cat([edge_index[1], edge_index[0]])
        order = torch.argsort(src)
        self.src_sorted = src[order].contiguous()
        self.dst_sorted = dst[order].contiguous()

    def neighbors(self, nid: int) -> set[int]:
        lo = int(torch.searchsorted(self.src_sorted, torch.tensor(nid), right=False))
        hi = int(torch.searchsorted(self.src_sorted, torch.tensor(nid), right=True))
        return set(self.dst_sorted[lo:hi].tolist())


def build(
    graph: PrimeGraph,
    *,
    per_template: int,
    seed: int,
    tokenizer=None,
    max_attempts_factor: int = 50,
) -> tuple[list[Item], dict]:
    rng = random.Random(seed)
    nbr = NeighborIndex(graph.edge_index)
    type_id = {v: k for k, v in graph.node_type_dict.items()}
    nodes_of = {
        t: [
            n
            for n in (graph.node_types == i).nonzero().flatten().tolist()
            if name_ok(graph.name(n), tokenizer)
        ]
        for t, i in type_id.items()
    }

    used: set[int] = set()
    items: list[Item] = []
    stats: dict = {}

    for key, tmpl in CHAINING_TEMPLATES.items():
        h1_t, r1, br_t, r2, tail_t = key
        adj1 = graph.typed_adjacency(r1, h1_t, br_t)
        adj2 = graph.typed_adjacency(r2, br_t, tail_t)

        e1_pool = [n for n in nodes_of[h1_t] if n not in adj1]
        e2_pool = [n for n in nodes_of[br_t] if n not in adj2]
        e3_pool = list(nodes_of[tail_t])
        rng.shuffle(e1_pool); rng.shuffle(e2_pool); rng.shuffle(e3_pool)

        made = 0
        rejected_shortcut = 0
        attempts = 0
        limit = per_template * max_attempts_factor
        i1 = i2 = i3 = 0

        while made < per_template and attempts < limit:
            attempts += 1
            if i1 >= len(e1_pool) or i2 >= len(e2_pool) or i3 >= len(e3_pool):
                break
            e1, e2, e3 = e1_pool[i1], e2_pool[i2], e3_pool[i3]
            i1 += 1; i2 += 1; i3 += 1

            if e1 in used or e2 in used or e3 in used:
                continue
            if len({e1, e2, e3}) != 3:
                continue
            # constraint 2: no direct E1->E3 shortcut
            if e3 in nbr.neighbors(e1):
                rejected_shortcut += 1
                continue

            n1, n2, n3 = graph.name(e1), graph.name(e2), graph.name(e3)
            items.append(
                Item(
                    task_id=f"{h1_t}|{r1}|{r2}|{tail_t}|{made:04d}",
                    template_key=list(key),
                    e1_id=e1, e2_id=e2, e3_id=e3,
                    e1_name=n1, e2_name=n2, e3_name=n3,
                    fact1_prompt=single_hop_question(r1, h1_t, n1, br_t),
                    fact1_answer=n2,
                    fact2_prompt=single_hop_question(r2, br_t, n2, tail_t),
                    fact2_answer=n3,
                    chain_prompt=tmpl["template"].format(entity_1=n1),
                    chain_answer=n3,
                )
            )
            used |= {e1, e2, e3}
            made += 1

        stats["|".join(key)] = {
            "made": made,
            "e1_pool": len(e1_pool),
            "e2_pool": len(e2_pool),
            "rejected_shortcut": rejected_shortcut,
            "attempts": attempts,
        }

    return items, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-template", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/tasks/pilot.json")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=revision_for(args.tokenizer))
    g = PrimeGraph()
    items, stats = build(
        g, per_template=args.per_template, seed=args.seed, tokenizer=tokenizer
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "design": "B-prime",
        "seed": args.seed,
        "per_template": args.per_template,
        "system_prompt": SYSTEM_PROMPT,
        "constraints": [
            "E1 has no existing r1 edge to bridge type (pure addition)",
            "E2 has no existing r2 edge to tail type (pure addition)",
            "E3 not a graph neighbour of E1 (no direct shortcut)",
            "each entity used at most once",
        ],
        "stats": stats,
        "items": [asdict(i) for i in items],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"wrote {len(items)} items -> {out}")
    for k, v in stats.items():
        print(f"  {v['made']:>4}  {k}   (shortcut-rejected {v['rejected_shortcut']})")
    if items:
        it = items[0]
        print("\n--- example ---")
        print("fact1 :", it.fact1_prompt, "->", it.fact1_answer)
        print("fact2 :", it.fact2_prompt, "->", it.fact2_answer)
        print("chain :", it.chain_prompt, "->", it.chain_answer)


if __name__ == "__main__":
    main()
