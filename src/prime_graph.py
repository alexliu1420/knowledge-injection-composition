"""STaRK-Prime (PrimeKG) graph loading and chaining-path extraction.

Replaces the source paper's LLM-based data generator. A chaining task is
    E1 --r1--> E2 --r2--> E3
which is a graph query, not a generation problem (results/phase1-log.md, decision D1).

We adopt their published task *design* -- the nine (h1_type, r1, bridge_type, r2,
tail_type) templates and question wordings hardcoded in their analysis/layer_patching.py
-- and supply our own instances from a peer-reviewed source.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

DEFAULT_ROOT = Path("data/stark/prime/processed")

# Verbatim from external/mem2gen/analysis/layer_patching.py CHAINING_TEMPLATES.
# Key: (h1_type, r1, bridge_type, r2, tail_type)
CHAINING_TEMPLATES: dict[tuple[str, ...], dict[str, str]] = {
    ("anatomy", "expression present", "gene/protein", "target", "drug"): {
        "template": "Which drug targets the genes or proteins which are expressed in {entity_1}?",
        "relation1": "expressed in", "relation2": "targets"},
    ("anatomy", "expression present", "gene/protein", "enzyme", "drug"): {
        "template": "Which drug is catalyzed by the genes or proteins which are expressed in {entity_1}?",
        "relation1": "expressed in", "relation2": "catalyzed by"},
    ("cellular_component", "interacts with", "gene/protein", "carrier", "drug"): {
        "template": "Which drug is carried by genes or proteins that interact with {entity_1}?",
        "relation1": "interacts with", "relation2": "carried by"},
    ("molecular_function", "interacts with", "gene/protein", "target", "drug"): {
        "template": "Which drug targets the genes or proteins that interact with {entity_1}?",
        "relation1": "interacts with", "relation2": "targets"},
    ("effect/phenotype", "side effect", "drug", "synergistic interaction", "drug"): {
        "template": "Which drug has a synergistica interaction with the drug that has the side effect {entity_1}?",
        "relation1": "has the side effect", "relation2": "has a synergistic interaction with"},
    ("disease", "indication", "drug", "contraindication", "disease"): {
        "template": "Which disease is a contraindication for the drugs that is indicated for {entity_1}?",
        "relation1": "indicated for", "relation2": "contraindication for"},
    ("disease", "parent-child", "disease", "phenotype present", "effect/phenotype"): {
        "template": "Which phenotype is present in the disease that is a sub type or super type of {entity_1}?",
        "relation1": "a sub type or super type of", "relation2": "present in"},
    ("gene/protein", "transporter", "drug", "side effect", "effect/phenotype"): {
        "template": "Which effect is a side effect of the drug that is transported by {entity_1}?",
        "relation1": "transported by", "relation2": "a side effect of"},
    ("drug", "transporter", "gene/protein", "interacts with", "exposure"): {
        "template": "Which exposure acts on the gene or protein that transports {entity_1}?",
        "relation1": "transports", "relation2": "acts on"},
}


@dataclass
class ChainPath:
    """One E1 -r1-> E2 -r2-> E3 path, with the ambiguity bookkeeping the eval needs."""

    template_key: tuple[str, ...]
    e1_id: int
    e2_id: int
    e3_id: int
    e1_name: str
    e2_name: str
    e3_name: str
    n_bridges: int          # how many r1-neighbours of the right type E1 has
    n_tails_from_e2: int    # how many r2-neighbours of the right type E2 has
    all_valid_e3: list[str] # every reachable tail name -- the gold_list for eval


class PrimeGraph:
    def __init__(self, root: str | Path = DEFAULT_ROOT):
        root = Path(root)
        self.node_type_dict: dict[int, str] = pickle.load(open(root / "node_type_dict.pkl", "rb"))
        self.edge_type_dict: dict[int, str] = pickle.load(open(root / "edge_type_dict.pkl", "rb"))
        self.node_info: dict = pickle.load(open(root / "node_info.pkl", "rb"))
        self.edge_index: torch.Tensor = torch.load(root / "edge_index.pt", weights_only=False)
        self.edge_types: torch.Tensor = torch.load(root / "edge_types.pt", weights_only=False)
        self.node_types: torch.Tensor = torch.load(root / "node_types.pt", weights_only=False)

        self._type_id = {v: k for k, v in self.node_type_dict.items()}
        self._rel_id = {v: k for k, v in self.edge_type_dict.items()}
        self._adj: dict[int, dict[int, list[int]]] = {}

    # -- basics -------------------------------------------------------------
    def name(self, nid: int) -> str:
        return self.node_info[nid]["name"]

    def ntype(self, nid: int) -> str:
        return self.node_type_dict[int(self.node_types[nid])]

    def typed_adjacency(self, rel: str, head_type: str, tail_type: str) -> dict[int, list[int]]:
        """head -> tails for `rel`, oriented head_type -> tail_type.

        PrimeKG's stored edge direction does NOT reliably match a template's semantic
        direction, and it differs per relation:
            expression present  stored gene/protein -> anatomy  (template needs anatomy -> gene)
            interacts with      stored gene/protein -> molecular_function
            indication          stored in both directions
            target              stored in both directions, partially

        So we orient by node type rather than by storage order. Every edge of this
        relation whose endpoints match {head_type, tail_type} is emitted head->tail,
        in whichever storage order it appears. This is immune to per-relation
        directionality quirks and is why a naive head->tail lookup silently fails here.
        """
        key = (self._rel_id[rel], head_type, tail_type)
        if key not in self._adj:
            rid = self._rel_id[rel]
            h_tid, t_tid = self._type_id[head_type], self._type_id[tail_type]
            mask = self.edge_types == rid
            src = self.edge_index[0][mask]
            dst = self.edge_index[1][mask]
            st = self.node_types[src]
            dt = self.node_types[dst]

            fwd = (st == h_tid) & (dt == t_tid)
            rev = (st == t_tid) & (dt == h_tid)

            d: dict[int, set[int]] = defaultdict(set)
            for s, t in zip(src[fwd].tolist(), dst[fwd].tolist()):
                d[s].add(t)
            for s, t in zip(src[rev].tolist(), dst[rev].tolist()):
                d[t].add(s)  # reversed into head_type -> tail_type
            self._adj[key] = {k: sorted(v) for k, v in d.items()}
        return self._adj[key]

    def edge_direction_report(self, rel: str, sample: int = 20000) -> dict:
        """Is this relation stored in both directions? Determines whether a template's
        stated direction is respected by a naive head->tail lookup."""
        rid = self._rel_id[rel]
        mask = self.edge_types == rid
        src = self.edge_index[0][mask][:sample].tolist()
        dst = self.edge_index[1][mask][:sample].tolist()
        pairs = set(zip(src, dst))
        both = sum(1 for a, b in pairs if (b, a) in pairs)
        head_types: dict[str, int] = defaultdict(int)
        tail_types: dict[str, int] = defaultdict(int)
        for a, b in list(pairs)[:5000]:
            head_types[self.ntype(a)] += 1
            tail_types[self.ntype(b)] += 1
        return {
            "relation": rel,
            "sampled_edges": len(pairs),
            "symmetric_fraction": round(both / max(len(pairs), 1), 3),
            "head_types": dict(sorted(head_types.items(), key=lambda x: -x[1])[:4]),
            "tail_types": dict(sorted(tail_types.items(), key=lambda x: -x[1])[:4]),
        }

    # -- path extraction ----------------------------------------------------
    def find_chains(
        self,
        template_key: tuple[str, ...],
        *,
        max_paths: int = 500,
        require_unique_bridge: bool = True,
        require_unique_tail: bool = True,
    ) -> list[ChainPath]:
        """Extract E1 -r1-> E2 -r2-> E3 paths matching the template's type signature.

        require_unique_bridge / require_unique_tail control ambiguity. PrimeKG relations
        are many-to-many, so without them the chaining question admits many correct
        answers and 'wrong' is unmeasurable. Recorded as spec-gap T10/T12: the source
        release does not state how they handled this.
        """
        h1_t, r1, br_t, r2, tail_t = template_key
        adj1 = self.typed_adjacency(r1, h1_t, br_t)
        adj2 = self.typed_adjacency(r2, br_t, tail_t)

        out: list[ChainPath] = []
        for e1, bridges_typed in adj1.items():
            if require_unique_bridge and len(bridges_typed) != 1:
                continue

            for e2 in bridges_typed:
                tails = adj2.get(e2, [])
                if not tails:
                    continue
                if require_unique_tail and len(tails) != 1:
                    continue

                # every tail reachable from E1 via any typed bridge -- the honest gold set
                all_tails = {t for b in bridges_typed for t in adj2.get(b, [])}
                e3 = tails[0]
                out.append(
                    ChainPath(
                        template_key=template_key,
                        e1_id=e1, e2_id=e2, e3_id=e3,
                        e1_name=self.name(e1), e2_name=self.name(e2), e3_name=self.name(e3),
                        n_bridges=len(bridges_typed),
                        n_tails_from_e2=len(tails),
                        all_valid_e3=sorted(self.name(t) for t in all_tails),
                    )
                )
                if len(out) >= max_paths:
                    return out
        return out
