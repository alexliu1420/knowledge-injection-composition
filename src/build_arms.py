"""Build the four Track-B training arms from the anchored dataset.

Every arm trains **two items per fact**, so token count, example count and schedule are
matched and only the IDENTITY of the second item varies. A naive `fact2` vs
`fact2 + anchor` comparison would confound the intervention with more data, which
docs/BASELINES.md standing requirement 2 forbids.

    A  baseline   second item = fact2 repeated          controls count and schedule
    B  treatment  second item = the anchor E1-r1->E2     the intervention
    C  volume     second item = another item's anchor    "any extra known fact helps"
    D  salience   second item = another fact about E2    bridge salience vs the relation

**D is the decisive control.** A11 found the anchor effect is relational rather than
entity familiarity, so it predicts B > D. If B == D the mechanism is bridge salience and
A11 is contradicted. That is a differential prediction between two TREATMENT arms, which
"more training helps" cannot produce.

Parity requirement: arm B trains a fact the base model knows discriminatively (every
anchored item has anchor_margin > 0 by construction). D must too, or the arms differ in
whether the extra fact is known rather than in what it is about. So D candidates are
scored with the same distractor procedure and only positive-margin facts are used.

All four arms are restricted to the items that have a valid D candidate, so every
comparison is PAIRED on identical items.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_dataset import name_ok, single_hop_question
from eval_runner import continuous_scores
from prime_graph import PrimeGraph
from model_pin import revision_for


def d_candidates(g: PrimeGraph, items: list[dict], max_per_item: int) -> dict[str, list[dict]]:
    """For each item, other facts about E2: E2 -r3-> X with r3 not in {r1, r2}.

    Excluded by construction:
      neighbour == e3  -- that is fact2, the injected fact itself
      neighbour == e1  -- that is the anchor link, which is arm B
      r3 in {r1, r2}   -- same relation as either hop
    """
    wanted = {int(it["e2_id"]) for it in items}
    inc: dict[int, list[tuple[int, int]]] = defaultdict(list)   # e2 -> [(rel_id, tail)]
    src = g.edge_index[0].tolist()
    dst = g.edge_index[1].tolist()
    ets = g.edge_types.tolist()
    for a, b, r in zip(src, dst, ets):
        if a in wanted:
            inc[a].append((r, b))

    out: dict[str, list[dict]] = {}
    for it in items:
        e2 = int(it["e2_id"])
        r1, r2 = it["template_key"][1], it["template_key"][3]
        e2_type = it["template_key"][2]
        seen, cands = set(), []
        for rid, tail in inc.get(e2, []):
            rel = g.edge_type_dict[int(rid)]
            if rel in (r1, r2) or tail in (int(it["e1_id"]), int(it["e3_id"])):
                continue
            nm = g.name(tail)
            if not name_ok(nm) or (rel, nm) in seen:
                continue
            seen.add((rel, nm))
            cands.append({
                "prompt": single_hop_question(rel, e2_type, it["e2_name"], g.ntype(tail)),
                "answer": nm, "rel": rel, "tail_type": g.ntype(tail),
            })
            if len(cands) >= max_per_item:
                break
        if cands:
            out[it["task_id"]] = cands
    return out


def verify(model, tok, cands: dict[str, list[dict]], n_distractors: int,
           batch_size: int, seed: int) -> dict[str, dict]:
    """Keep, per item, the highest-margin candidate with margin > 0.

    Same distractor procedure as verify_anchors: same-relation alternatives, so the
    measure is discrimination rather than raw likelihood.
    """
    rng = random.Random(seed)
    by_rel: dict[str, list[str]] = defaultdict(list)
    for cl in cands.values():
        for c in cl:
            by_rel[c["rel"]].append(c["answer"])

    prompts, golds, owners = [], [], []
    for tid, cl in cands.items():
        for j, c in enumerate(cl):
            pool = [a for a in by_rel[c["rel"]] if a != c["answer"]]
            if len(pool) < n_distractors:
                continue
            prompts.append(c["prompt"]); golds.append(c["answer"]); owners.append((tid, j, True))
            for d in rng.sample(pool, n_distractors):
                prompts.append(c["prompt"]); golds.append(d); owners.append((tid, j, False))

    print(f"  scoring {len(prompts)} prompts for D-candidate verification")
    sc = continuous_scores(model, tok, prompts, golds, batch_size=batch_size)

    corr: dict[tuple[str, int], float] = {}
    dist: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (tid, j, is_c), lp in zip(owners, sc["logprob_mean"]):
        (corr.__setitem__((tid, j), lp) if is_c else dist[(tid, j)].append(lp))

    best: dict[str, dict] = {}
    for (tid, j), lp in corr.items():
        ds = dist.get((tid, j), [])
        if not ds:
            continue
        margin = lp - max(ds)
        if margin <= 0:
            continue
        if tid not in best or margin > best[tid]["d_margin"]:
            best[tid] = {**cands[tid][j], "d_margin": round(margin, 4)}
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tasks/anchored7.json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--n-distractors", type=int, default=4)
    ap.add_argument("--max-cands", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="data/tasks")
    ap.add_argument("--reuse", default=None,
                    help="reuse verified D facts from a previous results/arm_build.json. "
                         "D verification is the only GPU step, so this makes a rebuild "
                         "CPU-only -- needed to fix a construction bug without evicting "
                         "a training job from an 8 GB card.")
    args = ap.parse_args()

    payload = json.loads(Path(args.data).read_text(encoding="utf-8"))
    items = payload["items"]
    print(f"{len(items)} anchored items")

    if args.reuse and Path(args.reuse).exists():
        prev = json.loads(Path(args.reuse).read_text(encoding="utf-8"))
        best = prev["d_facts"]
        print(f"  reusing {len(best)} verified D facts from {args.reuse} (no GPU)")
    else:
        g = PrimeGraph()
        cands = d_candidates(g, items, args.max_cands)
        print(f"  {len(cands)}/{len(items)} items have >=1 structurally valid D candidate")

        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]
        tok = AutoTokenizer.from_pretrained(args.model, revision=revision_for(args.model))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision_for(args.model), dtype=dtype).to("cuda").eval()
        best = verify(model, tok, cands, args.n_distractors, args.batch_size, args.seed)
        del model
        torch.cuda.empty_cache()

    keep = [it for it in items if it["task_id"] in best]
    print(f"  {len(keep)}/{len(items)} items have a POSITIVE-margin D fact -- all arms "
          f"restricted to these so comparisons are paired")
    if not keep:
        raise SystemExit("no usable D candidates; arm D cannot be built")

    # Arm C: another item's anchor -- known discriminatively (parity with B) but about an
    # unrelated entity. Drawn from a DIFFERENT template so E1/E2 types differ too.
    #
    # Donors MUST come from items OUTSIDE the arm set. Drawing them from `keep` -- the
    # obvious implementation, and the one built first -- trained 53.5% of the arm's own
    # anchors as some other item's aug fact, making arm C half of arm B and destroying
    # prediction P4. The excluded items have verified anchors and are never evaluated,
    # so they are donors with no path to contamination.
    rng = random.Random(args.seed)
    kept_ids = {it["task_id"] for it in keep}
    donors = [it for it in items if it["task_id"] not in kept_ids]
    if not donors:
        raise SystemExit("no donor items outside the arm set; arm C would be contaminated")
    by_t = defaultdict(list)
    for it in donors:
        by_t["|".join(it["template_key"])].append(it)
    c_src: dict[str, dict] = {}
    for it in keep:
        t = "|".join(it["template_key"])
        pool = [o for o in donors
                if "|".join(o["template_key"]) != t and o["e2_id"] != it["e2_id"]]
        if not pool:
            pool = [o for o in donors if o["e2_id"] != it["e2_id"]]
        c_src[it["task_id"]] = rng.choice(pool)
    print(f"  arm C donors drawn from {len(donors)} items OUTSIDE the arm set "
          f"({len({v['task_id'] for v in c_src.values()})} distinct used)")

    arms = {
        "A": ("fact2 repeated (token/count-matched baseline)",
              lambda it: (it["fact2_prompt"], it["fact2_answer"], {})),
        "B": ("the anchor link E1-r1->E2 (TREATMENT)",
              lambda it: (it["fact1_prompt"], it["fact1_answer"],
                          {"aug_margin": it["anchor_margin"]})),
        "C": ("another item's anchor -- unrelated entity",
              lambda it: (c_src[it["task_id"]]["fact1_prompt"],
                          c_src[it["task_id"]]["fact1_answer"],
                          {"aug_margin": c_src[it["task_id"]]["anchor_margin"],
                           "aug_src": c_src[it["task_id"]]["task_id"]})),
        "D": ("a different known fact about E2",
              lambda it: (best[it["task_id"]]["prompt"], best[it["task_id"]]["answer"],
                          {"aug_margin": best[it["task_id"]]["d_margin"],
                           "aug_rel": best[it["task_id"]]["rel"]})),
    }

    for arm, (desc, fn) in arms.items():
        rows = []
        for it in keep:
            p, a, extra = fn(it)
            rows.append({**it, "aug_prompt": p, "aug_answer": a, **extra})
        out = {
            **{k: v for k, v in payload.items() if k != "items"},
            "arm": arm,
            "arm_description": desc,
            "built_from": args.data,
            # Two items per fact. train_inject reads this to decide what to train on.
            "injected_facts": ["fact2", "aug"],
            "items": rows,
        }
        p = Path(args.outdir) / f"arm{arm}.json"
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        margins = [r["aug_margin"] for r in rows if "aug_margin" in r]
        mtxt = (f"  aug margin mean {sum(margins)/len(margins):+.3f} "
                f"min {min(margins):+.3f}") if margins else "  (aug = fact2, no margin)"
        print(f"  arm{arm}: {len(rows)} items -> {p}   {desc}\n    {mtxt}")

    Path("results/arm_build.json").write_text(json.dumps({
        "n_anchored": len(items), "n_kept": len(keep),
        "d_facts": {k: v for k, v in best.items()},
    }, indent=2), encoding="utf-8")
    print("\nwrote results/arm_build.json")


if __name__ == "__main__":
    main()
