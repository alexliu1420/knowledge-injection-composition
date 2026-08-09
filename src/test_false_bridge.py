"""Is `chain+E2` composition, or one-hop lookup on whatever token is in context?

A13 established a dissociation: supplying E2 in context doubles composition (0.220 ->
0.498), but TRAINING the model to generate E2 (0.000 -> 1.000) buys nothing anchor-
specific. The sceptical reading of that is deflationary and has to be tested:

    when E2 is supplied, the model may not be composing at all -- it may be reading the
    supplied token and doing a single-hop lookup of the fact it memorised about it.

If so, `chain+E2` = 0.498 is not evidence about composition, and A9/A10/A13 all need
restating a second time.

The test: supply a FALSE bridge E2' taken from another item of the SAME template, whose
own fact2 was also injected -- so the model has a stored answer E3' for it. Then ask the
original chaining question and score BOTH possible answers.

    follows the supplied bridge -> outputs E3'   (one-hop lookup; ignores truth)
    uses its own path           -> outputs E3    (the true chain answer)

Four conditions on the same checkpoint:

    chain            no bridge supplied
    chain+TRUE       the real bridge          (reproduces the 0.498)
    chain+FALSE      a same-template decoy whose fact2 was injected
    direct-FALSE     "What <e3type> has <r2> relation with <e2type> E2'?"  -- confirms
                     the model actually stored E3' and can produce it when asked plainly

Decisive comparison is `follow rate under FALSE` against `correct rate under TRUE`:

    similar   -> the model routes on the supplied token regardless of truth.
                 `chain+E2` measures lookup, not composition. Deflates A9/A13.
    far lower -> the supplied bridge is used as evidence, not as a pointer, and the
                 model resists a bridge inconsistent with what it knows. A13 stands.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_runner import generate, match_strict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="results/tmpl7/anchored_s0/checkpoint-final")
    ap.add_argument("--data", default="data/tasks/anchored7.json")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/false_bridge.json")
    args = ap.parse_args()

    items = json.loads(Path(args.data).read_text(encoding="utf-8"))["items"]

    # decoy from the SAME template: same entity types, same relations, same phrasing --
    # so the only thing that differs is whether the bridge is the true one.
    by_t = defaultdict(list)
    for it in items:
        by_t["|".join(it["template_key"])].append(it)
    rng = random.Random(args.seed)
    rows = []
    for it in items:
        pool = [o for o in by_t["|".join(it["template_key"])] if o["e2_id"] != it["e2_id"]]
        if not pool:
            continue
        dec = rng.choice(pool)
        bridge_type = it["template_key"][2]
        rows.append({
            "task_id": it["task_id"],
            "true_e2": it["anchor_answer"], "false_e2": dec["anchor_answer"],
            "true_ans": it["chain_answer"], "false_ans": dec["fact2_answer"],
            "chain": it["chain_prompt"],
            "chain_true": f"{it['anchor_answer']} is the {bridge_type} referred to below.\n{it['chain_prompt']}",
            "chain_false": f"{dec['anchor_answer']} is the {bridge_type} referred to below.\n{it['chain_prompt']}",
            "direct_false": dec["fact2_prompt"],
        })
    # a decoy whose answer collides with the true answer cannot separate the hypotheses
    rows = [r for r in rows if r["false_ans"].strip().lower() != r["true_ans"].strip().lower()]
    print(f"{len(rows)} items with a usable same-template decoy")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to("cuda").eval()
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()

    preds = {}
    for cond in ("chain", "chain_true", "chain_false", "direct_false"):
        preds[cond] = generate(model, tok, [r[cond] for r in rows],
                               batch_size=args.batch_size, max_new_tokens=24)
    del model
    torch.cuda.empty_cache()

    for i, r in enumerate(rows):
        for cond in ("chain", "chain_true", "chain_false", "direct_false"):
            r[f"{cond}_pred"] = preds[cond][i]
            r[f"{cond}_true"] = float(match_strict([r["true_ans"]], preds[cond][i]))
            r[f"{cond}_false"] = float(match_strict([r["false_ans"]], preds[cond][i]))

    def m(k):
        return st.mean(r[k] for r in rows)

    print()
    print(f"  {'condition':<16}{'-> TRUE answer':>16}{'-> FALSE answer':>17}")
    for cond, lbl in (("chain", "chain"), ("chain_true", "chain + TRUE E2"),
                      ("chain_false", "chain + FALSE E2"), ("direct_false", "direct(FALSE E2)")):
        print(f"  {lbl:<16}{m(cond + '_true'):>16.4f}{m(cond + '_false'):>17.4f}")

    stored = m("direct_false_false")
    follow = m("chain_false_false")
    correct = m("chain_true_true")
    print()
    print(f"  decoy's fact IS stored (direct)          : {stored:.4f}")
    print(f"  FOLLOW rate  (chain+FALSE -> decoy's ans): {follow:.4f}")
    print(f"  CORRECT rate (chain+TRUE  -> true ans)   : {correct:.4f}")
    print()
    if stored < 0.5:
        print("  !! the decoy's fact is not reliably stored -- the follow rate has no")
        print("     ceiling to be compared against; test is inconclusive.")
    elif follow >= 0.7 * correct:
        print("  -> The model routes on the SUPPLIED TOKEN regardless of truth.")
        print("     `chain+E2` measures one-hop lookup, not composition. A9/A13 deflate.")
    elif follow <= 0.3 * correct:
        print("  -> The model RESISTS a bridge inconsistent with what it knows; the")
        print("     supplied bridge is evidence, not a pointer. A13 stands.")
    else:
        print("  -> Intermediate: partially token-driven. Report the ratio, claim neither.")

    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
