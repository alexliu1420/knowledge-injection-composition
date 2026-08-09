"""Supply the bridge entity in context, bypassing retrieval.

Two questions, one experiment (docs/INTERPRETATIONS.md I1):

**Q1 -- is the anchor a RETRIEVAL PATH or a REPRESENTATIONAL SCAFFOLD?**
    If composition fails because E1->E2 is not traversable, then handing the model E2
    should rescue it, and the dependence on anchor margin should vanish.
    If the anchor instead supplies a representation that E3 attaches to, supplying E2
    in the prompt changes little and the margin dependence survives.

**Q2 -- why does `transporter` compose at exactly 0.000?**
    Same logic at the relation level. If supplying E2 rescues transporter items, the
    bridge was merely unreachable. If they stay at zero, the relation itself is not
    composable and no amount of anchoring would help.

Three conditions per item, all evaluated on the SAME trained checkpoint:

    chain      "Which drug is carried by proteins that interact with {E1}?"
    chain+E2   same question, with the bridge stated in the prompt
    direct     "What drug has carrier relation with gene/protein {E2}?"   (= the
               injected fact, verbatim -- a ceiling, and a check that the fact is there)

`direct` matters: if chain+E2 falls well short of it, the model can recall the fact when
asked plainly but not when the same fact must be used inside a compositional frame --
which would locate the failure in composition rather than in retrieval OR storage.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_runner import generate, match_strict


def build_conditions(items: list[dict]) -> dict[str, list[str]]:
    chain, chain_e2, direct = [], [], []
    for it in items:
        chain.append(it["chain_prompt"])
        # state the bridge, then ask the original question unchanged
        chain_e2.append(
            f"{it['anchor_answer']} is the {it['template_key'][2]} referred to below.\n"
            f"{it['chain_prompt']}"
        )
        direct.append(it["fact2_prompt"])
    return {"chain": chain, "chain+E2": chain_e2, "direct": direct}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="results/tmpl7/anchored_s0/checkpoint-final")
    ap.add_argument("--control-adapter", default="results/tmpl7/control_s0/checkpoint-final")
    ap.add_argument("--data", default="data/tasks/anchored7.json")
    ap.add_argument("--control-data", default="data/tasks/anchored7_control.json")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="results/e2_in_context.json")
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    out: dict = {}
    for arm, adapter, data_path in (("anchored", args.adapter, args.data),
                                    ("control", args.control_adapter, args.control_data)):
        items = json.loads(Path(data_path).read_text(encoding="utf-8"))["items"]
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to("cuda").eval()
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()

        conds = build_conditions(items)
        scores: dict[str, list[float]] = {}
        for name, prompts in conds.items():
            preds = generate(model, tok, prompts, batch_size=args.batch_size, max_new_tokens=24)
            gold = [it["chain_answer"] for it in items]  # same target in all conditions
            scores[name] = [float(match_strict([g], p)) for g, p in zip(gold, preds)]

        rows = []
        for i, it in enumerate(items):
            rows.append({
                "task_id": it["task_id"],
                "r1": it["template_key"][1],
                "template": "|".join(it["template_key"]),
                "margin": it["anchor_margin"],
                **{k: v[i] for k, v in scores.items()},
            })
        out[arm] = rows
        del model
        torch.cuda.empty_cache()

        print(f"\n=== {arm} (n={len(rows)}) ===")
        for k in conds:
            print(f"  {k:<10} {st.mean(r[k] for r in rows):.4f}")

        print(f"  {'first hop':<22}{'chain':>8}{'chain+E2':>10}{'direct':>9}{'n':>5}")
        by = defaultdict(list)
        for r in rows:
            by[r["r1"]].append(r)
        for r1, rs in sorted(by.items(), key=lambda kv: -st.mean(x["chain"] for x in kv[1])):
            print(f"  {r1:<22}{st.mean(x['chain'] for x in rs):>8.3f}"
                  f"{st.mean(x['chain+E2'] for x in rs):>10.3f}"
                  f"{st.mean(x['direct'] for x in rs):>9.3f}{len(rs):>5}")

    # Q1: does the margin dependence survive when E2 is supplied?
    import sys

    sys.path.insert(0, "src")
    from analyze_anchor_dose import spearman

    pooled = out["anchored"] + out["control"]
    print("\n=== Q1: anchor-margin dependence, by condition ===")
    for cond in ("chain", "chain+E2"):
        rho, p = spearman([r["margin"] for r in pooled], [r[cond] for r in pooled])
        print(f"  {cond:<10} rho={rho:+.4f}  p={p:.2e}")
    print("  survives  -> anchor is a representational SCAFFOLD")
    print("  collapses -> anchor is a RETRIEVAL PATH")

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
