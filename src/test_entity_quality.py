"""Is the mechanism bridge-entity REPRESENTATION QUALITY rather than traversability?

The E2-in-context experiment showed the anchor-margin dependence SURVIVES when the
bridge is handed to the model (rho +0.234 -> +0.294). If the anchor were a route from
E1 to E2, supplying E2 should have erased it.

Anchor margin = logprob(E2 | E1-question) - best distractor. That confounds:
    (a) how well E1 -> E2 is known          -- traversability
    (b) how well E2 is represented at all   -- entity quality

This measures (b) WITHOUT reference to E1, on the BASE model:

    fam_type     length-normalised logprob of the E2 name given only its type
                 ("gene/protein: ") -- E1 never appears
    fam_ctr      fam_type centred within bridge type. Raw name logprob is confounded
                 by tokenisation and string frequency, which differ systematically
                 between drugs, genes and diseases; centring removes the between-type
                 shift while leaving within-type ordering untouched.

Decisive test is not which correlates better, but whether anchor margin still predicts
composition ONCE ENTITY QUALITY IS CONTROLLED FOR. Partial correlation via rank
residuals: regress both on entity quality, correlate what is left.

    margin retains predictive power  -> traversability contributes independently
    margin's power vanishes          -> anchor margin was a proxy for entity quality
                                        all along, and the simpler claim is correct
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "src")
from analyze_anchor_dose import spearman  # noqa: E402
from model_pin import revision_for

# torch/transformers are imported inside main(). `ranks` and `residualise` below are
# pure Python and are imported by analyze_entity_quality.py, which reports from saved
# JSON and must run in an environment with no GPU stack installed.


def ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def residualise(y: list[float], x: list[float]) -> list[float]:
    """Rank-residuals of y after removing linear dependence on rank(x)."""
    ry, rx = ranks(y), ranks(x)
    mx, my = st.mean(rx), st.mean(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    if sxx == 0:
        return ry
    b = sum((a - mx) * (c - my) for a, c in zip(rx, ry)) / sxx
    return [c - (my + b * (a - mx)) for a, c in zip(rx, ry)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="results/entity_quality.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from eval_runner import continuous_scores

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision_for(args.model))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # BASE model: entity quality is a pre-treatment property
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision_for(args.model), dtype=dtype).to("cuda").eval()

    rows = []
    for dp, ep in (("data/tasks/anchored7.json", "results/tmpl7/anchored_s0/eval_final.json"),
                   ("data/tasks/anchored7_control.json", "results/tmpl7/control_s0/eval_final.json")):
        meta = {d["task_id"]: d for d in json.loads(Path(dp).read_text(encoding="utf-8"))["items"]}
        for r in json.loads(Path(ep).read_text(encoding="utf-8"))["per_item"]:
            d = meta.get(r["task_id"])
            if d:
                rows.append({"task_id": r["task_id"], "e2": d["e2_name"],
                             "type": d["template_key"][2], "margin": d["anchor_margin"],
                             "template": "|".join(d["template_key"]),
                             "correct": float(r["chain_strict"])})

    print(f"{len(rows)} items; scoring entity quality on the BASE model")

    # E2 familiarity, no reference to E1 anywhere in the prompt
    q_type = [f"{r['type']}: " for r in rows]
    a_e2 = [r["e2"] for r in rows]
    fam = continuous_scores(model, tok, q_type, a_e2, batch_size=args.batch_size)
    for i, r in enumerate(rows):
        r["fam_type"] = fam["logprob_mean"][i]

    # centre within bridge type
    by_type: dict[str, list[float]] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r["fam_type"])
    type_mean = {t: st.mean(v) for t, v in by_type.items()}
    for r in rows:
        r["fam_ctr"] = r["fam_type"] - type_mean[r["type"]]

    del model
    torch.cuda.empty_cache()

    y = [r["correct"] for r in rows]
    print()
    print("=== predictors of composition (pooled) ===")
    preds = {}
    for k in ("margin", "fam_type", "fam_ctr"):
        x = [r[k] for r in rows]
        rho, p = spearman(x, y)
        preds[k] = rho
        print(f"  {k:<10} rho={rho:+.4f}  p={p:.2e}")

    print()
    print("=== do the two measure the same thing? ===")
    for k in ("fam_type", "fam_ctr"):
        rho_mm, _ = spearman([r["margin"] for r in rows], [r[k] for r in rows])
        print(f"  margin vs {k:<9}: rho={rho_mm:+.4f}")

    print()
    print("=== DECISIVE: partial correlations ===")
    for k in ("fam_type", "fam_ctr"):
        fam_v = [r[k] for r in rows]
        mar_v = [r["margin"] for r in rows]
        rho_pm, p_pm = spearman(residualise(mar_v, fam_v), residualise(y, fam_v))
        rho_pf, p_pf = spearman(residualise(fam_v, mar_v), residualise(y, mar_v))
        print(f"  controlling for {k}:")
        print(f"    margin         -> composition : rho={rho_pm:+.4f} p={p_pm:.2e}")
        print(f"    entity quality -> composition : rho={rho_pf:+.4f} p={p_pf:.2e}")
        if abs(rho_pm) > 0.10 and abs(rho_pm) > abs(rho_pf):
            print("    -> margin retains independent power; traversability contributes.")
        elif abs(rho_pf) > 0.10 and abs(rho_pf) > abs(rho_pm):
            print("    -> entity quality retains it; margin was a proxy.")
        else:
            print("    -> neither retains clear independent power; largely redundant,")
            print("       or entity quality is too weak a measure to separate them.")

    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
