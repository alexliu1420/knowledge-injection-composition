"""Dose-response: does composition scale with how recoverable the anchor is?

The binary anchored-vs-control split throws information away. Every item carries a
continuous `anchor_margin` -- logprob(correct hop-1 answer) minus the best distractor,
measured on the BASE model before any injection. The split at margin 0 was a
convenience, not a natural boundary, and the control arm's items are not
"unanchored" so much as "less strongly anchored".

So instead of comparing two groups, pool all items and ask directly:

    does per-item composition success increase with per-item anchor margin?

This is a stronger test of the mechanism than the group comparison:

  - it uses all ~503 items rather than splitting them
  - the predictor is measured on the BASE model, before the treatment, so it cannot
    have been influenced by injection
  - template is included as a stratifier, so an effect cannot come from the anchored
    templates simply being easier -- which is soundness-audit item A1, answered
    within-template rather than between-groups
  - a monotone relationship is far harder to produce by confound than a two-group
    difference

Reports: pooled correlation, per-template correlation, and composition by margin
quintile.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path


def spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Rank correlation + two-sided p (normal approximation, n>=10).

    Returns NaN when either vector is constant. A constant outcome means the
    correlation is UNDEFINED, not zero -- and returning 0.0 there caused two
    zero-composition templates to be counted as evidence AGAINST the hypothesis
    when they contained no test at all.
    """
    n = len(xs)
    if n < 10:
        return float("nan"), float("nan")
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan"), float("nan")

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):           # average ties
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    rho = num / den if den else 0.0
    z = rho * math.sqrt(n - 1)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return rho, p


def load_arm(data_path: str, eval_path: str) -> list[dict]:
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))["items"]
    margins = {d["task_id"]: d["anchor_margin"] for d in data}
    tmpl = {d["task_id"]: "|".join(d["template_key"]) for d in data}
    ev = json.loads(Path(eval_path).read_text(encoding="utf-8"))["per_item"]
    out = []
    for r in ev:
        tid = r["task_id"]
        if tid not in margins:
            continue
        out.append({
            "task_id": tid,
            "template": tmpl[tid],
            "anchor_margin": margins[tid],
            "chain_correct": int(r["chain_strict"]),
            "chain_logprob": r.get("chain_logprob"),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchored-data", default="data/tasks/anchored.json")
    ap.add_argument("--anchored-eval", nargs="+",
                    default=["results/anchored_dense/eval_final.json"],
                    help="one or more eval files; several pools across seeds")
    ap.add_argument("--control-data", default="data/tasks/anchored_control.json")
    ap.add_argument("--control-eval", nargs="+",
                    default=["results/anchored_control/eval_final.json"])
    ap.add_argument("--out", default="results/anchor_dose.json")
    args = ap.parse_args()

    # Pooling several seeds is the reported analysis: each seed contributes its own
    # per-item outcome against the same pre-treatment margin, so n scales with seeds.
    rows = []
    for ev in args.anchored_eval:
        if Path(ev).exists():
            rows += load_arm(args.anchored_data, ev)
    n_anchored = len(rows)
    for ev in args.control_eval:
        if Path(ev).exists():
            rows += load_arm(args.control_data, ev)
    n_seeds = max(len(args.anchored_eval), len(args.control_eval))
    print(f"{len(rows)} item-evaluations pooled over {n_seeds} seed(s) "
          f"({n_anchored} anchored, {len(rows)-n_anchored} control)")

    margins = [r["anchor_margin"] for r in rows]
    correct = [float(r["chain_correct"]) for r in rows]
    lps = [r["chain_logprob"] for r in rows if r["chain_logprob"] is not None]

    rho_c, p_c = spearman(margins, correct)
    print()
    print("=== pooled dose-response ===")
    print(f"  anchor margin vs composition CORRECT : rho = {rho_c:+.4f}  p = {p_c:.2e}")
    if len(lps) == len(rows):
        rho_l, p_l = spearman(margins, [r["chain_logprob"] for r in rows])
        print(f"  anchor margin vs composition LOGPROB : rho = {rho_l:+.4f}  p = {p_l:.2e}")

    # quintiles
    order = sorted(rows, key=lambda r: r["anchor_margin"])
    q = max(len(order) // 5, 1)
    print()
    print(f"  {'quintile':<10}{'n':>5}{'margin med':>12}{'composition':>13}")
    for i in range(5):
        chunk = order[i * q: (i + 1) * q] if i < 4 else order[4 * q:]
        if not chunk:
            continue
        print(f"  Q{i+1:<9}{len(chunk):>5}{st.median(r['anchor_margin'] for r in chunk):>12.3f}"
              f"{st.mean(r['chain_correct'] for r in chunk):>13.3f}")

    # within-template: answers soundness-audit A1 without relying on the group split
    print()
    print("=== within-template (controls for template difficulty) ===")
    by_t = defaultdict(list)
    for r in rows:
        by_t[r["template"]].append(r)
    per_t = {}
    for t, rs in sorted(by_t.items()):
        if len(rs) < 15:
            print(f"  {t[:52]:<54} n={len(rs):>3}  (too few)")
            continue
        rho, p = spearman([r["anchor_margin"] for r in rs],
                          [float(r["chain_correct"]) for r in rs])
        per_t[t] = rho
        print(f"  {t[:52]:<54} n={len(rs):>3}  rho={rho:+.4f}  p={p:.3f}")

    if per_t:
        pos = sum(1 for v in per_t.values() if v > 0)
        print()
        print(f"  templates with POSITIVE rho: {pos}/{len(per_t)}")
        print(f"  mean within-template rho   : {st.mean(per_t.values()):+.4f}")
        print()
        if pos == len(per_t):
            print("  The relationship holds inside every template, so it cannot be")
            print("  produced by the anchored templates simply being easier (audit A1).")
        elif pos >= len(per_t) * 0.6:
            print("  Mostly consistent within templates; some heterogeneity.")
        else:
            print("  NOT consistent within templates -- the pooled effect is likely")
            print("  driven by between-template differences, not by anchor strength.")

    Path(args.out).write_text(json.dumps({
        "n_pooled": len(rows), "n_anchored": n_anchored,
        "rho_correct": rho_c, "p_correct": p_c,
        "within_template_rho": per_t,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
