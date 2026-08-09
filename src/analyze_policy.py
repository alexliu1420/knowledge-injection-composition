"""Does the pre-treatment predictor support a usable POLICY?

The draft's conclusion could not say what a practitioner should do with `anchor_margin`,
because no policy built on it had been tested. This tests the obvious one, and it needs
no new training: every item already has a pre-treatment margin and a post-injection
outcome, so a gate is just a threshold applied to data we hold.

    gate(tau):  inject facts with anchor_margin >= tau into the weights.
                Facts below tau are DEFERRED -- left to retrieval, or not committed.

Two things have to be true for this to be worth reporting:

  1. **It beats random selection at the same yield.** A gate that commits 50% of facts
     and gets a better composition rate than committing a random 50% is doing work.
  2. **The threshold transfers.** Choosing tau on one seed and evaluating it on another
     is the honest test; choosing tau on the data you report is not a policy, it is
     hindsight.

Reported as a yield/quality curve rather than a single operating point, because the
right tau depends on how much a practitioner values coverage against usability, and we
have no basis for choosing that for them.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

SEEDS = (0, 1, 2)


def load() -> list[dict]:
    rows = []
    for s in SEEDS:
        for dp, ev in (("data/tasks/anchored7.json", f"results/tmpl7/anchored_s{s}/eval_final.json"),
                       ("data/tasks/anchored7_control.json", f"results/tmpl7/control_s{s}/eval_final.json")):
            meta = {it["task_id"]: it for it in json.loads(Path(dp).read_text(encoding="utf-8"))["items"]}
            for r in json.loads(Path(ev).read_text(encoding="utf-8"))["per_item"]:
                d = meta.get(r["task_id"])
                if d:
                    rows.append({"seed": s, "tid": r["task_id"], "m": d["anchor_margin"],
                                 "y": float(r["chain_strict"]),
                                 "t": "|".join(d["template_key"])})
    return rows


def curve(rows: list[dict]) -> list[tuple[float, float, float, float]]:
    """(yield, gated composition, random-selection baseline, tau) over thresholds."""
    base = st.mean(r["y"] for r in rows)
    out = []
    for pct in range(0, 100, 5):
        srt = sorted(rows, key=lambda r: -r["m"])
        k = max(1, int(len(srt) * (100 - pct) / 100))
        sel = srt[:k]
        # random selection at the same yield has expected composition = base rate
        out.append((k / len(rows), st.mean(r["y"] for r in sel), base, sel[-1]["m"]))
    return out


def main() -> None:
    rows = load()
    base = st.mean(r["y"] for r in rows)
    print(f"{len(rows)} item-evaluations (3 seeds x 2 arms).  "
          f"Commit-everything composition = {base:.4f}")

    print()
    print("=== gate on pre-treatment anchor margin ===")
    print(f"  {'yield':>7}{'tau':>8}{'gated':>9}{'random':>9}{'lift':>9}")
    for y, g, r, tau in curve(rows):
        if round(y * 100) % 20 == 0 or abs(y - 0.5) < 0.03:
            print(f"  {y:>7.1%}{tau:>8.2f}{g:>9.4f}{r:>9.4f}{g - r:>+9.4f}")

    # 1 · does it beat random at matched yield?  (it must, if rho > 0 -- the question
    #     is by how much, and whether the magnitude is practically interesting)
    half = [c for c in curve(rows) if abs(c[0] - 0.5) < 0.03][0]
    print()
    print(f"  Commit the top half: composition {half[1]:.4f} vs {half[2]:.4f} "
          f"if you commit a random half  ->  {half[1] / half[2]:.2f}x")

    # 2 · THE HONEST TEST: pick tau on one seed, evaluate on the others
    print()
    print("=== does the threshold TRANSFER across seeds? ===")
    print("  (choose tau on one seed to hit 50% yield, apply it to the held-out seeds)")
    lifts = []
    for fit in SEEDS:
        f = [r for r in rows if r["seed"] == fit]
        tau = sorted((r["m"] for r in f), reverse=True)[len(f) // 2]
        held = [r for r in rows if r["seed"] != fit]
        sel = [r for r in held if r["m"] >= tau]
        if not sel:
            continue
        g = st.mean(r["y"] for r in sel)
        b = st.mean(r["y"] for r in held)
        lifts.append(g - b)
        print(f"    fit on seed {fit}: tau={tau:+.2f}  held-out yield {len(sel)/len(held):.1%}  "
              f"composition {g:.4f} vs {b:.4f} base  lift {g - b:+.4f}")
    if lifts:
        print(f"    mean held-out lift {st.mean(lifts):+.4f}")

    # 3 · does it survive within template? a gate that only sorts templates is much
    #     less useful -- a practitioner can already see the template.
    print()
    print("=== within-template: is the gate doing more than ranking templates? ===")
    by = defaultdict(list)
    for r in rows:
        by[r["t"]].append(r)
    ws = []
    for t, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(rs) < 60:
            continue
        srt = sorted(rs, key=lambda r: -r["m"])
        top = srt[:len(srt) // 2]
        g, b = st.mean(r["y"] for r in top), st.mean(r["y"] for r in rs)
        ws.append(g - b)
        print(f"    {t[:46]:<48} n={len(rs):>4}  top-half {g:.3f} vs {b:.3f}  {g - b:+.4f}")
    if ws:
        print(f"    mean within-template lift {st.mean(ws):+.4f}  "
              f"positive in {sum(1 for v in ws if v > 0)}/{len(ws)}")

    Path("results/policy.json").write_text(json.dumps({
        "base_rate": base,
        "curve": [{"yield": y, "tau": tau, "gated": g, "random": r}
                  for y, g, r, tau in curve(rows)],
        "held_out_lift_mean": st.mean(lifts) if lifts else None,
        "within_template_lift_mean": st.mean(ws) if ws else None,
    }, indent=2), encoding="utf-8")
    print("\nwrote results/policy.json")


if __name__ == "__main__":
    main()
