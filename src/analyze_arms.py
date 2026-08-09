"""Track B analysis: the four token-matched arms, evaluated against pre-declared predictions.

Predictions were fixed in docs/OBJECTIVES.md before any arm ran:

    P1  B > A          the intervention does something
    P2  B <= 0.498     the E2-in-context ceiling; exceeding it breaks the retrieval reading
    P3  B > D          DECISIVE -- if B == D the mechanism is bridge salience, not the
                       relational link, which contradicts A11
    P4  B > C, C ~= A  nothing is anchor-specific if any known fact suffices
    P5  B's gain is largest for LOW-margin items

All arms train the same 114 items, so every contrast is PAIRED and tested with McNemar's
exact test rather than a two-sample comparison. Items also cluster by template, so the
difference in means additionally gets a template-level (cluster) bootstrap -- item-level
intervals are anti-conservative here because items share entities and a model.

**Primary comparison is at `final`, not `mem100`** -- the opposite of the anchored-vs-
control convention, deliberately.

`mem100` fires when every INJECTED fact is memorised, and the arms inject different
second facts. Arm A's aug IS fact2, so it saturates almost immediately; arms B/C/D must
additionally memorise a distinct fact, so their mem100 fires later, giving them MORE
total training. Comparing there would hand the treatment arms extra compute and call it
matched recall.

At `final` every arm has had identical compute -- same 114 items, same 228 rows, same 40
epochs -- so epochs are matched by construction and recall only needs verifying (it is,
below). `mem100` was correct for anchored-vs-control, where the two datasets differed in
size and saturation; here the same logic points the other way.

Also reported at `mem100` for completeness. Composition emerges AFTER memorisation
saturates (A6: ep12->40, 0.150->0.233), so `final` additionally carries more signal.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

ARMS = {
    "A": "baseline  (fact2 repeated)",
    "B": "TREATMENT (anchor link E1-r1->E2)",
    "C": "control   (unrelated known fact)",
    "D": "control   (other fact about E2)",
}
CEILING = 0.4978   # chain+E2, anchored arm, results/e2_in_context.json


def binom_two_sided(b: int, c: int) -> float:
    """Exact McNemar: P(|X - n/2| >= |b - n/2|) for X ~ Bin(n, 0.5), n = b + c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def cluster_bootstrap(pairs: list[tuple[str, float, float]], n: int, seed: int) -> tuple[float, float]:
    """CI on mean(y2 - y1), resampling TEMPLATES with replacement."""
    by = defaultdict(list)
    for t, y1, y2 in pairs:
        by[t].append(y2 - y1)
    keys = list(by)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        vals = []
        for k in (rng.choice(keys) for _ in keys):
            vals.extend(by[k])
        if vals:
            out.append(st.mean(vals))
    out.sort()
    if not out:
        return float("nan"), float("nan")
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def load(arm: str, root: str, ckpt: str) -> dict[str, dict] | None:
    p = Path(root) / f"arm{arm}_s0" / f"eval_{ckpt}.json"
    if not p.exists():
        return None
    return {r["task_id"]: r for r in json.loads(p.read_text(encoding="utf-8"))["per_item"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/arms")
    ap.add_argument("--checkpoint", default="final", choices=["mem100", "final"],
                    help="'final' is primary: identical compute across arms. See module "
                         "docstring for why mem100 is NOT the matched comparison here.")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = {it["task_id"]: it for it in
            json.loads(Path("data/tasks/armB.json").read_text(encoding="utf-8"))["items"]}
    data = {a: load(a, args.root, args.checkpoint) for a in ARMS}
    have = [a for a, v in data.items() if v]
    missing = [a for a in ARMS if a not in have]
    if missing:
        print(f"  !! not yet run: {', '.join('arm' + m for m in missing)}")
    if "B" not in have:
        print("  arm B (the treatment) has not run -- nothing to analyse")
        return

    common = set.intersection(*(set(data[a]) for a in have))
    print(f"checkpoint={args.checkpoint}   arms={have}   paired items n={len(common)}")

    print()
    print(f"  {'arm':<5}{'composition':>13}{'fact2 recall':>14}{'anchor gen':>12}   note")
    for a in have:
        rows = [data[a][t] for t in sorted(common)]
        comp = st.mean(r["chain_strict"] for r in rows)
        f2 = st.mean(r["fact2_strict"] for r in rows)
        f1 = st.mean(r["fact1_strict"] for r in rows)
        print(f"  {a:<5}{comp:>13.4f}{f2:>14.4f}{f1:>12.4f}   {ARMS[a]}")

    # Manipulation check -- arm B must actually have learned to generate the anchor.
    if "B" in have:
        f1b = st.mean(data["B"][t]["fact1_strict"] for t in common)
        print()
        print(f"  MANIPULATION CHECK  arm B anchor generation = {f1b:.4f}")
        if f1b < 0.5:
            print("  !! the anchor was NOT learned generatively -- arm B is void, do not")
            print("    interpret any contrast below as a test of the hypothesis.")

    # Arms C and D train an aug fact that `evaluate_dataset` never probes -- it only
    # scores fact1/fact2/chain. If an arm silently failed to learn its extra fact, it is
    # not the control it claims to be and the contrast is unfair. `mem_strict` in
    # history.json covers ALL injected facts (fact2 AND aug), so it is the check.
    print()
    print("  AUG LEARNED?  final-epoch memorisation over all injected facts")
    for a in have:
        hp = Path(args.root) / f"arm{a}_s0" / "history.json"
        if not hp.exists():
            print(f"    arm{a}: no history.json")
            continue
        h = json.loads(hp.read_text(encoding="utf-8"))
        m = h[-1].get("mem_strict", float("nan"))
        flag = "" if m >= 0.98 else "   !! extra fact NOT learned -- arm is not a fair control"
        print(f"    arm{a}: mem_strict = {m:.4f}{flag}")

    # Matched recall: the primary comparison is only valid if fact2 recall is equal.
    recalls = {a: st.mean(data[a][t]["fact2_strict"] for t in common) for a in have}
    spread = max(recalls.values()) - min(recalls.values())
    print(f"  MATCHED RECALL      spread across arms = {spread:.4f}"
          + ("   OK" if spread <= 0.02 else "   !! >2pp, comparison is confounded"))

    print()
    print("=== pre-declared predictions ===")
    tests = [("P1", "B", "A"), ("P3", "B", "D"), ("P4a", "B", "C"), ("P4b", "C", "A")]
    for label, x, y in tests:
        if x not in have or y not in have:
            print(f"  {label:<4} arm{x} vs arm{y}: not yet available")
            continue
        b = sum(1 for t in common if data[x][t]["chain_strict"] and not data[y][t]["chain_strict"])
        c = sum(1 for t in common if data[y][t]["chain_strict"] and not data[x][t]["chain_strict"])
        p = binom_two_sided(b, c)
        d = st.mean(data[x][t]["chain_strict"] - data[y][t]["chain_strict"] for t in common)
        lo, hi = cluster_bootstrap(
            [("|".join(meta[t]["template_key"]), float(data[y][t]["chain_strict"]),
              float(data[x][t]["chain_strict"])) for t in common], args.boot, args.seed)
        print(f"  {label:<4} arm{x} - arm{y} = {d:+.4f}   McNemar {x}+/{y}+ = {b}/{c}, p={p:.4f}"
              f"   cluster CI [{lo:+.4f}, {hi:+.4f}]")

    if "B" in have:
        compB = st.mean(data["B"][t]["chain_strict"] for t in common)
        print()
        print(f"  P2   arm B = {compB:.4f} vs ceiling {CEILING:.4f}   "
              + ("OK, at or below the ceiling" if compB <= CEILING + 0.02
                 else "!! EXCEEDS the ceiling -- the retrieval reading is wrong"))

    # P5: is the gain concentrated in low-margin items?
    if "B" in have and "A" in have:
        print()
        print("  P5   gain by pre-treatment anchor-margin tercile")
        order = sorted(common, key=lambda t: meta[t]["anchor_margin"])
        k = max(len(order) // 3, 1)
        print(f"       {'tercile':<10}{'n':>5}{'margin med':>12}{'armA':>9}{'armB':>9}{'gain':>9}")
        gains = []
        for i, lbl in enumerate(("low", "mid", "high")):
            chunk = order[i * k:(i + 1) * k] if i < 2 else order[2 * k:]
            if not chunk:
                continue
            a_ = st.mean(data["A"][t]["chain_strict"] for t in chunk)
            b_ = st.mean(data["B"][t]["chain_strict"] for t in chunk)
            gains.append(b_ - a_)
            print(f"       {lbl:<10}{len(chunk):>5}"
                  f"{st.median(meta[t]['anchor_margin'] for t in chunk):>12.3f}"
                  f"{a_:>9.3f}{b_:>9.3f}{b_ - a_:>+9.3f}")
        if len(gains) == 3:
            print(f"       low-tercile gain {gains[0]:+.3f} vs high {gains[2]:+.3f} -> "
                  + ("P5 supported" if gains[0] > gains[2] else "P5 NOT supported (gain is not margin-dependent)"))

    Path("results/arms_analysis.json").write_text(json.dumps({
        "checkpoint": args.checkpoint, "n": len(common), "arms": have,
        "composition": {a: st.mean(data[a][t]["chain_strict"] for t in common) for a in have},
        "fact2_recall": recalls,
    }, indent=2), encoding="utf-8")
    print("\nwrote results/arms_analysis.json")


if __name__ == "__main__":
    main()
