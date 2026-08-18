"""C2: null-controlled analysis of self-patching.

The published recovery figure (chaining 7.8% -> 44%) is the effect at the BEST layer
pair, selected post hoc from ~784 candidates. That statistic is positively biased by
construction: taking a maximum over many comparisons produces a positive value even
when every individual comparison is null.

Measured on an uninjected model, the null is not subtle:

    mean delta over all pairs   -0.029      (patching usually HURTS)
    fraction of pairs improving  28%
    per-item MAX delta          up to +0.667

So a per-item maximum near +0.67 is what NOISE looks like here. Reporting an injected
effect against zero cannot distinguish relocation from selection.

This compares like with like: run the identical sweep on the base model and on an
injected checkpoint, and ask whether the injected per-item maxima exceed the
distribution of base per-item maxima. Same grid, same items, same selection procedure
on both sides -- so the selection bias is present in both and cancels.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def _p95(values: list[float]) -> float:
    """Linear-interpolated 95th percentile.

    The summary table and the null threshold must use the same estimator; they did not,
    so the two p95 columns were not comparable. int(0.95 * (n - 1)) also collapses to
    the median at n = 3.
    """
    s = sorted(values)
    pos = 0.95 * (len(s) - 1)
    lo, frac = int(pos), pos - int(pos)
    return s[lo] + frac * (s[min(lo + 1, len(s) - 1)] - s[lo])


def per_item_stats(results: list[dict]) -> dict:
    """Collapse a sweep to one number per item, plus the full pair distribution."""
    maxima, means, all_pairs = [], [], []
    for r in results:
        vals = list(r.get("grid", {}).values())
        if not vals:
            continue
        maxima.append(max(vals))
        means.append(st.mean(vals))
        all_pairs.extend(vals)
    return {
        "n_items": len(maxima),
        "n_pairs": len(all_pairs),
        "max_mean": round(st.mean(maxima), 5) if maxima else None,
        "max_median": round(st.median(maxima), 5) if maxima else None,
        "max_p95": round(_p95(maxima), 5) if maxima else None,
        "pair_mean": round(st.mean(all_pairs), 5) if all_pairs else None,
        "pair_frac_positive": round(sum(1 for v in all_pairs if v > 0) / max(len(all_pairs), 1), 4),
        "_maxima": maxima,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="sweep JSON from the UNINJECTED model")
    ap.add_argument("--injected", required=True, help="sweep JSON from an injected checkpoint")
    ap.add_argument("--out", default="results/patching_analysis.json")
    args = ap.parse_args()

    base = per_item_stats(json.loads(Path(args.base).read_text(encoding="utf-8")))
    inj = per_item_stats(json.loads(Path(args.injected).read_text(encoding="utf-8")))

    print(f"{'':<26} {'base (null)':>14} {'injected':>14}")
    print("-" * 56)
    for k, label in (
        ("n_items", "items"), ("n_pairs", "layer pairs"),
        ("pair_mean", "mean delta, all pairs"),
        ("pair_frac_positive", "fraction improving"),
        ("max_mean", "per-item MAX, mean"),
        ("max_median", "per-item MAX, median"),
        ("max_p95", "per-item MAX, p95"),
    ):
        print(f"{label:<26} {str(base[k]):>14} {str(inj[k]):>14}")

    bm, im = base["_maxima"], inj["_maxima"]
    verdict: dict = {"base": {k: v for k, v in base.items() if not k.startswith("_")},
                     "injected": {k: v for k, v in inj.items() if not k.startswith("_")}}

    MIN_N = 20  # below this, a p95 is not a p95

    if bm and im:
        # Ties must count as half, or comparing a distribution against ITSELF returns
        # far below 0.5 -- caught exactly that way in the self-check.
        wins = sum(1 for a in im for b in bm if a > b)
        ties = sum(1 for a in im for b in bm if a == b)
        auc = (wins + 0.5 * ties) / max(len(im) * len(bm), 1)

        # Linear-interpolated percentile. int(0.95*(n-1)) collapses to the MEDIAN at
        # n=3, silently reporting a "p95" that is nothing of the sort.
        null_p95 = _p95(bm)
        above = sum(1 for v in im if v > null_p95) / len(im)

        if len(bm) < MIN_N or len(im) < MIN_N:
            print()
            print(f"  !! n = {len(bm)} base / {len(im)} injected items. Below {MIN_N} the")
            print(f"     p95 and AUC are not meaningful. Treat the numbers below as a")
            print(f"     plumbing check, not a result.")
        verdict["null_p95"] = round(null_p95, 5)
        verdict["frac_injected_above_null_p95"] = round(above, 4)
        verdict["auc_injected_vs_base"] = round(auc, 4)

        print()
        print("=== null-controlled comparison ===")
        print(f"  null (base) p95 of per-item max : {null_p95:+.5f}")
        print(f"  injected items above that       : {above:.1%}   (chance = 5%)")
        print(f"  AUC, injected max vs base max   : {auc:.4f}   (chance = 0.5)")
        print()
        if len(bm) < MIN_N or len(im) < MIN_N:
            print("  VERDICT WITHHELD -- sample too small.")
        elif auc > 0.65 and above > 0.20:
            print("  Patching recovers something that selection alone does not explain.")
        elif auc < 0.55:
            print("  The apparent recovery is NOT distinguishable from selection over")
            print("  layer pairs. The published oracle figure should be read with that")
            print("  in mind -- it is the same statistic on the same kind of grid.")
        else:
            print("  Marginal. More items before claiming either way.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
