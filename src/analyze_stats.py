"""Statistical treatment for the manuscript (gaps G5 and G6).

Two problems with the numbers as reported so far.

**G5 -- item non-independence.** Items share entities, templates and a single model,
so treating 227 items (or 908 item-distractor pairs) as independent overstates
significance. Every SE quoted so far is anti-conservative. Fix: bootstrap resampling
at the TEMPLATE level (cluster bootstrap), which respects the dependence structure,
plus per-item bootstrap reported alongside so the difference is visible.

**G6 -- audit item A4.** Memorisation discrimination has been reported pooled across
fact1 and fact2. In the anchored design fact1 is the untrained anchor and fact2 is
the injected fact; pooling them could hide the anchor DEGRADING during injection,
which would be an alternative explanation for anything we attribute to anchoring.
Fix: report them separately.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path


def cluster_bootstrap(
    groups: dict[str, list[float]], n_boot: int = 5000, seed: int = 0
) -> tuple[float, float, float]:
    """Resample CLUSTERS with replacement, then items within them.

    Resampling items alone would treat 120 items from one template as 120 independent
    observations, which they are not.
    """
    rng = random.Random(seed)
    keys = list(groups)
    point = st.mean(v for vs in groups.values() for v in vs)
    boots = []
    for _ in range(n_boot):
        vals = []
        for _ in keys:
            k = rng.choice(keys)
            src = groups[k]
            vals.extend(rng.choice(src) for _ in range(len(src)))
        if vals:
            boots.append(st.mean(vals))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    return point, lo, hi


def item_bootstrap(values: list[float], n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [rng.choice(values) for _ in values]
        boots.append(st.mean(s))
    boots.sort()
    return boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]


def load_arm(data_path: str, eval_path: str) -> dict[str, list[float]]:
    """Per-template composition outcomes."""
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))["items"]
    tmpl = {d["task_id"]: "|".join(d["template_key"]) for d in data}
    ev = json.loads(Path(eval_path).read_text(encoding="utf-8"))["per_item"]
    g: dict[str, list[float]] = defaultdict(list)
    for r in ev:
        if r["task_id"] in tmpl:
            g[tmpl[r["task_id"]]].append(float(r["chain_strict"]))
    return dict(g)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default="results/stats.json")
    args = ap.parse_args()

    arms = {
        "anchored": ("data/tasks/anchored.json", "results/anchored_dense/eval_final.json"),
        "control": ("data/tasks/anchored_control.json", "results/anchored_control/eval_final.json"),
    }

    out: dict = {}
    print("=== G5: composition accuracy with cluster-bootstrap CIs ===")
    print(f"{'arm':<12}{'n':>5}{'point':>9}{'cluster 95% CI':>24}{'item 95% CI (too narrow)':>28}")
    print("-" * 80)
    for name, (dp, ep) in arms.items():
        if not Path(ep).exists():
            print(f"{name:<12}  (eval missing)")
            continue
        g = load_arm(dp, ep)
        flat = [v for vs in g.values() for v in vs]
        pt, lo, hi = cluster_bootstrap(g, args.n_boot)
        ilo, ihi = item_bootstrap(flat, args.n_boot)
        out[name] = {"n": len(flat), "point": round(pt, 4),
                     "cluster_ci": [round(lo, 4), round(hi, 4)],
                     "item_ci": [round(ilo, 4), round(ihi, 4)],
                     "n_templates": len(g)}
        print(f"{name:<12}{len(flat):>5}{pt:>9.4f}"
              f"{f'[{lo:.4f}, {hi:.4f}]':>24}{f'[{ilo:.4f}, {ihi:.4f}]':>28}")

    if "anchored" in out and "control" in out:
        a, c = out["anchored"], out["control"]
        overlap = not (a["cluster_ci"][0] > c["cluster_ci"][1] or c["cluster_ci"][0] > a["cluster_ci"][1])
        print()
        print(f"  cluster CIs overlap: {overlap}")
        print(f"  ratio (anchored/control): {a['point']/max(c['point'],1e-9):.2f}x")
        widen = ((a["cluster_ci"][1]-a["cluster_ci"][0]) /
                 max(a["item_ci"][1]-a["item_ci"][0], 1e-9))
        print(f"  clustering widens the anchored CI by {widen:.2f}x "
              f"-- the amount by which item-level statistics were overstating precision")

    # ---- G6 / audit A4
    print()
    print("=== G6 (audit A4): anchor vs injected discrimination, reported separately ===")
    for label, path in (("anchored", "results/gate_anchored.json"),
                        ("control", "results/gate_control.json")):
        if not Path(path).exists():
            continue
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        print(f"\n  {label}:")
        print(f"    {'checkpoint':<34}{'anchor(f1)':>12}{'injected(f2)':>14}{'chain':>9}")
        for k, v in d.items():
            print(f"    {k.split('/')[-1]:<34}{v['fact1']['discrimination']:>12.4f}"
                  f"{v['fact2']['discrimination']:>14.4f}{v['chain']['discrimination']:>9.4f}")
        base = d.get("base", {}).get("fact1", {}).get("discrimination")
        trained = [v["fact1"]["discrimination"] for k, v in d.items() if k != "base"]
        if base is not None and trained:
            delta = min(trained) - base
            out[f"{label}_anchor_drift"] = round(delta, 4)
            verdict = ("anchor intact" if delta > -0.05 else
                       "ANCHOR DEGRADED -- alternative explanation in play")
            print(f"    anchor discrimination change, base -> worst trained: "
                  f"{delta:+.4f}   ({verdict})")

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
