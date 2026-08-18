"""Item-clustered dose-response, replacing the pseudoreplicated normal approximation.

Treating 1554 "observations" -- 518 items scored under three seeds, with the predictor
repeated three times, all from one model -- as independent makes a normal-approximation
p-value invalid. Two things follow:

  1 · The unit of analysis is the ITEM. Each item contributes one outcome, its mean over
      seeds, against one margin. n falls from 1554 to 518, which is the honest n.

  2 · Uncertainty comes from resampling, not from a closed form. Items are resampled with
      replacement for the item-level interval; TEMPLATES are resampled for the
      cluster-level interval, since items within a template share entities and phrasing.
      The template-level interval is the one to report.

Two designs are supported. With --pooled-root, all facts come from ONE adapter and the
full sample is interpretable. Without it, the two margin groups were trained as separate
adapters with different fact counts, so pooling them lets margin sign identify which model
answered; that pooled figure is printed for comparison and labelled as confounded.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "common/src")
from analyze_anchor_dose import spearman  # noqa: E402

S1 = Path(".")


def load_pooled(seeds: list[int], root: str) -> list[dict]:
    """Both margin signs from ONE adapter -- the direct test of the E1 confound.

    In the tmpl7 design the two arms were trained separately, so margin sign also
    identified which model answered. Here all 518 facts are in a single training run, so
    a margin/outcome association cannot be produced by a difference between models.
    """
    meta = {i["task_id"]: i for i in
            json.loads((S1 / "data/tasks/pooled7.json").read_text(encoding="utf-8"))["items"]}
    acc: dict[str, list[float]] = defaultdict(list)
    for s in seeds:
        p = Path(root) / f"s{s}" / "eval_final.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text(encoding="utf-8"))["per_item"]:
            if r["task_id"] in meta:
                acc[r["task_id"]].append(float(r["chain_strict"]))
    return [{"task_id": k, "arm": meta[k].get("arm", "?"), "margin": meta[k]["anchor_margin"],
             "template": "|".join(meta[k]["template_key"]),
             "y": st.mean(v), "n_seeds": len(v)} for k, v in acc.items()]


def load(arm: str, seeds: list[int]) -> list[dict]:
    tag = "anchored" if arm == "anchored" else "control"
    data = "anchored7.json" if arm == "anchored" else "anchored7_control.json"
    meta = {i["task_id"]: i for i in
            json.loads((S1 / f"data/tasks/{data}").read_text(encoding="utf-8"))["items"]}
    acc: dict[str, list[float]] = defaultdict(list)
    for s in seeds:
        ev = json.loads((S1 / f"results/tmpl7/{tag}_s{s}/eval_final.json").read_text(encoding="utf-8"))
        for r in ev["per_item"]:
            if r["task_id"] in meta:
                acc[r["task_id"]].append(float(r["chain_strict"]))
    return [{"task_id": k, "arm": arm, "margin": meta[k]["anchor_margin"],
             "template": "|".join(meta[k]["template_key"]),
             "y": st.mean(v), "n_seeds": len(v)} for k, v in acc.items()]


def boot_rho(rows: list[dict], by: str, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile CI on Spearman rho, resampling whole clusters with replacement."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["task_id"] if by == "item" else r["template"]].append(r)
    keys = list(groups)
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        s: list[dict] = []
        for k in (rng.choice(keys) for _ in keys):
            s.extend(groups[k])
        rho, _ = spearman([r["margin"] for r in s], [r["y"] for r in s])
        if rho == rho:
            out.append(rho)
    out.sort()
    if not out:
        return float("nan"), float("nan")
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--perm", type=int, default=10000)
    ap.add_argument("--pooled-root", default=None,
                    help="single-adapter run root, e.g. results/pooled7; "
                         "both margin signs come from one training run")
    ap.add_argument("--out", default="results/dose_clustered.json")
    args = ap.parse_args()

    if args.pooled_root:
        rows = load_pooled(args.seeds, args.pooled_root)
        anch = [r for r in rows if r["arm"] == "anchored"]
        ctrl = [r for r in rows if r["arm"] != "anchored"]
        got = sorted({s for s in args.seeds
                      if (Path(args.pooled_root) / f"s{s}" / "eval_final.json").exists()})
        print(f"SINGLE-ADAPTER design: all {len(rows)} facts trained in one run, seeds {got}")
        print("Margin sign no longer identifies which model answered, so the pooled row")
        print("below is the unconfounded estimate -- this is the direct test of E1.")
        if got != sorted(args.seeds):
            print(f"*** PARTIAL: {len(got)} of {len(args.seeds)} seeds present. Preliminary. ***")
    else:
        anch, ctrl = load("anchored", args.seeds), load("control", args.seeds)
    print(f"unit of analysis = item.  anchored {len(anch)}, control {len(ctrl)}, "
          f"each averaged over {args.seeds}")

    out = {}
    print()
    print(f"  {'scope':<34}{'n':>6}{'rho':>9}{'item CI':>20}{'template CI':>22}")
    for label, rows, flag in (("anchored arm", anch, ""),
                              ("control arm", ctrl, ""),
                              ("pooled", anch + ctrl,
                               "" if args.pooled_root else "  <- CONFOUNDED (E1)")):
        rho, _ = spearman([r["margin"] for r in rows], [r["y"] for r in rows])
        il, ih = boot_rho(rows, "item", args.boot, 0)
        tl, th = boot_rho(rows, "template", args.boot, 0)
        out[label] = {"n": len(rows), "rho": rho, "item_ci": [il, ih], "template_ci": [tl, th]}
        print(f"  {label:<34}{len(rows):>6}{rho:>+9.4f}"
              f"   [{il:+.3f}, {ih:+.3f}]   [{tl:+.3f}, {th:+.3f}]{flag}")

    print()
    print("  The template-level interval is the one to report: items within a template")
    print("  share entities and phrasing, so item-level resampling understates uncertainty.")
    print(f"  Templates available: {len({r['template'] for r in anch + ctrl})}")

    # Which scope is unconfounded depends on the design: with separate adapters only the
    # within-arm view is interpretable; with a single adapter the full 518 items are.
    focus, focus_name = ((anch + ctrl, "all 518, single adapter") if args.pooled_root
                         else (anch, "anchored arm"))
    print()
    print(f"=== within template, {focus_name} (the unconfounded scope) ===")
    by = defaultdict(list)
    for r in focus:
        by[r["template"]].append(r)
    vals = []
    for t, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(rs) < 20 or len(set(r["y"] for r in rs)) < 2:
            print(f"  {t[:44]:<46} n={len(rs):>3}  (too few / constant)")
            continue
        rho, _ = spearman([r["margin"] for r in rs], [r["y"] for r in rs])
        vals.append(rho)
        print(f"  {t[:44]:<46} n={len(rs):>3}  rho={rho:+.4f}")
    if vals:
        print(f"\n  mean {st.mean(vals):+.4f}, positive in {sum(1 for v in vals if v>0)}/{len(vals)}")
    out["within_template_anchored"] = vals

    print()
    print(f"=== leave-one-template-out, {focus_name} ===")
    print("  Six templates cannot support inference about a template population, so instead")
    print("  we report whether any single one carries the association.")
    full, _ = spearman([r["margin"] for r in focus], [r["y"] for r in focus])
    loo = []
    for t in sorted(by):
        keep = [r for r in focus if r["template"] != t]
        rho, _ = spearman([r["margin"] for r in keep], [r["y"] for r in keep])
        loo.append({"dropped": t, "n": len(keep), "rho": rho})
        print(f"  drop {t[:40]:<42} n={len(keep):>3}  rho={rho:+.4f}  ({rho - full:+.4f})")
    print(f"  full sample rho={full:+.4f}; leave-one-out range "
          f"[{min(d['rho'] for d in loo):+.4f}, {max(d['rho'] for d in loo):+.4f}]")
    out["leave_one_template_out"] = {"full_rho": full, "folds": loo}

    print()
    print(f"=== template-stratified permutation test, {focus_name} ===")
    print("  Margins are shuffled WITHIN template, so the null preserves template composition")
    print("  and only breaks the item-level margin/outcome link. This replaces the normal")
    print("  approximation, which would assume 1554 independent observations.")
    rng = random.Random(0)
    null = []
    for _ in range(args.perm):
        perm: list[dict] = []
        for t, rs in by.items():
            ms = [r["margin"] for r in rs]
            rng.shuffle(ms)
            perm.extend({**r, "margin": m} for r, m in zip(rs, ms))
        rho, _ = spearman([r["margin"] for r in perm], [r["y"] for r in perm])
        if rho == rho:
            null.append(rho)
    # two-sided, +1 correction so p is never zero with a finite permutation count
    n_ge = sum(1 for v in null if abs(v) >= abs(full))
    p = (n_ge + 1) / (len(null) + 1)
    print(f"  observed rho = {full:+.4f}")
    print(f"  null over {len(null)} shuffles: mean {st.mean(null):+.4f}, "
          f"sd {st.pstdev(null):.4f}, |rho| >= observed in {n_ge}")
    print(f"  two-sided p = {p:.5f}")
    out["permutation_within_template"] = {"rho": full, "n_perm": len(null), "n_ge": n_ge, "p": p}

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
