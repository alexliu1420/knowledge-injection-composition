"""Within-template follow-up to test_entity_quality.py.

The pooled partial correlations said anchor margin keeps its predictive power once
entity familiarity is controlled for, and that familiarity itself goes NEGATIVE. Both
could be between-template artefacts, so redo them inside each template, where template
difficulty, bridge type and relation are all held fixed by construction.

Also tests the natural explanation of a negative familiarity effect: a well-known
bridge entity carries more prior associations, so an injected fact about it has more
competitors. Graph degree of E2 is a direct proxy for that.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
from analyze_anchor_dose import spearman  # noqa: E402
from test_entity_quality import residualise  # noqa: E402


def partial(y, x, ctrl):
    """rho(x, y | ctrl) via rank residuals."""
    return spearman(residualise(x, ctrl), residualise(y, ctrl))


def main() -> None:
    rows = json.loads(Path("results/entity_quality.json").read_text(encoding="utf-8"))

    # e2_id is not carried in the scored rows; join it back by task_id
    e2id = {}
    for dp in ("data/tasks/anchored7.json", "data/tasks/anchored7_control.json"):
        for d in json.loads(Path(dp).read_text(encoding="utf-8"))["items"]:
            e2id[d["task_id"]] = d["e2_id"]
    for r in rows:
        r["e2_id"] = e2id.get(r["task_id"])

    # attach E2 graph degree
    deg_path = Path("results/e2_degree.json")
    if deg_path.exists():
        deg = json.loads(deg_path.read_text(encoding="utf-8"))
        for r in rows:
            r["degree"] = deg.get(str(r["e2_id"]))

    by = defaultdict(list)
    for r in rows:
        by[r["template"]].append(r)

    print(f"{'template':<46}{'n':>4}{'mrg|fam':>10}{'fam|mrg':>10}{'fam raw':>10}")
    print("-" * 80)
    keep_m, keep_f = [], []
    for t, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(rs) < 15:
            continue
        y = [r["correct"] for r in rs]
        m = [r["margin"] for r in rs]
        f = [r["fam_type"] for r in rs]
        if len(set(y)) < 2:
            print(f"{t[:44]:<46}{len(rs):>4}   (composition constant -- no test)")
            continue
        rho_m, _ = partial(y, m, f)
        rho_f, _ = partial(y, f, m)
        rho_raw, _ = spearman(f, y)
        keep_m.append(rho_m)
        keep_f.append(rho_f)
        print(f"{t[:44]:<46}{len(rs):>4}{rho_m:>+10.3f}{rho_f:>+10.3f}{rho_raw:>+10.3f}")

    if keep_m:
        print()
        print(f"  templates tested                       : {len(keep_m)}")
        print(f"  mean within-template rho(margin | fam) : {st.mean(keep_m):+.4f}"
              f"   positive in {sum(1 for v in keep_m if v > 0)}/{len(keep_m)}")
        print(f"  mean within-template rho(fam | margin) : {st.mean(keep_f):+.4f}"
              f"   negative in {sum(1 for v in keep_f if v < 0)}/{len(keep_f)}")
        print()
        if st.mean(keep_m) > 0.10 and sum(1 for v in keep_m if v > 0) >= 0.7 * len(keep_m):
            print("  Margin's independent power holds WITHIN templates -- not a")
            print("  between-template artefact. Entity familiarity does not explain it.")
        else:
            print("  Margin's independent power does NOT hold within templates --")
            print("  the pooled partial was driven by between-template variation.")

    if any(r.get("degree") is not None for r in rows):
        d = [r for r in rows if r.get("degree") is not None]
        print()
        print("=== is a well-connected bridge entity harder to inject into? ===")
        rho_d, p_d = spearman([r["degree"] for r in d], [r["correct"] for r in d])
        rho_df, _ = spearman([r["degree"] for r in d], [r["fam_type"] for r in d])
        print(f"  n={len(d)}  degree vs composition : rho={rho_d:+.4f} p={p_d:.2e}")
        print(f"           degree vs familiarity : rho={rho_df:+.4f}")
        rho_dp, p_dp = partial([r["correct"] for r in d], [r["degree"] for r in d],
                               [r["margin"] for r in d])
        print(f"  degree -> composition, controlling margin : rho={rho_dp:+.4f} p={p_dp:.2e}")

        # degree varies enormously between entity types, so the pooled effect could be
        # nothing but a drug-vs-gene contrast. Template holds bridge type fixed.
        print()
        print(f"  {'template':<46}{'n':>4}{'deg raw':>10}{'deg|mrg':>10}")
        wd = []
        for t, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
            rs = [r for r in rs if r.get("degree") is not None]
            if len(rs) < 15 or len(set(r["correct"] for r in rs)) < 2:
                continue
            yv = [r["correct"] for r in rs]
            dv = [r["degree"] for r in rs]
            raw, _ = spearman(dv, yv)
            pr, _ = partial(yv, dv, [r["margin"] for r in rs])
            wd.append(pr)
            print(f"  {t[:44]:<46}{len(rs):>4}{raw:>+10.3f}{pr:>+10.3f}")
        if wd:
            print()
            print(f"  mean within-template rho(degree | margin): {st.mean(wd):+.4f}"
                  f"   negative in {sum(1 for v in wd if v < 0)}/{len(wd)}")
            if st.mean(wd) < -0.10 and sum(1 for v in wd if v < 0) >= 0.7 * len(wd):
                print("  Holds within templates -- not a bridge-type contrast.")
            else:
                print("  Does NOT hold within templates -- the pooled degree effect is")
                print("  most likely a between-template / bridge-type contrast.")
    else:
        print()
        print("  (no results/e2_degree.json -- run src/dump_e2_degree.py for the")
        print("   competing-associations test)")


if __name__ == "__main__":
    main()
