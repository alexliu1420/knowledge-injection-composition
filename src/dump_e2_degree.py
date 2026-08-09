"""Graph degree of every bridge entity, as a proxy for competing prior associations.

Entity familiarity predicts composition NEGATIVELY once anchor margin is controlled for
(within-template mean rho -0.178, negative in 3/4 templates). The obvious reading is
proactive interference: a well-known bridge entity already participates in many
relations, so an injected fact about it competes with everything else the model knows.

Degree is the structural version of that claim and is independent of the model. If the
negative familiarity effect is really about competing associations, degree should show
it too; if degree shows nothing, the familiarity effect is about something else.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from prime_graph import PrimeGraph


def main() -> None:
    g = PrimeGraph()

    wanted = set()
    for dp in ("data/tasks/anchored7.json", "data/tasks/anchored7_control.json"):
        for d in json.loads(Path(dp).read_text(encoding="utf-8"))["items"]:
            wanted.add(int(d["e2_id"]))
    print(f"{len(wanted)} distinct bridge entities")

    # undirected degree over the full edge list
    deg: Counter[int] = Counter()
    src = g.edge_index[0].tolist()
    dst = g.edge_index[1].tolist()
    for a, b in zip(src, dst):
        if a in wanted:
            deg[a] += 1
        if b in wanted:
            deg[b] += 1

    out = {str(k): deg.get(k, 0) for k in wanted}
    Path("results/e2_degree.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    vals = sorted(out.values())
    print(f"degree  min={vals[0]}  median={vals[len(vals)//2]}  max={vals[-1]}")
    print("wrote results/e2_degree.json")


if __name__ == "__main__":
    main()
