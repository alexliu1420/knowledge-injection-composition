"""I4 sweep driver: does isolation buy retention by spending integration?

Sweeps the isolation fraction and, at MATCHED RECALL (the mem100 checkpoint),
records both axes:

    retention   -- held-out perplexity delta vs pre-injection
    integration -- two-hop composition accuracy

plus the MEASURED update-support overlap, which is the real x-axis. The knob is a
hyperparameter; overlap is the quantity the claim is about, and the two are not the
same -- on Qwen-2.5-0.5B, keeping the top 10% of gradients still leaves 42% overlap.
Sweeping on the knob alone risks a false null from a range where overlap barely moves.

Each configuration runs as a separate process. That isolates GPU memory, survives a
single failure, and means the same config list can be dispatched in parallel on
rented hardware without modification -- `--emit-jobs` writes exactly that list.

Outcomes and how each reads (pre-declared, docs/LEADS.md):
  monotone opposing trends  -> the tradeoff is real, measured within one method
  dense dominates both      -> the field is on a dominated point, not a frontier
  integration flat (<3pp)   -> our clearest prediction fails on its easiest case
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Reaches genuinely low overlap. 1.0 -> 0.1 alone would sweep a range where measured
# overlap barely moves (0.42 at fraction 0.1), which is how a false null happens.
DEFAULT_FRACTIONS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]


def config_name(fraction: float, seed: int, model: str) -> str:
    tag = model.split("/")[-1].replace("Qwen2.5-", "q").replace("-Instruct", "")
    return f"{tag}_f{fraction:g}_s{seed}"


def build_jobs(args) -> list[dict]:
    jobs = []
    for frac in args.fractions:
        for seed in range(args.seeds):
            name = config_name(frac, seed, args.model)
            out = str(Path(args.out) / name)
            cmd = [
                sys.executable, "-u", "src/train_inject.py",
                "--model", args.model,
                "--data", args.data,
                "--epochs", str(args.epochs),
                "--isolation", str(frac),
                "--measure-overlap", str(args.measure_overlap),
                "--seed", str(seed),
                "--eval-subset", str(args.eval_subset),
                "--precision", args.precision,
                "--out", out,
            ]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            jobs.append({"name": name, "fraction": frac, "seed": seed, "out": out, "cmd": cmd})
    return jobs


def collect(jobs: list[dict]) -> list[dict]:
    rows = []
    for j in jobs:
        p = Path(j["out"]) / "eval_mem100.json"
        fallback = Path(j["out"]) / "eval_final.json"
        src = p if p.exists() else (fallback if fallback.exists() else None)
        if src is None:
            rows.append({**{k: j[k] for k in ("name", "fraction", "seed")}, "status": "missing"})
            continue
        d = json.loads(src.read_text(encoding="utf-8"))
        s = d.get("summary", {})
        ov = (d.get("isolation") or {}).get("measured_overlap") or {}
        rows.append({
            "name": j["name"], "fraction": j["fraction"], "seed": j["seed"],
            "at": "mem100" if src == p else "final",
            "measured_overlap": ov.get("mean"),
            "memorization": round((s.get("fact1_strict", 0) + s.get("fact2_strict", 0)) / 2, 4),
            "integration": s.get("chain_strict"),
            "integration_theirs": s.get("chain_theirs"),
            "retention_delta_ppl": d.get("retention_delta_ppl"),
            "mem100_epoch": d.get("mem100_epoch"),
            "status": "ok",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data/tasks/pilot.json")
    ap.add_argument("--fractions", type=float, nargs="+", default=DEFAULT_FRACTIONS)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eval-subset", type=int, default=60)
    ap.add_argument("--measure-overlap", type=int, default=24)
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/sweep_i4")
    ap.add_argument("--emit-jobs", action="store_true",
                    help="write the job list and exit, for parallel dispatch on rented hardware")
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)

    (out / "jobs.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    print(f"{len(jobs)} configs: fractions={args.fractions} seeds={args.seeds} model={args.model}")

    if args.emit_jobs:
        print(f"job list -> {out/'jobs.json'} (dispatch these in parallel; nothing run locally)")
        return

    if not args.collect_only:
        for i, j in enumerate(jobs, 1):
            if (Path(j["out"]) / "eval_final.json").exists():
                print(f"[{i}/{len(jobs)}] {j['name']}: already complete, skipping")
                continue
            print(f"[{i}/{len(jobs)}] {j['name']}: running")
            t0 = time.time()
            Path(j["out"]).mkdir(parents=True, exist_ok=True)
            # Stream to a file rather than capture_output. capture_output can leave
            # stdout/stderr as None, and concatenating them then raises AFTER the job
            # has already succeeded -- which killed the remaining configs once, losing
            # ~5h of idle GPU. Log-writing must never be able to abort the sweep.
            try:
                with open(Path(j["out"]) / "stdout.log", "w", encoding="utf-8") as fh:
                    r = subprocess.run(j["cmd"], stdout=fh, stderr=subprocess.STDOUT)
                status = "ok" if r.returncode == 0 else f"FAILED rc={r.returncode}"
            except Exception as exc:  # never let one config take down the sweep
                status = f"ERROR {type(exc).__name__}: {exc}"
            print(f"    {status} in {(time.time()-t0)/60:.1f} min", flush=True)

    rows = collect(jobs)
    (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\n{'frac':>7} {'overlap':>9} {'memoriz':>9} {'integr':>8} {'ret_dppl':>9} {'ep':>4}")
    print("-" * 52)
    for r in sorted([r for r in rows if r["status"] == "ok"],
                    key=lambda x: -(x["fraction"] or 0)):
        ov = f"{r['measured_overlap']:.4f}" if r["measured_overlap"] is not None else "  --  "
        rd = f"{r['retention_delta_ppl']:+.3f}" if r["retention_delta_ppl"] is not None else "  --  "
        print(f"{r['fraction']:>7g} {ov:>9} {r['memorization']:>9.4f} "
              f"{r['integration']:>8.4f} {rd:>9} {str(r['mem100_epoch']):>4}")
    print(f"\nwrote {out/'summary.json'}")


if __name__ == "__main__":
    main()
