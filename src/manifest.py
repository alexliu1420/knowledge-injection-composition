"""Run provenance capture.

Every run emits a manifest. A run whose provenance was not captured did not happen
(docs/MANUSCRIPT-REQUIREMENTS.md section 4).

Captures the fields the NeurIPS reproducibility checklist requires and that cannot
be reconstructed after the fact: exact command, git state, full environment,
precision, hardware, and seeds.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=15
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    # bool(None) is False, so a failed git call previously recorded "dirty": false --
    # asserting a clean tree when the truth was that provenance could not be determined.
    # None now means unknown, and is distinguishable from a genuine clean tree.
    return {
        "commit": commit,
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
        "available": commit is not None,
    }


def _hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_capability"] = "sm_%d%d" % torch.cuda.get_device_capability()
            info["gpu_total_gb"] = round(props.total_memory / 1e9, 2)
            info["cuda_version"] = torch.version.cuda
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
    except Exception as exc:  # pragma: no cover - torch always present in practice
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _model_revisions() -> dict[str, Any]:
    """Which base-model weights this run used.

    Scripts previously took a bare model name, so `main` at run time decided the
    weights and nothing recorded which snapshot that was. Both the pin and the cache
    state are stored, so a divergence is visible in the deposit rather than hidden.
    """
    try:
        from model_pin import REVISIONS, resolved
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    cache = resolved()
    return {"pinned": dict(REVISIONS), "cache_refs_main": cache,
            "match": all(cache.get(k) == v for k, v in REVISIONS.items() if k in cache)}


def _packages() -> dict[str, str]:
    """Full resolved environment. Not a curated subset -- versions drift silently."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        pkgs = {}
        for line in out.stdout.splitlines():
            if "==" in line:
                name, _, ver = line.partition("==")
                pkgs[name.strip().lower()] = ver.strip()
        return pkgs
    except Exception:
        return {}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    *,
    run_name: str,
    precision: str,
    seeds: list[int] | None = None,
    hyperparams: dict[str, Any] | None = None,
    search_budget: dict[str, Any] | None = None,
    data_files: dict[str, str | Path] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Assemble a run manifest.

    precision is mandatory and unabbreviated ('fp16', 'bf16', 'fp32'): HindsightBench
    (EVIDENCE-LOG H3) shows the same weights give different results across precisions,
    and our hardware forces a deviation from the source paper's fp32.

    search_budget records how hyperparameters were chosen, not just what they were --
    required to support the tuned-baseline claim in docs/BASELINES.md.
    """
    return {
        "run_name": run_name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "precision": precision,
        "seeds": seeds or [],
        "hyperparams": hyperparams or {},
        "search_budget": search_budget or {},
        "data_files": {
            k: {"path": str(v), "sha256": file_sha256(v) if Path(v).exists() else None}
            for k, v in (data_files or {}).items()
        },
        "model_revisions": _model_revisions(),
        "git": _git_state(),
        "hardware": _hardware(),
        "packages": _packages(),
        "notes": notes,
    }


def write_manifest(manifest: dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    m = build_manifest(run_name="env-check", precision="n/a", notes="environment probe")
    print(json.dumps({k: m[k] for k in ("hardware", "git")}, indent=2))
    print(f"\n{len(m['packages'])} packages resolved")
