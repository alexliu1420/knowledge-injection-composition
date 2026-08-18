"""Pinned Hugging Face revisions for the base models.

A bare model name resolves to whatever `main` points at when the script runs, so results
from different dates are not guaranteed to come from the same weights and a reader
re-running the deposit could silently get different ones.

The hashes below are the snapshots present in the cache these experiments were run from.
They bind runs made from this release onward; manifests written before the pin existed
record the model name only, and say so. `revision_for` returns None for anything that is
not a pinned hub id -- local adapter directories and unknown models pass through
unchanged.
"""

from __future__ import annotations

from pathlib import Path

REVISIONS: dict[str, str] = {
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
}


def revision_for(model: str | Path | None) -> str | None:
    """The pinned revision for a hub id, or None for local paths and unpinned models."""
    if model is None:
        return None
    m = str(model)
    if Path(m).exists():
        return None
    return REVISIONS.get(m)


def resolved() -> dict[str, str]:
    """What the local cache currently has, for the manifest.

    Reports the cache state rather than the pin, so a mismatch between the two is
    visible in the deposit instead of being asserted away.
    """
    out: dict[str, str] = {}
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for name in REVISIONS:
        ref = hub / f"models--{name.replace('/', '--')}" / "refs" / "main"
        if ref.exists():
            out[name] = ref.read_text(encoding="utf-8").strip()
    return out
