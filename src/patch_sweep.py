"""Self-patching layer sweep (BUILD-SPEC B2), on nnsight.

Relocates a representation that already exists inside the model from one layer to
another, at entity-anchor positions, and measures whether the answer improves. If a
fact is stored but sits off the computation path, moving it should recover the answer.

Deliberately an INDEPENDENT reimplementation of the source paper's transformer_lens
version (spec-gaps E5). Independence is the point: C2b asks whether our probe agrees
with theirs, which is only meaningful if the two do not share an implementation. It
is also 3.3x cheaper -- HookedTransformer holds both an HF model and its own copy and
reached 10.66 GB at 1.5B on an 8.59 GB card, while nnsight does the same work in 3.19.

Anchor positions come from character offsets rather than token-string matching. Their
EntityPositionParser matches normalised token strings and falls back to fuzzy matching;
offset mapping is exact and cannot silently mis-locate an entity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from model_pin import revision_for


def find_entity_positions(tokenizer, text: str, entity: str) -> list[int]:
    """Token indices covering `entity` in `text`, via character offsets (exact)."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    lo = text.lower().find(entity.lower())
    if lo < 0:
        return []
    hi = lo + len(entity)
    return [
        i
        for i, (s, e) in enumerate(enc["offset_mapping"])
        if e > s and not (e <= lo or s >= hi)
    ]


@dataclass
class SweepResult:
    task_id: str
    baseline_mrr: float
    baseline_logprob: float
    n_layers: int
    anchor_positions: list[int]
    grid: dict[str, float]  # "src>tgt" -> mrr delta
    best_pair: tuple[int, int] | None
    best_delta: float


def _metrics(logits: torch.Tensor, answer_id: int, pos: int) -> tuple[float, float]:
    """(MRR, logprob) of `answer_id` at `pos`.

    MRR = 1/rank matches their get_prediction_metrics(metric_type='mrr'); logprob is
    added because MRR is discrete and saturates, which hides small effects.
    """
    row = logits[0, pos, :].float()
    rank = int((row.argsort(descending=True) == answer_id).nonzero()[0, 0]) + 1
    logprob = float(torch.log_softmax(row, dim=-1)[answer_id])
    return 1.0 / rank, logprob


class PatchSweeper:
    def __init__(self, model_name: str, precision: str = "fp16", device: str = "cuda",
                 adapter: str | None = None):
        """`adapter` runs the sweep on an INJECTED checkpoint.

        C2 requires the identical sweep on both an uninjected and an injected model --
        the selection bias from taking a max over ~784 pairs is present on both sides
        and only cancels if the procedure is the same. Without this argument only the
        base half was runnable, which is why the null was measured and the comparison
        never was.

        The adapter is merged into the weights before nnsight wraps the model, so the
        traced module tree is an ordinary dense model and layer indices mean the same
        thing in both conditions.
        """
        from nnsight import LanguageModel

        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
        if adapter:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_name, revision=revision_for(model_name))
            hf = AutoModelForCausalLM.from_pretrained(model_name, revision=revision_for(model_name), dtype=dtype)
            hf = PeftModel.from_pretrained(hf, adapter).merge_and_unload().eval()
            self.lm = LanguageModel(hf, tokenizer=tok, device_map=device, dispatch=True)
        else:
            self.lm = LanguageModel(model_name, device_map=device, dtype=dtype, dispatch=True)
        self.tok = self.lm.tokenizer
        self.n_layers = len(self.lm.model.layers)
        self.precision = precision
        self.model_name = model_name

    def _prompt(self, question: str, system: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True,
        )

    def sweep(
        self,
        question: str,
        answer: str,
        anchor_entity: str,
        *,
        system: str,
        layers: list[int] | None = None,
    ) -> SweepResult:
        text = self._prompt(question, system)
        positions = find_entity_positions(self.tok, text, anchor_entity)
        if not positions:
            return SweepResult("", 0.0, 0.0, self.n_layers, [], {}, None, 0.0)

        ans_id = self.tok(answer, add_special_tokens=False)["input_ids"][0]
        n_tok = len(self.tok(text, add_special_tokens=False)["input_ids"])
        last = n_tok - 1
        layers = layers or list(range(self.n_layers))

        # one pass: cache every layer's residual output, plus the clean baseline
        acts: dict[int, torch.Tensor] = {}
        base: dict[str, torch.Tensor] = {}
        with self.lm.trace(text):
            for i in range(self.n_layers):
                acts[i] = self.lm.model.layers[i].output.save()
            base["logits"] = self.lm.lm_head.output.save()

        cached = {i: acts[i].detach().clone() for i in layers}
        b_mrr, b_lp = _metrics(base["logits"], ans_id, last)

        grid: dict[str, float] = {}
        best_pair, best_delta = None, float("-inf")
        for src in layers:
            src_act = cached[src]
            for tgt in layers:
                if src == tgt:
                    continue
                out: dict[str, torch.Tensor] = {}
                with self.lm.trace(text):
                    for p in positions:
                        self.lm.model.layers[tgt].output[:, p, :] = src_act[:, p, :]
                    out["logits"] = self.lm.lm_head.output.save()
                mrr, _ = _metrics(out["logits"], ans_id, last)
                delta = mrr - b_mrr
                grid[f"{src}>{tgt}"] = round(delta, 6)
                if delta > best_delta:
                    best_delta, best_pair = delta, (src, tgt)

        return SweepResult(
            task_id="", baseline_mrr=round(b_mrr, 6), baseline_logprob=round(b_lp, 4),
            n_layers=self.n_layers, anchor_positions=positions, grid=grid,
            best_pair=best_pair, best_delta=round(best_delta, 6),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="data/tasks/pilot.json")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--stride", type=int, default=4, help="coarse grid: every Nth layer")
    ap.add_argument("--adapter", default=None,
                    help="LoRA checkpoint -> sweep the INJECTED model (C2's other half)")
    ap.add_argument("--out", default="results/patch_sweep_smoke.json")
    args = ap.parse_args()

    payload = json.loads(Path(args.data).read_text(encoding="utf-8"))
    items = payload["items"][: args.limit]
    system = payload["system_prompt"]

    sw = PatchSweeper(args.model, args.precision, adapter=args.adapter)
    layers = list(range(0, sw.n_layers, args.stride))
    print(f"{sw.model_name}{' +adapter' if args.adapter else ' (BASE)'} | "
          f"{sw.n_layers} layers | coarse grid {layers} "
          f"({len(layers)*(len(layers)-1)} pairs/item)")

    results = []
    for it in items:
        r = sw.sweep(it["chain_prompt"], it["chain_answer"], it["e1_name"],
                     system=system, layers=layers)
        r.task_id = it["task_id"]
        results.append(r.__dict__)
        print(f"  {it['task_id'][:44]:<46} anchors={len(r.anchor_positions)} "
              f"base_mrr={r.baseline_mrr:.4f} best={r.best_pair} delta={r.best_delta:+.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
