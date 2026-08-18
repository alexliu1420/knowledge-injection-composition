"""Fact-injection training (LoRA), with per-epoch memorization tracking.

Hyperparameters follow the source paper exactly where it specifies them
(results/spec-gaps.md R1-R3): LoRA r=16 alpha=32 dropout=0.05 on
{q,k,v,o,gate,up,down}_proj; AdamW lr 2e-4 wd 0.01; batch 1 x grad-accum 8; 50 epochs.

Per-epoch we evaluate memorization on a fixed subset and record the first epoch at
which it saturates. Their trainer saves a dedicated `mem100` checkpoint at that point
(spec-gaps R4a) and we do the same: it is the natural MATCHED-RECALL anchor, which is
what lead I4 needs to compare integration across isolation levels fairly.

Full evaluation (both memorization probes + chaining, both matchers) runs at mem100
and at the final epoch -- running it every epoch would cost more than the training.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_runner import (  # noqa: F401
    SYSTEM_PROMPT,
    continuous_scores,
    evaluate_dataset,
    generate,
    match_strict,
    match_theirs,
)
from isolation import (
    apply_mask,
    enforced_mask,
    enforced_support,
    mean_pairwise_overlap,
    topk_mask_gradients,
    trainable_named_params,
    update_support,
)
from retention import load_heldout, measure as measure_retention
from model_pin import revision_for


FIXED_SCALE = 1024.0  # manual loss scale for the enforced path (see training loop)


class FactDataset(Dataset):
    """One training example per injected fact: (question -> answer)."""

    def __init__(self, items: list[dict], tokenizer, max_len: int = 256,
                 inject: tuple[str, ...] = ("fact1", "fact2")):
        """`inject` names which facts are trained on.

        The anchored design (B'') injects ONLY fact2 -- fact1 is the pretrained anchor,
        verified per item to be recoverable by the base model. Training on it would
        destroy the property the whole design exists for, and would do so silently:
        the run would look normal and the anchor would stop being pretrained knowledge.
        """
        self.rows: list[tuple[str, str]] = []
        for it in items:
            for f in inject:
                self.rows.append((it[f"{f}_prompt"], it[f"{f}_answer"]))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        # fact_id travels with the example: the enforced-isolation mask is keyed on it,
        # so the mask must follow the fact through shuffling, not the batch position.
        q, a = self.rows[i]
        prompt = self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
        )
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = self.tok(a + self.tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p_ids + a_ids)[: self.max_len]
        # loss on the answer only -- we are injecting the fact, not the question
        labels = ([-100] * len(p_ids) + a_ids)[: self.max_len]
        return {"input_ids": ids, "labels": labels, "fact_id": i}


def collate(batch: list[dict], pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    return {
        "input_ids": torch.tensor([b["input_ids"] + [pad_id] * (n - len(b["input_ids"])) for b in batch]),
        "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
        "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]),
        "fact_ids": [b.get("fact_id", 0) for b in batch],
    }


def model_inputs(batch: dict, device: str = "cuda") -> dict:
    """Tensor fields only -- fact_ids is bookkeeping, not a model argument."""
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


@torch.no_grad()
def epoch_probe(model, tok, items: list[dict], batch_size: int,
                inject: tuple[str, ...] = ("fact1", "fact2")) -> dict:
    """Per-epoch memorization AND composition on a fixed subset.

    Composition is measured every epoch, not only at mem100/final. Two endpoints
    cannot distinguish a temporal LAG (composition emerges some epochs after
    memorization saturates -- EVIDENCE-LOG A3 reports 4.6-5.5) from a PLATEAU
    (it never emerges). Those are different claims and the second is stronger.

    It also matters for I4: if isolation changes *when* composition appears rather
    than *whether*, endpoint-only measurement misses the effect entirely. This run
    is the dense (fraction=1.0) baseline every sweep arm is compared against, so it
    must be instrumented identically to them.

    Cost is ~10% of epoch time. Greedy decoding draws no RNG, so adding this does
    not perturb the training trajectory.
    """
    # Memorization is scored over INJECTED facts only. In the anchored design fact1 is
    # pretrained knowledge that is never trained; counting it would cap memorization
    # near 50%, so mem100 would never fire and the matched-recall anchor would vanish.
    mem_q = [it[f"{f}_prompt"] for f in inject for it in items]
    mem_g = [it[f"{f}_answer"] for f in inject for it in items]
    chain_q = [it["chain_prompt"] for it in items]
    chain_g = [it["chain_answer"] for it in items]

    was_training = model.training
    model.eval()
    mem_p = generate(model, tok, mem_q, batch_size=batch_size, max_new_tokens=24)
    chain_p = generate(model, tok, chain_q, batch_size=batch_size, max_new_tokens=24)

    # Continuous scores every epoch. Accuracy on 60 items cannot separate 0.05 from
    # 0.033 -- that is 3 items against 2 -- so the composition TRAJECTORY (which peaks
    # before memorization saturates and then declines) is invisible on accuracy alone.
    # Log-probability has ~3.3 nats of range where accuracy has 0.026. One forward pass
    # per item, no generation: ~3% of epoch time.
    mem_c = continuous_scores(model, tok, mem_q, mem_g, batch_size=batch_size)
    chain_c = continuous_scores(model, tok, chain_q, chain_g, batch_size=batch_size)

    if was_training:
        model.train()

    def rate(fn, golds, preds):
        return sum(fn([g], p) for g, p in zip(golds, preds)) / max(len(golds), 1)

    def mean(xs):
        xs = [x for x in xs if x == x]
        return sum(xs) / max(len(xs), 1)

    return {
        "mem_strict": round(rate(match_strict, mem_g, mem_p), 4),
        "mem_theirs": round(rate(match_theirs, mem_g, mem_p), 4),
        "chain_strict": round(rate(match_strict, chain_g, chain_p), 4),
        "chain_theirs": round(rate(match_theirs, chain_g, chain_p), 4),
        "mem_logprob": round(mean(mem_c["logprob_mean"]), 4),
        "chain_logprob": round(mean(chain_c["logprob_mean"]), 4),
        "mem_mrr": round(mean(mem_c["mrr_first"]), 5),
        "chain_mrr": round(mean(chain_c["mrr_first"]), 5),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data/tasks/pilot.json")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--micro-bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--eval-subset", type=int, default=60, help="items used for per-epoch memorization tracking")
    ap.add_argument("--eval-bs", type=int, default=16)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--isolation", type=float, default=1.0,
        help="lead I4 knob: fraction of LoRA gradient entries kept per step. "
             "1.0 = dense (standard LoRA), small = isolated updates.",
    )
    ap.add_argument(
        "--iso-mode", default="natural", choices=["natural", "enforced"],
        help="natural = mask by gradient magnitude (what a selection method achieves; "
             "plateaus near 0.33 overlap). enforced = deterministic per-fact disjoint "
             "subsets (what a structural constraint imposes; reaches ~0 overlap).",
    )
    ap.add_argument(
        "--measure-overlap", type=int, default=0,
        help="if >0, measure mean pairwise update-support overlap over N facts before training",
    )
    ap.add_argument(
        "--checkpoint-every", type=int, default=0,
        help="save an adapter every N epochs (0 = only mem100 and final). "
             "Enables reconstructing a distractor-CONTROLLED trajectory post-hoc: "
             "raw per-epoch log-probability is confounded by time-varying format "
             "learning, and only saved checkpoints can be re-scored against distractors.",
    )
    ap.add_argument("--out", default="results/inject_pilot")
    args = ap.parse_args()

    from manifest import build_manifest

    torch.manual_seed(args.seed)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(args.data).read_text(encoding="utf-8"))
    items = payload["items"][: args.limit] if args.limit else payload["items"]
    eval_items = items[:: max(1, len(items) // args.eval_subset)][: args.eval_subset]

    tok = AutoTokenizer.from_pretrained(args.model, revision=revision_for(args.model))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision_for(args.model), dtype=dtype).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    peft_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # Respect the dataset's own declaration of which facts are injected. The anchored
    # design injects fact2 only; training the anchor would silently void the design.
    inject = tuple(payload.get("injected_facts", ["fact1", "fact2"]))
    print(f"  injecting: {inject}" + ("" if len(inject) == 2 else "   (anchor NOT trained)"))
    ds = FactDataset(items, tok, inject=inject)
    dl = DataLoader(ds, batch_size=args.micro_bs, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16"))

    print(f"{len(items)} items -> {len(ds)} facts | eval subset {len(eval_items)} | "
          f"{args.epochs} epochs | isolation={args.isolation}")

    # Measured isolation: what fraction of each fact's update support is shared with
    # another fact's? This is the x-axis of the I4 curve -- a measured quantity, not
    # the hyperparameter. Taken at init so it characterises the configuration rather
    # than a particular point in training.
    enforced = args.iso_mode == "enforced"
    # fp32 accumulation buffer: fp16 grads underflow badly once masked to a few percent
    accum = (
        {n: torch.zeros_like(p, dtype=torch.float32) for n, p in trainable_named_params(model)}
        if enforced
        else {}
    )
    nonfinite = 0

    overlap_stats = None
    if args.measure_overlap > 0:
        sups = []
        for i in range(min(args.measure_overlap, len(ds))):
            if enforced:
                # mask is gradient-independent, so no backward pass is needed
                sups.append(enforced_support(model, i, args.isolation, salt=args.seed))
            else:
                model.zero_grad(set_to_none=True)
                b = collate([ds[i]], tok.pad_token_id)
                model(**model_inputs(b)).loss.backward()
                sups.append(update_support(model, args.isolation))
        model.zero_grad(set_to_none=True)
        overlap_stats = mean_pairwise_overlap(sups)
        print(f"  measured update-support overlap ({args.iso_mode}, "
              f"fraction={args.isolation}) over {len(sups)} facts: {overlap_stats}")
        (out / "overlap.json").write_text(json.dumps(overlap_stats, indent=2), encoding="utf-8")

    # Retention baseline BEFORE any injection. Damage is only interpretable as a
    # delta -- absolute perplexity varies with model and corpus and says nothing on
    # its own. This is the other half of the I4 curve.
    heldout = load_heldout()
    ret_base = measure_retention(model, tok, texts=heldout)
    print(f"  retention baseline (pre-injection): {ret_base}")

    history: list[dict] = []
    mem100_epoch = None
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(dl):
            fact_ids = batch["fact_ids"]
            inputs = model_inputs(batch)

            # Enforced mode masks PER FACT, so each micro-batch's gradient must be
            # masked before it accumulates. Natural mode selects by magnitude on the
            # accumulated gradient, so it masks once per optimizer step.
            if enforced:
                opt.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.float16, enabled=(args.precision == "fp16")):
                loss = model(**inputs).loss / args.grad_accum
            if enforced:
                (loss * FIXED_SCALE).backward()
            else:
                scaler.scale(loss).backward()

            if enforced:
                # GradScaler.unscale_ may be called only once per optimizer step, but
                # enforced mode masks EVERY micro-batch. So this path uses a fixed
                # manual loss scale and does its own unscaling. The natural path keeps
                # the scaler untouched, so it stays bit-comparable to the dense baseline
                # already running.
                apply_mask(model, enforced_mask(model, fact_ids[0], args.isolation, salt=args.seed))
                for n_, p_ in trainable_named_params(model):
                    if p_.grad is not None:
                        g = p_.grad.detach().float() / FIXED_SCALE
                        if torch.isfinite(g).all():
                            accum[n_] += g
                        else:
                            nonfinite += 1

            if (step + 1) % args.grad_accum == 0:
                if enforced:
                    for n_, p_ in trainable_named_params(model):
                        p_.grad = accum[n_].to(p_.dtype)
                        accum[n_].zero_()
                    opt.step()
                else:
                    if args.isolation < 1.0:
                        # unscale first so the mask is applied to true gradients and the
                        # scaler's inf/nan check still sees a consistent state
                        scaler.unscale_(opt)
                        topk_mask_gradients(model, args.isolation)
                    scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
            tot += loss.item() * args.grad_accum; nb += 1

        probe = epoch_probe(model, tok, eval_items, args.eval_bs, inject=inject)
        strict = probe["mem_strict"]
        rec = {"epoch": epoch, "loss": round(tot / max(nb, 1), 4), **probe,
               "elapsed_s": round(time.time() - t_start, 1)}
        history.append(rec)
        print(f"  ep{epoch:>3} loss={rec['loss']:.4f} mem={strict:.3f} "
              f"chain={probe['chain_strict']:.3f} "
              f"| logprob mem={probe['mem_logprob']:+.3f} chain={probe['chain_logprob']:+.3f} "
              f"| chain_mrr={probe['chain_mrr']:.4f} ({rec['elapsed_s']:.0f}s)")
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            model.save_pretrained(out / f"checkpoint-epoch{epoch:03d}")

        if mem100_epoch is None and strict >= 0.99:
            mem100_epoch = epoch
            model.save_pretrained(out / "checkpoint-mem100")
            print(f"  >>> memorization saturated at epoch {epoch}; saved checkpoint-mem100")
            full = evaluate_dataset(model, tok, items, batch_size=args.eval_bs)
            full["retention"] = measure_retention(model, tok, texts=heldout)
            full["retention_baseline"] = ret_base
            full["retention_delta_ppl"] = round(
                full["retention"]["heldout_ppl"] - ret_base["heldout_ppl"], 4
            )
            (out / "eval_mem100.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
            print("  mem100 full eval:", full["summary"])
            print(f"  mem100 retention: ppl {ret_base['heldout_ppl']:.3f} -> "
                  f"{full['retention']['heldout_ppl']:.3f} "
                  f"(delta {full['retention_delta_ppl']:+.3f})")

    model.save_pretrained(out / "checkpoint-final")
    full = evaluate_dataset(model, tok, items, batch_size=args.eval_bs)
    full["manifest"] = build_manifest(
        run_name=out.name, precision=args.precision, seeds=[args.seed],
        hyperparams={k: v for k, v in vars(args).items()},
        data_files={"tasks": args.data},
        notes=f"mem100_epoch={mem100_epoch}",
    )
    full["retention"] = measure_retention(model, tok, texts=heldout)
    full["retention_baseline"] = ret_base
    full["retention_delta_ppl"] = round(
        full["retention"]["heldout_ppl"] - ret_base["heldout_ppl"], 4
    )
    full["isolation"] = {
        "fraction": args.isolation,
        "mode": args.iso_mode,
        "measured_overlap": overlap_stats,
    }
    full["mem100_epoch"] = mem100_epoch
    (out / "eval_final.json").write_text(json.dumps(full, indent=2), encoding="utf-8")

    print("\n=== final ===")
    for k, v in full["summary"].items():
        print(f"  {k:<16} {v}")
    print(f"  mem100_epoch     {mem100_epoch}")


if __name__ == "__main__":
    main()
