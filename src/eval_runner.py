"""Evaluation runner (BUILD-SPEC B1).

Scores memorization (single-hop recall of each injected fact) and generalization
(two-hop composition) with TWO matchers, always reported together:

  match_theirs -- verbatim reimplementation of eval_callback.py's substring_match,
                  including the `pred in gold` clause. Kept for comparability.
  match_strict -- normalised containment of gold in pred, with a length floor.

Reporting both is not padding. Their matcher accepts a prediction that is merely a
*substring of* the gold answer, so "a" scores correct against "Fazadinium bromide"
(spec-gaps E2a). Since our flagship claim (I4) is a difference in composition scores,
a matcher that can manufacture that difference has to be measured, not inherited.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a biomedical assistant. "
    "Answer the question with the most appropriate entity name."
)


def match_theirs(gold_list: list[str], pred: str) -> bool:
    """external/mem2gen/eval_callback.py::substring_match, verbatim behaviour."""
    for gold in gold_list:
        g = gold.strip().lower()
        p = pred.strip().lower()
        if g in p or p in g:  # NOTE: the second clause is the unsound one
            return True
    return False


_norm_re = re.compile(r"[^a-z0-9]+")


def _norm(s: str) -> str:
    return _norm_re.sub(" ", s.strip().lower()).strip()


def match_strict(gold_list: list[str], pred: str) -> bool:
    """Gold must appear in the prediction. Directional, normalised, length-floored."""
    p = _norm(pred)
    if len(p) < 3:
        return False
    return any(_norm(g) and _norm(g) in p for g in gold_list)


@torch.no_grad()
def continuous_scores(
    model, tokenizer, questions: list[str], answers: list[str], *, batch_size: int = 8
) -> dict[str, list[float]]:
    """Teacher-forced continuous scoring of the gold answer.

    Thresholded accuracy floors out: the dense baseline scores 7/270 on composition,
    so a predicted DECREASE under isolation (I4) has almost no room to register and
    7-vs-3 is not statistically separable. These measures do not floor.

      logprob_mean -- mean log P(answer token | prefix), length-normalised
      mrr_first    -- 1/rank of the first answer token (comparable to their MRR)
      rank_first   -- raw rank, for reporting

    One forward pass per item, no generation, so this is cheaper than the accuracy
    probe it supplements -- not replaces: accuracy stays as the interpretable headline.
    """
    was_training = model.training
    model.eval()
    out: dict[str, list[float]] = {"logprob_mean": [], "mrr_first": [], "rank_first": []}

    for i in range(0, len(questions), batch_size):
        for q, a in zip(questions[i : i + batch_size], answers[i : i + batch_size]):
            prompt = build_prompt(tokenizer, q)
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tokenizer(a, add_special_tokens=False)["input_ids"]
            if not a_ids:
                out["logprob_mean"].append(float("nan"))
                out["mrr_first"].append(0.0)
                out["rank_first"].append(float("inf"))
                continue

            ids = torch.tensor([p_ids + a_ids], device=model.device)
            logits = model(input_ids=ids).logits[0].float()
            logprobs = torch.log_softmax(logits, dim=-1)

            # position predicting answer token j is (len(p_ids) + j - 1)
            lps = [
                float(logprobs[len(p_ids) + j - 1, tid]) for j, tid in enumerate(a_ids)
            ]
            first_pos = len(p_ids) - 1
            row = logits[first_pos]
            rank = int((row.argsort(descending=True) == a_ids[0]).nonzero()[0, 0]) + 1

            out["logprob_mean"].append(sum(lps) / len(lps))
            out["mrr_first"].append(1.0 / rank)
            out["rank_first"].append(float(rank))

    if was_training:
        model.train()
    return out


@torch.no_grad()
def distractor_control(
    model,
    tokenizer,
    items: list[dict],
    prompt_key: str,
    answer_key: str,
    *,
    n_distractors: int = 4,
    seed: int = 0,
    batch_size: int = 8,
) -> dict:
    """Format-invariant scoring: correct answer vs same-type distractors.

    Raw log-probability of the gold answer rises for two reasons that cannot be
    separated from it: the model learning to EMIT short biomedical entity names in
    the answer slot (which lifts every plausible entity), and the model actually
    composing the injected facts. After one epoch of training, chain logprob had
    already moved from -5.30 to -2.81 while memorization was still 0.000 -- almost
    certainly format, not composition.

    A distractor of the SAME entity type under the SAME prompt absorbs the format
    gain, so the contrast isolates what we want:

        delta      = logprob(correct) - mean logprob(distractors)
        discrim    = P(logprob(correct) > logprob(distractor))    chance = 0.5

    Distractors are drawn from other items sharing the template key, so entity type
    and question form match. This is the foil control from the original E1 design,
    applied where it is actually needed.
    """
    rng = random.Random(seed)
    by_template: dict[str, list[str]] = {}
    for it in items:
        by_template.setdefault("|".join(it["template_key"]), []).append(it[answer_key])

    prompts_c, golds_c, prompts_d, golds_d, owner = [], [], [], [], []
    for idx, it in enumerate(items):
        key = "|".join(it["template_key"])
        pool = [a for a in by_template[key] if a != it[answer_key]]
        if not pool:
            continue
        prompts_c.append(it[prompt_key])
        golds_c.append(it[answer_key])
        for d in rng.sample(pool, min(n_distractors, len(pool))):
            prompts_d.append(it[prompt_key])
            golds_d.append(d)
            owner.append(len(prompts_c) - 1)

    sc = continuous_scores(model, tokenizer, prompts_c, golds_c, batch_size=batch_size)
    sd = continuous_scores(model, tokenizer, prompts_d, golds_d, batch_size=batch_size)

    per_item_delta, wins, total = [], 0, 0
    grouped: dict[int, list[float]] = {}
    for j, o in enumerate(owner):
        grouped.setdefault(o, []).append(sd["logprob_mean"][j])
    for i, lp_c in enumerate(sc["logprob_mean"]):
        ds = grouped.get(i, [])
        if not ds:
            continue
        per_item_delta.append(lp_c - sum(ds) / len(ds))
        wins += sum(1 for d in ds if lp_c > d)
        total += len(ds)

    n = max(len(per_item_delta), 1)
    return {
        "n_items": len(per_item_delta),
        "n_distractors": n_distractors,
        "logprob_correct": round(sum(sc["logprob_mean"]) / max(len(sc["logprob_mean"]), 1), 4),
        "logprob_distractor": round(sum(sd["logprob_mean"]) / max(len(sd["logprob_mean"]), 1), 4),
        "delta_logprob": round(sum(per_item_delta) / n, 4),
        "discrimination": round(wins / max(total, 1), 4),  # chance = 0.5
    }


def build_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def generate(model, tokenizer, questions: list[str], *, batch_size: int, max_new_tokens: int) -> list[str]:
    outs: list[str] = []
    for i in range(0, len(questions), batch_size):
        chunk = questions[i : i + batch_size]
        texts = [build_prompt(tokenizer, q) for q in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # greedy, matching their setup
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        for j in range(len(chunk)):
            new = gen[j, enc["input_ids"].shape[1] :]
            outs.append(tokenizer.decode(new, skip_special_tokens=True).strip())
    return outs


def evaluate_dataset(model, tokenizer, items: list[dict], *, batch_size=8, max_new_tokens=32) -> dict:
    """Three probes per item: fact1, fact2 (memorization) and chain (generalization)."""
    probes = {
        "fact1": ("fact1_prompt", "fact1_answer"),
        "fact2": ("fact2_prompt", "fact2_answer"),
        "chain": ("chain_prompt", "chain_answer"),
    }
    results: dict = {"per_item": [], "summary": {}}
    preds: dict[str, list[str]] = {}

    # Must be explicit: when called mid-training the model is in train() mode and
    # LoRA dropout stays active during generation, silently degrading every score.
    was_training = model.training
    model.eval()

    for probe, (pk, _) in probes.items():
        t0 = time.time()
        preds[probe] = generate(
            model, tokenizer, [it[pk] for it in items],
            batch_size=batch_size, max_new_tokens=max_new_tokens,
        )
        print(f"  {probe}: {len(items)} prompts in {time.time()-t0:.1f}s")

    for idx, it in enumerate(items):
        row = {"task_id": it["task_id"], "template_key": it["template_key"]}
        for probe, (_, ak) in probes.items():
            gold = [it[ak]]
            pred = preds[probe][idx]
            row[f"{probe}_pred"] = pred
            row[f"{probe}_gold"] = it[ak]
            row[f"{probe}_theirs"] = match_theirs(gold, pred)
            row[f"{probe}_strict"] = match_strict(gold, pred)
        results["per_item"].append(row)

    # continuous scores -- see continuous_scores() for why accuracy alone is insufficient
    cont: dict[str, dict[str, list[float]]] = {}
    for probe, (pk, ak) in probes.items():
        cont[probe] = continuous_scores(
            model, tokenizer, [it[pk] for it in items], [it[ak] for it in items],
            batch_size=batch_size,
        )
    for idx, row in enumerate(results["per_item"]):
        for probe in probes:
            row[f"{probe}_logprob"] = round(cont[probe]["logprob_mean"][idx], 4)
            row[f"{probe}_mrr"] = round(cont[probe]["mrr_first"][idx], 6)

    if was_training:
        model.train()

    n = max(len(items), 1)
    for probe in probes:
        for m in ("theirs", "strict"):
            results["summary"][f"{probe}_{m}"] = round(
                sum(r[f"{probe}_{m}"] for r in results["per_item"]) / n, 4
            )
        lp = [v for v in cont[probe]["logprob_mean"] if v == v]
        results["summary"][f"{probe}_logprob"] = round(sum(lp) / max(len(lp), 1), 4)
        results["summary"][f"{probe}_mrr"] = round(
            sum(cont[probe]["mrr_first"]) / n, 6
        )
    results["summary"]["n_items"] = len(items)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter path")
    ap.add_argument("--data", default="data/tasks/pilot.json")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/eval_base.json")
    args = ap.parse_args()

    from manifest import build_manifest, write_manifest

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    payload = json.loads(Path(args.data).read_text(encoding="utf-8"))
    items = payload["items"][: args.limit] if args.limit else payload["items"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to("cuda").eval()
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()

    print(f"evaluating {len(items)} items | {args.model} | {args.precision} | adapter={args.adapter}")
    res = evaluate_dataset(
        model, tokenizer, items,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res["manifest"] = build_manifest(
        run_name=out.stem,
        precision=args.precision,
        hyperparams={
            "model": args.model, "adapter": args.adapter,
            "max_new_tokens": args.max_new_tokens, "decoding": "greedy",
        },
        data_files={"tasks": args.data},
    )
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\n=== summary ===")
    for k, v in res["summary"].items():
        print(f"  {k:<16} {v}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
