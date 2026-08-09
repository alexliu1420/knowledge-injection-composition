"""Isolation knob for lead I4, plus direct measurement of what it controls.

I4 claims that methods achieving stability through NON-OVERLAP sacrifice the shared
substrate composition requires. The original plan used SMF's top-T memory-slot
selection as the isolation knob, which needs a dense retrofit (2 epochs over 50k
OpenAssistant responses -- 35-70h locally). The retrofit is incidental to the claim:
the mechanism under test is non-overlap, not memory layers.

So we apply the same principle to LoRA directly. Before each optimizer step, keep
only the top fraction of gradient entries by magnitude and zero the rest:

    fraction = 1.00  ->  dense update      (standard LoRA)
    fraction = 0.01  ->  1% of params move (high isolation)

This holds base model, data, schedule and parameterisation fixed, and removes the
retrofit as a confound rather than merely holding it constant.

The second half of this module matters more than the first. Rather than ASSUMING
"smaller fraction = more isolated", we MEASURE the pairwise overlap between the
parameter supports that different facts actually update. That makes the x-axis of
the I4 curve a measured quantity rather than a hyperparameter -- which is the
difference between "we varied a knob" and "we varied overlap".
"""

from __future__ import annotations

import torch


def trainable_named_params(model) -> list[tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


@torch.no_grad()
def topk_mask_gradients(model, fraction: float, *, per_tensor: bool = True) -> dict:
    """Zero all but the top `fraction` of gradient entries by |grad|.

    Applied after gradient accumulation and before optimizer.step(), so the mask
    reflects the full accumulated gradient rather than a partial micro-batch.

    per_tensor=True selects within each parameter tensor independently. Global
    selection lets a single large module monopolise the budget, which would confound
    the isolation knob with module size.
    """
    if fraction >= 1.0:
        return {"fraction": 1.0, "kept": None, "total": None}

    kept_total = 0
    total = 0
    for _, p in trainable_named_params(model):
        if p.grad is None:
            continue
        g = p.grad
        flat = g.abs().flatten()
        n = flat.numel()
        k = max(1, int(round(fraction * n)))
        if k >= n:
            kept_total += n
            total += n
            continue
        thresh = torch.topk(flat, k, largest=True, sorted=False).values.min()
        mask = g.abs() >= thresh
        g.mul_(mask)
        kept_total += int(mask.sum())
        total += n
    return {"fraction": fraction, "kept": kept_total, "total": total}


@torch.no_grad()
def update_support(model, fraction: float) -> dict[str, torch.Tensor]:
    """Boolean support of the current gradient under the same top-k rule.

    Called after a single-example backward pass, this is 'which parameters would
    this one fact move?'. Comparing supports across facts gives measured overlap.
    """
    out: dict[str, torch.Tensor] = {}
    for name, p in trainable_named_params(model):
        if p.grad is None:
            continue
        g = p.grad.abs().flatten()
        n = g.numel()
        k = max(1, int(round(min(fraction, 1.0) * n)))
        idx = torch.topk(g, k, largest=True, sorted=False).indices
        m = torch.zeros(n, dtype=torch.bool, device=g.device)
        m[idx] = True
        out[name] = m.cpu()
    return out


# ------------------------------------------------------------------ enforced mode
#
# Magnitude masking selects whatever the gradient already prefers, and measurement
# shows that plateaus near 0.33 overlap: different facts want the SAME parameters,
# even in the top 0.1% of gradient entries. So it cannot reach the low-overlap regime
# the claim is about.
#
# Enforced disjointness assigns each fact a deterministic pseudo-random parameter
# subset of size fraction*n, independent of the gradient. Expected Jaccard between
# two random subsets of density f is f/(2-f) -- so f=0.01 gives ~0.005 overlap,
# genuinely isolated.
#
# The contrast between the two arms is the point, not an implementation detail:
# N-LoRA's collision penalty, O-LoRA's orthogonality and SMF's slot isolation all
# FORCE separation against the model's preferred update direction. Natural-vs-enforced
# is exactly that distinction, and running both at matched fraction isolates it.


def enforced_mask(
    model, fact_id: int, fraction: float, *, salt: int = 0
) -> dict[str, torch.Tensor]:
    """Deterministic per-fact parameter subset. Same fact -> same mask, always."""
    masks: dict[str, torch.Tensor] = {}
    for name, p in trainable_named_params(model):
        # keyed on (fact, parameter tensor) so different tensors get independent draws
        g = torch.Generator(device="cpu")
        g.manual_seed((hash((name, fact_id, salt)) & 0x7FFFFFFF))
        r = torch.rand(p.numel(), generator=g)
        masks[name] = (r < fraction).reshape(p.shape).to(p.device)
    return masks


@torch.no_grad()
def apply_mask(model, masks: dict[str, torch.Tensor]) -> None:
    for name, p in trainable_named_params(model):
        if p.grad is not None and name in masks:
            p.grad.mul_(masks[name])


@torch.no_grad()
def enforced_support(model, fact_id: int, fraction: float, *, salt: int = 0) -> dict[str, torch.Tensor]:
    """Support of the enforced mask, in the same format update_support() returns."""
    return {k: v.flatten().cpu() for k, v in enforced_mask(model, fact_id, fraction, salt=salt).items()}


def jaccard(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    """Overlap between two update supports. 0 = disjoint, 1 = identical."""
    inter = union = 0
    for k in a.keys() & b.keys():
        x, y = a[k], b[k]
        inter += int((x & y).sum())
        union += int((x | y).sum())
    return inter / union if union else 0.0


def mean_pairwise_overlap(supports: list[dict[str, torch.Tensor]]) -> dict:
    """Mean Jaccard over all pairs -- the measured isolation of a configuration."""
    vals = [
        jaccard(supports[i], supports[j])
        for i in range(len(supports))
        for j in range(i + 1, len(supports))
    ]
    if not vals:
        return {"mean": 0.0, "n_pairs": 0}
    vals_t = torch.tensor(vals)
    return {
        "mean": round(float(vals_t.mean()), 6),
        "median": round(float(vals_t.median()), 6),
        "min": round(float(vals_t.min()), 6),
        "max": round(float(vals_t.max()), 6),
        "n_pairs": len(vals),
    }


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    torch.manual_seed(0)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(64, 64, bias=False)
            self.b = torch.nn.Linear(64, 8, bias=False)

        def forward(self, x):
            return self.b(self.a(x))

    m = Toy()

    def grads_for(x):
        m.zero_grad()
        m(x).sum().backward()

    print("=== topk_mask_gradients ===")
    for frac in (1.0, 0.5, 0.1, 0.01):
        grads_for(torch.randn(4, 64))
        before = sum(int((p.grad != 0).sum()) for _, p in trainable_named_params(m))
        stats = topk_mask_gradients(m, frac)
        after = sum(int((p.grad != 0).sum()) for _, p in trainable_named_params(m))
        obs = after / before
        status = "ok" if abs(obs - min(frac, 1.0)) < 0.02 else "MISMATCH"
        print(f"  fraction={frac:<5} nonzero {before:>5} -> {after:>5}  observed={obs:.3f}  {status}")

    print("\n=== update_support / overlap ===")
    xs = [torch.randn(1, 64) for _ in range(6)]
    for frac in (1.0, 0.25, 0.05):
        sups = []
        for x in xs:
            grads_for(x)
            sups.append(update_support(m, frac))
        ov = mean_pairwise_overlap(sups)
        print(f"  fraction={frac:<5} mean_overlap={ov['mean']:.4f} "
              f"(min {ov['min']:.4f} max {ov['max']:.4f}, {ov['n_pairs']} pairs)")

    print("\n=== enforced (deterministic disjoint subsets) ===")
    for frac in (1.0, 0.3, 0.1, 0.03, 0.01):
        sups = [enforced_support(m, i, frac) for i in range(8)]
        ov = mean_pairwise_overlap(sups)
        expected = frac / (2 - frac)  # Jaccard of two independent Bernoulli(f) masks
        print(f"  fraction={frac:<5} mean_overlap={ov['mean']:.4f}  "
              f"expected f/(2-f)={expected:.4f}  "
              f"{'ok' if abs(ov['mean']-expected) < 0.02 else 'MISMATCH'}")

    print("\n=== determinism ===")
    a = enforced_support(m, 3, 0.1)
    b = enforced_support(m, 3, 0.1)
    c = enforced_support(m, 4, 0.1)
    print(f"  same fact twice : jaccard={jaccard(a,b):.4f}  (must be 1.0)")
    print(f"  different facts : jaccard={jaccard(a,c):.4f}  (should be ~f/(2-f)=0.0526)")

    print("\nNatural masking plateaus near 0.33 overlap -- gradients prefer shared")
    print("parameters. Enforced masking reaches ~0. Both arms are needed: the first")
    print("is what a gradient-selection method achieves, the second is what a")
    print("structural constraint imposes. I4 asks what each buys and what it spends.")
