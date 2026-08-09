# Stored, Not Integrated

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21865285.svg)](https://doi.org/10.5281/zenodo.21865285)

Code, data and results for **"Stored, Not Integrated: A Pre-Treatment Predictor and Three Controls for Knowledge Injection."**

Fine-tuning stores new facts almost perfectly and leaves them unusable. In this setting injected-fact recall is **1.000** while two-hop composition using the same fact is **0.22**. This repository asks what determines which side of that gap a fact lands on, and whether it can be known before the fact is injected.

**Manuscript: [`paper/stored_not_integrated.pdf`](paper/stored_not_integrated.pdf)** (Markdown source alongside it)

## Findings

- Composition scales with a per-item measurement taken on the **base model before injection** - Spearman rho = +0.225 (n = 1554, three seeds), positive within five of six templates.
- The dependence is **relational**, not entity familiarity: it survives controlling for how well the model knows the bridge entity, which alone predicts nothing (rho = -0.023, p = 0.61).
- **Retrieval capability is not the bottleneck.** Training the model to generate the bridge entity, 0.000 to 1.000, yields no gain over a control that trains an unrelated fact.
- Supplying the bridge in context raises composition 0.220 to 0.498, but a **false-bridge control** shows about 73% of that gain does not depend on the bridge being correct.
- A **commit/defer gate** on the predictor raises composition among committed facts from 0.144 to 0.207, and the threshold transfers across seeds.
- Two experiments returned negative results: an intervention derived from the diagnosis did not beat its control, and a test of template generalisation did not reach significance.
- A published self-patching recovery figure is measured against a baseline of zero; on an uninjected model the per-item maximum of the same statistic is +0.80.

## Quick start

Regenerate every figure in the manuscript from the deposited results. No GPU, no model download, no dataset:

```
python -m pip install matplotlib
python src/make_figures.py
```

## Reproducing the reported analyses

These run from the included JSON and need only `matplotlib` (some need nothing beyond the standard library):

```
python src/analyze_anchor_dose.py --anchored-data data/tasks/anchored7.json --anchored-eval results/tmpl7/anchored_s0/eval_final.json results/tmpl7/anchored_s1/eval_final.json results/tmpl7/anchored_s2/eval_final.json --control-data data/tasks/anchored7_control.json --control-eval results/tmpl7/control_s0/eval_final.json results/tmpl7/control_s1/eval_final.json results/tmpl7/control_s2/eval_final.json
```

reproduces the headline dose-response: rho = +0.2254 over 1554 item-evaluations pooled across three seeds, with the within-template breakdown.

```
python src/analyze_arms.py
python src/analyze_policy.py
python src/analyze_entity_quality.py
python src/analyze_patching.py --base results/patch_sweep_base.json --injected results/patch_sweep_injected.json
```

giving, in order: the four-arm comparison with McNemar tests and a template-level cluster bootstrap; the commit/defer gate with held-out threshold transfer; the relational-versus-entity-familiarity partial correlations; and the null-controlled self-patching comparison.

Run these from the repository root.

## Reproducing from scratch

Requires a GPU (runs used a single 8 GB card, about 1.9 h per training run), the packages in `requirements.txt`, and the source knowledge graph, which is not redistributed here.

```
python src/probe_anchors.py
python src/build_dataset_anchored.py
python src/build_arms.py
python src/train_inject.py --data data/tasks/anchored7.json --epochs 40 --seed 0 --out results/tmpl7/anchored_s0
python src/test_e2_in_context.py
python src/test_false_bridge.py
python src/patch_sweep.py --model Qwen/Qwen2.5-1.5B-Instruct --data data/tasks/anchored7.json --limit 60 --stride 4 --out results/patch_sweep_base.json
```

## Layout

| path | contents |
|---|---|
| `src/` | dataset construction, training, evaluation, analysis, figures |
| `data/tasks/` | constructed chaining datasets: anchored arm, matched control, four token-matched arms |
| `results/` | per-item evaluation output and the JSON every figure is computed from |
| `paper/` | manuscript (Markdown source and built PDF), its eleven figures, and `build_paper.py` |

### Scripts by manuscript section

| section | script |
|---|---|
| 3 task construction | `prime_graph.py`, `build_dataset_anchored.py`, `probe_anchors.py` |
| 4 measurement | `eval_runner.py` |
| 5.1 dose-response | `analyze_anchor_dose.py` |
| 5.2 relational vs entity familiarity | `test_entity_quality.py`, `analyze_entity_quality.py`, `dump_e2_degree.py` |
| 5.3 decomposition | `test_e2_in_context.py` |
| 5.4 false-intermediate control | `test_false_bridge.py` |
| 5.5 four arms | `build_arms.py`, `analyze_arms.py` |
| 5.6 cluster bootstrap | `analyze_stats.py` |
| 5.7 isolation overlap | `isolation.py`, `sweep_isolation.py` |
| 5.8 commit/defer gate | `analyze_policy.py` |
| 6 self-patching null | `patch_sweep.py`, `analyze_patching.py` |
| figures | `make_figures.py` |
| manuscript PDF | `paper/build_paper.py` (needs pandoc + XeLaTeX) |

## Data

Chaining tasks are constructed by graph traversal over **PrimeKG / STaRK-Prime**, which is **not redistributed here**; `build_dataset_anchored.py` takes it as input. The constructed task files are included.

LoRA checkpoints (about 7 GB) are not included. Every reported analysis runs from the JSON in `results/`.

## Setup

Qwen2.5-1.5B-Instruct, LoRA r=16, alpha=32, dropout 0.05 on all seven projection modules; AdamW, lr 2e-4, weight decay 0.01, batch 1 with gradient accumulation 8, 40 epochs, fp16. Only the second hop is injected; the anchor is pretrained and left untrained, except in the arm experiment where training it is the manipulation.

## Method notes

Three measurement decisions carry most of the weight (manuscript section 4):

- **Two matchers, always both.** A `pred in gold` substring match scores `"a"` correct against `"Fazadinium bromide"`.
- **Accuracy floors, and raw log-probability is confounded** by format learning: it moved 3.3 nats after one epoch while memorisation was still 0.000. A same-type distractor control absorbs that gain; base-model composition discrimination is 0.4972, exactly chance.
- **Matched recall.** Every integration comparison is taken where single-fact recall is equal across conditions.

Figures are generated directly from `results/*.json`; no figure value is hand-entered.

## License

Code: MIT. Documentation, manuscript and figures: CC BY 4.0. See [`LICENSE`](LICENSE).

## Citation

See [`CITATION.md`](CITATION.md).

## Authorship and tooling

One human author, working with Claude (Anthropic) as an execution tool. Stated as a split rather than a single sentence, because the division of labour bears on how the interpretive claims should be read.

**The human author** set the research direction and the problem framing; chose which questions were worth asking and which lines to abandon; and set the methodological standards the work was held to - a strong baseline before any intervention, matched compute and matched recall on every comparison, designs audited *before* long runs rather than after, novelty checked before writing, and negative results documented rather than quietly dropped. They made the resource and scoping decisions, decided when to stop diagnosing and attempt an intervention, and take full responsibility for every claim made here.

**Claude** did the execution: literature search, all implementation (dataset construction from the knowledge graph, training, evaluation, analysis, figures), the statistics, and the drafting. Within that direction it also proposed specific experimental designs and interpretations - among them the four-arm token-matched comparison, the false-intermediate control, and the entity-quality test.

**Why the distinction matters.** Claude proposed, and later withdrew, six interpretations over the course of the project. Each was overturned by a test - five by a follow-up experiment, one by a figure that contradicted its own caption. The standards that forced those tests to be run came from the author: insist on a control, audit the design before committing GPU time, check novelty before claiming it. **The claims that survive in the manuscript are the ones that came through that process, not the ones first proposed.** Two construction bugs were also caught by a pre-run audit rather than by the runs failing - an arm control that was 53.5% contaminated, and an evaluation subset covering three of six templates - either of which would have produced clean-looking but wrong numbers.

Independent line-by-line review of the five analysis scripts behind the headline numbers has not yet been completed; the manuscript's *Authorship and tooling* section says so explicitly.
