# Stored, Not Integrated

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21865285.svg)](https://doi.org/10.5281/zenodo.21865285)

Code, data and results for **"Stored, Not Integrated: A Pre-Treatment Predictor and Three Controls for Knowledge Injection."**

Fine-tuning stores new facts almost perfectly and leaves them unusable. In this setting injected-fact recall is **1.000** while two-hop composition using the same fact is **0.22**. This repository asks what determines which side of that gap a fact lands on, and whether it can be known before the fact is injected.

**Manuscript: [`paper/stored_not_integrated.pdf`](paper/stored_not_integrated.pdf)** (Markdown source alongside it)

## Findings

- Composition scales with a per-item measurement taken on the **base model before injection**. With all 518 facts injected in a single adapter, Spearman rho = **+0.362** over 518 items across three seeds; template-bootstrap CI [+0.204, +0.435]; template-stratified permutation p = 0.0001. Positive within five of five testable templates.
- The association is **not a restatement of bridge-entity familiarity**: controlling for how well the base model already knows the bridge entity attenuates it but does not remove it (+0.362 to +0.219).
- **Training explicit first-hop generation did not improve chaining.** Raising bridge-entity generation from 0.000 to 1.000 yields no gain over a control that trains an unrelated fact.
- Supplying the bridge in context raises composition 0.220 to 0.498, but a **false-bridge control** shows a quantity equal to about **68%** of that gain is followed regardless of whether the bridge is correct.
- Two experiments returned negative results: an intervention derived from the diagnosis did not beat its control, and a test of template generalisation did not reach significance.
- A published self-patching recovery figure is measured against a baseline of zero; on an uninjected model the per-item maximum of the same statistic is +0.80.

## Quick start

Regenerate every figure in the manuscript from the deposited results. No GPU, no model download, no dataset:

```
python -m pip install matplotlib
python src/make_figures.py
```

## Reproducing the reported analyses

These run from the included JSON and need only `matplotlib` (some need nothing beyond the standard library). Run them from the repository root.

The headline dose-response, in the single-adapter design that the manuscript reports:

```
python src/analyze_dose_clustered.py --pooled-root results/pooled7
```

reproduces rho = +0.3619 over 518 items, the template-level bootstrap intervals, the leave-one-template-out sensitivity and the template-stratified permutation test. The unit of analysis is the item, seed-averaged; the earlier release pooled 1554 item-evaluations and treated them as independent, which is corrected here.

The same script on the original separate-adapter dataset, for comparison:

```
python src/analyze_dose_clustered.py
```

The remaining analyses:

```
python src/analyze_arms.py
python src/analyze_entity_quality.py --arm pooled --pooled-root results/pooled7
python src/analyze_patching.py --base results/patch_sweep_base.json --injected results/patch_sweep_injected.json
```

giving, in order: the four-arm comparison with McNemar tests and a template-level cluster bootstrap; the entity-familiarity partial correlations; and the null-controlled self-patching comparison.

## Reproducing from scratch

Requires a GPU (runs used a single 8 GB card, about 1.9 h per training run for the arm datasets and about 3.6 h for the 518-fact single adapter), the packages in `requirements.txt`, and the source knowledge graph, which is not redistributed here.

```
python src/probe_anchors.py
python src/build_dataset_anchored.py
python src/build_arms.py
python src/train_inject.py --data data/tasks/pooled7.json --epochs 40 --seed 0 --out results/pooled7/s0
python src/test_e2_in_context.py
python src/test_false_bridge.py
python src/patch_sweep.py --model Qwen/Qwen2.5-1.5B-Instruct --data data/tasks/anchored7.json --limit 60 --stride 4 --out results/patch_sweep_base.json
```

Base-model weights are pinned to explicit revisions in `src/model_pin.py`.

## Layout

| path | contents |
|---|---|
| `src/` | dataset construction, training, evaluation, analysis, figures |
| `data/tasks/` | constructed chaining datasets: the single-adapter set, the anchored arm, the matched control, and four token-matched arms |
| `results/` | per-item evaluation output and the JSON every figure is computed from |
| `paper/` | manuscript (Markdown source and built PDF), its ten figures, and `build_paper.py` |

### Scripts by manuscript section

| section | script |
|---|---|
| 3 task construction | `prime_graph.py`, `build_dataset_anchored.py`, `probe_anchors.py` |
| 4 measurement | `eval_runner.py` |
| 5.1 dose-response | `analyze_dose_clustered.py`, `analyze_anchor_dose.py` |
| 5.2 entity-familiarity control | `test_entity_quality.py`, `analyze_entity_quality.py`, `dump_e2_degree.py` |
| 5.3 decomposition | `test_e2_in_context.py` |
| 5.4 false-intermediate control | `test_false_bridge.py` |
| 5.5 four arms | `build_arms.py`, `analyze_arms.py` |
| 5.6 cluster bootstrap | `analyze_stats.py` |
| 5.7 isolation overlap | `isolation.py`, `sweep_isolation.py` |
| 6 self-patching null | `patch_sweep.py`, `analyze_patching.py` |
| figures | `make_figures.py` |
| manuscript PDF | `paper/build_paper.py` (needs pandoc + XeLaTeX) |

A commit/defer gate reported in v0.1.0 was withdrawn; neither its analysis nor its figure is shipped, because nothing in the current manuscript depends on them. See [`CHANGELOG.md`](CHANGELOG.md).

## Data

Chaining tasks are constructed by graph traversal over **PrimeKG / STaRK-Prime**, which is **not redistributed here**; `build_dataset_anchored.py` takes it as input. The constructed task files are included.

LoRA checkpoints (about 7 GB) are not included. Every reported analysis runs from the JSON in `results/`.

## Provenance

Each run directory carries a manifest recording hyperparameters, precision, package
versions, hardware and the sha256 of its input dataset. Two caveats a reader should know:

- **Model revisions are pinned from v0.2.0 onward.** `src/model_pin.py` fixes the exact
  Hugging Face snapshot for each base model, and the manifest records both the pin and the
  revision present in the local cache. Of the 23 deposited manifests, 2 were produced after
  this was introduced and record it; the other 21 record the model name only. No manifest
  carries a git commit, because the working tree is not a git repository.
- **One recorded dataset hash does not match its deposited file.**
  `data/tasks/anchored.json` was regenerated after the `anchored_dense` run. The item set is
  identical (227 task_ids) and no scored gold answer differs — the change is per-item
  metadata from a later probe pass — but the original bytes are not recoverable, so this is
  disclosed rather than repaired. Every other recorded hash verifies.

Version history is in [`CHANGELOG.md`](CHANGELOG.md). v0.1.0 contained errors that v0.2.0
corrects, including one withdrawn result; check it before citing any number from that
release.

## Setup

Qwen2.5-1.5B-Instruct, LoRA r=16, alpha=32, dropout 0.05 on all seven projection modules; AdamW, lr 2e-4, weight decay 0.01, batch 1 with gradient accumulation 8, 40 epochs, fp16. Only the second hop is injected; the anchor is pretrained and left untrained, except in the arm experiment where training it is the manipulation.

## Method notes

Three measurement decisions carry most of the weight (manuscript section 4):

- **Two matchers, always both.** A `pred in gold` substring match scores `"a"` correct against `"Fazadinium bromide"`.
- **Accuracy floors, and raw log-probability is confounded** by format learning: it moved +0.88 nats after one epoch while memorisation was still 0.003. A same-type distractor control absorbs that gain; base-model composition discrimination is 0.4972, exactly chance.
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

**Review status.** v0.1.0 shipped before the analysis scripts had an independent line-by-line review. That review has since been carried out and found errors in five of them, including two arithmetic errors and the confounded design behind the headline number. Every finding was re-verified against the deposited data before being accepted, and the resulting changes are listed in the manuscript.
