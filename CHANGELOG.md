# Changelog

All notable changes to this record. Versions are Zenodo releases; the concept DOI
[10.5281/zenodo.21865285](https://doi.org/10.5281/zenodo.21865285) always resolves to the
most recent one.

## v0.2.0

An independent review of v0.1.0's deposited code and data found errors in the analysis.
Every finding was re-verified against the deposited files before being accepted. Where a
correction was possible by re-running the experiment rather than by narrowing the claim,
that is what was done.

### Corrected

| area | change |
|---|---|
| **§5.1 headline** | Replaced by a single-adapter replication. The two margin groups had been trained as separate adapters holding 223 and 295 facts, so in the pooled analysis the sign of the predictor also identified which model answered. All 518 facts are now injected in one adapter: ρ = +0.362, template-bootstrap CI [+0.204, +0.435], permutation p = 0.0001. Removing the confound *raised* the estimate from +0.271 |
| **§5.1 inference** | Normal-approximation p-values over 1554 item-evaluations replaced by template-level bootstrap intervals over 518 items; three seeds of the same items, from one model, are not independent observations. Leave-one-template-out sensitivity and a template-stratified permutation test added |
| **§5.2** | The entity-familiarity control carried the same confound and scored outcomes on one seed while §5.1 used three. Recomputed on the single adapter, seed-averaged: margin +0.362, controlling familiarity +0.219, CI [+0.044, +0.410], permutation p = 0.0001. The conclusion holds, but all four published numbers change, and the control **attenuates** the effect rather than leaving it unchanged as previously reported |
| **§5.4** | The truth-insensitive fraction corrected from 73% to **68%**: the base rate at which the decoy answer appears anyway had not been subtracted. Causal phrasing replaced with descriptive |
| **§5.5** | Arm results reported per seed rather than mixing a two-seed mean with a one-seed test. McNemar is computed per seed on binary outcomes; averaging across seeds first makes a tied item score 0.5 and corrupts the discordance counts. The diversity reading is stated as a non-detection |
| **§4, Fig 9** | Three values in the measurement section were hardcoded rather than computed, and two did not match the deposit: composition accuracy in the unanchored condition is **0.0111** (not 0.026), and the one-epoch log-probability movement is **+0.88 nats** (not 3.3) |
| **§3** | The claim that the control closes the fact-count confound "by construction" is removed; it matches the arms on item selection, not on training environment. Wording now describes what the construction matches rather than what it isolates |
| **§5.3, §5.7, §6** | Claims narrowed to the evidence: training first-hop generation is no longer described as a manipulation of general retrieval capability; the 0.5B result is a same-direction observation, not a replication; the self-patching comparison is a directional argument, not a lower bound |
| **§5.3** | Form-robust recall (0.71–0.81) distinguished from trained-form recall (1.000) |
| **§2** | Multi-hop editing work added (Zhang et al., 2025; Yang et al., 2026) |

### Withdrawn

- **The commit/defer gate** (§5.8), one of five contributions in v0.1.0. The threshold was
  selected without using outcomes and every fact was trained, so the reported transfer
  measured neither generalisation nor deferral. The section states the withdrawal; the
  figure and analysis behind it are no longer shipped.

### Repository

- Base-model weights pinned to explicit revisions (`src/model_pin.py`). The pin binds runs
  made from this version onward: of the 23 deposited run manifests, 2 record it and 21
  predate it, recording the model name only. The working tree is not a git repository, so
  no manifest carries a commit hash; the git block previously recorded `"dirty": false`
  when the check had simply failed, and now records `null` for unknown.
- The release build runs the deposited figure script against the deposited data and fails
  if any figure cannot be regenerated. Defects introduced after v0.1.0 and fixed here:
  path defaults pointing at a directory the deposit does not have; a module imported but
  not shipped; two data files the measurement figure reads that were never deposited,
  whose absence the figure script swallowed while still exiting successfully; Figure 1
  hand-entering two values computable from the deposited gate check; and the LaTeX
  preamble `build_paper.py` requires, which was never deposited, so the PDF could not be
  rebuilt from the repository at all.
- `make_figures.py` records the sha256 of every result file it reads, and `build_paper.py`
  refuses to build a PDF whose figures were generated from different data. The check is on
  content rather than timestamps, which are not meaningful in a freshly copied tree.
- Recorded dataset hashes are verified at build time. One does not match:
  `data/tasks/anchored.json` was regenerated after the `anchored_dense` run. The item set
  is identical (227 task_ids) and no scored gold answer differs; the change is per-item
  metadata from a later probe pass. The original bytes are not recoverable, so this is
  disclosed rather than repaired.

## v0.1.0

Initial public release: code, constructed datasets, per-item results and manuscript.
Superseded by v0.2.0 — see the corrections above before using any number from it.
