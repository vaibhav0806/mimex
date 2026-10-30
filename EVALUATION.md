# Mimex v2 evaluation

Completed 2026-10-29 on the local M4 Pro. V2 passed the predeclared development criterion and was activated after that experiment. [V3](EVALUATION_V3.md) is now selected locally. This historical report describes a stronger **older-data research baseline**, not validated Pangram parity or reliable contemporary AI detection.

## Final held-out comparison

All three settings below use the same 1,592 documents: 796 human and 796 AI examples from custom, source-disjoint partitions of the public labelled RAID dataset. These are not official hidden-test or leaderboard results.

| Model setting | AI detected | Human false positives |
|---|---:|---:|
| V1, original HC3 threshold | 426/796 (53.52%) | 68/796 (8.54%) |
| V1, recalibrated on RAID | 325/796 (40.83%) | 12/796 (1.51%) |
| V2, full-document RAID model | 653/796 (82.04%) | 8/796 (1.01%) |

V2's human false-positive 95% Wilson interval is **0.51%–1.97%**; its AI detection interval is **79.22%–84.55%**. Observed human errors were lower than v1, but the intervals overlap: this sample does not establish a statistically significant FPR improvement or guarantee sub-1% deployment FPR.

## Development selection and training

Development detection: 41.60% for recalibrated v1 versus 85.21% for v2. V2 development FPR was 1.25%, below the predeclared 3% promotion ceiling.

Selection was frozen at `2026-10-29T11:14:55.539825+00:00` before either final-test evaluation. V2's threshold is `0.9878852272544723`. It was chosen only from clean calibration data, targeting 1% empirical human false positives. No post-test tuning was performed.

ModernBERT-base was fine-tuned for two epochs on 4,800 documents from nine generators and eight domains. The second epoch included two extra copies of the 480 hardest training examples, for 5,760 presentations. Total: 10,560 presentations, 2,640 optimizer updates, batch size 4, learning rate 2e-5. Training plus its preliminary calibration/saving took 16.76 minutes. Clean-data recalibration and external evaluation were additional.

V2 changes the training corpus, sampling, hard-example weighting, scoring and calibration together. This experiment does not isolate the contribution of each change.

## Data integrity

- Training: 4,800 documents; development: 798; calibration: 1,588; final test: 1,592.
- Human/AI pairs share a source ID and stay within one partition. Text lengths are matched within a 0.5–2 ratio and restricted to 50–1,000 words.
- Exact selected text, human titles and 100-word human prefixes are deduplicated; exact HC3 examples are excluded.
- An approximate bottom-8 word-shingle audit found repetitive GPT-2 outputs across splits. Eleven later-partition pairs were removed before candidate selection/final testing. Training bytes remained unchanged. A repeat audit found no remaining matches at its 0.8 Jaccard threshold; this is not an exhaustive near-duplicate guarantee.
- Author identities are unavailable, so author-disjoint evaluation is not claimed.
- GPT-4 and Cohere-chat were excluded from training, development and calibration.

## Generator results

| Generator | AI detected |
|---|---:|
| chatgpt | 64/69 (92.75%) |
| cohere | 31/68 (45.59%) |
| cohere-chat (unseen in training) | 51/68 (75.00%) |
| gpt2 | 72/84 (85.71%) |
| gpt3 | 39/45 (86.67%) |
| gpt4 (unseen in training) | 51/70 (72.86%) |
| llama-chat | 78/83 (93.98%) |
| mistral | 60/82 (73.17%) |
| mistral-chat | 74/79 (93.67%) |
| mpt | 66/79 (83.54%) |
| mpt-chat | 67/69 (97.10%) |

Combined unseen-generator detection: **102/138 (73.91%)**. These are older GPT-4/Cohere versions, not evidence about their current successors.

## Writing categories

| Category | AI detected | Human false positives |
|---|---:|---:|
| abstracts | 77/100 (77.00%) | 1/100 |
| books | 85/100 (85.00%) | 0/100 |
| news | 73/100 (73.00%) | 1/100 |
| poetry | 77/100 (77.00%) | 1/100 |
| recipes | 96/100 (96.00%) | 0/100 |
| reddit | 75/100 (75.00%) | 1/100 |
| reviews | 87/96 (90.62%) | 1/96 |
| wiki | 83/100 (83.00%) | 3/100 |

## Robustness and failure cases

| Paired formatting check | AI detected | Human false positives |
|---|---:|---:|
| original | 67/80 (83.75%) | 1/80 (1.25%) |
| whitespace | 67/80 (83.75%) | 1/80 (1.25%) |
| uppercase | 12/80 (15.00%) | 0/80 (0.00%) |
| zero_width_spaces | 58/80 (72.50%) | 4/80 (5.00%) |

For 80 preselected RAID AI documents, detection fell from **67/80 (83.75%)** on originals to **42/80 (52.50%)** on their upstream paraphrases. The paraphrase detection 95% interval is 41.70%–63.08%. Original IDs, attack IDs, scores and retrieval provenance are saved. This is a small paired subset, not the full RAID adversarial benchmark.

**Fresh assistant-written checks: 0/4 flagged.** These four session-generated examples were not used for training, calibration or selection. There are no matched human controls, so this is a qualitative failure demonstration, not a representative estimate of current-model performance.

Uppercase text, paraphrasing and fresh generated prose expose substantial weaknesses. No extra fitting was performed after observing these final checks.

## App behavior and verification

- Full-document inference uses 256-token windows, 64-token overlap and mean log-odds aggregation. The document threshold is calibrated for this scoring rule.
- Inputs over 16,384 tokens are rejected; none of an accepted document is silently dropped. Inputs longer than the evaluated 1,000-word range receive an explicit notice.
- Scores are not calibrated authorship probabilities. Per-window scores are descriptive; mixed authorship is unvalidated.
- Ten unit tests passed. Ruff lint/format checks passed. CLI and HTTP scores agreed for the long-text fixture.
- Live HTTP checks passed for the v2 health endpoint, full-document coverage, short text, invalid types, wrong content type, token limit and body-size limit.
- Chrome completed a full-document analysis of a 396-token functional fixture across two windows. The v2 preview remains open; the user's existing text in the older tab was preserved. Screenshot: `runs/browser-v2-check.png`.

## Reproducibility and artifacts

- [Protocol and integrity amendment](EXPERIMENT.md)
- [Commands and licensing](README.md)
- `runs/selection.json`: frozen selection, checkpoint weight hash and metadata hash.
- `runs/comparison.json`: compact before/after results.
- `runs/baseline-final.json`, `runs/v2-final.json`: per-example and subgroup results.
- `runs/robustness.json`: paired formatting/paraphrase scores and provenance.
- `runs/fresh-assistant-checks.json`: the four qualitative outcomes.
- `checkpoints/raid-v2/mimex.json`: training and calibration provenance.

Weights SHA-256: `e33ccf58a96a8c5ecacf737ccdd2bc4e5ada4dfea0180bd6853481e4d41859a4`.

Data and weights remain local and excluded from Git. The final test is now consumed for this research cycle; future model selection requires a fresh final holdout. The remaining research requirement is genuinely contemporary, diverse generator data plus independent human controls and external validation.
