# Mimex v3 evaluation

Completed locally on 2026-10-29. V3 passed the frozen development improvement
criterion and is selected in the local experimental app. It performs strongly
within DetectRL-X, but **fails to generalize reliably to the independent
parliamentary test**. This is a completed research iteration, not Pangram parity
or a broadly reliable authorship detector. No fourth training run was started.

Follow-up: [three additional external corpora and the RAID regression check](EVALUATION_EXTERNAL_V3.md)
show substantial human false-positive problems, including 50% on selected human
CS224N abstracts. Those tests use these same frozen v3 weights and threshold.

## Final held-out results

The main final test contains 900 human and 900 AI documents, disjoint from our
training, development and calibration sources. All thresholds were fixed before
this test was scored.

| Main-test setting | AI detected | Human false positives |
|---|---:|---:|
| V2, original saved threshold | 324/900 (36.00%) | 147/900 (16.33%) |
| V2, recalibrated on new human data | 27/900 (3.00%) | 9/900 (1.00%) |
| V3 | 900/900 (100.00%) | 14/900 (1.56%) |

V3's AI detection 95% Wilson interval is **99.57%–100%**; its human false-positive
interval is **0.93%–2.59%**. The 1% calibration target is not a guarantee of 1%
errors on new data. Perfect observed AI recall on this corpus does not establish
universal detection, as the separate external test demonstrates.

| V3 original-generator subset | AI detected |
|---|---:|
| DeepSeek-V3 | 207/207 |
| GPT-4o | 228/228 |
| Qwen-Max | 229/229 |
| Gemini-2.5-Flash, original-generator holdout | 236/236 |

The held-out Gemini recall interval is 98.40%–100%. These source-model labels do
not establish transfer to future versions, other generation settings or domains.

| V3 writing category | AI detected | Human false positives |
|---|---:|---:|
| Academic | 150/150 | 0/150 |
| News | 150/150 | 0/150 |
| Novel | 150/150 | 4/150 |
| SEO | 150/150 | 6/150 |
| Webtext | 150/150 | 1/150 |
| Wiki | 150/150 | 3/150 |

SEO human FPR is 4%; aggregate performance hides this weaker subgroup. All
subgroup counts and confidence intervals are preserved in the JSON reports.

## Paraphrases, formatting and consumed checks

On 331 paired final examples, v3 detects all 331 originals and **296/331 rewrites
(89.43%)**, with a rewrite recall interval of 85.65%–92.30%. Recalibrated v2 detects
3/331 rewrites (0.91%). The rewrite comparison includes length changes, as
documented below, and covers one upstream attack family.

For both uppercase and inserted zero-width spaces, all 1,800 v3 final inputs
normalize to the same model inputs as their clean counterparts. Accordingly,
AI detection remains 900/900 and human false positives remain 14/900. Recalibrated
v2 detects 0/900 AI texts under either transformation, with 0/900 human flags.
V3's result is a deliberate preprocessing property, not learned immunity to
arbitrary rewriting or obfuscation.

The four previously used assistant-written checks change from 0/4 flags under v2
to 2/4 under v3, using each model's saved threshold. These are consumed qualitative
examples without matched human controls, not fresh benchmark evidence.

## Development decision

The predetermined candidate passed the criterion in [EXPERIMENT_V3.md](EXPERIMENT_V3.md):
at least five percentage points more clean and paraphrase detection than v2,
with human false positives at most 3% under each tested formatting condition.
The decision and checkpoint hashes were frozen at
`2026-10-29T12:08:12.438267+00:00`, before final-test scoring.

| Development setting | AI detected | Human false positives |
|---|---:|---:|
| V2, original saved threshold | 272/600 (45.33%) | 100/600 (16.67%) |
| V2, recalibrated on new human data | 11/600 (1.83%) | 3/600 (0.50%) |
| V3 | 599/600 (99.83%) | 16/600 (2.67%) |

On the 221 paired development rewrites, detection was 1/221 (0.45%) for
recalibrated v2 and 195/221 (88.24%) for v3. The promotion decision is a research
improvement gate, not evidence that the model meets a deployment false-positive
target. V3's development FPR interval is 1.65%–4.29% (95% Wilson).

V3's threshold is `0.8630226609167616`; v2's comparison threshold is
`0.9999999978513959`. Both were selected from the same 900 clean human calibration
examples with a 1% empirical FPR target. The original v2 checkpoint and its saved
threshold remain unchanged. This strong threshold shift demonstrates v2's
poor transfer to the new human distribution.

## Training and data

ModernBERT-base continues from the frozen v2 weights. Two epochs used 6,000 and
7,200 example presentations, respectively, with batch size 4 and learning rate
1e-5. Epoch two includes two extra presentations of each of the 600 hardest
training examples. Total: 13,200 presentations and 3,300 optimizer updates.
Training plus calibration and saving took 20.26 minutes on the M4 Pro's MPS GPU.
No paid generation or remote training was used.

The 6,000 training documents comprise 4,800 DetectRL-X documents and 1,200 old RAID
training examples. The modern portion contains 460 AI paraphrases. Development
has 1,200 documents; calibration and final testing each have 1,800. Every split
has equal human and AI counts.

- [DetectRL-X](https://huggingface.co/datasets/WUJUNCHAO/DetectRL-X), revision
  `ccc9a0a74793dd3448e0da1650b430f5df1b0402`, upstream MIT. The two pinned public
  files are SHA-256 verified. Generator labels include DeepSeek-V3, GPT-4o,
  Qwen-Max and Gemini-2.5-Flash.
- Six upstream categories: Academic, News, Novel, SEO, Webtext and Wiki.
  These are source labels; "Novel" also includes technical book excerpts.
- These are custom source-grouped partitions across the public open release,
  **not the official DetectRL-X benchmark**. Some upstream test sources and
  rewrites deliberately enter our training partition, but no selected source
  enters more than one of our own partitions.
- Canonical full human-source hashes keep selected pairs and variants together.
  All prior HC3/RAID partitions are used as exclusion lists. The bottom-eight
  word-five-shingle filter is approximate. The additional 0.8 Jaccard audit found
  no cross-split matches; it does not prove the absence of every near duplicate.
  Exact normalized-text checks include held-out paraphrase variants.
- Human and original AI crops have equal lengths, restricted to 100–400 whitespace
  words. These are excerpts rather than full source documents. Author identifiers
  are unavailable, so author-disjoint evaluation is not claimed.
- Gemini-2.5-Flash original outputs are excluded from training, development and
  calibration. Its final subset contains 236 AI examples. Paraphraser lineage
  has not been independently verified; this is an original-generator holdout.
- Paired final rewrite testing covers 331 AI examples. In 292 pairs, original and
  rewritten crops differ in length, so this check also includes a length change.
  Both versions remain in the same source partition.

Versioned preprocessing applies NFKC, removes five specified invisible format
characters and casefolds English text before the existing whitespace, punctuation
and contraction cleanup. It is applied to both labels during training and
inference. V1/v2 preserve legacy preprocessing. Uppercase and inserted zero-width
spaces that normalize to identical model inputs reuse the original score in
evaluation; other changed inputs are actually rescored. This is invariance by
preprocessing, not evidence that the model learned arbitrary attack robustness.

Data, initialization, normalization and training change together. This experiment
does not isolate the contribution of any one change.

## Independent external test

The [parliamentary dataset](https://zenodo.org/records/20611280) by Suvanto et al.
describes GPT-5-mini and Gemini-3.1-Flash-Lite generations, without per-row model
attribution. Labels are explicitly 0 for human and 1 for generated text. Only
`UK/Base/Test.txt` is used; no training, validation or unlabelled forward material
is used for fitting or selection. ZIP MD5 was verified.

Length and approximate-overlap filtering leaves 179 examples: 117 human and
62 AI. Another 1,456 examples were excluded as too short. Upstream text is already
tokenized, so whitespace counts include its tokenization conventions. Crops keep
the first 400 whitespace units. This small, narrow sample tests domain and model
transfer together; it cannot establish broad current-model performance.

Human Hansard text is attributed under Open Parliament Licence v3.0. The files
remain local; no blanket MIT claim is made for this external dataset.

| Independent parliamentary setting | AI detected | Human false positives |
|---|---:|---:|
| V2, original saved threshold | 1/62 (1.61%) | 3/117 (2.56%) |
| V2, recalibrated on DetectRL-X | 0/62 (0.00%) | 0/117 (0.00%) |
| V3 | 6/62 (9.68%) | 0/117 (0.00%) |

V3's external AI detection interval is **4.51%–19.55%**. Zero observed human
errors still permits an upper 95% Wilson bound of **3.18%**. This is a clear
transfer failure; the main-test 100% figure must not be presented as general
current-model performance. The next research step is more diverse, matched data
and fresh external holdouts, not an assumption that more epochs on this dataset
will solve the problem. Neither thresholds nor weights were changed after these
results were observed.

## Local app verification

- Frozen weight and metadata hashes verified before selecting `checkpoints/v3`.
  The old v2 checkpoint remains available; only the `checkpoints/active` symlink
  changed. The server listens on `127.0.0.1:7860`.
- Twelve unit tests and Ruff lint/format checks passed. Tests include versioned
  normalization, legacy behavior, complete window coverage and input integrity.
- CLI and HTTP agree on the functional fixture's score. All 450 normalized tokens
  are covered across three overlapping windows; accepted input is not truncated.
- HTTP checks passed for formatting invariance, short text, the evaluated length
  range, malformed JSON, invalid types, wrong content type, body and token limits.
- Chrome visibly completed the same full-document analysis. The v3 preview is
  open, and the user's earlier text was preserved. Screenshot:
  `runs/browser-v3-check.png`. This is a functional test, not accuracy evidence.
- Inference accepts at least 50 words but marks inputs outside the evaluated
  100–400-word range. It rejects over 16,384 normalized tokens. Per-window scores
  are uncalibrated, and scores do not establish authorship.

## Artifacts and scope

- `runs/v3-selection.json`: frozen development decision and checkpoint hashes.
- `runs/v3-baseline-development.json`, `runs/v3-development.json`: development scores.
- `runs/v2-v3-final.json`, `runs/v3-v3-final.json`: final per-example and subgroup scores.
- `runs/v2-v3-external.json`, `runs/v3-v3-external.json`: independent external scores.
- `runs/v3-comparison.json`: completed comparison summary.
- `runs/v2-v3-consumed-smoke.json`, `runs/v3-v3-consumed-smoke.json`: four old
  assistant-generated examples. These are consumed qualitative regression checks,
  not fresh final-test evidence; their predictions use each saved model threshold.
- `checkpoints/v3/mimex.json`: training, calibration and source provenance.

Weights SHA-256: `7c0e4e7e95455f625b59ada628cb647566fddc11633588c76152db653b43a279`.

Scores are not authorship probabilities. Mixed authorship, non-English prose and
arbitrary short or long documents remain unvalidated. Final-test data is now
consumed for this research cycle and must not drive further model or threshold
selection.
