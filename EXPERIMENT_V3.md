# Mimex v3 protocol

Written before v3 training, development scoring or final-test scoring.

## Objective and sources

Improve detection of newer generators and paraphrases while controlling human
false positives, and remove the observed case/invisible-character sensitivity.
No paid generation is used.

Use DetectRL-X English public open files at revision
`ccc9a0a74793dd3448e0da1650b430f5df1b0402` (upstream MIT):
`Binary/binary_general_open.json` and
`Binary/binary_decoder_paraphrasing_open.json`. Original generator labels are
DeepSeek-V3, Gemini-2.5-Flash, GPT-4o and Qwen-Max. Paraphraser model lineage has
not been independently verified. This is newer coverage, not a claim to cover
all October 2026 frontier models.

Defactify was considered but excluded: no dataset license was declared in its
card, and visible examples contained API errors and obvious prompt/formatting
artifacts. Do not use it in this experiment.

## Data construction

- Remove the documented summarization instruction wrapper from source articles.
- Group on the canonical full human source, so variants remain together. Source
  author IDs are unavailable; no author-disjoint claim is made.
- Use custom splits across the public open train/test pool. Some upstream `test`
  sources and their rewrites deliberately enter our training split; none enters
  more than one of our splits. This is not the official DetectRL-X benchmark.
- Sample matched 100–400-word crops at a shared deterministic relative position
  in each human/AI pair. Equalize lengths; these are excerpts, not full articles.
- Exclude exact/approximate text overlaps with all earlier HC3/RAID partitions.
  Candidate near duplicates use bottom-eight word-five-shingle hashes, requiring
  two shared signature hashes and >=90% shingle containment for rejection.
- Domain quotas: 400 train, 100 development, 150 calibration, 150 test pairs
  for each of six domains. Original Gemini-2.5-Flash outputs are excluded from
  training/development/calibration and allowed only in final testing.
- Where a training source has a rewrite, deterministically choose the rewrite
  for half of those sources. One human/AI pair per source; no artificial label
  changes to human controls. Actual training paraphrases: 460 AI examples.
- Replay 600 source pairs (1,200 documents) only from the old RAID training set.
  No old development/calibration/test data is used for replay.
- Final counts: 6,000 train, 1,200 development, 1,800 calibration, 1,800 test.
  The post-construction near-duplicate audit found no cross-split matches at
  its 0.8 Jaccard cutoff; this audit is approximate, not exhaustive.
- Seal optional paired rewrite texts with their source partition. Development
  has 221 rewrite examples; calibration has 325; final test has 331.

## Model

Initialize from the frozen v2 checkpoint. Train two epochs, batch size 4,
learning rate 1e-5, seed 42, 256-token windows, 64-token overlap. In epoch two,
oversample the hardest 10% of training examples twice, as in v2. Training uses
random document windows; evaluation uses the mean of all window log-odds.

Versioned preprocessing `english_casefold_v1` applies NFKC, removes U+200B,
U+200C, U+200D, U+2060 and U+FEFF, and casefolds before the existing whitespace,
punctuation-spacing and contraction normalization. Apply it symmetrically to
both classes and at training/inference time. This makes the tested uppercase
and invisible-space variants equivalent without duplicating identical training
examples. Preserve v1/v2's legacy preprocessing.

Select the document threshold only using clean calibration humans, targeting
at most 1% empirical false positives. Preserve the old v2 checkpoint unchanged;
recalibrate its comparison threshold on the same new calibration humans in
memory, without modifying its saved metadata.

## Selection and final tests

Promote v3 only if, on development:

1. Clean AI detection improves by at least 5 percentage points over recalibrated v2.
2. Paired-paraphrase AI detection improves by at least 5 percentage points.
3. Human false positives are <=3% for clean, uppercase and invisible-space inputs.

Evaluate one predetermined candidate. Freeze candidate weight/metadata hashes
and the decision before final-test scoring. Do not tune using the final results.
Report counts and uncertainty overall, by domain, original generator, and
original-generator-held-out subset. Also report paired paraphrases and symmetric
uppercase/zero-width perturbations. Keep all failed cases in the reports.

Additional external final check: eligible English samples from
`UK/Base/Test.txt` in the parliamentary detection dataset at
https://zenodo.org/records/20611280 (ZIP MD5
`fec1073de34859862a94101ea660dc35`). Authors describe Gemini-3.1-Flash-Lite and
GPT-5-mini generations; per-row model attribution is unavailable. Use no
parliamentary training, validation or unlabelled forward data. Treat this as a
separate domain-transfer test, not a model-specific leaderboard result.

Retest the four earlier assistant-generated examples only as consumed qualitative
regression checks, never as selection evidence or a fresh benchmark. General
frontier-model reliability is not an assumed outcome of this experiment.
