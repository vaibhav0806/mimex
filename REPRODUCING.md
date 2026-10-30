# Reproducing the experiments

This guide records the v2/v3 training and evaluation workflows. Start with the
[README](README.md) to install Mimex and train the original HC3 baseline.
These experiments require local datasets and checkpoints; trained weights and raw
run artifacts are not distributed in this repository.

## Reproduce v2

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The original run used an
M4 Pro with 24 GB RAM and PyTorch MPS. `uv.lock` pins Python packages. Fixed seeds
and model revisions are recorded; MPS can still produce numerical variation.

```sh
uv sync --frozen
uv run mimex doctor

# HC3 is also used as an exact-text exclusion list.
uv run mimex prepare
# Train the comparison baseline (skip if checkpoints/hc3 already exists).
uv run mimex train

mkdir -p data/raw
curl --fail --location https://dataset.raid-bench.xyz/train_none.csv \
  --output data/raw/raid-train-none.csv
uv run mimex prepare-raid
uv run mimex audit-raid --clean-output data/raid-clean
uv run mimex audit-raid --data data/raid-clean \
  --output runs/raid-clean-duplicate-audit.json

uv run mimex train --data data/raid --output checkpoints/raid-v2 \
  --epochs 2 --batch-size 4 --full-document --mine-hard
uv run mimex collect-paraphrases
uv run python scripts/evaluate_v2.py
```

The 802 MB RAID download is checksum-verified. Paraphrase collection streams the
larger upstream file (roughly 12 GB transferred) and retains only 80 preselected
examples; it does not store the full file. Preparation/training refuse to overwrite
populated destinations. Choose new paths for a new experiment.

The evaluation script recalibrates v2 on the cleaned calibration split, compares
against the recalibrated baseline on development data, and records a selection
in `runs/selection.json` **before** evaluating the final test. It then runs paired
formatting/paraphrase checks and writes `runs/comparison.json`. It does not activate
or publish a model. The promotion criterion is improved development detection
with development false positives <=3%; the calibration target remains 1%.

To activate a checkpoint that passes that criterion locally:

```sh
# If no active link exists yet:
ln -s raid-v2 checkpoints/active
uv run mimex serve
```

For individual stages:

```sh
uv run mimex calibrate --data data/raid-clean --checkpoint checkpoints/raid-v2
uv run mimex assess --data data/raid-clean --checkpoint checkpoints/raid-v2 \
  --split development --output runs/v2-development.json
uv run mimex assess --data data/raid-clean --checkpoint checkpoints/raid-v2 \
  --split test --output runs/v2-final.json
uv run mimex robustness
```

Keep final-test results outside all subsequent model/threshold selection. A new
research cycle needs a fresh final holdout. The custom RAID splits are not the
official hidden-test benchmark and are not a leaderboard result.

## Reproduce v3

V3 starts from `checkpoints/raid-v2` and requires the original `data/hc3` and
`data/raid` partitions for overlap exclusion and training-only replay. Its frozen
plan is in [EXPERIMENT_V3.md](EXPERIMENT_V3.md). Prepare a new destination rather
than overwriting an existing experiment.

Download the two DetectRL-X files at the pinned revision (about 1.74 GB combined):

```sh
uv run python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

root = Path('data/raw')
root.mkdir(parents=True, exist_ok=True)
for name in ['binary_general_open.json', 'binary_decoder_paraphrasing_open.json']:
    downloaded = hf_hub_download(
        repo_id='WUJUNCHAO/DetectRL-X', repo_type='dataset',
        revision='ccc9a0a74793dd3448e0da1650b430f5df1b0402',
        filename='Binary/' + name,
    )
    target = root / name
    if not target.exists():
        target.symlink_to(downloaded)
PY
curl --fail --location \
  'https://zenodo.org/records/20611280/files/ParliamentaryLLMDetect.zip?download=1' \
  --output data/raw/parliamentary.zip

uv run mimex prepare-v3
uv run mimex audit-raid --data data/v3 --output runs/v3-duplicate-audit.json
uv run mimex prepare-v3-external
uv run mimex train --data data/v3 --output checkpoints/v3 \
  --epochs 2 --batch-size 4 --full-document --mine-hard \
  --initial-checkpoint checkpoints/raid-v2 \
  --normalization english_casefold_v1 --learning-rate 0.00001
uv run python scripts/evaluate_v3.py
```

Preparation verifies upstream checksums. Evaluation compares both checkpoints at
thresholds calibrated on the same new human calibration set, freezes the
development decision in `runs/v3-selection.json`, and only then scores final
tests. It leaves the active model unchanged. `runs/v3-comparison.json` records
the result, including failed promotion criteria and external-domain failures.

These are custom source-grouped splits of a public release, not the official
DetectRL-X benchmark. Evaluation covers English 100–400-word excerpts. The
original Gemini generator is held out; paraphraser lineage is not independently
verified. The separate parliamentary test has only 117 human and 62 AI examples
after length filtering, so its uncertainty is substantial.

After training v3, select it explicitly:

```sh
uv run mimex serve --checkpoint checkpoints/v3
uv run mimex predict --checkpoint checkpoints/v3 --file article.txt
```

## Additional external evaluation

The [external evaluation protocol](EXTERNAL_V3_PROTOCOL.md) fixes the checkpoint,
sampling and threshold before testing Ghostbuster, M4 and the Liang et al. bias
study. [Results and subgroup failures](EVALUATION_EXTERNAL_V3.md) include counts,
confidence intervals, AI-edited writing and the consumed RAID regression check.

```sh
# Requires gh (GitHub CLI) with working access to public repositories.
uv run python scripts/evaluate_external_v3.py --prepare
uv run python scripts/evaluate_external_v3.py --evaluate
```

This script deliberately requires the original frozen v3 weight/metadata hashes
in the protocol and the prior HC3/RAID/v3 data for overlap checks. A freshly
retrained checkpoint may have different hashes; define a new protocol and fresh
holdouts for a new experiment rather than treating it as the historical run.
Preparation refuses to overwrite a populated selected suite. The four old local
assistant-written checks are optional qualitative fixtures; scripts record their
absence when `data/fresh-assistant-checks.jsonl` is unavailable. They are not
distributed with the repository.

## Data integrity and scoring

- Source IDs keep a human document and its AI variants in the same partition.
- Initially domain-balanced pairs have similar lengths (ratio 0.5–2), with 50–1,000 words
  per example. This does not eliminate every length or style confound.
- Exact text, human titles and human prefixes are deduplicated. An approximate
  word-shingle audit removes detected cross-split near duplicates as whole pairs.
  Author IDs are unavailable; this is not author-disjoint evaluation.
- GPT-4 and Cohere-chat are excluded from v2 training/development/calibration and
  appear only in its final test.
- Epoch two oversamples the hardest 10% of **training** examples. No development,
  calibration or test examples contribute gradients.
- V2 samples a document window during training. Inference averages window log-odds
  and converts that average to a score before applying the document threshold.
- File hashes, training configuration, calibration and revisions are saved with
  the checkpoint. Reports preserve per-example scores and subgroup counts.

## Licenses and attribution

Code is MIT licensed. ModernBERT-base weights are Apache-2.0. RAID's upstream card
lists MIT. HC3 is CC-BY-SA-4.0 and is not relicensed under Mimex's code license.
Datasets and derived artifacts retain their applicable upstream obligations.
This repository does not bundle datasets or publish trained weights.

- [RAID](https://huggingface.co/datasets/liamdugan/raid): Dugan et al.,
  [ACL 2024 paper](https://aclanthology.org/2024.acl-long.674/).
- [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3): Guo et al.,
  [2023 paper](https://arxiv.org/abs/2301.07597).
- [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base).
- [DetectRL-X](https://huggingface.co/datasets/WUJUNCHAO/DetectRL-X) (MIT),
  [ACL 2026 paper](https://aclanthology.org/2026.acl-long.1773/).
- [Parliamentary detection dataset](https://zenodo.org/records/20611280),
  Suvanto et al.; human Hansard source attributed under Open Parliament Licence
  v3.0. These external texts remain local and are not used for training.
