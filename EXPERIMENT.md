# Mimex v2 experiment protocol

Protocol written before scoring the RAID development or final test partitions.

- Data: checksum-pinned public labelled RAID `train_none.csv` (2024-era models).
  These are custom partitions, not the official hidden RAID test.
- Each of eight domains contributes 300 training, 50 development, 100 calibration
  and 100 final-test human/AI pairs. Pairing preserves source/topic, chooses a
  similar-length variant (ratio 0.5–2), and limits human/AI text to 50–1,000 words.
- Group by original human source ID. Exclude duplicate human titles, human
  100-word prefixes and exact selected text; exclude exact HC3 examples.
- Exclude GPT-4 and Cohere-chat from training, development and calibration.
  Final-test examples can include them. No author-disjoint claim is possible.
- Evaluate the original HC3 model as-is and with a threshold recalibrated only
  on RAID calibration human text. Keep the original checkpoint unchanged.
- Candidate: ModernBERT-base, full-parameter training, 2 epochs, batch 4,
  learning rate 2e-5. Sample one overlapping window per document per epoch.
  After epoch 1, mine the highest-error 10% of training examples using prefix
  scores; include two extra copies of these documents in epoch 2. No development
  or calibration examples feed gradients.
- Document score: sigmoid of mean window log-odds; 256 tokens per window,
  64-token overlap. Recalibrate its threshold on full calibration documents
  targeting 1% empirical human false positives. Reject inputs over 16,384 tokens
  rather than silently truncating. Window scores are descriptive, not separately
  calibrated authorship verdicts.
- Compare on development: promote the candidate only if detection exceeds the
  RAID-recalibrated baseline and observed human false positives are <=3%.
  Freeze the chosen checkpoint/threshold before final-test evaluation. Do not
  retrain or adjust thresholds in response to final-test results.
- Report confusion counts, uncertainty, domain, generator and unseen-generator
  results. Keep per-example scores and false-positive/false-negative IDs.
- Test deterministic formatting perturbations separately; these are smoke
  robustness checks, not a substitute for RAID's official adversarial benchmark.
- Fresh assistant-generated examples may be used only as explicitly labelled
  qualitative checks, never as evidence of measured multi-model performance.

Training and test data have old generators; contemporary-model coverage and
external author-disjoint validation remain separate research requirements.

## Integrity amendment, before candidate selection or final-test scoring

A bottom-8 word-shingle candidate audit found 49 high-similarity cross-split
links involving 11 later-partition GPT-2 outputs. These were repetitive movie
review generations with different titles. Remove the entire human/AI pair for
all 11 later-partition documents (1 development, 6 calibration, 4 test pairs).
Training remains byte-identical. The final sizes are 4,800 training, 798
development, 1,588 calibration and 1,592 test documents. Use `data/raid-clean`
for every comparison and recalibrate the candidate after training on this clean
calibration partition. The preliminary training-time calibration is superseded.
This amendment follows a content-integrity audit, not candidate/test performance.
The approximate audit is not proof that all near duplicates have been removed.

RAID paraphrase examples will be streamed directly from the upstream full CSV
because the filtered viewer endpoint timed out. Select ten original AI IDs per
domain from the original final-test partition before seeing predictions; exclude
any pairs removed by the integrity amendment when reporting paired results.
No paraphrase example feeds training or threshold selection.
