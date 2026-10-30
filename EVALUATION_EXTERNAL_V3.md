# V3 additional transfer evaluation

Completed 2026-10-29. **The additional tests confirm substantial generalization and human false-positive problems.** No model weights, normalization, scoring rule or threshold were changed. The local app still selects the same frozen v3 checkpoint; these results do not establish deployment readiness.

## New external corpora

2,639 new excerpts were scored: 2,564 binary-labelled examples plus 75 AI-edited human examples reported separately. These are deterministic, filtered subsets, not the full official benchmark protocols. No aggregate accuracy across unrelated corpora is used as a success claim.

| Corpus | AI detected | Human false positives |
|---|---:|---:|
| Ghostbuster | 514/600 (85.67%) | 14/300 (4.67%) |
| M4 | 336/500 (67.20%) | 22/500 (4.40%) |
| Liang et al. bias study, binary subsets | 310/315 (98.41%) | 84/349 (24.07%) |

| Corpus | AI detection 95% Wilson interval | Human FPR 95% Wilson interval |
|---|---:|---:|
| ghostbuster | 82.64%–88.24% | 2.80%–7.68% |
| m4 | 62.97%–71.17% | 2.92%–6.57% |
| bias | 96.34%–99.32% | 19.88%–28.82% |

The previously evaluated parliamentary test remains 6/62 AI detections (9.68%) and 0/117 human flags; it was not rerun or counted among the new examples. Its zero observed false positives have a 95% upper bound of 3.18%. See [the original v3 report](EVALUATION_V3.md).

## Ghostbuster: domain and generator transfer

100 source groups were selected in each domain, retaining one human, one GPT and one Claude excerpt per group. Generator names below are the upstream file labels. The underlying generators are from the older Ghostbuster release, not current versions.

| Domain | AI detected | Human false positives |
|---|---:|---:|
| essay | 170/200 (85.00%) | 3/100 (3.00%) |
| reuter | 164/200 (82.00%) | 0/100 (0.00%) |
| wp | 180/200 (90.00%) | 11/100 (11.00%) |

| Upstream generator | AI detected |
|---|---:|
| gpt | 293/300 (97.67%) |
| claude | 221/300 (73.67%) |

The weakest human subgroup is creative writing (`wp`): 11/100 falsely flagged. GPT and Claude variants can share a source; the document-level Wilson intervals do not account for that clustering.

## M4: five writing domains

50 eligible human/AI pairs per domain/generator were selected for ChatGPT and davinci, with duplicate human sources rejected before adding a later pair. This gives 100 humans and 100 AI excerpts per domain. The first released human and machine text are used when a record contains lists.

| Domain | AI detected | Human false positives |
|---|---:|---:|
| arxiv | 71/100 (71.00%) | 10/100 (10.00%) |
| peerread | 97/100 (97.00%) | 2/100 (2.00%) |
| reddit | 56/100 (56.00%) | 6/100 (6.00%) |
| wikihow | 37/100 (37.00%) | 2/100 (2.00%) |
| wikipedia | 75/100 (75.00%) | 2/100 (2.00%) |

| Upstream generator | AI detected |
|---|---:|
| chatgpt | 211/250 (84.40%) |
| davinci | 125/250 (50.00%) |

The model misses 63/100 wikiHow AI excerpts while falsely flagging 10/100 human arXiv excerpts. These failures point in different directions; increasing sensitivity alone does not resolve them.

## Bias-study conditions

AI-edited human documents are excluded from binary error rates. Subsets differ in length, topic and writing context; differences do not by themselves establish a causal effect of native-language status. Counts below are after the declared length and duplicate filters, not the original full cohorts.

| Human subset | False positives | 95% Wilson interval |
|---|---:|---:|
| CS224N_real_145 | 66/132 (50.00%) | 41.59%–58.41% |
| CollegeEssay_real_70 | 14/70 (20.00%) | 12.30%–30.82% |
| HewlettStudentEssay_real_88 | 0/88 (0.00%) | 0.00%–4.18% |
| TOEFL_real_91 | 4/59 (6.78%) | 2.67%–16.18% |

In particular, 66/132 human CS224N abstracts and 14/70 human college essays were falsely flagged. The selected non-native TOEFL cohort had 4/59 false flags. A zero count in the Hewlett cohort does not prove a zero population error rate.

| Generated subset | AI detected |
|---|---:|
| CS224N_gpt3PromptEng_145 | 111/115 (96.52%) |
| CS224N_gpt3_145 | 138/138 (100.00%) |
| CollegeEssay_gpt3PromptEng_31 | 30/31 (96.77%) |
| CollegeEssay_gpt3_31 | 31/31 (100.00%) |

| AI-edited human subset | Flagged, not binary accuracy |
|---|---:|
| HewlettStudentEssay_GPTsimplify_88 | 24/55 (43.64%) |
| TOEFL_gpt4polished_91 | 20/20 (100.00%) |

Prompt-engineered and original generated subsets are reported as separate conditions. Filtering gives different sample counts, so these percentages are not a matched-pair estimate of attack effectiveness. Likewise, the AI-polished/simplified conditions do not measure human FPR.

## Consumed RAID regression

This is the old cleaned 1,592-document RAID final test, already consumed in the v2 cycle. It is an in-family regression check, **not new external evidence**.

| Saved checkpoint | AI detected | Human false positives |
|---|---:|---:|
| V2, historical result | 653/796 (82.04%) | 8/796 (1.01%) |
| V3 | 678/796 (85.18%) | 41/796 (5.15%) |

V3 catches 25 more AI examples but falsely flags 33 more human examples on this same old test. Both models use their own saved thresholds. V3 is therefore not an across-the-board improvement over v2.

## Method, exclusions and reproducibility

- The [protocol](EXTERNAL_V3_PROTOCOL.md) preceded scoring. The complete suite was sealed at `2026-10-29T12:37:13.472332+00:00`; scoring ran from `2026-10-29T12:37:45.062971+00:00` to `2026-10-29T12:40:38.837102+00:00`.
- Fixed threshold: `0.8630226609167616`. Weight and metadata hashes were checked before and after evaluation. No training, threshold tuning, external detector calls or paid generation occurred.
- All new examples require at least 100 whitespace words and are cropped to the first 400, symmetrically for all labels. Lengths are not further matched within each human/AI group. Results apply to excerpts. This excludes many short TOEFL and AI-polished samples; it does not characterize the entire original cohorts.
- The approximate prior-overlap index covered every earlier HC3/RAID/v3 partition, optional v3 paraphrases, the parliamentary test and consumed assistant checks. No prior overlap was detected among inspected candidates. The bottom-eight word-five-shingle method is approximate and cannot guarantee absence of every duplicate or encoder-pretraining exposure.
- Ghostbuster rejected four too-short candidate groups before reaching its quotas. M4 rejected 19 too-short pairs and 17 within-corpus duplicate pairs among candidates inspected before its quotas. The bias study excluded 178 short documents and eight within-condition near duplicates. Detailed counters are retained in the suite manifest.
- Deliberate bias-study variants are preserved as separate conditions. No cross-corpus pooled estimate or claim of independent authors is made. Confidence intervals are document-level Wilson intervals, not source-cluster bootstrap estimates.
- Every downloaded data file was checked against its pinned Git blob hash, with SHA-256 also recorded. Source code from those repositories was not executed. Files and model weights remain local and excluded from Git.
- Preparation revealed a JSONL parsing bug: Python `splitlines()` split valid Unicode separators inside strings. The reader now uses newline-delimited file iteration; a regression test preserves U+2028/U+0085 inside strings. A second test verifies AI-edited human rows cannot inflate binary false positives. Fourteen tests passed; Ruff checks passed.

Reproduce with Python/uv and the GitHub CLI configured for public repository access:

```sh
uv run python scripts/evaluate_external_v3.py --prepare
uv run python scripts/evaluate_external_v3.py --evaluate
```

Preparation refuses to overwrite an existing selected suite. Cached raw files are verified and reused. This run consumed these datasets as evaluation evidence; further model selection needs fresh holdouts.

## Sources and artifacts

- [Ghostbuster data](https://github.com/vivek3141/ghostbuster-data), Verma et al.; repository license CC BY 3.0. Revision `86ebd72590556a81622986fab736ab9227a948af`.
- [M4](https://github.com/mbzuai-nlp/M4), Wang et al.; revision `628bf0fcb2e8c6b7ffa71fd1af6be413aced8f7d`. No explicit repository license was found; used locally for research evaluation, not redistributed or relicensed.
- [GPT Detector Bias data](https://github.com/Weixin-Liang/ChatGPT-Detector-Bias), Liang et al.; revision `1d05ed8242d4694a0ed7b73a79c52be8864470f9`. README displays an MIT badge, without a standalone license file. Underlying source-text rights are not assumed to transfer.
- `data/external-v3/manifest.json`: source URLs, revisions, 1,370 file hashes, exclusions, selected-suite hashes and counts.
- `runs/external-v3/comparison.json`: completed metrics and subgroup summary.
- `runs/external-v3/{ghostbuster,m4,bias,raid-consumed}.json`: individual predictions and all subgroup intervals.
- `scripts/evaluate_external_v3.py`: pinned preparation and evaluation workflow.

**Conclusion:** v3 learned useful signals for some older generated writing but lacks reliable transfer and human specificity. Diverse human controls, difficult writing domains and fresh current-generator samples are higher priorities than additional epochs on the same corpus. No v4 training was started.
