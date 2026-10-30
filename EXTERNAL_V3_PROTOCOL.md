# Frozen v3 transfer evaluation

Declared before scoring these additional corpora, 2026-10-29. No training,
threshold adjustment or candidate selection is performed.

- Checkpoint: `checkpoints/v3`; weights SHA-256
  `7c0e4e7e95455f625b59ada628cb647566fddc11633588c76152db653b43a279`;
  metadata SHA-256
  `2fcb075a169319fdfb9f3d53ac4b2c2f8ccac5d005fb49ed1c1d67c684ee5b57`.
  Saved threshold: `0.8630226609167616`; saved preprocessing and window scoring.
- Ghostbuster data, revision `86ebd72590556a81622986fab736ab9227a948af`:
  essays, Reuters news and writing prompts; human, GPT and Claude files.
  Take up to 100 eligible source groups per domain from the first 150 shared
  source IDs in deterministic SHA-256 order. Retain all three variants together.
- M4, revision `628bf0fcb2e8c6b7ffa71fd1af6be413aced8f7d`:
  arXiv, PeerRead, Reddit, wikiHow and Wikipedia; ChatGPT and davinci files.
  Take up to 50 eligible human/AI pairs per domain/generator in deterministic
  hash order. Use the first released human and machine text when fields are lists.
- Liang et al. bias data, revision `1d05ed8242d4694a0ed7b73a79c52be8864470f9`:
  all eligible examples from its ten `data.json` subsets. Keep untouched human,
  generated, and AI-edited-human categories distinct. For the last category,
  report flag rate, not human FPR or binary authorship accuracy.
- Require at least 100 whitespace words; score the first 400. Apply the same
  length rule to all labels. Report excluded counts and unequal subgroup sizes.
  Results concern excerpts, not necessarily complete original documents.
- Exclude approximate/exact overlap with all prior HC3, RAID and v3 partitions,
  the consumed parliamentary set and earlier qualitative assistant checks.
  Reuse the existing bottom-eight word-five-shingle containment filter; this is
  approximate, not a guarantee against every overlap or pretraining exposure.
  Deduplicate within each corpus; preserve deliberate bias-study variants as
  separate conditions and do not pool their rates into one headline accuracy.
- Optionally score the existing cleaned RAID final split as a **consumed
  in-family regression check**, clearly separate from new external evidence.
- Save the entire selected suite and hashes before loading the model. Report
  detection, human FPR, counts, 95% Wilson intervals, domain/generator/subset
  breakdowns and per-example scores. AI variants sharing a source are correlated;
  document-level Wilson intervals do not account for source clustering.
- No paid generation, source model inference, third-party detector queries or
  redistribution. Upstream provenance and license notes remain attached. M4's
  repository has no explicit license; bias-study README displays MIT without a
  standalone license file. No blanket MIT claim is made for acquired corpora.
- All corpora become consumed evaluation evidence after this run. Do not tune
  v3 on them or imply these mostly 2023-era generators validate current models.
