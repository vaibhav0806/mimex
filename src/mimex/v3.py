"""Versioned modern-generator data, preserving v1/v2 and a fresh final holdout."""

import hashlib
import heapq
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import ijson

from mimex.data import normalize_for_model, read_rows, verify_data

DATASET = "WUJUNCHAO/DetectRL-X"
REVISION = "ccc9a0a74793dd3448e0da1650b430f5df1b0402"
NORMALIZATION = "english_casefold_v1"
HELD_OUT_GENERATOR = "gemini-2.5-flash"
QUOTAS = {"train": 400, "development": 100, "calibration": 150, "test": 150}
RAW_SHA256 = {
    "binary_general_open.json": "815be1667a5e7404d9e513ab339c214c713bd7ac4eb46f375fa0a26bf2dfc15f",
    "binary_decoder_paraphrasing_open.json": "95ce677961f9cb9de51226f40aa515bcc36f6d036b4181bcd14dc1f35641ca58",
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def clean(text):
    text = re.sub(
        r"^Given a document to generate the summary of the document\.\s*Article:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(text.split())


def canonical(text):
    return normalize_for_model(clean(text), NORMALIZATION)


def rank(text):
    return digest("mimex-v3-2026:" + text)


def shingles(text):
    words = canonical(text).split()
    return {
        hashlib.blake2b(" ".join(words[i : i + 5]).encode(), digest_size=8).digest()
        for i in range(len(words) - 4)
    }


class DuplicateIndex:
    def __init__(self):
        self.exact = set()
        self.documents = []
        self.buckets = defaultdict(list)

    def matches(self, text):
        if digest(canonical(text)) in self.exact:
            return True
        tokens = shingles(text)
        if not tokens:
            return True
        candidates = Counter(
            i for h in heapq.nsmallest(8, tokens) for i in self.buckets[h]
        )
        for i, count in candidates.items():
            other = self.documents[i]
            if count >= 2 and len(tokens & other) / min(len(tokens), len(other)) >= 0.9:
                return True
        return False

    def add(self, text):
        self.exact.add(digest(canonical(text)))
        tokens = shingles(text)
        for h in heapq.nsmallest(8, tokens):
            self.buckets[h].append(len(self.documents))
        self.documents.append(tokens)


def crop_pair(human, generated, group):
    h, a = clean(human).split(), clean(generated).split()
    length = min(400, len(h), len(a))
    if length < 100:
        return None
    fraction = int(rank(group + ":crop")[:8], 16) / 0xFFFFFFFF
    hi, ai = int(fraction * (len(h) - length)), int(fraction * (len(a) - length))
    return " ".join(h[hi : hi + length]), " ".join(a[ai : ai + length])


def write_dataset(destination, splits, manifest):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest["splits"] = {}
    for name, rows in splits.items():
        path = destination / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest["splits"][name] = {
            "rows": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "generators": dict(Counter(r["generator"] for r in rows)),
        }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_data(destination)


def prepare(raw_directory="data/raw", destination="data/v3"):
    destination, raw_directory = Path(destination), Path(raw_directory)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("V3 destination is not empty.")
    base, variants, checksums = {}, {}, {}
    for filename in [
        "binary_general_open.json",
        "binary_decoder_paraphrasing_open.json",
    ]:
        path = raw_directory / filename
        with path.open("rb") as f:
            checksums[filename] = hashlib.file_digest(f, "sha256").hexdigest()
        if checksums[filename] != RAW_SHA256[filename]:
            raise ValueError(f"Unexpected DetectRL-X checksum: {filename}")
        with path.open("rb") as f:
            for row in ijson.items(f, "item"):
                if row["lang"] != "english":
                    continue
                group = digest(canonical(row["human_written_text"]))
                key = (group, row["model"])
                if filename == "binary_general_open.json":
                    base[key] = row
                else:
                    variants[key] = row["llm_generated_text"]
    index = DuplicateIndex()
    old_manifests = {}
    for name in ["hc3", "raid"]:
        previous, previous_manifest = verify_data(Path("data") / name)
        old_manifests[name] = previous_manifest
        for rows in previous.values():
            for row in rows:
                index.add(row["text"])
    prior_count = len(index.documents)
    splits = {name: [] for name in QUOTAS}
    counts = {name: Counter() for name in QUOTAS}
    rejected, accepted_groups = Counter(), set()
    domains = sorted({r["domain"] for r in base.values()})
    for (group, model), raw in sorted(
        base.items(), key=lambda kv: rank(kv[0][0] + kv[0][1])
    ):
        domain = raw["domain"]
        split = next(
            (s for s, count in QUOTAS.items() if counts[s][domain] < count), None
        )
        if split is None or group in accepted_groups:
            continue
        if split != "test" and model == HELD_OUT_GENERATOR:
            continue
        pair = crop_pair(raw["human_written_text"], raw["llm_generated_text"], group)
        if pair is None:
            rejected["too_short"] += 1
            continue
        human, generated = pair
        if any(
            marker in canonical(generated)
            for marker in [
                "error communicating with",
                "as an ai language model",
                "i cannot fulfill",
                "i can't fulfill",
            ]
        ):
            rejected["api_error_or_refusal"] += 1
            continue
        if index.matches(human) or index.matches(generated):
            rejected["duplicate_or_overlap"] += 1
            continue
        alternative = variants.get((group, model))
        rewritten = None
        if alternative and canonical(alternative) != canonical(
            raw["llm_generated_text"]
        ):
            cropped = crop_pair(raw["human_written_text"], alternative, group)
            if cropped and not index.matches(cropped[1]):
                rewritten = cropped[1]
        # A source contributes only one labelled human/AI pair. All its variants stay here.
        use_rewrite = (
            split == "train"
            and rewritten is not None
            and int(rank(group + ":variant")[:8], 16) % 2 == 0
        )
        ai_text = rewritten if use_rewrite else generated
        if len(ai_text.split()) != len(human.split()):
            # Keep length control exact even when the rewrite is shorter.
            length = min(len(ai_text.split()), len(human.split()))
            human = " ".join(human.split()[:length])
            ai_text = " ".join(ai_text.split()[:length])
        for label, text in [(0, human), (1, ai_text)]:
            row = {
                "id": digest(group + model + str(label)),
                "group": group,
                "text": text,
                "label": label,
                "source": domain,
                "generator": model if label else "human",
                "paired_generator": model,
                "upstream_split": raw["split"],
                "attack": "decoder_paraphrasing" if label and use_rewrite else "none",
            }
            if label and rewritten is not None and split != "train":
                row["paraphrase_text"] = rewritten
            splits[split].append(row)
        counts[split][domain] += 1
        accepted_groups.add(group)
        for text in {human, generated, ai_text, rewritten} - {None}:
            index.add(text)
    for split, limit in QUOTAS.items():
        if any(counts[split][d] < limit for d in domains):
            raise ValueError(
                f"Insufficient eligible groups for {split}: {dict(counts[split])}; exclusions {dict(rejected)}"
            )
    # Replay only old training pairs, never old calibration/development/test examples.
    prior_train = read_rows("data/raid/train.jsonl")
    replay = defaultdict(list)
    for row in prior_train:
        replay[row["group"]].append(row)
    selected_replay = sorted(replay, key=lambda group: rank("replay:" + group))[:600]
    for group in selected_replay:
        for row in replay[group]:
            splits["train"].append(
                {
                    **row,
                    "id": "replay:" + row["id"],
                    "group": "raid:" + group,
                    "source": "raid/" + row["source"],
                }
            )
    manifest = {
        "dataset": "DetectRL-X English custom source-grouped v3 splits plus RAID training replay",
        "revision": REVISION,
        "source_url": "https://huggingface.co/datasets/" + DATASET,
        "license": "MIT (both upstream dataset cards)",
        "raw_sha256": checksums,
        "normalization": NORMALIZATION,
        "held_out_generators": [HELD_OUT_GENERATOR],
        "selection": "hashed source order, matched 100–400-word crops, grouped variants; custom splits of the public open release",
        "rejected": dict(rejected),
        "prior_documents_indexed": prior_count,
        "prior_manifests": old_manifests,
        "near_duplicate_method": "bottom-8 word-5-shingle candidates, >=2 shared hashes, >=90% containment rejection; approximate",
        "training_paraphrases": sum(
            r["label"] == 1 and r.get("attack") == "decoder_paraphrasing"
            for r in splits["train"]
        ),
        "replay_documents": 1200,
        "model_scope": "English DetectRL-X 2025-era generators; 100–400-word crops; full-document scores outside that range unvalidated",
        "evaluated_max_words": 400,
        "evaluated_min_words": 100,
        "limitations": [
            "author IDs unavailable",
            "heldout generator is based on original-generator metadata; paraphraser lineage not independently verified",
            "matched crops rather than full original documents",
            "custom source splits, not official hidden benchmark",
        ],
    }
    write_dataset(destination, splits, manifest)
    return manifest


def prepare_external(destination="data/v3-parliament.jsonl"):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("External dataset destination exists.")
    archive = Path("data/raw/parliamentary.zip")
    if (
        hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest()
        != "fec1073de34859862a94101ea660dc35"
    ):
        raise ValueError("Parliamentary archive checksum mismatch.")
    index = DuplicateIndex()
    for directory in ["data/hc3", "data/raid", "data/v3"]:
        splits, _ = verify_data(directory)
        for rows in splits.values():
            for row in rows:
                index.add(row["text"])
                if row.get("paraphrase_text"):
                    index.add(row["paraphrase_text"])
    rejected, selected = Counter(), []
    with zipfile.ZipFile(archive) as z:
        text = z.read("UK/Base/Test.txt").decode()
    for number, line in enumerate(text.splitlines()):
        label, text = line.split("\t", 1)
        if label not in {"0", "1"}:
            raise ValueError("Unknown parliamentary ground-truth label.")
        words = clean(text).split()
        if len(words) < 100:
            rejected["too_short"] += 1
            continue
        text = " ".join(words[:400])
        if index.matches(text):
            rejected["overlap"] += 1
            continue
        index.add(text)
        selected.append(
            {
                "id": f"parliament-test-{number}",
                "group": digest(canonical(text)),
                "text": text,
                "label": int(label),
                "source": "UK_parliament",
                "generator": "unspecified_GPT5mini_or_Gemini3.1FlashLite"
                if label == "1"
                else "human",
            }
        )
    destination.write_text("".join(json.dumps(r) + "\n" for r in selected))
    provenance = {
        "source": "https://zenodo.org/records/20611280",
        "archive_md5": "fec1073de34859862a94101ea660dc35",
        "member": "UK/Base/Test.txt",
        "selection": "100-word minimum; first 400 words; approximate overlap exclusion; no model scores used",
        "license_note": "Source attributes human texts under Open Parliament Licence v3.0; retained locally with attribution",
        "rows": len(selected),
        "labels": dict(Counter(r["label"] for r in selected)),
        "rejected": dict(rejected),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "limitations": [
            "per-row generator unavailable",
            "source/author IDs unavailable",
            "upstream tokenized text; domain transfer plus generator change",
        ],
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return provenance


def scores(loaded, rows):
    from mimex.detector import score_documents

    model, tokenizer, device, metadata = loaded
    return [
        r["score"]
        for r in score_documents(
            model,
            tokenizer,
            rows,
            device,
            metadata["max_length"],
            normalization=metadata.get("normalization", "legacy"),
        )
    ]


def assess_partition(loaded, rows, threshold):
    from mimex.metrics import evaluate_scores
    from mimex.raid import compare_reports

    originals = scores(loaded, rows)
    result = {
        "clean": compare_reports(rows, originals, threshold, {HELD_OUT_GENERATOR}),
        "predictions": [
            {
                "id": r["id"],
                "group": r["group"],
                "label": r["label"],
                "source": r["source"],
                "generator": r["generator"],
                "score": s,
            }
            for r, s in zip(rows, originals, strict=True)
        ],
    }
    human_indices = [i for i, r in enumerate(rows) if r["label"] == 0]
    result["formatting"] = {}
    for condition, transform in [
        ("uppercase", str.upper),
        ("zero_width_spaces", lambda t: t.replace(" ", "\u200b ")),
    ]:
        transformed = [{**r, "text": transform(r["text"])} for r in rows]
        normalization = loaded[3].get("normalization", "legacy")
        changed = [
            i
            for i, (a, b) in enumerate(zip(rows, transformed, strict=True))
            if normalize_for_model(a["text"], normalization)
            != normalize_for_model(b["text"], normalization)
        ]
        transformed_scores = list(originals)
        for i, score in zip(
            changed, scores(loaded, [transformed[i] for i in changed]), strict=True
        ):
            transformed_scores[i] = score
        human_scores = [transformed_scores[i] for i in human_indices]
        result[condition + "_humans"] = evaluate_scores(
            human_scores, [0] * len(human_indices), threshold
        )
        result["formatting"][condition] = {
            "metrics": evaluate_scores(
                transformed_scores, [r["label"] for r in rows], threshold
            ),
            "identical_normalized_inputs": len(rows) - len(changed),
            "scores": transformed_scores,
        }
    rewritten = [
        {**r, "text": r["paraphrase_text"]} for r in rows if r.get("paraphrase_text")
    ]
    rewritten_scores = scores(loaded, rewritten)
    by_id = {r["id"]: s for r, s in zip(rows, originals, strict=True)}
    result["paraphrase"] = {
        "original": evaluate_scores(
            [by_id[r["id"]] for r in rewritten], [1] * len(rewritten), threshold
        ),
        "rewritten": evaluate_scores(rewritten_scores, [1] * len(rewritten), threshold),
        "predictions": [
            {"id": r["id"], "score": s}
            for r, s in zip(rewritten, rewritten_scores, strict=True)
        ],
    }
    return result
