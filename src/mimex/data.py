import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

from huggingface_hub import hf_hub_download

DATASET = "Hello-SimpleAI/HC3"
REVISION = "4d0ff18143b5a7e1b1e79beb540c04549d1e59d3"


def normalize(text):
    text = " ".join(text.split())
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return re.sub(r"\b(\w+)\s+(n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1\2", text)


def normalize_for_model(text, version="legacy"):
    if version == "english_casefold_v1":
        text = unicodedata.normalize("NFKC", text)
        text = text.translate(dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff")))
        text = text.casefold()
    elif version != "legacy":
        raise ValueError(f"Unknown normalization version: {version}")
    return normalize(text)


def fingerprint(text):
    return hashlib.sha256(normalize(text).casefold().encode()).hexdigest()


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepare(destination, pairs=1500):
    if pairs < 10:
        raise ValueError("Use at least 10 pairs to populate all three splits.")
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Dataset directory is not empty; choose a new destination.")
    raw = Path(
        hf_hub_download(DATASET, "all.jsonl", repo_type="dataset", revision=REVISION)
    )
    candidates = read_rows(raw)
    random.Random(42).shuffle(candidates)
    selected, seen_text, seen_question = [], set(), set()
    for row in candidates:
        question = fingerprint(row["question"])
        human = next(
            (normalize(t) for t in row["human_answers"] if 50 <= len(t.split()) <= 600),
            None,
        )
        ai = next(
            (
                normalize(t)
                for t in row["chatgpt_answers"]
                if 50 <= len(t.split()) <= 600
            ),
            None,
        )
        if not human or not ai or question in seen_question:
            continue
        hashes = {fingerprint(human), fingerprint(ai)}
        if len(hashes) != 2 or hashes & seen_text:
            continue
        seen_text.update(hashes)
        seen_question.add(question)
        selected.append(
            [
                {
                    "text": text,
                    "label": label,
                    "group": question,
                    "source": row["source"],
                    "generator": "human" if label == 0 else "HC3-era ChatGPT",
                }
                for label, text in enumerate([human, ai])
            ]
        )
        if len(selected) == pairs:
            break
    if len(selected) < pairs:
        raise ValueError(f"Only {len(selected)} eligible pairs available.")
    destination.mkdir(parents=True, exist_ok=True)
    a, b = int(pairs * 0.7), int(pairs * 0.85)
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "seed": 42,
        "license": "CC-BY-SA-4.0 (upstream dataset)",
        "source_url": "https://huggingface.co/datasets/Hello-SimpleAI/HC3",
        "normalization": "whitespace, punctuation spacing, English contractions",
        "limitations": [
            "older ChatGPT only",
            "author identities unavailable",
            "exact deduplication only; near duplicates may remain",
        ],
        "splits": {},
    }
    for name, groups in [
        ("train", selected[:a]),
        ("calibration", selected[a:b]),
        ("test", selected[b:]),
    ]:
        rows = [r for group in groups for r in group]
        path = destination / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest["splits"][name] = {
            "rows": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_data(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    all_groups, all_text = set(), set()
    splits = {}
    for required in ["train", "calibration", "test"]:
        if required not in manifest["splits"]:
            raise ValueError(f"Missing {required} split.")
    for split in manifest["splits"]:
        if split not in {"train", "development", "calibration", "test"}:
            raise ValueError(f"Unknown split: {split}")
        path = directory / f"{split}.jsonl"
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != manifest["splits"][split]["sha256"]
        ):
            raise ValueError(f"{split} data changed after preparation.")
        rows = read_rows(path)
        groups, texts = (
            {r["group"] for r in rows},
            {fingerprint(r["text"]) for r in rows},
        )
        if groups & all_groups or texts & all_text:
            raise ValueError("Data leakage across splits.")
        if {r["label"] for r in rows} != {0, 1}:
            raise ValueError(f"{split} must contain both labels.")
        all_groups.update(groups)
        all_text.update(texts)
        splits[split] = rows
    return splits, manifest
