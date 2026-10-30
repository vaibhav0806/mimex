"""Small, source-disjoint experiments using the public labelled RAID partition."""

import csv
import hashlib
import heapq
import io
import json
import math
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from mimex.data import fingerprint, normalize, verify_data

URL = "https://dataset.raid-bench.xyz/train_none.csv"
SHA256 = "c5467bca6fc7f5c728c676450c7f84ce401df6c6ccc6d82c47e3b5f3c6d6fce4"
HELD_OUT_MODELS = {"gpt4", "cohere-chat"}
QUOTAS = {"train": 300, "development": 50, "calibration": 100, "test": 100}


def rank(value):
    return hashlib.sha256(("mimex-42:" + value).encode()).hexdigest()


def audit_duplicates(data):
    """Approximate candidate search, followed by exact word-shingle Jaccard checks."""
    splits, _ = verify_data(data)
    inverted, documents, matches = defaultdict(list), [], []
    for split, rows in splits.items():
        for row in rows:
            words = normalize(row["text"]).casefold().split()
            shingles = {
                hashlib.blake2b(
                    " ".join(words[i : i + 5]).encode(), digest_size=8
                ).digest()
                for i in range(len(words) - 4)
            }
            signature = heapq.nsmallest(8, shingles)
            candidate_counts = Counter(i for s in signature for i in inverted[s])
            for i, count in candidate_counts.items():
                other_split, other_id, other_shingles = documents[i]
                if count < 2 or other_split == split:
                    continue
                similarity = len(shingles & other_shingles) / len(
                    shingles | other_shingles
                )
                if similarity >= 0.8:
                    matches.append(
                        {
                            "split_a": other_split,
                            "id_a": other_id,
                            "split_b": split,
                            "id_b": row["id"],
                            "jaccard": similarity,
                        }
                    )
            for s in signature:
                inverted[s].append(len(documents))
            documents.append((split, row["id"], shingles))
    return {
        "documents": len(documents),
        "method": "bottom-8 word-5-shingle hashes; >=2 shared hashes proposes candidates; exact Jaccard >=0.8",
        "exhaustive": False,
        "cross_split_matches": matches,
    }


def collect_paraphrases(data, destination, per_domain=10):
    """Stream upstream attacks, retaining only preselected final-test variants."""
    from datetime import UTC, datetime

    destination = Path(destination)
    if destination.exists():
        raise ValueError("Paraphrase output already exists.")
    splits, manifest = verify_data(data)
    selected, counts = {}, Counter()
    for row in sorted(splits["test"], key=lambda r: rank(r["id"])):
        if row["label"] == 1 and counts[row["source"]] < per_domain:
            selected[row["id"]] = row
            counts[row["source"]] += 1
    url = "https://dataset.raid-bench.xyz/train.csv"
    found = {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(10_000_000)
    with (
        urllib.request.urlopen(url, timeout=60) as response,
        destination.open("w") as output,
    ):
        for index, raw in enumerate(csv.DictReader(io.TextIOWrapper(response)), 1):
            original = selected.get(raw["adv_source_id"])
            if raw["attack"] == "paraphrase" and original is not None:
                if (
                    raw["source_id"] != original["group"]
                    or raw["model"] != original["generator"]
                ):
                    raise ValueError("Unexpected attack provenance.")
                row = {
                    **original,
                    "id": raw["id"],
                    "text": normalize(raw["generation"]),
                    "attack": "paraphrase",
                    "original_id": original["id"],
                }
                if original["id"] not in found:
                    found[original["id"]] = row
                    output.write(json.dumps(row) + "\n")
                    output.flush()
            if index % 100_000 == 0:
                print(
                    f"Scanned {index:,} upstream rows; collected {len(found)}/{len(selected)} paraphrases",
                    flush=True,
                )
            if len(found) == len(selected):
                break
    report = {
        "source_url": url,
        "collected_at": datetime.now(UTC).isoformat(),
        "selection": "10 AI examples per domain, hashed ID order, from untouched custom final-test partition",
        "requested": len(selected),
        "found": len(found),
        "missing_original_ids": sorted(set(selected) - set(found)),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "data_manifest": manifest,
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def clean_audited_splits(data, destination, audit):
    splits, manifest = verify_data(data)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Clean dataset destination is not empty.")
    rejected_ids = {r["id_b"] for r in audit["cross_split_matches"]}
    removed = {}
    destination.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        groups = {r["group"] for r in rows if r["id"] in rejected_ids}
        if split == "train" and groups:
            raise ValueError("Audit cleaning must preserve training data.")
        removed[split] = sorted(groups)
        kept = [r for r in rows if r["group"] not in groups]
        path = destination / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in kept))
        manifest["splits"][split] = {
            "rows": len(kept),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "generators": dict(Counter(r["generator"] for r in kept)),
        }
    manifest["duplicate_audit"] = {
        "method": audit["method"],
        "exhaustive": False,
        "removed_groups": removed,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_data(destination)
    return manifest


def prepare_raid(raw, destination, exclude=Path("data/hc3")):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Dataset directory is not empty; choose a new destination.")
    with Path(raw).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != SHA256:
        raise ValueError("Unexpected RAID download checksum.")
    excluded = set()
    if exclude is not None:
        original, _ = verify_data(exclude)
        excluded = {fingerprint(r["text"]) for rows in original.values() for r in rows}
    humans = {}
    with Path(raw).open() as stream:
        for row in csv.DictReader(stream):
            if row["model"] == "human" and 50 <= len(row["generation"].split()) <= 1000:
                humans[row["source_id"]] = row
    # Choose one variant per generator by closest length, with deterministic ties.
    candidates = defaultdict(dict)
    with Path(raw).open() as stream:
        for row in csv.DictReader(stream):
            human = humans.get(row["source_id"])
            if human is None or row["model"] == "human" or row["attack"] != "none":
                continue
            words, human_words = (
                len(row["generation"].split()),
                len(human["generation"].split()),
            )
            if not 50 <= words <= 1000 or not 0.5 <= words / human_words <= 2:
                continue
            key = (abs(math.log(words / human_words)), rank(row["id"]))
            old = candidates[row["source_id"]].get(row["model"])
            if old is None or key < old[0]:
                candidates[row["source_id"]][row["model"]] = (key, row)
    splits = {name: [] for name in QUOTAS}
    counters = {name: Counter() for name in QUOTAS}
    seen_text, seen_title, seen_prefix = set(excluded), set(), set()
    domains = sorted({row["domain"] for row in humans.values()})
    for source_id in sorted(humans, key=rank):
        human = humans[source_id]
        domain = human["domain"]
        split = next(
            (s for s, limit in QUOTAS.items() if counters[s][domain] < limit), None
        )
        if split is None:
            continue
        text = normalize(human["generation"])
        title = (
            fingerprint(domain + ":" + human["title"])
            if human["title"].strip()
            else None
        )
        prefix = fingerprint(" ".join(text.split()[:100]))
        if (
            fingerprint(text) in seen_text
            or prefix in seen_prefix
            or (title and title in seen_title)
        ):
            continue
        pool = candidates[source_id]
        if split != "test":
            pool = {k: v for k, v in pool.items() if k not in HELD_OUT_MODELS}
        if not pool:
            continue
        models = sorted(pool, key=lambda m: rank(source_id + m))
        chosen = next(
            (
                pool[m][1]
                for m in models
                if fingerprint(pool[m][1]["generation"]) not in seen_text
            ),
            None,
        )
        if chosen is None:
            continue
        ai_text = normalize(chosen["generation"])
        if fingerprint(ai_text) == fingerprint(text):
            continue
        for label, content, original in [(0, text, human), (1, ai_text, chosen)]:
            splits[split].append(
                {
                    "id": original["id"],
                    "text": content,
                    "label": label,
                    "group": source_id,
                    "source": domain,
                    "generator": original["model"],
                    "decoding": original["decoding"],
                    "attack": original["attack"],
                    "paired_generator": chosen["model"],
                }
            )
        seen_text.update([fingerprint(text), fingerprint(ai_text)])
        seen_prefix.add(prefix)
        if title:
            seen_title.add(title)
        counters[split][domain] += 1
    for split, limit in QUOTAS.items():
        if any(counters[split][d] != limit for d in domains):
            raise ValueError(
                f"Insufficient eligible pairs: {split}: {dict(counters[split])}"
            )
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "RAID public labelled train partition, custom source-disjoint splits",
        "source_url": URL,
        "raw_sha256": digest,
        "license": "MIT (upstream dataset card)",
        "seed": 42,
        "selection": "source hash order; one human/AI pair per source; length ratio 0.5–2; balanced domains",
        "normalization": "whitespace, punctuation spacing, English contractions",
        "held_out_generators": sorted(HELD_OUT_MODELS),
        "limitations": [
            "2024-era generators",
            "author IDs unavailable",
            "exact/title/prefix dedup only; near duplicates may remain",
            "custom split, not official RAID hidden test",
        ],
        "splits": {},
    }
    for split, rows in splits.items():
        path = destination / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest["splits"][split] = {
            "rows": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "generators": dict(Counter(r["generator"] for r in rows)),
        }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_data(destination)
    return manifest


def compare_reports(rows, scores, threshold, held_out_generators=HELD_OUT_MODELS):
    from mimex.metrics import evaluate_scores

    result = {"overall": evaluate_scores(scores, [r["label"] for r in rows], threshold)}
    for field in ["source", "paired_generator"]:
        groups = {}
        for value in sorted({r[field] for r in rows}):
            indices = [i for i, r in enumerate(rows) if r[field] == value]
            groups[value] = evaluate_scores(
                [scores[i] for i in indices],
                [rows[i]["label"] for i in indices],
                threshold,
            )
        result["by_domain" if field == "source" else "by_generator"] = groups
    indices = [
        i
        for i, row in enumerate(rows)
        if row["paired_generator"] in held_out_generators
    ]
    if indices:
        result["held_out_generators"] = evaluate_scores(
            [scores[i] for i in indices], [rows[i]["label"] for i in indices], threshold
        )
    return result


def assess(data, checkpoint, split="development", output=None, recalibrate=False):
    from mimex.detector import load, score_documents, score_rows
    from mimex.metrics import calibrate

    splits, manifest = verify_data(data)
    rows = splits[split]
    model, tokenizer, device, metadata = load(checkpoint)
    mode = metadata.get("scoring", "prefix")
    threshold = metadata["threshold"]
    if recalibrate:
        calibration = splits["calibration"]
        calibration_scores = (
            [
                r["score"]
                for r in score_documents(
                    model,
                    tokenizer,
                    calibration,
                    device,
                    metadata["max_length"],
                    normalization=metadata.get("normalization", "legacy"),
                )
            ]
            if mode == "window_mean_logit"
            else score_rows(
                model,
                tokenizer,
                calibration,
                device,
                metadata["max_length"],
                normalization=metadata.get("normalization", "legacy"),
            )
        )
        threshold = calibrate(calibration_scores, [r["label"] for r in calibration])
    if mode == "window_mean_logit":
        scores = [
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
    else:
        scores = score_rows(
            model,
            tokenizer,
            rows,
            device,
            metadata["max_length"],
            normalization=metadata.get("normalization", "legacy"),
        )
    report = {
        "checkpoint": str(checkpoint),
        "split": split,
        "scoring": mode,
        "data_manifest": manifest,
        "metrics": compare_reports(
            rows,
            scores,
            threshold,
            manifest.get("held_out_generators", HELD_OUT_MODELS),
        ),
        "recalibrated_on_raid": recalibrate,
        "predictions": [
            {
                "id": r["id"],
                "group": r["group"],
                "label": r["label"],
                "score": s,
                "source": r["source"],
                "generator": r["generator"],
            }
            for r, s in zip(rows, scores, strict=True)
        ],
    }
    if recalibrate:
        report["original_threshold_metrics"] = compare_reports(
            rows,
            scores,
            metadata["threshold"],
            manifest.get("held_out_generators", HELD_OUT_MODELS),
        )
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def robustness(data, checkpoint, paraphrases, output):
    from mimex.detector import load, score_documents
    from mimex.metrics import evaluate_scores

    splits, manifest = verify_data(data)
    loaded = load(checkpoint)
    model, tokenizer, device, metadata = loaded
    if (
        metadata.get("scoring") != "window_mean_logit"
        or metadata["data_manifest"] != manifest
    ):
        raise ValueError(
            "Robustness checks require the calibrated full-document checkpoint and matching data."
        )
    threshold = metadata["threshold"]
    controls = {r["group"]: r for r in splits["test"] if r["label"] == 0}
    selected, counts = [], Counter()
    for row in sorted(splits["test"], key=lambda r: rank(r["id"])):
        if row["label"] == 1 and counts[row["source"]] < 10:
            selected.extend([controls[row["group"]], row])
            counts[row["source"]] += 1
    report = {"checkpoint": str(checkpoint), "threshold": threshold, "formatting": {}}
    transformations = {
        "original": lambda t: t,
        "whitespace": lambda t: "\n\t".join(t.split()),
        "uppercase": str.upper,
        "zero_width_spaces": lambda t: t.replace(" ", "\u200b "),
    }
    for name, transform in transformations.items():
        rows = [{**r, "text": transform(r["text"])} for r in selected]
        scores = [
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
        report["formatting"][name] = {
            "metrics": evaluate_scores(scores, [r["label"] for r in rows], threshold),
            "scores": scores,
            "ids": [r["id"] for r in rows],
        }
    if paraphrases is not None:
        path = Path(paraphrases)
        provenance = json.loads(path.with_suffix(".manifest.json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != provenance["sha256"]:
            raise ValueError("Paraphrase data checksum mismatch.")
        originals = {r["id"]: r for r in splits["test"] if r["label"] == 1}
        attacks = [json.loads(line) for line in path.read_text().splitlines()]
        attacks = [r for r in attacks if r["original_id"] in originals]
        for row in attacks:
            original = originals[row["original_id"]]
            if (
                row["group"] != original["group"]
                or row["generator"] != original["generator"]
            ):
                raise ValueError(
                    "Paraphrase does not match its original test document."
                )
        before = [originals[r["original_id"]] for r in attacks]
        before_scores = [
            r["score"]
            for r in score_documents(
                model,
                tokenizer,
                before,
                device,
                metadata["max_length"],
                normalization=metadata.get("normalization", "legacy"),
            )
        ]
        after_scores = [
            r["score"]
            for r in score_documents(
                model,
                tokenizer,
                attacks,
                device,
                metadata["max_length"],
                normalization=metadata.get("normalization", "legacy"),
            )
        ]
        report["paraphrase"] = {
            "provenance": provenance,
            "original": evaluate_scores(before_scores, [1] * len(attacks), threshold),
            "paraphrased": evaluate_scores(after_scores, [1] * len(attacks), threshold),
            "pairs": [
                {
                    "original_id": r["original_id"],
                    "attack_id": r["id"],
                    "source": r["source"],
                    "generator": r["generator"],
                    "before": a,
                    "after": b,
                }
                for r, a, b in zip(attacks, before_scores, after_scores, strict=True)
            ],
        }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report
