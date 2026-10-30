"""Frozen-checkpoint transfer tests; see EXTERNAL_V3_PROTOCOL.md."""

import argparse
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from mimex.data import read_rows, verify_data
from mimex.metrics import evaluate_scores
from mimex.v3 import DuplicateIndex, canonical, scores

RAW = Path("data/raw/external-v3")
DEST = Path("data/external-v3")
OUT = Path("runs/external-v3")
SOURCES = {
    "ghostbuster": (
        "vivek3141/ghostbuster-data",
        "86ebd72590556a81622986fab736ab9227a948af",
    ),
    "m4": ("mbzuai-nlp/M4", "628bf0fcb2e8c6b7ffa71fd1af6be413aced8f7d"),
    "bias": (
        "Weixin-Liang/ChatGPT-Detector-Bias",
        "1d05ed8242d4694a0ed7b73a79c52be8864470f9",
    ),
}
WEIGHTS = "7c0e4e7e95455f625b59ada628cb647566fddc11633588c76152db653b43a279"
METADATA = "2fcb075a169319fdfb9f3d53ac4b2c2f8ccac5d005fb49ed1c1d67c684ee5b57"


def log(message):
    print(datetime.now(UTC).isoformat(), message, flush=True)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def order(value):
    return hashlib.sha256(("external-v3:" + value).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def tree(source, revision, filename, recursive=True):
    path = RAW / filename
    if not path.exists():
        repo = SOURCES[source][0]
        result = subprocess.check_output(
            [
                "gh",
                "api",
                f"repos/{repo}/git/trees/{revision}"
                + ("?recursive=1" if recursive else ""),
            ]
        )
        path.write_bytes(result)
    result = json.loads(path.read_text())
    if result.get("truncated"):
        raise ValueError(f"Incomplete source tree: {filename}")
    return result["tree"]


def acquire(job):
    source, entry = job
    path = RAW / source / entry["path"]
    repo, revision = SOURCES[source]
    url = f"https://raw.githubusercontent.com/{repo}/{revision}/{entry['path']}"
    if path.exists():
        body = path.read_bytes()
    else:
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    body = response.read()
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    blob = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
    if blob != entry["sha"]:
        raise ValueError(f"Upstream Git blob mismatch: {path}")
    return {
        "source": source,
        "path": entry["path"],
        "url": url,
        "git_blob": blob,
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def download(jobs):
    records = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(acquire, job) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 100 == 0:
                log(f"Downloaded/verified {len(records)}/{len(jobs)} source files")
    return sorted(records, key=lambda r: (r["source"], r["path"]))


def clipped(text):
    words = text.split()
    return " ".join(words[:400]) if len(words) >= 100 else None


def first(text):
    if isinstance(text, list):
        return text[0] if text else ""
    if not isinstance(text, str):
        raise TypeError("Unexpected upstream text type")
    return text


def make_row(
    dataset, group, condition, text, label, domain, generator, category="binary"
):
    return {
        "id": f"{dataset}:{group}:{condition}",
        "group": group,
        "text": text,
        "label": label,
        "domain": domain,
        "generator": generator,
        "condition": condition,
        "category": category,
    }


def prepare():
    if DEST.exists() and any(DEST.iterdir()):
        raise ValueError("External suite already exists; use --evaluate to reuse it.")
    RAW.mkdir(parents=True, exist_ok=True)
    ghost_root = tree(
        "ghostbuster", SOURCES["ghostbuster"][1], "ghostbuster-root-tree.json", False
    )
    jobs, ghost_groups = [], {}
    for domain in ["essay", "reuter", "wp"]:
        revision = next(r["sha"] for r in ghost_root if r["path"] == domain)
        entries = tree("ghostbuster", revision, f"ghostbuster-{domain}-tree.json")
        variants = {}
        for model in ["human", "gpt", "claude"]:
            variants[model] = {
                r["path"][len(model) + 1 :]: {**r, "path": domain + "/" + r["path"]}
                for r in entries
                if r["type"] == "blob"
                and r["path"].startswith(model + "/")
                and r["path"].endswith(".txt")
                and "/logprobs/" not in r["path"]
            }
        common = set.intersection(*(set(v) for v in variants.values()))
        selected = sorted(common, key=lambda s: order(domain + "/" + s))[:150]
        ghost_groups[domain] = [
            (key, {m: variants[m][key] for m in variants}) for key in selected
        ]
        for _, group in ghost_groups[domain]:
            jobs.extend(("ghostbuster", e) for e in group.values())
    m4_entries = [
        r
        for r in tree("m4", SOURCES["m4"][1], "M4-tree.json")
        if r["type"] == "blob"
        and r["path"].lower()
        in {
            f"data/{domain}_{model}.jsonl"
            for domain in ["arxiv", "peerread", "reddit", "wikihow", "wikipedia"]
            for model in ["chatgpt", "davinci"]
        }
    ]
    if len(m4_entries) != 10:
        raise ValueError("Missing a declared M4 source file")
    bias_entries = [
        r
        for r in tree("bias", SOURCES["bias"][1], "ChatGPT-Detector-Bias-tree.json")
        if r["path"].endswith("/data.json")
    ]
    if len(bias_entries) != 10:
        raise ValueError("Unexpected bias-study subsets")
    jobs += [("m4", r) for r in m4_entries] + [("bias", r) for r in bias_entries]
    log(f"Acquiring {len(jobs)} pinned raw files; no upstream code is executed")
    provenance = download(jobs)
    log("Indexing earlier corpora for overlap exclusions")
    prior = DuplicateIndex()
    prior_hashes = {}
    for name in ["hc3", "raid", "v3"]:
        splits, _ = verify_data(Path("data") / name)
        prior_hashes[name] = sha(Path("data") / name / "manifest.json")
        for rows in splits.values():
            for row in rows:
                prior.add(row["text"])
                if row.get("paraphrase_text"):
                    prior.add(row["paraphrase_text"])
    for path in [
        Path("data/v3-parliament.jsonl"),
        Path("data/fresh-assistant-checks.jsonl"),
    ]:
        if path.name == "fresh-assistant-checks.jsonl" and not path.exists():
            prior_hashes[str(path)] = None
            log(
                "Optional consumed qualitative fixture unavailable; recorded in manifest"
            )
            continue
        prior_hashes[str(path)] = sha(path)
        for row in read_rows(path):
            prior.add(row["text"])
    suites, excluded = {}, defaultdict(Counter)

    def eligible(dataset, texts, seen):
        if any(t is None for t in texts):
            excluded[dataset]["too_short_groups_or_rows"] += 1
            return False
        if len({canonical(t) for t in texts}) != len(texts):
            excluded[dataset]["identical_text_within_group"] += 1
            return False
        if any(prior.matches(t) for t in texts):
            excluded[dataset]["prior_overlap_groups_or_rows"] += 1
            return False
        if any(seen.matches(t) for t in texts):
            excluded[dataset]["within_corpus_duplicate_groups_or_rows"] += 1
            return False
        for text in texts:
            seen.add(text)
        return True

    rows, seen = [], DuplicateIndex()
    for domain, groups in ghost_groups.items():
        accepted = 0
        for key, variants in groups:
            texts = {
                m: clipped((RAW / "ghostbuster" / e["path"]).read_text())
                for m, e in variants.items()
            }
            if not eligible("ghostbuster", list(texts.values()), seen):
                continue
            for model, text in texts.items():
                rows.append(
                    make_row(
                        "ghostbuster",
                        domain + "/" + key,
                        model,
                        text,
                        int(model != "human"),
                        domain,
                        model,
                    )
                )
            accepted += 1
            if accepted == 100:
                break
        log(f"Ghostbuster {domain}: {accepted} source groups")
    suites["ghostbuster"] = rows

    rows, seen = [], DuplicateIndex()
    for entry in sorted(m4_entries, key=lambda r: r["path"]):
        domain, model = Path(entry["path"]).stem.lower().split("_", 1)
        candidates = read_rows(RAW / "m4" / entry["path"])
        candidates.sort(
            key=lambda r: order(
                domain
                + model
                + str(r.get("source_ID", r.get("source_id")))
                + canonical(first(r["human_text"]))
            )
        )
        accepted = 0
        for raw in candidates:
            human, ai = (
                clipped(first(raw["human_text"])),
                clipped(first(raw["machine_text"])),
            )
            if not eligible("m4/" + domain + "/" + model, [human, ai], seen):
                continue
            group = (
                domain
                + "/"
                + str(raw.get("source_ID", raw.get("source_id")))
                + "/"
                + order(human)[:16]
            )
            rows += [
                make_row("m4", group, model + "/human", human, 0, domain, "human"),
                make_row("m4", group, model, ai, 1, domain, model),
            ]
            accepted += 1
            if accepted == 50:
                break
        log(f"M4 {domain}/{model}: {accepted} source pairs")
    suites["m4"] = rows

    rows = []
    for entry in sorted(bias_entries, key=lambda r: r["path"]):
        subset = Path(entry["path"]).parent.name
        mixed = "polished" in subset.lower() or "simplify" in subset.lower()
        label = None if mixed else (0 if "_real_" in subset else 1)
        seen = DuplicateIndex()
        for i, raw in enumerate(json.loads((RAW / "bias" / entry["path"]).read_text())):
            text = clipped(raw["document"])
            if eligible("bias/" + subset, [text], seen):
                rows.append(
                    make_row(
                        "bias",
                        subset + "/" + str(i),
                        subset,
                        text,
                        label,
                        subset.split("_")[0],
                        "mixed" if mixed else ("human" if label == 0 else "gpt3.5"),
                        "ai_edited_human" if mixed else "binary",
                    )
                )
    suites["bias"] = rows
    log("Sealing the complete suite before model scoring")
    manifest = {
        "sealed_at": datetime.now(UTC).isoformat(),
        "sources": SOURCES,
        "protocol_sha256": sha("EXTERNAL_V3_PROTOCOL.md"),
        "checkpoint_weights_sha256": WEIGHTS,
        "checkpoint_metadata_sha256": METADATA,
        "selection": "fixed source/hash order; >=100 words; first 400 words; prior and within-corpus approximate overlap rejection",
        "prior_manifests_sha256": prior_hashes,
        "source_files": provenance,
        "excluded": dict(excluded),
        "datasets": {},
    }
    DEST.mkdir(parents=True, exist_ok=True)
    for name, rows in suites.items():
        if not rows or len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Empty suite or duplicate IDs in {name}")
        path = DEST / (name + ".jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest["datasets"][name] = {
            "rows": len(rows),
            "sha256": sha(path),
            "labels": dict(Counter(str(r["label"]) for r in rows)),
        }
    save(DEST / "manifest.json", manifest)
    log(json.dumps(manifest["datasets"]))


def measure(loaded, rows):
    threshold = loaded[3]["threshold"]
    predictions = []
    for start in range(0, len(rows), 200):
        batch = rows[start : start + 200]
        for row, score in zip(batch, scores(loaded, batch), strict=True):
            predictions.append(
                {k: v for k, v in row.items() if k != "text"}
                | {"score": score, "flagged": score >= threshold}
            )
        log(f"Scored {len(predictions)}/{len(rows)}")

    def metrics(group):
        return evaluate_scores(
            [r["score"] for r in group], [r["label"] for r in group], threshold
        )

    binary = [r for r in predictions if r["label"] is not None]
    result = {"metrics": metrics(binary), "predictions": predictions, "subgroups": {}}
    for field in ["domain", "generator", "condition"]:
        result["subgroups"][field] = {}
        for value in sorted({r[field] for r in predictions}):
            group = [r for r in predictions if r[field] == value]
            if all(r["label"] is not None for r in group):
                result["subgroups"][field][value] = metrics(group)
            else:
                mixed = [r for r in group if r["label"] is None]
                result["subgroups"][field][value] = {
                    "ai_edited_human_examples": len(mixed),
                    "flagged": sum(r["flagged"] for r in mixed),
                    "note": "Flag rate only; mixed authorship has no binary ground truth here",
                }
                labelled = [r for r in group if r["label"] is not None]
                if labelled:
                    result["subgroups"][field][value]["binary_metrics"] = metrics(
                        labelled
                    )
    return result


def evaluate():
    from mimex.detector import load

    if (
        sha("checkpoints/v3/model.safetensors") != WEIGHTS
        or sha("checkpoints/v3/mimex.json") != METADATA
    ):
        raise ValueError("Frozen checkpoint changed")
    manifest = json.loads((DEST / "manifest.json").read_text())
    if manifest["protocol_sha256"] != sha("EXTERNAL_V3_PROTOCOL.md"):
        raise ValueError("Protocol changed after selection")
    suites = {}
    for name, record in manifest["datasets"].items():
        path = DEST / (name + ".jsonl")
        if sha(path) != record["sha256"]:
            raise ValueError("Sealed evaluation file changed")
        suites[name] = read_rows(path)
    loaded = load("checkpoints/v3")
    summary = {
        "started_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": sha(DEST / "manifest.json"),
        "checkpoint_weights_sha256": WEIGHTS,
        "threshold": loaded[3]["threshold"],
        "datasets": {},
    }
    for name, rows in suites.items():
        log("Evaluating " + name)
        result = measure(loaded, rows)
        save(OUT / (name + ".json"), result)
        summary["datasets"][name] = {
            k: v for k, v in result.items() if k != "predictions"
        }
        log(name + " " + json.dumps(result["metrics"]))
    splits, _ = verify_data("data/raid-clean")
    rows = [
        {
            **r,
            "domain": r["source"],
            "condition": r["generator"],
            "category": "consumed_regression",
        }
        for r in splits["test"]
    ]
    log("Evaluating consumed RAID regression (not new external evidence)")
    result = measure(loaded, rows)
    save(OUT / "raid-consumed.json", result)
    summary["consumed_raid_regression"] = {
        k: v for k, v in result.items() if k != "predictions"
    }
    if (
        sha("checkpoints/v3/model.safetensors") != WEIGHTS
        or sha("checkpoints/v3/mimex.json") != METADATA
    ):
        raise ValueError("Checkpoint changed during evaluation")
    summary["completed_at"] = datetime.now(UTC).isoformat()
    save(OUT / "comparison.json", summary)
    log("COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.prepare or not args.evaluate:
        prepare()
    if args.evaluate or not args.prepare:
        evaluate()
