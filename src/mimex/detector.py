import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mimex.benchmark import MODEL_ID, MODEL_REVISION, select_device
from mimex.data import normalize, normalize_for_model, verify_data
from mimex.metrics import calibrate, evaluate_scores

WINDOW_OVERLAP = 64
MAX_DOCUMENT_TOKENS = 16_384


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1 + exp)


def window_ranges(tokens, capacity=254, overlap=WINDOW_OVERLAP):
    if tokens < 1 or not 0 <= overlap < capacity:
        raise ValueError("Invalid token count, window capacity or overlap.")
    ranges = []
    start = 0
    while True:
        end = min(start + capacity, tokens)
        ranges.append((start, end))
        if end == tokens:
            return ranges
        start = end - overlap


def score_documents(
    model, tokenizer, rows, device, length=256, batch_size=8, normalization="legacy"
):
    """Mean window log-odds, calibrated separately from the old prefix detector."""
    model.eval()
    results = []
    with torch.inference_mode():
        for row in rows:
            text = normalize_for_model(row["text"], normalization)
            count = len(
                tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
            )
            if count > MAX_DOCUMENT_TOKENS:
                raise ValueError(
                    f"Document exceeds {MAX_DOCUMENT_TOKENS:,} tokens; split it into smaller documents."
                )
            encoded = tokenizer(
                text,
                max_length=length,
                truncation=True,
                stride=WINDOW_OVERLAP,
                return_overflowing_tokens=True,
                padding="max_length",
                return_tensors="pt",
            )
            encoded.pop("overflow_to_sample_mapping")
            margins = []
            for start in range(0, len(encoded["input_ids"]), batch_size):
                batch = {
                    k: v[start : start + batch_size].to(device)
                    for k, v in encoded.items()
                }
                logits = model(**batch).logits
                margins.extend((logits[:, 1] - logits[:, 0]).cpu().tolist())
            if not all(math.isfinite(m) for m in margins):
                raise RuntimeError("Non-finite model output.")
            ranges = window_ranges(
                count, length - tokenizer.num_special_tokens_to_add(False)
            )
            if len(ranges) != len(margins):
                raise RuntimeError("Tokenizer window coverage mismatch.")
            results.append(
                {
                    "score": sigmoid(sum(margins) / len(margins)),
                    "tokens": count,
                    "windows": [
                        {"start_token": a, "end_token": b, "score": sigmoid(m)}
                        for (a, b), m in zip(ranges, margins, strict=True)
                    ],
                }
            )
    return results


def score_rows(
    model, tokenizer, rows, device, length, batch_size=8, normalization="legacy"
):
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = tokenizer(
                [
                    normalize_for_model(r["text"], normalization)
                    for r in rows[start : start + batch_size]
                ],
                padding="max_length",
                truncation=True,
                max_length=length,
                return_tensors="pt",
            ).to(device)
            scores.extend(model(**batch).logits.softmax(-1)[:, 1].cpu().tolist())
    return scores


def train(
    data,
    output,
    epochs=2,
    batch_size=2,
    device="auto",
    length=256,
    full_document=False,
    mine_hard=False,
    initial_checkpoint=None,
    normalization="legacy",
    learning_rate=2e-5,
):
    output = Path(output)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be positive and finite.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Checkpoint directory is not empty; choose a new output.")
    splits, manifest = verify_data(data)
    target = select_device(device)
    revision = MODEL_REVISION
    torch.manual_seed(42)
    model_source = initial_checkpoint or MODEL_ID
    source_options = (
        {"local_files_only": True} if initial_checkpoint else {"revision": revision}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_source, **source_options)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        **source_options,
        num_labels=2,
        attn_implementation="sdpa",
        use_safetensors=True,
        trust_remote_code=False,
        id2label={0: "human", 1: "ai"},
        label2id={"human": 0, "ai": 1},
    ).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    rows = splits["train"]
    encoded = tokenizer(
        [normalize_for_model(r["text"], normalization) for r in rows],
        padding="max_length",
        truncation=True,
        max_length=length,
        return_tensors="pt",
        **(
            {"return_overflowing_tokens": True, "stride": WINDOW_OVERLAP}
            if full_document
            else {}
        ),
    )
    mapping = encoded.pop("overflow_to_sample_mapping", torch.arange(len(rows)))
    windows_by_row = [[] for _ in rows]
    for window, row_index in enumerate(mapping.tolist()):
        windows_by_row[row_index].append(window)
    labels = torch.tensor([r["label"] for r in rows])
    started = time.monotonic()
    hard_indices = []
    history = []
    for epoch in range(epochs):
        indices = list(range(len(rows))) + hard_indices * 2
        random.Random(42 + epoch).shuffle(indices)
        chooser = random.Random(1024 + epoch)
        model.train()
        loss_total = 0.0
        for step, start in enumerate(range(0, len(indices), batch_size), 1):
            ix = indices[start : start + batch_size]
            window_ix = [chooser.choice(windows_by_row[i]) for i in ix]
            batch = {k: v[window_ix].to(target) for k, v in encoded.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch, labels=labels[ix].to(target)).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += loss.item() * len(ix)
            if step % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, examples {min(start + batch_size, len(indices))}/{len(indices)}, loss {loss.item():.4f}",
                    flush=True,
                )
        history.append(
            {
                "epoch": epoch + 1,
                "mean_training_loss": loss_total / len(indices),
                "examples": len(indices),
                "oversampled_hard_examples": len(hard_indices),
            }
        )
        if mine_hard and epoch < epochs - 1:
            # Mine only the training partition. Development/calibration/test never feed gradients.
            train_scores = score_rows(
                model, tokenizer, rows, target, length, normalization=normalization
            )
            difficulty = [
                abs(score - r["label"])
                for r, score in zip(rows, train_scores, strict=True)
            ]
            hard_indices = sorted(
                range(len(rows)), key=lambda i: difficulty[i], reverse=True
            )[: max(1, len(rows) // 10)]
            print(
                f"Mining {len(hard_indices)} hardest training examples for the next epoch",
                flush=True,
            )
    calibration = splits["calibration"]
    scores = (
        [
            r["score"]
            for r in score_documents(
                model,
                tokenizer,
                calibration,
                target,
                length,
                normalization=normalization,
            )
        ]
        if full_document
        else score_rows(
            model, tokenizer, calibration, target, length, normalization=normalization
        )
    )
    threshold = calibrate(scores, [r["label"] for r in calibration])
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    metadata = {
        "base_model": MODEL_ID,
        "revision": revision,
        "max_length": length,
        "threshold": threshold,
        "calibration_target_fpr": 0.01,
        "calibration_metrics": evaluate_scores(
            scores, [r["label"] for r in calibration], threshold
        ),
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": 42,
        "learning_rate": learning_rate,
        "normalization": normalization,
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "evaluated_min_words": manifest.get("evaluated_min_words", 50),
        "evaluated_max_words": manifest.get("evaluated_max_words", 1000),
        "device": str(target),
        "training_seconds": time.monotonic() - started,
        "data_manifest": manifest,
        "scoring": "window_mean_logit" if full_document else "prefix",
        "window_overlap": WINDOW_OVERLAP if full_document else 0,
        "hard_mining": mine_hard,
        "training_history": history,
        "scope": manifest.get("model_scope")
        or (
            "English RAID 2024-era writing; overlapping windows; current generators unvalidated"
            if full_document
            else "English HC3 answers; first 256 tokens; older ChatGPT"
        ),
        "score_is_calibrated_probability": False,
    }
    if initial_checkpoint:
        parent = Path(initial_checkpoint)
        metadata["initial_metadata_sha256"] = hashlib.sha256(
            (parent / "mimex.json").read_bytes()
        ).hexdigest()
        with (parent / "model.safetensors").open("rb") as f:
            metadata["initial_weights_sha256"] = hashlib.file_digest(
                f, "sha256"
            ).hexdigest()
    (output / "mimex.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load(checkpoint, device="auto"):
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "mimex.json").read_text())
    target = select_device(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        local_files_only=True,
        use_safetensors=True,
        attn_implementation="sdpa",
        trust_remote_code=False,
    ).to(target)
    return model, tokenizer, target, metadata


def recalibrate_checkpoint(data, checkpoint):
    splits, manifest = verify_data(data)
    model, tokenizer, target, metadata = load(checkpoint)
    if (
        manifest["splits"]["train"]["sha256"]
        != metadata["data_manifest"]["splits"]["train"]["sha256"]
    ):
        raise ValueError("Recalibration requires the identical training partition.")
    rows = splits["calibration"]
    scores = (
        [
            r["score"]
            for r in score_documents(
                model,
                tokenizer,
                rows,
                target,
                metadata["max_length"],
                normalization=metadata.get("normalization", "legacy"),
            )
        ]
        if metadata.get("scoring") == "window_mean_logit"
        else score_rows(
            model,
            tokenizer,
            rows,
            target,
            metadata["max_length"],
            normalization=metadata.get("normalization", "legacy"),
        )
    )
    threshold = calibrate(scores, [r["label"] for r in rows])
    metadata["previous_calibration"] = {
        "threshold": metadata["threshold"],
        "metrics": metadata["calibration_metrics"],
        "data_manifest": metadata["data_manifest"],
    }
    metadata["threshold"] = threshold
    metadata["calibration_metrics"] = evaluate_scores(
        scores, [r["label"] for r in rows], threshold
    )
    metadata["data_manifest"] = manifest
    path = Path(checkpoint) / "mimex.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(path)
    return metadata["calibration_metrics"]


def evaluate(data, checkpoint):
    splits, manifest = verify_data(data)
    model, tokenizer, target, metadata = load(checkpoint)
    if manifest != metadata["data_manifest"]:
        raise ValueError("Evaluation data does not match the checkpoint manifest.")
    rows = splits["test"]
    scores = (
        [
            r["score"]
            for r in score_documents(
                model,
                tokenizer,
                rows,
                target,
                metadata["max_length"],
                normalization=metadata.get("normalization", "legacy"),
            )
        ]
        if metadata.get("scoring") == "window_mean_logit"
        else score_rows(
            model,
            tokenizer,
            rows,
            target,
            metadata["max_length"],
            normalization=metadata.get("normalization", "legacy"),
        )
    )
    result = {
        "split": "source-disjoint test",
        "dataset": manifest["dataset"],
        "metrics": evaluate_scores(
            scores, [r["label"] for r in rows], metadata["threshold"]
        ),
        "per_source": {},
    }
    for source in sorted({r["source"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["source"] == source]
        result["per_source"][source] = evaluate_scores(
            [scores[i] for i in indices],
            [rows[i]["label"] for i in indices],
            metadata["threshold"],
        )
    return result


def predict(text, loaded):
    model, tokenizer, target, metadata = loaded
    if len(text.split()) < 50:
        return {"verdict": "insufficient_text", "minimum_words": 50}
    if metadata.get("scoring") == "window_mean_logit":
        document = score_documents(
            model,
            tokenizer,
            [{"text": text}],
            target,
            metadata["max_length"],
            normalization=metadata.get("normalization", "legacy"),
        )[0]
        return {
            "verdict": "ai_signal"
            if document["score"] >= metadata["threshold"]
            else "below_ai_threshold",
            "ai_score": document["score"],
            "threshold": metadata["threshold"],
            "tokens_in_input": document["tokens"],
            "tokens_scored": document["tokens"],
            "truncated": False,
            "windows_analyzed": len(document["windows"]),
            "outside_evaluated_length_range": not metadata.get(
                "evaluated_min_words", 50
            )
            <= len(text.split())
            <= metadata.get("evaluated_max_words", 1000),
            "evaluated_word_range": [
                metadata.get("evaluated_min_words", 50),
                metadata.get("evaluated_max_words", 1000),
            ],
            "windows": document["windows"],
            "scoring": "window_mean_logit",
            "scope": metadata["scope"],
            "note": "Experimental document score, not an authorship probability. Window scores are uncalibrated; mixed authorship is not validated.",
        }
    text = normalize(text)
    count = len(tokenizer(text, add_special_tokens=True)["input_ids"])
    score = score_rows(
        model,
        tokenizer,
        [{"text": text}],
        target,
        metadata["max_length"],
        1,
        normalization=metadata.get("normalization", "legacy"),
    )[0]
    return {
        "verdict": "ai_signal"
        if score >= metadata["threshold"]
        else "below_ai_threshold",
        "ai_score": score,
        "threshold": metadata["threshold"],
        "tokens_in_input": count,
        "tokens_scored": min(count, metadata["max_length"]),
        "truncated": count > metadata["max_length"],
        "scope": metadata["scope"],
        "note": "Model score is not an authorship probability. Below threshold does not prove human authorship.",
    }
