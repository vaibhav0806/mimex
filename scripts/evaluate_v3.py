"""One frozen development decision followed by untouched final evaluations."""

import gc
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from mimex.data import read_rows, verify_data
from mimex.detector import load, predict
from mimex.metrics import calibrate, evaluate_scores
from mimex.v3 import NORMALIZATION, assess_partition, scores


def log(message, result=None):
    print(
        datetime.now(UTC).isoformat(),
        message,
        json.dumps(result) if result is not None else "",
        flush=True,
    )


def save(name, value):
    Path("runs").mkdir(exist_ok=True)
    Path(f"runs/{name}.json").write_text(json.dumps(value, indent=2) + "\n")


def clear():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def brief(report):
    return {
        "clean": report["clean"]["overall"],
        "paraphrase": report["paraphrase"]["rewritten"],
    }


def external_check(loaded, threshold):
    path = Path("data/v3-parliament.jsonl")
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("External test data changed.")
    rows = read_rows(path)
    predictions = scores(loaded, rows)
    return {
        "provenance": manifest,
        "metrics": evaluate_scores(predictions, [r["label"] for r in rows], threshold),
        "predictions": [
            {"id": r["id"], "label": r["label"], "score": s}
            for r, s in zip(rows, predictions, strict=True)
        ],
    }


def main():
    splits, manifest = verify_data("data/v3")
    calibration_humans = [r for r in splits["calibration"] if r["label"] == 0]
    log("V2 comparison calibration and development")
    loaded = load("checkpoints/raid-v2")
    baseline_threshold = calibrate(
        scores(loaded, calibration_humans), [0] * len(calibration_humans)
    )
    baseline_dev = assess_partition(loaded, splits["development"], baseline_threshold)
    save("v3-baseline-development", baseline_dev)
    log("Baseline development", brief(baseline_dev))
    del loaded
    clear()

    log("V3 development")
    loaded = load("checkpoints/v3")
    metadata = loaded[3]
    if (
        metadata["data_manifest"] != manifest
        or metadata["normalization"] != NORMALIZATION
    ):
        raise ValueError("V3 checkpoint provenance mismatch.")
    threshold = metadata["threshold"]
    candidate_dev = assess_partition(loaded, splits["development"], threshold)
    save("v3-development", candidate_dev)
    log("Candidate development", brief(candidate_dev))
    del loaded
    clear()
    clean_gain = (
        candidate_dev["clean"]["overall"]["ai_detection_rate"]
        - baseline_dev["clean"]["overall"]["ai_detection_rate"]
    )
    rewrite_gain = (
        candidate_dev["paraphrase"]["rewritten"]["ai_detection_rate"]
        - baseline_dev["paraphrase"]["rewritten"]["ai_detection_rate"]
    )
    false_positives = [
        candidate_dev["clean"]["overall"]["false_positive_rate"],
        candidate_dev["uppercase_humans"]["false_positive_rate"],
        candidate_dev["zero_width_spaces_humans"]["false_positive_rate"],
    ]
    promote = (
        clean_gain >= 0.05 and rewrite_gain >= 0.05 and max(false_positives) <= 0.03
    )
    with Path("checkpoints/v3/model.safetensors").open("rb") as f:
        weights_hash = hashlib.file_digest(f, "sha256").hexdigest()
    selection = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "promote_candidate": promote,
        "clean_detection_gain": clean_gain,
        "paraphrase_detection_gain": rewrite_gain,
        "candidate_development_fprs": false_positives,
        "baseline_threshold": baseline_threshold,
        "candidate_threshold": threshold,
        "criterion": "clean and paraphrase development detection each improve >=5pp; FPR <=3% for all tested human conditions",
        "candidate_weights_sha256": weights_hash,
        "candidate_metadata_sha256": hashlib.sha256(
            Path("checkpoints/v3/mimex.json").read_bytes()
        ).hexdigest(),
    }
    save("v3-selection", selection)
    log("Selection frozen before final tests", selection)

    final = {}
    for name, checkpoint, cutoff in [
        ("v2", "checkpoints/raid-v2", baseline_threshold),
        ("v3", "checkpoints/v3", threshold),
    ]:
        log(f"Final test: {name}")
        loaded = load(checkpoint)
        report = assess_partition(loaded, splits["test"], cutoff)
        save(f"{name}-v3-final", report)
        log(f"{name} final", brief(report))
        external = external_check(loaded, cutoff)
        save(f"{name}-v3-external", external)
        log(f"{name} independent parliamentary test", external["metrics"])
        diagnostic_path = Path("data/fresh-assistant-checks.jsonl")
        diagnostics = [
            {"id": r["id"], "prediction": predict(r["text"], loaded)}
            for r in (read_rows(diagnostic_path) if diagnostic_path.exists() else [])
        ]
        if not diagnostic_path.exists():
            log(f"{name}: optional consumed qualitative checks unavailable; skipped")
        save(f"{name}-v3-consumed-smoke", diagnostics)
        final[name] = {
            "clean": report["clean"],
            "paraphrase": {
                k: report["paraphrase"][k] for k in ["original", "rewritten"]
            },
            "formatting": {k: v["metrics"] for k, v in report["formatting"].items()},
            "external": external["metrics"],
            "consumed_smoke_flagged": sum(
                r["prediction"]["verdict"] == "ai_signal" for r in diagnostics
            )
            if diagnostics
            else None,
        }
        del loaded
        clear()
    save(
        "v3-comparison",
        {
            "selection": selection,
            "development": {"v2": brief(baseline_dev), "v3": brief(candidate_dev)},
            "final": final,
        },
    )
    log("COMPLETE")


if __name__ == "__main__":
    main()
