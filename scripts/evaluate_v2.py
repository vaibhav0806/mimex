import gc
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from mimex.detector import recalibrate_checkpoint
from mimex.raid import assess, robustness


def clear():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def log(message, result=None):
    print(
        datetime.now(UTC).isoformat(),
        message,
        json.dumps(result) if result else "",
        flush=True,
    )


log("Re-evaluating baseline on cleaned development data")
baseline = assess(
    "data/raid-clean",
    "checkpoints/hc3",
    "development",
    "runs/baseline-clean-development.json",
    True,
)
log("Baseline development", baseline["metrics"]["overall"])
clear()
log("Recalibrating candidate using clean calibration data only")
calibration = recalibrate_checkpoint("data/raid-clean", "checkpoints/raid-v2")
log("Clean calibration", calibration)
clear()
log("Evaluating candidate on development")
candidate = assess(
    "data/raid-clean", "checkpoints/raid-v2", "development", "runs/v2-development.json"
)
log("Candidate development", candidate["metrics"]["overall"])
clear()
b, c = baseline["metrics"]["overall"], candidate["metrics"]["overall"]
promote = (
    c["ai_detection_rate"] > b["ai_detection_rate"] and c["false_positive_rate"] <= 0.03
)
with open("checkpoints/raid-v2/model.safetensors", "rb") as f:
    weights_hash = hashlib.file_digest(f, "sha256").hexdigest()
selection = {
    "frozen_at": datetime.now(UTC).isoformat(),
    "promote_candidate": promote,
    "criterion": "development AI detection > recalibrated baseline and development FPR <=3%",
    "baseline_development": b,
    "candidate_development": c,
    "candidate_weights_sha256": weights_hash,
    "candidate_metadata_sha256": hashlib.sha256(
        Path("checkpoints/raid-v2/mimex.json").read_bytes()
    ).hexdigest(),
}
Path("runs/selection.json").write_text(json.dumps(selection, indent=2) + "\n")
log("Selection frozen before final test", {"promote_candidate": promote})
log("Final test: baseline")
bt = assess(
    "data/raid-clean", "checkpoints/hc3", "test", "runs/baseline-final.json", True
)
log("Baseline final", bt["metrics"]["overall"])
clear()
log("Final test: candidate")
ct = assess("data/raid-clean", "checkpoints/raid-v2", "test", "runs/v2-final.json")
log("Candidate final", ct["metrics"]["overall"])
clear()
log("Paired robustness checks")
r = robustness(
    "data/raid-clean",
    "checkpoints/raid-v2",
    "data/raid-paraphrases.jsonl",
    "runs/robustness.json",
)
log("Paraphrase result", {k: r["paraphrase"][k] for k in ["original", "paraphrased"]})
summary = {
    "selection": selection,
    "baseline_original": bt["original_threshold_metrics"],
    "baseline_recalibrated": bt["metrics"],
    "candidate": ct["metrics"],
    "formatting": {k: v["metrics"] for k, v in r["formatting"].items()},
    "paraphrase": {k: r["paraphrase"][k] for k in ["original", "paraphrased"]},
}
Path("runs/comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
log("COMPLETE")
