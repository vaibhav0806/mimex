import math
import platform
import statistics
import time
from datetime import UTC, datetime

import torch
import transformers
from transformers import AutoModelForSequenceClassification

MODEL_ID = "answerdotai/ModernBERT-base"
MODEL_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"


def select_device(requested: str) -> torch.device:
    available = {
        "cpu": True,
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }
    if requested == "auto":
        requested = next(name for name in ["mps", "cuda", "cpu"] if available[name])
    if not available[requested]:
        raise ValueError(f"The requested device ({requested}) is unavailable.")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def run_benchmark(
    *,
    device: str,
    batch_size: int,
    sequence_length: int,
    warmup: int,
    steps: int,
    dataset_size: int,
    epochs: int,
) -> dict:
    target = select_device(device)
    torch.manual_seed(42)
    print(
        f"Loading {MODEL_ID} on {target}. First run downloads the model weights.",
        flush=True,
    )
    revision = MODEL_REVISION
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        revision=revision,
        num_labels=2,
        attn_implementation="sdpa",
        trust_remote_code=False,
        use_safetensors=True,
    )
    if not 2 <= sequence_length <= model.config.max_position_embeddings:
        raise ValueError(
            f"Sequence length must be between 2 and {model.config.max_position_embeddings}."
        )
    model.to(device=target, dtype=torch.float32)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, sequence_length))
    input_ids[:, 0] = model.config.cls_token_id
    input_ids[:, -1] = model.config.sep_token_id
    batch = {
        "input_ids": input_ids.to(target),
        "attention_mask": torch.ones_like(input_ids).to(target),
        "labels": (torch.arange(batch_size) % 2).to(target),
    }
    durations = []
    print(
        "Synthetic tokens and arbitrary labels: this measures speed, not accuracy.",
        flush=True,
    )
    for step in range(warmup + steps):
        synchronize(target)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        if not torch.isfinite(loss).item():
            raise RuntimeError("Training produced a non-finite loss.")
        loss.backward()
        optimizer.step()
        synchronize(target)
        elapsed = time.perf_counter() - started
        if step >= warmup:
            durations.append(elapsed)
        phase = "warmup" if step < warmup else "measured"
        print(f"  {phase} step {step + 1}/{warmup + steps}: {elapsed:.3f}s", flush=True)

    mean_step = statistics.mean(durations)
    epoch_seconds = math.ceil(dataset_size / batch_size) * mean_step
    result = {
        "kind": "synthetic-training-benchmark",
        "is_trained_detector": False,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model": MODEL_ID,
        "model_revision": revision,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(target),
        "dtype": "float32",
        "attention": "sdpa",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "seconds_per_step": round(mean_step, 4),
        "step_seconds": [round(value, 4) for value in durations],
        "examples_per_second": round(batch_size / mean_step, 2),
        "tokens_per_second": round(batch_size * sequence_length / mean_step, 2),
        "projection": {
            "examples": dataset_size,
            "epochs": epochs,
            "minutes_per_epoch": round(epoch_seconds / 60, 1),
            "total_training_hours": round(epoch_seconds * epochs / 3600, 2),
            "excludes": [
                "model loading",
                "tokenization and data loading",
                "validation",
                "checkpoint writing",
                "changes in throughput during longer runs",
            ],
        },
    }
    if target.type == "mps":
        result["mps_active_gib_at_end"] = round(
            torch.mps.current_allocated_memory() / 1024**3, 3
        )
        result["mps_driver_gib_at_end"] = round(
            torch.mps.driver_allocated_memory() / 1024**3, 3
        )
    return result
