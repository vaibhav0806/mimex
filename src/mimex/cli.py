import argparse
import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def doctor() -> None:
    import torch
    import transformers

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "free_disk_gib": round(shutil.disk_usage(Path.cwd()).free / 1024**3, 1),
    }
    if platform.system() == "Darwin":
        info["chip"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        info["unified_memory_gib"] = round(
            int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 1024**3
        )
    print(json.dumps(info, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mimex detector experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="check Python, packages and GPU availability")
    benchmark = commands.add_parser(
        "benchmark", help="measure training throughput using synthetic tokens"
    )
    benchmark.add_argument(
        "--device", choices=["auto", "mps", "cuda", "cpu"], default="auto"
    )
    benchmark.add_argument("--batch-size", type=positive_int, default=2)
    benchmark.add_argument("--sequence-length", type=positive_int, default=256)
    benchmark.add_argument("--warmup", type=positive_int, default=2)
    benchmark.add_argument("--steps", type=positive_int, default=10)
    benchmark.add_argument("--dataset-size", type=positive_int, default=10_000)
    benchmark.add_argument("--epochs", type=positive_int, default=3)
    benchmark.add_argument("--output", type=Path)
    prepare = commands.add_parser(
        "prepare", help="download and split a reproducible HC3 subset"
    )
    prepare.add_argument("--data", type=Path, default=Path("data/hc3"))
    prepare.add_argument("--pairs", type=positive_int, default=1500)
    train = commands.add_parser(
        "train", help="fine-tune the detector and calibrate its threshold"
    )
    train.add_argument("--data", type=Path, default=Path("data/hc3"))
    train.add_argument("--output", type=Path, default=Path("checkpoints/hc3"))
    train.add_argument("--epochs", type=positive_int, default=2)
    train.add_argument("--batch-size", type=positive_int, default=2)
    train.add_argument("--full-document", action="store_true")
    train.add_argument("--mine-hard", action="store_true")
    train.add_argument("--initial-checkpoint", type=Path)
    train.add_argument(
        "--normalization", choices=["legacy", "english_casefold_v1"], default="legacy"
    )
    train.add_argument("--learning-rate", type=float, default=2e-5)
    raid = commands.add_parser(
        "prepare-raid", help="prepare source-disjoint, multi-generator RAID pairs"
    )
    raid.add_argument("--raw", type=Path, default=Path("data/raw/raid-train-none.csv"))
    raid.add_argument("--data", type=Path, default=Path("data/raid"))
    v3 = commands.add_parser(
        "prepare-v3",
        help="prepare modern-generator grouped data with training-only replay",
    )
    v3.add_argument("--raw", type=Path, default=Path("data/raw"))
    v3.add_argument("--output", type=Path, default=Path("data/v3"))
    external = commands.add_parser(
        "prepare-v3-external", help="prepare the independent parliamentary test"
    )
    external.add_argument(
        "--output", type=Path, default=Path("data/v3-parliament.jsonl")
    )
    assess = commands.add_parser(
        "assess", help="evaluate a fixed checkpoint on a RAID partition"
    )
    assess.add_argument("--data", type=Path, default=Path("data/raid"))
    assess.add_argument("--checkpoint", type=Path, default=Path("checkpoints/hc3"))
    assess.add_argument(
        "--split", choices=["development", "test"], default="development"
    )
    assess.add_argument("--output", type=Path, required=True)
    assess.add_argument(
        "--recalibrate",
        action="store_true",
        help="select threshold on this dataset's calibration split",
    )
    audit = commands.add_parser(
        "audit-raid",
        help="audit near duplicates and optionally remove later split pairs",
    )
    audit.add_argument("--data", type=Path, default=Path("data/raid"))
    audit.add_argument(
        "--output", type=Path, default=Path("runs/raid-duplicate-audit.json")
    )
    audit.add_argument("--clean-output", type=Path)
    recalibrate = commands.add_parser(
        "calibrate",
        help="recalibrate a checkpoint while preserving its training partition",
    )
    recalibrate.add_argument("--data", type=Path, default=Path("data/raid-clean"))
    recalibrate.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/raid-v2")
    )
    collect = commands.add_parser(
        "collect-paraphrases", help="stream selected held-out RAID paraphrases"
    )
    collect.add_argument("--data", type=Path, default=Path("data/raid"))
    collect.add_argument(
        "--output", type=Path, default=Path("data/raid-paraphrases.jsonl")
    )
    robust = commands.add_parser(
        "robustness", help="paired formatting and paraphrase checks"
    )
    robust.add_argument("--data", type=Path, default=Path("data/raid-clean"))
    robust.add_argument("--checkpoint", type=Path, default=Path("checkpoints/raid-v2"))
    robust.add_argument(
        "--paraphrases", type=Path, default=Path("data/raid-paraphrases.jsonl")
    )
    robust.add_argument("--output", type=Path, default=Path("runs/robustness.json"))
    for name in ["evaluate", "predict", "serve"]:
        command = commands.add_parser(name)
        default_checkpoint = (
            Path("checkpoints/active")
            if name != "evaluate" and Path("checkpoints/active").exists()
            else Path("checkpoints/hc3")
        )
        command.add_argument("--checkpoint", type=Path, default=default_checkpoint)
        if name == "evaluate":
            command.add_argument("--data", type=Path, default=Path("data/hc3"))
            command.add_argument(
                "--output", type=Path, default=Path("runs/evaluation.json")
            )
        elif name == "predict":
            command.add_argument(
                "--file", type=Path, help="read a text file; otherwise read stdin"
            )
        else:
            command.add_argument("--port", type=positive_int, default=7860)
    args = parser.parse_args()
    if args.command == "prepare-v3":
        from mimex.v3 import prepare

        report = prepare(args.raw, args.output)
        print(
            json.dumps(
                {
                    "splits": {k: v["rows"] for k, v in report["splits"].items()},
                    "training_paraphrases": report["training_paraphrases"],
                    "rejected": report["rejected"],
                },
                indent=2,
            )
        )
        return
    if args.command == "prepare-v3-external":
        from mimex.v3 import prepare_external

        print(json.dumps(prepare_external(args.output), indent=2))
        return
    if args.command == "calibrate":
        from mimex.detector import recalibrate_checkpoint

        print(json.dumps(recalibrate_checkpoint(args.data, args.checkpoint), indent=2))
        return
    if args.command == "audit-raid":
        from mimex.raid import audit_duplicates, clean_audited_splits

        report = audit_duplicates(args.data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if args.clean_output:
            clean_audited_splits(args.data, args.clean_output, report)
        print(
            json.dumps(
                {
                    "documents": report["documents"],
                    "cross_split_matches": len(report["cross_split_matches"]),
                }
            )
        )
        return
    if args.command == "collect-paraphrases":
        from mimex.raid import collect_paraphrases

        report = collect_paraphrases(args.data, args.output)
        print(json.dumps({k: report[k] for k in ["requested", "found", "sha256"]}))
        return
    if args.command == "robustness":
        from mimex.raid import robustness

        report = robustness(args.data, args.checkpoint, args.paraphrases, args.output)
        print(
            json.dumps(
                {
                    name: value["metrics"]
                    for name, value in report["formatting"].items()
                },
                indent=2,
            )
        )
        if "paraphrase" in report:
            print(
                json.dumps(
                    {k: report["paraphrase"][k] for k in ["original", "paraphrased"]},
                    indent=2,
                )
            )
        return
    if args.command == "prepare-raid":
        from mimex.raid import prepare_raid

        print(json.dumps(prepare_raid(args.raw, args.data), indent=2))
        return
    if args.command == "assess":
        from mimex.raid import assess

        result = assess(
            args.data, args.checkpoint, args.split, args.output, args.recalibrate
        )
        print(json.dumps(result["metrics"], indent=2))
        return
    if args.command in {"prepare", "train", "evaluate", "predict", "serve"}:
        import sys

        from mimex import data, detector

        if args.command == "prepare":
            result = data.prepare(args.data, args.pairs)
        elif args.command == "train":
            result = detector.train(
                args.data,
                args.output,
                args.epochs,
                args.batch_size,
                full_document=args.full_document,
                mine_hard=args.mine_hard,
                initial_checkpoint=args.initial_checkpoint,
                normalization=args.normalization,
                learning_rate=args.learning_rate,
            )
        elif args.command == "evaluate":
            result = detector.evaluate(args.data, args.checkpoint)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        elif args.command == "predict":
            text = args.file.read_text() if args.file else sys.stdin.read()
            result = detector.predict(text, detector.load(args.checkpoint))
        else:
            from mimex.server import serve

            serve(args.checkpoint, args.port)
            return
        print(json.dumps(result, indent=2))
        return
    if args.command == "doctor":
        doctor()
        return

    from mimex.benchmark import run_benchmark

    try:
        result = run_benchmark(
            device=args.device,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            warmup=args.warmup,
            steps=args.steps,
            dataset_size=args.dataset_size,
            epochs=args.epochs,
        )
    except ValueError as error:
        parser.error(str(error))
    output = args.output or Path("runs") / (
        datetime.now(UTC).strftime("benchmark-%Y%m%dT%H%M%S%fZ.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nSaved benchmark: {output.resolve()}")


if __name__ == "__main__":
    main()
