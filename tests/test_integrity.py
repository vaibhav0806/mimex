import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mimex.data import normalize, verify_data
from mimex.metrics import calibrate, evaluate_scores
from mimex.raid import clean_audited_splits


class IntegrityTests(unittest.TestCase):
    def test_threshold_ties_cannot_exceed_budget(self):
        scores = [0.8] * 100 + [0.99]
        labels = [0] * 100 + [1]
        threshold = calibrate(scores, labels, 0.01)
        report = evaluate_scores(scores, labels, threshold)
        self.assertEqual(report["fp"], 0)
        self.assertEqual(report["tp"], 1)

    def test_zero_errors_still_have_uncertainty(self):
        report = evaluate_scores([0.1] * 100, [0] * 100, 0.5)
        self.assertGreater(report["fpr_95pct_wilson_interval"][1], 0.03)

    def test_confusion_counts(self):
        report = evaluate_scores([0.1, 0.9, 0.2, 0.8], [0, 0, 1, 1], 0.5)
        self.assertEqual([report[k] for k in ["tp", "tn", "fp", "fn"]], [1, 1, 1, 1])

    def test_normalization(self):
        self.assertEqual(
            normalize("I do n't agree , really ."), "I don't agree, really."
        )

    def test_cross_split_question_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest = {"splits": {}}
            for split in ["train", "calibration", "test"]:
                rows = [
                    {"text": f"{split} text {label}", "label": label, "group": "shared"}
                    for label in [0, 1]
                ]
                path = directory / f"{split}.jsonl"
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                manifest["splits"][split] = {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                }
            (directory / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "leakage"):
                verify_data(directory)

    def test_audit_removes_whole_pairs_and_preserves_training_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "clean"
            source.mkdir()
            manifest = {"splits": {}}
            for split in ["train", "development", "calibration", "test"]:
                rows = [
                    {
                        "id": f"{split}-{pair}-{label}",
                        "text": f"Unique {split} sample {pair} label {label}",
                        "group": f"{split}-{pair}",
                        "label": label,
                        "generator": "human" if label == 0 else "test-model",
                    }
                    for pair in [0, 1]
                    for label in [0, 1]
                ]
                path = source / f"{split}.jsonl"
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                manifest["splits"][split] = {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                }
            (source / "manifest.json").write_text(json.dumps(manifest))
            audit = {
                "method": "test fixture",
                "cross_split_matches": [{"id_b": "calibration-0-1"}],
            }
            clean_audited_splits(source, target, audit)
            self.assertEqual(
                (source / "train.jsonl").read_bytes(),
                (target / "train.jsonl").read_bytes(),
            )
            splits, _ = verify_data(target)
            self.assertEqual(
                {r["group"] for r in splits["calibration"]}, {"calibration-1"}
            )
            self.assertEqual({r["label"] for r in splits["calibration"]}, {0, 1})


if __name__ == "__main__":
    unittest.main()
