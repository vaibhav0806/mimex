import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mimex.data import read_rows
from scripts.evaluate_external_v3 import make_row, measure


class ExternalEvaluationTests(unittest.TestCase):
    def test_unicode_line_separators_stay_inside_json_strings(self):
        rows = [{"text": "first\u2028second\u0085third"}, {"text": "next document"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            )
            self.assertEqual(read_rows(path), rows)

    def test_ai_edited_humans_are_not_counted_as_binary_errors(self):
        rows = [
            make_row("test", "h", "original", "human", 0, "essay", "human"),
            make_row("test", "a", "generated", "ai", 1, "essay", "model"),
            make_row(
                "test",
                "m",
                "polished",
                "mixed",
                None,
                "essay",
                "mixed",
                "ai_edited_human",
            ),
        ]
        with (
            patch("scripts.evaluate_external_v3.scores", return_value=[0.1, 0.9, 0.9]),
            patch("scripts.evaluate_external_v3.log"),
        ):
            report = measure((None, None, None, {"threshold": 0.5}), rows)
        self.assertEqual(report["metrics"]["examples"], 2)
        self.assertEqual(report["metrics"]["fp"], 0)
        self.assertEqual(report["metrics"]["tp"], 1)
        domain = report["subgroups"]["domain"]["essay"]
        self.assertEqual(domain["ai_edited_human_examples"], 1)
        self.assertEqual(domain["flagged"], 1)
        self.assertEqual(domain["binary_metrics"]["examples"], 2)
        self.assertIsNone(report["predictions"][2]["label"])


if __name__ == "__main__":
    unittest.main()
