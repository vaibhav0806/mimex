import unittest

import torch

from mimex.data import normalize_for_model
from mimex.detector import MAX_DOCUMENT_TOKENS, score_documents, sigmoid, window_ranges


class ToyTokenizer:
    """Deterministic tokens for checking inference coverage without model downloads."""

    def num_special_tokens_to_add(self, pair):
        return 0

    def __call__(self, text, **kwargs):
        tokens = [2 if word == "signal" else 1 for word in text.split()]
        if not kwargs.get("return_overflowing_tokens"):
            return {"input_ids": tokens}
        width, overlap = kwargs["max_length"], kwargs["stride"]
        windows = []
        while tokens:
            part = tokens[:width]
            windows.append(part + [0] * (width - len(part)))
            if len(tokens) <= width:
                break
            tokens = tokens[width - overlap :]
        return {
            "input_ids": torch.tensor(windows),
            "overflow_to_sample_mapping": torch.zeros(len(windows)),
        }


class ToyClassifier(torch.nn.Module):
    def forward(self, input_ids):
        from types import SimpleNamespace

        margin = torch.where((input_ids == 2).any(dim=1), 4.0, -4.0)
        return SimpleNamespace(
            logits=torch.stack([torch.zeros_like(margin), margin], dim=1)
        )


class DocumentTests(unittest.TestCase):
    def test_v3_canonicalizes_formatting_for_both_classes(self):
        for word in ["ordinary", "signal"]:
            text = (word + " ") * 300
            variants = [text, text.upper(), text.replace(" ", "\u200b ")]
            results = score_documents(
                ToyClassifier(),
                ToyTokenizer(),
                [{"text": t} for t in variants],
                "cpu",
                normalization="english_casefold_v1",
            )
            self.assertEqual([r["score"] for r in results], [results[0]["score"]] * 3)

    def test_legacy_normalization_remains_case_sensitive(self):
        self.assertEqual(normalize_for_model("Hello WORLD", "legacy"), "Hello WORLD")
        self.assertEqual(
            normalize_for_model("Hello WORLD", "english_casefold_v1"), "hello world"
        )
        with self.assertRaisesRegex(ValueError, "Unknown normalization"):
            normalize_for_model("text", "future-version")

    def test_tail_changes_score_even_when_prefix_identical(self):
        prefix = "ordinary " * 400
        rows = [
            {"text": prefix + "ordinary " * 200},
            {"text": prefix + "signal " * 200},
        ]
        results = score_documents(ToyClassifier(), ToyTokenizer(), rows, "cpu")
        self.assertGreater(results[1]["score"], results[0]["score"])
        self.assertEqual(results[1]["tokens"], 600)
        self.assertEqual(results[1]["windows"][-1]["end_token"], 600)

    def test_windows_cover_every_token_without_gaps(self):
        for count in [1, 254, 255, 444, 445, MAX_DOCUMENT_TOKENS]:
            covered = set()
            for start, end in window_ranges(count):
                self.assertLessEqual(end - start, 254)
                covered.update(range(start, end))
            self.assertEqual(covered, set(range(count)))

    def test_oversized_document_is_rejected_not_truncated(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            score_documents(
                ToyClassifier(),
                ToyTokenizer(),
                [{"text": "word " * (MAX_DOCUMENT_TOKENS + 1)}],
                "cpu",
            )

    def test_extreme_logits_do_not_overflow(self):
        self.assertEqual(sigmoid(1000), 1)
        self.assertEqual(sigmoid(-1000), 0)


if __name__ == "__main__":
    unittest.main()
