import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from isl.model import load_recogniser
from tests.test_prepare import load_script

train = load_script("train")


class MetricsTests(unittest.TestCase):
    def test_metrics_only_score_words_in_the_test_set(self):
        probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.5, 0.4, 0.1]])
        result = train.metrics(probabilities, np.array([0, 1, 1]), ["a", "b", "c"])
        self.assertAlmostEqual(result["top1"], 1 / 3)
        self.assertAlmostEqual(result["top3"], 1.0)
        self.assertEqual(result["per_word"], {"a": 1.0, "b": 0.0})
        self.assertAlmostEqual(result["macro_f1"], 1 / 3, places=3)  # F1 a = 2/3, b = 0; "c" isn't averaged in


class TrainSmokeTests(unittest.TestCase):
    def test_smoke_run_writes_model_labels_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("builtins.print"):
                train.main(["--smoke", "--out", tmp])
            out = Path(tmp)
            for relative in ("models/isl50_bigru.keras", "models/labels.json", "reports/metrics.json",
                             "reports/confusion_matrix.png", "reports/results.md"):
                self.assertTrue((out / relative).exists(), relative)
            report = json.loads((out / "reports/metrics.json").read_text())
            self.assertEqual(set(report), {"counts", "baseline", "bigru", "inference_ms", "paper_top1",
                                           "epochs_trained"})
            self.assertIn("| Model | Top-1 |", (out / "reports/results.md").read_text(encoding="utf-8"))
            recogniser = load_recogniser(out / "models/isl50_bigru.keras", out / "models/labels.json")
            self.assertEqual(len(recogniser.predict(np.zeros((32, 184), np.float32))), 3)
