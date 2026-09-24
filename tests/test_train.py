import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from isl import features as F
from isl.model import load_recogniser
from tests.fakes import frame, make_raw
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

    def test_runs_are_summarised_as_mean_and_spread(self):
        summary = train.summarise_runs([{"top1": 0.9, "top3": 1.0, "macro_f1": 0.8},
                                        {"top1": 0.8, "top3": 1.0, "macro_f1": 0.6}])
        self.assertAlmostEqual(summary["top1_mean"], 0.85)
        self.assertAlmostEqual(summary["top1_std"], 0.0707, places=4)  # sample standard deviation
        self.assertAlmostEqual(summary["top3_std"], 0.0)
        self.assertAlmostEqual(summary["macro_f1_mean"], 0.7)


class TrainingDataTests(unittest.TestCase):
    def test_training_inputs_are_seen_through_random_closer_cameras(self):
        # a sign made at waist height (wrists 1.4 shoulder widths down) is out of view for the closer cameras
        clip = np.stack([F.frame_features(make_raw("both", wrist_y=340))[0]] * 20)
        rng = np.random.default_rng(0)
        outs = [train.training_sequence(clip, rng) for _ in range(200)]
        self.assertTrue(all(out.shape == (32, 184) for out in outs))
        no_hands = sum(not out[:, F.PRESENCE_SLICE].any() for out in outs)
        self.assertTrue(30 < no_hands < 120, no_hands)

    def test_validation_covers_the_full_view_and_closer_views(self):
        lowered = F.frame_features(make_raw("both", wrist_y=400))[0]
        clips = [np.stack([lowered] * 5 + [frame(True)] * 10 + [lowered] * 5)] * 2
        X, y = train.validation_set(clips, np.array([3, 4]))
        views = 1 + len(train.VAL_VIEWS)
        self.assertEqual(X.shape, (2 * views, 32, 184))
        np.testing.assert_array_equal(y, np.tile([3, 4], views))
        np.testing.assert_allclose(X[0], F.to_sequence(clips[0]))
        self.assertFalse(np.allclose(X[2], X[0]))  # a closer camera loses the lowered hands


class TrainSmokeTests(unittest.TestCase):
    def test_smoke_run_writes_model_labels_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("builtins.print"):
                train.main(["--smoke", "--seeds", "2", "--out", tmp])
            out = Path(tmp)
            for relative in ("models/isl50_bigru.keras", "models/labels.json", "reports/metrics.json",
                             "reports/confusion_matrix.png", "reports/results.md"):
                self.assertTrue((out / relative).exists(), relative)
            report = json.loads((out / "reports/metrics.json").read_text())
            self.assertEqual(set(report), {"counts", "baseline", "bigru", "bigru_seeds", "saved_seed",
                                           "inference_ms", "paper_top1", "epochs_trained"})
            runs = report["bigru_seeds"]["runs"]
            self.assertEqual([run["seed"] for run in runs], [train.SEED, train.SEED + 1])
            # the saved model is the run with the best validation accuracy, never chosen by test accuracy
            self.assertEqual(report["saved_seed"], max(runs, key=lambda run: run["val_accuracy"])["seed"])
            results = (out / "reports/results.md").read_text(encoding="utf-8")
            self.assertIn("| Model | Top-1 |", results)
            self.assertIn("mean ± std over 2 training runs", results)
            recogniser = load_recogniser(out / "models/isl50_bigru.keras", out / "models/labels.json")
            self.assertEqual(len(recogniser.predict(np.zeros((32, 184), np.float32))), 3)
