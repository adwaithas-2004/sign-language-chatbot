import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from isl.model import build_model, load_recogniser, save_labels


class ModelTests(unittest.TestCase):
    def test_output_is_a_probability_per_class(self):
        model = build_model(5)
        out = np.asarray(model(np.zeros((2, 32, 184), np.float32), training=False))
        self.assertEqual(out.shape, (2, 5))
        np.testing.assert_allclose(out.sum(axis=1), 1, rtol=1e-5)

    def test_save_load_and_predict_top_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            build_model(4).save(tmp / "m.keras")
            save_labels(tmp / "labels.json", ["a", "b", "c", "d"], ["A", "B", "C", "D"])
            recogniser = load_recogniser(tmp / "m.keras", tmp / "labels.json")
            guesses = recogniser.predict(np.zeros((32, 184), np.float32))  # an empty sign must not crash
        self.assertEqual(len(guesses), 3)
        probabilities = [p for _, p in guesses]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))
        self.assertTrue({word for word, _ in guesses} <= {"A", "B", "C", "D"})

    def test_missing_model_explains_how_to_get_one(self):
        with self.assertRaises(SystemExit) as caught:
            load_recogniser(Path("missing.keras"), Path("missing.json"))
        self.assertIn("scripts/train.py", str(caught.exception))

    def test_feature_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            build_model(2).save(tmp / "m.keras")
            (tmp / "labels.json").write_text(json.dumps({"feature_version": 0, "sequence_length": 32,
                                                          "num_features": 184, "keys": ["a", "b"],
                                                          "display": ["a", "b"]}))
            with self.assertRaises(SystemExit) as caught:
                load_recogniser(tmp / "m.keras", tmp / "labels.json")
        self.assertIn("feature_version", str(caught.exception))
