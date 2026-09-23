import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from isl.model import Recogniser, build_model, load_recogniser, save_labels


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

    def test_prediction_is_fast_enough_for_the_live_app(self):
        # Calling the model eagerly ran the GRUs step by step: ~300 ms per sign, freezing the video
        model = build_model(50)
        recogniser = Recogniser(model, [str(i) for i in range(50)], [str(i) for i in range(50)])
        sequence = np.random.default_rng(0).normal(size=(32, 184)).astype(np.float32)
        recogniser.predict(sequence)  # warm-up
        start = time.perf_counter()
        for _ in range(10):
            guesses = recogniser.predict(sequence)
        self.assertLess((time.perf_counter() - start) / 10, 0.05)
        expected = np.asarray(model(sequence[None], training=False))[0]
        self.assertAlmostEqual(guesses[0][1], float(expected.max()), places=5)

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
