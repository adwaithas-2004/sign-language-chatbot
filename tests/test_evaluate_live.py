import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests.fakes import frame
from tests.test_prepare import load_script

evaluate_live = load_script("evaluate_live")


def save_clip(root, split, key, frames):
    path = Path(root) / split / key / f"{key}_{len(list((Path(root) / split).glob('*/*.npz')))}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=np.stack(frames), fps=25.0, key=key)


class LiveReplayTests(unittest.TestCase):
    def test_a_clip_with_a_raised_sign_produces_one_segment(self):
        clip = np.stack([frame(False)] * 10 + [frame(True)] * 20 + [frame(False)] * 5)
        signs = evaluate_live.replay(clip, 25.0)
        self.assertEqual(len(signs), 1)
        self.assertEqual(signs[0].shape, (32, 184))

    def test_a_clip_that_never_raises_the_hands_produces_nothing(self):
        self.assertEqual(evaluate_live.replay(np.stack([frame(False)] * 30), 25.0), [])

    def test_live_accuracy_and_trigger_rates(self):
        recogniser = mock.Mock(keys=["hello", "bank"], display=["hello", "bank"])
        recogniser.predict.return_value = [("hello", 0.9), ("bank", 0.1)]
        signed = [frame(False)] * 10 + [frame(True)] * 20 + [frame(False)] * 5
        with tempfile.TemporaryDirectory() as tmp:
            save_clip(tmp, "test", "hello", signed)
            save_clip(tmp, "test", "bank", [frame(False)] * 30)  # never starts a sign
            save_clip(tmp, "train", "bank", signed)
            result = evaluate_live.evaluate(tmp, recogniser)
        self.assertEqual((result["test_clips"], result["no_sign"], result["several_signs"]), (2, 1, 0))
        self.assertAlmostEqual(result["top1"], 0.5)
        self.assertEqual(result["trigger_rate"], {"bank": 0.5, "hello": 1.0})
        self.assertIn("bank (50%)", evaluate_live.live_markdown(result, recogniser))
