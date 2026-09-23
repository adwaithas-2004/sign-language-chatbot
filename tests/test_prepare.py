import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_script("prepare_include50")


class PrepareTests(unittest.TestCase):
    def test_test_videos_come_first_so_examples_are_test_videos(self):
        samples = prepare.all_samples()
        self.assertEqual(len(samples), 958)
        self.assertEqual((samples[0].split, samples[-1].split), ("test", "train"))

    def test_pending_skips_finished_videos(self):
        samples = prepare.all_samples()[:3]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(prepare, "LANDMARKS_DIR", Path(tmp)):
            done = prepare.landmarks_path(samples[0])
            done.parent.mkdir(parents=True)
            done.write_bytes(b"")
            half_written = prepare.landmarks_path(samples[1])  # a run stopped while saving this one
            half_written.parent.mkdir(parents=True, exist_ok=True)
            half_written.with_name(half_written.name + ".part").write_bytes(b"")
            self.assertEqual(prepare.pending(samples), samples[1:])

    def test_landmarks_path_layout(self):
        sample = prepare.all_samples()[0]
        path = prepare.landmarks_path(sample)
        self.assertEqual(path.parts[-3:-1], (sample.split, sample.key))
        self.assertEqual(path.suffix, ".npz")

    def test_unreadable_video_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.MOV"
            bad.write_bytes(b"not a video")
            with mock.patch.object(prepare, "LandmarkExtractor"), self.assertRaises(ValueError):
                prepare.video_landmarks(bad)
