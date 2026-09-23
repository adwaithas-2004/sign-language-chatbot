import importlib.util
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

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


class ParallelDownloadTests(unittest.TestCase):
    def test_downloads_overlap_but_videos_are_processed_in_order(self):
        # Zenodo gives each connection ~0.3 MB/s, so downloads must run in parallel
        samples = prepare.all_samples()[:12]
        active, peak, lock, processed = [0], [0], threading.Lock(), []

        def fake_fetch(member, dest):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            try:
                time.sleep(0.05)
                if member == "bad":
                    raise OSError("connection reset")
                Path(dest).write_bytes(b"video")
            finally:
                with lock:
                    active[0] -= 1

        def fake_landmarks(video):
            processed.append(Path(video).name)
            return np.zeros((5, 184), np.float32), 25.0

        index = {sample.path: sample.path for sample in samples}  # stand-ins for zip members
        index[samples[3].path] = "bad"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with mock.patch.multiple(prepare, LANDMARKS_DIR=tmp / "landmarks", EXAMPLES_DIR=tmp / "examples",
                                     TMP_DIR=tmp / "tmp", FAILED_PATH=tmp / "failed.txt",
                                     all_samples=lambda: samples, video_landmarks=fake_landmarks), \
                    mock.patch.object(prepare.include_data, "build_index", return_value=index), \
                    mock.patch.object(prepare.include_data, "fetch_video", side_effect=fake_fetch), \
                    mock.patch("builtins.print"):
                prepare.main([])
                self.assertEqual(len(list((tmp / "landmarks").rglob("*.npz"))), 11)
                self.assertIn(samples[3].path, (tmp / "failed.txt").read_text(encoding="utf-8"))
                self.assertEqual(list((tmp / "tmp").iterdir()), [])  # every downloaded video was cleaned up
                expected = [prepare.temp_video_path(s).name for i, s in enumerate(samples) if i != 3]
        self.assertGreater(peak[0], 1)
        self.assertLessEqual(peak[0], prepare.DOWNLOAD_WORKERS)
        self.assertEqual(processed, expected)
