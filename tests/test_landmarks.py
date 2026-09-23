import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from isl import landmarks


class ModelFileTests(unittest.TestCase):
    def test_missing_files_are_downloaded_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            urlopen = mock.Mock(side_effect=lambda url, timeout: io.BytesIO(b"model:" + url.encode()))
            paths = landmarks.ensure_model_files(Path(tmp), urlopen=urlopen)
            self.assertEqual(set(paths), set(landmarks.MODEL_URLS))
            self.assertTrue(all(path.read_bytes().startswith(b"model:") for path in paths.values()))
            landmarks.ensure_model_files(Path(tmp), urlopen=urlopen)
            self.assertEqual(urlopen.call_count, 2)

    def test_failed_download_says_where_to_get_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as caught:
                landmarks.ensure_model_files(Path(tmp), urlopen=mock.Mock(side_effect=OSError("offline")))
            self.assertIn("storage.googleapis.com", str(caught.exception))


@unittest.skipUnless(all((landmarks.MODELS_DIR / name).exists() for name in landmarks.MODEL_URLS),
                     "MediaPipe model files not downloaded yet")
class ExtractorTests(unittest.TestCase):
    def test_empty_frame_has_no_person_and_is_downscaled(self):
        with landmarks.LandmarkExtractor() as extractor:
            for i in range(3):
                raw = extractor.extract(np.zeros((720, 1280, 3), np.uint8), i * 33)
        self.assertIsNone(raw.pose)
        self.assertEqual(raw.hands, [])
        self.assertEqual((raw.width, raw.height), (640, 360))

    def test_repeated_timestamps_are_accepted(self):
        with landmarks.LandmarkExtractor() as extractor:
            extractor.extract(np.zeros((480, 640, 3), np.uint8), 100)
            extractor.extract(np.zeros((480, 640, 3), np.uint8), 100)
