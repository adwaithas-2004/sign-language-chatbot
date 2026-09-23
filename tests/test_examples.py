import tempfile
import unittest
from pathlib import Path
from unittest import mock

from isl import examples


class ExampleTests(unittest.TestCase):
    def test_words_and_display_names_find_their_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            for key in ("thankyou", "biglarge", "hello"):
                (Path(tmp) / f"{key}.MOV").write_bytes(b"")
            self.assertEqual(examples.example_path("thank you", Path(tmp)).name, "thankyou.MOV")
            self.assertEqual(examples.example_path("Big", Path(tmp)).name, "biglarge.MOV")
            self.assertEqual(examples.example_path("HELLO", Path(tmp)).name, "hello.MOV")
            self.assertIsNone(examples.example_path("penguin", Path(tmp)))

    def test_unknown_word_explains(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(examples, "EXAMPLES_DIR", Path(tmp)):
            with self.assertRaises(SystemExit) as caught:
                examples.main(["penguin"])
        self.assertIn("penguin", str(caught.exception))
