import os
import threading
import time
import unittest
from unittest import mock

import cv2
import numpy as np
from groq import APIConnectionError

import signlan
from isl.features import RawLandmarks
from tests.fakes import make_hand, make_raw


def fake_client(reply="I love you so much!"):
    client = mock.Mock()
    client.chat.completions.create.return_value.choices = [mock.Mock(message=mock.Mock(content=reply))]
    return client


class SentenceBuilderTests(unittest.TestCase):
    def test_pause_after_signing_sends_the_words(self):
        builder = signlan.SentenceBuilder(send_after=2.0)
        builder.add("hello", now=10.0)
        self.assertIsNone(builder.update(signing=False, now=11.0))
        self.assertAlmostEqual(builder.progress(11.0), 0.5)
        self.assertEqual(builder.update(signing=False, now=12.0), ["hello"])
        self.assertEqual(builder.words, [])

    def test_signing_again_restarts_the_pause(self):
        builder = signlan.SentenceBuilder(send_after=2.0)
        builder.add("hello", now=0.0)
        builder.update(signing=True, now=1.5)
        self.assertIsNone(builder.update(signing=False, now=2.5))
        builder.add("teacher", now=3.0)
        self.assertEqual(builder.update(signing=False, now=5.0), ["hello", "teacher"])

    def test_nothing_is_sent_without_words(self):
        builder = signlan.SentenceBuilder(send_after=2.0)
        self.assertIsNone(builder.update(signing=False, now=100.0))
        self.assertEqual(builder.progress(100.0), 0.0)

    def test_backspace_removes_the_last_word_and_gives_more_time(self):
        builder = signlan.SentenceBuilder(send_after=2.0)
        builder.add("hello", now=0.0)
        builder.add("dog", now=1.0)
        builder.remove_last(now=2.5)
        self.assertEqual(builder.words, ["hello"])
        self.assertIsNone(builder.update(signing=False, now=4.0))
        self.assertEqual(builder.update(signing=False, now=4.5), ["hello"])

    def test_backspace_on_an_empty_sentence_does_nothing(self):
        builder = signlan.SentenceBuilder()
        builder.remove_last(now=1.0)
        self.assertEqual(builder.words, [])
        self.assertIsNone(builder.update(signing=False, now=10.0))


class StatusTextTests(unittest.TestCase):
    def test_priorities(self):
        self.assertEqual(signlan.status_text("Thinking...", False, True, True), "Thinking...")
        self.assertEqual(signlan.status_text("", False, False, False), "Move back so your shoulders are visible")
        self.assertEqual(signlan.status_text("", True, True, False), "Signing...")
        self.assertIn("Pause to send", signlan.status_text("", True, False, True))
        self.assertEqual(signlan.status_text("", True, False, False), "Sign a word")


class SkeletonTests(unittest.TestCase):
    def test_skeleton_is_drawn_mirrored(self):
        display = np.zeros((480, 640, 3), np.uint8)
        raw = RawLandmarks(pose=None, hands=[make_hand((500.0, 200.0))], width=640, height=480)
        signlan.draw_skeleton(display, raw)
        columns = np.flatnonzero(display.any(axis=(0, 2)))
        self.assertTrue(columns.size)
        self.assertLess(columns.max(), 320)  # a hand on the camera image's right is on the preview's left


class InterpretTests(unittest.TestCase):
    def test_signs_are_sent_in_order_with_the_interpreter_prompt(self):
        client = fake_client()
        signlan.interpret(client, ["hello", "teacher"])
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], signlan.GROQ_MODEL)
        self.assertEqual(kwargs["messages"], [
            {"role": "system", "content": signlan.SYSTEM_PROMPT},
            {"role": "user", "content": "Signs: hello / teacher"},
        ])

    def test_reply_is_cleaned_up(self):
        self.assertEqual(signlan.interpret(fake_client(' "Hello, teacher!" '), ["hello"]), "Hello, teacher!")

    def test_api_error_returns_none(self):
        client = mock.Mock()
        client.chat.completions.create.side_effect = APIConnectionError(request=mock.Mock())
        with mock.patch("builtins.print"):
            self.assertIsNone(signlan.interpret(client, ["hello"]))

    def test_empty_reply_returns_none(self):
        self.assertIsNone(signlan.interpret(fake_client(""), ["hello"]))


class ResponderTests(unittest.TestCase):
    def test_replies_in_the_background_and_speaks(self):
        speaking, finish_speaking = threading.Event(), threading.Event()

        def slow_speak(text):
            speaking.set()
            finish_speaking.wait(5)

        responder = signlan.Responder(fake_client("Hello there!"))
        with mock.patch.object(signlan, "speak", side_effect=slow_speak) as speak, mock.patch("builtins.print"):
            responder.start(["hello"])
            self.assertTrue(speaking.wait(5))
            self.assertTrue(responder.busy)
            self.assertEqual(responder.status, "Speaking...")
            finish_speaking.set()
            responder.wait(5)
        self.assertFalse(responder.busy)
        self.assertEqual((responder.status, responder.heard, responder.reply), ("", ["hello"], "Hello there!"))
        speak.assert_called_once_with("Hello there!")

    def test_failed_reply_is_shown_and_not_spoken(self):
        client = mock.Mock()
        client.chat.completions.create.side_effect = APIConnectionError(request=mock.Mock())
        responder = signlan.Responder(client)
        with mock.patch.object(signlan, "speak") as speak, mock.patch("builtins.print"):
            responder.start(["hello"])
            responder.wait(5)
        speak.assert_not_called()
        self.assertIn("no reply", responder.reply)


class OverlayTests(unittest.TestCase):
    def test_long_text_is_wrapped_to_fit(self):
        text = "Said: " + "this is a fairly long sentence that will not fit on one line " * 2
        lines = signlan.wrap_text(text, 300, 0.7)
        self.assertGreater(len(lines), 1)
        self.assertEqual(" ".join(lines), " ".join(text.split()))
        for line in lines:
            self.assertLessEqual(cv2.getTextSize(line, signlan.FONT, 0.7, 2)[0][0], 300)

    def test_overlay_draws_on_the_frame(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        signlan.draw_overlay(frame, "hello (91%)", True, 0.5, "Thinking...", ["Signed: hello", "Said: Hello!"])
        self.assertTrue((frame != 128).any())


class MainLoopTests(unittest.TestCase):
    """Drive main() with scripted landmarks: sign, lower the hands, pause, hear the reply"""

    def test_sign_pause_send_speak(self):
        script = ["none"] * 5 + ["both"] * 20 + ["none"] * 90
        position = {"i": 0}

        class FakeCamera:
            def isOpened(self):
                return True

            def read(self):
                time.sleep(0.03)
                position["i"] += 1
                return True, np.zeros((480, 640, 3), dtype=np.uint8)

            def release(self):
                pass

        class FakeExtractor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

            def extract(self, frame, timestamp_ms):
                return make_raw(script[min(position["i"], len(script)) - 1])

        recogniser = mock.Mock()
        recogniser.predict.return_value = [("hello", 0.9), ("good", 0.05), ("bank", 0.01)]
        client = fake_client("Hello there!")
        spoken = []
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "test"}), \
                mock.patch.object(signlan, "load_recogniser", return_value=recogniser), \
                mock.patch.object(signlan, "LandmarkExtractor", FakeExtractor), \
                mock.patch.object(signlan, "Groq", return_value=client), \
                mock.patch.object(signlan, "speak", side_effect=spoken.append), \
                mock.patch.object(signlan.cv2, "VideoCapture", return_value=FakeCamera()), \
                mock.patch.object(signlan.cv2, "imshow"), \
                mock.patch.object(signlan.cv2, "destroyAllWindows"), \
                mock.patch.object(signlan.cv2, "getWindowProperty", return_value=1), \
                mock.patch.object(signlan.cv2, "waitKey",
                                  side_effect=lambda _: 27 if position["i"] >= len(script) else -1), \
                mock.patch("builtins.print"):
            signlan.main()

        recogniser.predict.assert_called_once()
        self.assertEqual(recogniser.predict.call_args.args[0].shape, (32, 184))
        self.assertEqual(client.chat.completions.create.call_args.kwargs["messages"][-1]["content"], "Signs: hello")
        self.assertEqual(spoken, ["Hello there!"])


if __name__ == "__main__":
    unittest.main()
