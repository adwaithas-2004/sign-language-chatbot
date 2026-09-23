import os
import threading
import time
import unittest
from unittest import mock

import cv2
import numpy as np
from groq import APIConnectionError

import signlan


def fake_client(reply="I love you so much!"):
    client = mock.Mock()
    client.chat.completions.create.return_value.choices = [mock.Mock(message=mock.Mock(content=reply))]
    return client


class ModelTests(unittest.TestCase):
    def test_teachable_machine_model_loads_and_matches_labels(self):
        model = signlan.load_model(signlan.MODEL_PATH, compile=False)
        labels = signlan.load_labels(signlan.LABELS_PATH)

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        prediction = model(signlan.preprocess(frame), training=False).numpy()

        self.assertEqual(prediction.shape, (1, len(labels)))
        self.assertAlmostEqual(float(prediction.sum()), 1.0, places=4)


class LabelTests(unittest.TestCase):
    def test_index_prefix_is_removed(self):
        self.assertEqual(signlan.load_labels(signlan.LABELS_PATH), ["I love you", "done", "Background"])

    def test_wake_word_and_background_are_real_labels(self):
        labels = signlan.load_labels(signlan.LABELS_PATH)
        self.assertIn(signlan.WAKE_WORD, labels)
        self.assertIn(signlan.BACKGROUND, labels)


class PreprocessTests(unittest.TestCase):
    def test_output_shape_and_range(self):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        image = signlan.preprocess(frame)
        self.assertEqual(image.shape, (1, 224, 224, 3))
        self.assertGreaterEqual(image.min(), -1.0)
        self.assertLessEqual(image.max(), 1.0)

    def test_bgr_is_converted_to_rgb(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # pure blue in OpenCV's BGR order
        pixel = signlan.preprocess(frame)[0, 112, 112]
        np.testing.assert_allclose(pixel, [-1.0, -1.0, 1.0])  # blue is the last channel in RGB


class SignTrackerTests(unittest.TestCase):
    def feed(self, tracker, frames):
        """frames: (label, confidence, timestamp) tuples"""
        return [tracker.update(label, conf, now=t) for label, conf, t in frames]

    def test_sign_fires_once_after_being_held(self):
        tracker = signlan.SignTracker(hold_seconds=1.0, threshold=80)
        results = self.feed(tracker, [("done", 95, t) for t in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)])
        self.assertEqual(results, [None, None, "done", None, None, None])

    def test_low_confidence_breaks_the_hold(self):
        tracker = signlan.SignTracker(hold_seconds=1.0, threshold=80)
        results = self.feed(tracker, [("done", 95, 0.0), ("done", 95, 0.5), ("done", 50, 0.9),
                                      ("done", 95, 1.0), ("done", 95, 1.5), ("done", 95, 2.0)])
        self.assertEqual(results, [None, None, None, None, None, "done"])

    def test_switching_sign_restarts_the_hold(self):
        tracker = signlan.SignTracker(hold_seconds=1.0, threshold=80)
        results = self.feed(tracker, [("done", 95, 0.0), ("I love you", 95, 0.9),
                                      ("I love you", 95, 1.5), ("I love you", 95, 2.0)])
        self.assertEqual(results, [None, None, None, "I love you"])

    def test_wake_word_held_after_waking_fires_only_once(self):
        # Regression: the frame after "done" woke the bot, "done" itself was sent to the chatbot
        tracker = signlan.SignTracker(hold_seconds=1.0, threshold=80)
        results = self.feed(tracker, [("done", 95, t / 10) for t in range(0, 40)])
        self.assertEqual([r for r in results if r], ["done"])

    def test_confidence_dip_while_still_holding_does_not_repeat_the_sign(self):
        # Regression: holding "I love you" with one frame dipping under the threshold accepted it twice
        tracker = signlan.SignTracker(hold_seconds=1.0, threshold=80, release_seconds=0.5)
        frames = ([("I love you", 99, t / 10) for t in range(0, 11)] + [("I love you", 70, 1.1)]
                  + [("I love you", 87, t / 10) for t in range(12, 40)])
        self.assertEqual([r for r in self.feed(tracker, frames) if r], ["I love you"])

    def test_brief_flicker_to_another_sign_does_not_repeat_the_sign(self):
        tracker = signlan.SignTracker(hold_seconds=1.0, threshold=80, release_seconds=0.5)
        frames = ([("done", 90, t / 10) for t in range(0, 11)] + [("I love you", 85, 1.1), ("Background", 60, 1.2)]
                  + [("done", 90, t / 10) for t in range(13, 40)])
        self.assertEqual([r for r in self.feed(tracker, frames) if r], ["done"])

    def test_same_sign_counts_again_after_hand_goes_down(self):
        tracker = signlan.SignTracker(hold_seconds=1.0, threshold=80, release_seconds=0.5)
        frames = ([("done", 90, t / 10) for t in range(0, 11)]
                  + [("Background", 60, t / 10) for t in range(11, 18)]  # hand down for 0.7s
                  + [("done", 90, t / 10) for t in range(18, 30)])
        results = self.feed(tracker, frames)
        self.assertEqual([r for r in results if r], ["done", "done"])
        self.assertGreaterEqual(results.index("done", 11), 28)  # second "done" needed its own full 1s hold

    def test_different_sign_counts_straight_away(self):
        tracker = signlan.SignTracker(hold_seconds=1.0, threshold=80, release_seconds=0.5)
        frames = [("done", 90, t / 10) for t in range(0, 11)] + [("I love you", 95, t / 10) for t in range(11, 25)]
        self.assertEqual([r for r in self.feed(tracker, frames) if r], ["done", "I love you"])

    def test_reset_keeps_the_last_sign_blocked(self):
        # While the bot is busy the hold is reset; a sign still held afterwards must not fire again
        tracker = signlan.SignTracker(hold_seconds=1.0, threshold=80, release_seconds=0.5)
        self.feed(tracker, [("done", 90, t / 10) for t in range(0, 11)])
        tracker.reset()
        results = self.feed(tracker, [("done", 90, t / 10) for t in range(40, 60)])
        self.assertEqual([r for r in results if r], [])

    def test_progress_fills_while_holding(self):
        tracker = signlan.SignTracker(hold_seconds=1.0, threshold=80)
        self.assertEqual(tracker.progress(now=0.0), 0.0)
        tracker.update("done", 95, now=0.0)
        self.assertAlmostEqual(tracker.progress(now=0.25), 0.25)
        tracker.update("done", 95, now=1.0)
        self.assertEqual(tracker.progress(now=5.0), 1.0)
        tracker.update("done", 10, now=5.0)
        self.assertEqual(tracker.progress(now=5.0), 0.0)


class SentenceBuilderTests(unittest.TestCase):
    def feed(self, signs):
        builder = signlan.SentenceBuilder()
        return builder, [builder.add(s) for s in signs]

    def test_wake_word_signs_wake_word_sends_the_sentence(self):
        builder, results = self.feed(["done", "I love you", "water", "done"])
        self.assertEqual(results, [None, None, None, ["I love you", "water"]])
        self.assertFalse(builder.collecting)

    def test_signs_before_the_wake_word_are_ignored(self):
        _, results = self.feed(["I love you", "done", "water", "done"])
        self.assertEqual(results[-1], ["water"])

    def test_background_is_never_part_of_the_sentence(self):
        _, results = self.feed(["done", "Background", "I love you", "Background", "done"])
        self.assertEqual(results[-1], ["I love you"])

    def test_wake_word_twice_cancels(self):
        builder, results = self.feed(["done", "done"])
        self.assertEqual(results, [None, []])
        self.assertFalse(builder.collecting)

    def test_signs_so_far_are_visible_while_collecting(self):
        builder, _ = self.feed(["done", "I love you"])
        self.assertTrue(builder.collecting)
        self.assertEqual(builder.signs, ["I love you"])


class InterpretTests(unittest.TestCase):
    def test_signs_are_sent_in_order_with_the_interpreter_prompt(self):
        client = fake_client()
        signlan.interpret(client, ["I love you", "water"])

        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], signlan.GROQ_MODEL)
        self.assertEqual(kwargs["messages"], [
            {"role": "system", "content": signlan.SYSTEM_PROMPT},
            {"role": "user", "content": "Signs: I love you / water"},
        ])

    def test_each_sentence_is_interpreted_on_its_own(self):
        client = fake_client()
        signlan.interpret(client, ["I love you"])
        signlan.interpret(client, ["water"])
        self.assertEqual(len(client.chat.completions.create.call_args.kwargs["messages"]), 2)

    def test_reply_is_cleaned_up(self):
        self.assertEqual(signlan.interpret(fake_client(' "I love you too!" '), ["I love you"]), "I love you too!")

    def test_api_error_returns_none(self):
        client = mock.Mock()
        client.chat.completions.create.side_effect = APIConnectionError(request=mock.Mock())
        with mock.patch("builtins.print"):
            self.assertIsNone(signlan.interpret(client, ["I love you"]))

    def test_empty_reply_returns_none(self):
        self.assertIsNone(signlan.interpret(fake_client(""), ["I love you"]))


class ResponderTests(unittest.TestCase):
    def test_replies_in_the_background_and_speaks(self):
        speaking = threading.Event()
        finish_speaking = threading.Event()

        def slow_speak(text):
            speaking.set()
            finish_speaking.wait(5)

        responder = signlan.Responder(fake_client("I love you too!"))
        with mock.patch.object(signlan, "speak", side_effect=slow_speak) as speak, \
                mock.patch("builtins.print"):
            responder.start(["I love you"])
            self.assertTrue(speaking.wait(5))
            self.assertTrue(responder.busy)  # start() returned while the reply is still being spoken
            self.assertEqual(responder.status, "Speaking...")

            finish_speaking.set()
            responder.wait(5)

        self.assertFalse(responder.busy)
        self.assertEqual(responder.status, "")
        self.assertEqual(responder.heard, ["I love you"])
        self.assertEqual(responder.reply, "I love you too!")
        speak.assert_called_once_with("I love you too!")

    def test_failed_reply_is_shown_and_not_spoken(self):
        client = mock.Mock()
        client.chat.completions.create.side_effect = APIConnectionError(request=mock.Mock())
        responder = signlan.Responder(client)
        with mock.patch.object(signlan, "speak") as speak, mock.patch("builtins.print"):
            responder.start(["I love you"])
            responder.wait(5)
        speak.assert_not_called()
        self.assertIn("no reply", responder.reply)
        self.assertEqual(responder.status, "")


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
        signlan.draw_overlay(frame, "done (95%)", True, 0.5, "Thinking...",
                             ["Signed: I love you", "Said: I love you too!"])
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertTrue((frame != 128).any())


class MainLoopTests(unittest.TestCase):
    """Drive main() with a scripted camera and model: wake, sign, send, get a spoken reply"""

    def test_full_conversation(self):
        labels = signlan.load_labels(signlan.LABELS_PATH)
        script = (["Background"] * 3 + ["done"] * 10 + ["Background"] * 3 + ["I love you"] * 10
                  + ["Background"] * 3 + ["done"] * 10 + ["Background"] * 15)
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

        class FakeModel:
            output_shape = (None, len(labels))

            def __call__(self, image, training=False):
                label = script[min(position["i"], len(script)) - 1]
                probabilities = np.full((1, len(labels)), 0.01, dtype=np.float32)
                probabilities[0, labels.index(label)] = 0.98
                return mock.Mock(numpy=lambda: probabilities)

        client = fake_client("I love you so much!")
        spoken = []
        real_tracker = signlan.SignTracker
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "test"}), \
                mock.patch.object(signlan, "load_model", return_value=FakeModel()), \
                mock.patch.object(signlan, "Groq", return_value=client), \
                mock.patch.object(signlan, "SignTracker", lambda: real_tracker(hold_seconds=0.15)), \
                mock.patch.object(signlan, "speak", side_effect=spoken.append), \
                mock.patch.object(signlan.cv2, "VideoCapture", return_value=FakeCamera()), \
                mock.patch.object(signlan.cv2, "imshow"), \
                mock.patch.object(signlan.cv2, "destroyAllWindows"), \
                mock.patch.object(signlan.cv2, "getWindowProperty", return_value=1), \
                mock.patch.object(signlan.cv2, "waitKey", side_effect=lambda _: 27 if position["i"] >= len(script) else -1), \
                mock.patch("builtins.print"):
            signlan.main()

        client.chat.completions.create.assert_called_once()
        user_message = client.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
        self.assertEqual(user_message, "Signs: I love you")
        self.assertEqual(spoken, ["I love you so much!"])


if __name__ == "__main__":
    unittest.main()
