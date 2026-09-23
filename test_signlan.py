import unittest
from unittest import mock

import numpy as np
from groq import APIConnectionError

import signlan


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

    def test_wake_word_held_after_waking_is_not_sent_as_a_command(self):
        # Regression: the frame after "done" woke the bot, "done" itself was sent to the chatbot
        tracker = signlan.SignTracker(hold_seconds=1.0, threshold=80)
        results = self.feed(tracker, [("done", 95, t / 10) for t in range(0, 40)])
        self.assertEqual([r for r in results if r], ["done"])


class ChatbotTests(unittest.TestCase):
    def test_api_error_returns_none_and_keeps_history_clean(self):
        client = mock.Mock()
        client.chat.completions.create.side_effect = APIConnectionError(request=mock.Mock())
        messages = [{"role": "system", "content": signlan.SYSTEM_PROMPT}]

        self.assertIsNone(signlan.customLLMBot(client, messages, "I love you"))
        self.assertEqual(len(messages), 1)

    def test_reply_is_returned_and_recorded(self):
        client = mock.Mock()
        client.chat.completions.create.return_value.choices = [
            mock.Mock(message=mock.Mock(content=" I love you too! "))]
        messages = [{"role": "system", "content": signlan.SYSTEM_PROMPT}]

        self.assertEqual(signlan.customLLMBot(client, messages, "I love you"), "I love you too!")
        self.assertEqual(messages[-1], {"role": "assistant", "content": "I love you too!"})
        self.assertEqual(client.chat.completions.create.call_args.kwargs["model"], signlan.GROQ_MODEL)


if __name__ == "__main__":
    unittest.main()
