import unittest

import numpy as np

from isl.features import NUM_FEATURES, frame_features
from isl.segmenter import DROPOUT_SECONDS, END_SECONDS, Segmenter
from tests.fakes import frame, make_raw

FRAMES = {
    True: frame(True),  # hands raised
    False: frame(False),  # no hand in view (lowered out of frame, or MediaPipe lost it)
    "lowered": frame_features(make_raw("both", wrist_y=400))[0],  # hands visible, below the threshold
    "lost": np.zeros(NUM_FEATURES, np.float32),  # shoulders out of view
}


class SegmenterTests(unittest.TestCase):
    def run_timeline(self, timeline, fps=30):
        """timeline: [(state, seconds)] with states from FRAMES; returns [(time, sequence)] for each sign emitted"""
        segmenter, now, emitted = Segmenter(), 0.0, []
        for state, seconds in timeline:
            for _ in range(round(seconds * fps)):
                result = segmenter.update(FRAMES[state], now)
                if result is not None:
                    emitted.append((now, result))
                now += 1 / fps
        return emitted

    def test_one_sign_ends_after_the_hands_are_lowered(self):
        emitted = self.run_timeline([(False, 0.5), (True, 1.0), ("lowered", 1.0)])
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0][1].shape, (32, 184))
        self.assertAlmostEqual(emitted[0][0], 1.5 + END_SECONDS, delta=0.1)

    def test_hands_leaving_the_view_end_the_sign_after_the_dropout_allowance(self):
        emitted = self.run_timeline([(False, 0.5), (True, 1.0), (False, 1.0)])
        self.assertEqual(len(emitted), 1)
        self.assertAlmostEqual(emitted[0][0], 1.5 + DROPOUT_SECONDS + END_SECONDS, delta=0.1)

    def test_hand_detection_dropouts_do_not_stop_a_sign_from_starting(self):
        # MediaPipe loses fast-moving hands for a frame or two (every INCLUDE "priest" video never started a sign)
        timeline = [(True, 2 / 30), (False, 1 / 30)] * 10 + [("lowered", 0.5)]
        self.assertEqual(len(self.run_timeline(timeline)), 1)

    def test_blips_are_ignored(self):
        self.assertEqual(self.run_timeline([(True, 0.05), (False, 0.5), (True, 0.2), (False, 1.0)]), [])

    def test_back_to_back_signs(self):
        self.assertEqual(len(self.run_timeline([(True, 0.8), ("lowered", 0.4), (True, 0.8), ("lowered", 0.5)])), 2)

    def test_a_brief_drop_does_not_split_a_sign(self):
        self.assertEqual(len(self.run_timeline([(True, 0.6), (False, 0.1), (True, 0.6), (False, 0.8)])), 1)

    def test_long_signs_are_cut_off(self):
        emitted = self.run_timeline([(True, 5.0)])
        self.assertEqual(len(emitted), 1)
        self.assertAlmostEqual(emitted[0][0], 4.0, delta=0.1)

    def test_losing_the_shoulders_mid_sign_discards_it(self):
        # Without the body there is nothing to normalise against, so the partial sign mustn't be guessed at
        self.assertEqual(self.run_timeline([(True, 0.6), ("lost", 0.5), ("lowered", 1.0)]), [])

    def test_a_brief_body_flicker_keeps_the_sign_without_empty_frames(self):
        emitted = self.run_timeline([(True, 0.5), ("lost", 0.1), (True, 0.4), ("lowered", 0.5)])
        self.assertEqual(len(emitted), 1)
        self.assertTrue(emitted[0][1].any(axis=1).all())

    def test_signing_flag_and_reset(self):
        segmenter = Segmenter()
        for i in range(6):
            segmenter.update(frame(True), i / 30)
        self.assertTrue(segmenter.signing)
        segmenter.reset()
        self.assertFalse(segmenter.signing)
