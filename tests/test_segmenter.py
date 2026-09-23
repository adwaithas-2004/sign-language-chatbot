import unittest

from isl.segmenter import Segmenter
from tests.fakes import frame


class SegmenterTests(unittest.TestCase):
    def run_timeline(self, timeline, fps=30):
        """timeline: [(hands up?, seconds)]; returns [(time, sequence)] for each sign emitted"""
        segmenter, now, emitted = Segmenter(), 0.0, []
        for up, seconds in timeline:
            for _ in range(round(seconds * fps)):
                result = segmenter.update(frame(up), now)
                if result is not None:
                    emitted.append((now, result))
                now += 1 / fps
        return emitted

    def test_one_sign_ends_after_the_hands_are_down(self):
        emitted = self.run_timeline([(False, 0.5), (True, 1.0), (False, 1.0)])
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0][1].shape, (32, 184))
        self.assertAlmostEqual(emitted[0][0], 1.5 + 0.3, delta=0.1)

    def test_blips_are_ignored(self):
        self.assertEqual(self.run_timeline([(True, 0.05), (False, 0.5), (True, 0.2), (False, 1.0)]), [])

    def test_back_to_back_signs(self):
        self.assertEqual(len(self.run_timeline([(True, 0.8), (False, 0.4), (True, 0.8), (False, 0.5)])), 2)

    def test_a_brief_drop_does_not_split_a_sign(self):
        self.assertEqual(len(self.run_timeline([(True, 0.6), (False, 0.1), (True, 0.6), (False, 0.5)])), 1)

    def test_long_signs_are_cut_off(self):
        emitted = self.run_timeline([(True, 5.0)])
        self.assertEqual(len(emitted), 1)
        self.assertAlmostEqual(emitted[0][0], 4.0, delta=0.1)

    def test_signing_flag_and_reset(self):
        segmenter = Segmenter()
        for i in range(6):
            segmenter.update(frame(True), i / 30)
        self.assertTrue(segmenter.signing)
        segmenter.reset()
        self.assertFalse(segmenter.signing)
