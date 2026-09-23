import unittest

import numpy as np

from isl import features as F
from isl.features import RawLandmarks
from tests.fakes import frame, make_hand, make_raw


class FrameFeatureTests(unittest.TestCase):
    def test_vector_layout(self):
        vector, found = F.frame_features(make_raw("both"))
        self.assertTrue(found)
        self.assertEqual(F.NUM_FEATURES, 184)
        self.assertEqual(vector.shape, (184,))
        self.assertEqual(vector.dtype, np.float32)
        np.testing.assert_array_equal(vector[F.PRESENCE_SLICE], [1, 1])
        # the shoulders are pose points 1 and 2 of 7, at +-0.5 shoulder widths from the origin
        np.testing.assert_allclose(vector[F.POSE_SLICE].reshape(7, 2)[1:3], [[0.5, 0.0], [-0.5, 0.0]])

    def test_moving_or_scaling_the_body_changes_nothing(self):
        raw = make_raw("both")
        shift, factor = np.array([123.0, -45.0], dtype=np.float32), 1.7
        pose = raw.pose.copy()
        pose[:, :2] = raw.pose[:, :2] * factor + shift
        moved = RawLandmarks(pose=pose, hands=[h * factor + shift for h in raw.hands], width=1280, height=720)
        np.testing.assert_allclose(F.frame_features(moved)[0], F.frame_features(raw)[0], atol=1e-5)

    def test_hands_are_matched_to_the_nearest_wrist(self):
        vector = F.frame_features(make_raw("right"))[0]
        np.testing.assert_array_equal(vector[F.PRESENCE_SLICE], [0, 1])
        self.assertFalse(vector[F.HAND_SLICES[0]].any())
        both = F.frame_features(make_raw("both"))[0]  # listed right-then-left, must still land left-then-right
        np.testing.assert_allclose(both[F.HAND_SLICES[1]], vector[F.HAND_SLICES[1]])

    def test_hidden_shoulders_give_an_empty_invalid_frame(self):
        vector, found = F.frame_features(make_raw("both", visibility=0.2))
        self.assertFalse(found)
        self.assertFalse(vector.any())
        vector, found = F.frame_features(RawLandmarks(pose=None, hands=[make_hand((300, 150))], width=640, height=480))
        self.assertFalse(found)
        self.assertFalse(F.hands_up(vector))


class HandsUpTests(unittest.TestCase):
    def test_raised_hand(self):
        self.assertTrue(F.hands_up(F.frame_features(make_raw("left", wrist_y=150))[0]))

    def test_lowered_or_missing_hands(self):
        self.assertFalse(F.hands_up(F.frame_features(make_raw("both", wrist_y=400))[0]))
        self.assertFalse(F.hands_up(F.frame_features(make_raw("none"))[0]))

    def test_sequence_gives_one_flag_per_frame(self):
        clip = np.stack([frame(True), frame(False), frame(True)])
        np.testing.assert_array_equal(F.hands_up(clip), [True, False, True])


class SequenceTests(unittest.TestCase):
    def test_resample_length_and_endpoints(self):
        clip = np.arange(10 * 184, dtype=np.float32).reshape(10, 184)
        out = F.resample(clip)
        self.assertEqual(out.shape, (32, 184))
        np.testing.assert_allclose(out[0], clip[0])
        np.testing.assert_allclose(out[-1], clip[-1])

    def test_resample_single_frame_and_empty(self):
        self.assertEqual(F.resample(np.ones((1, 184))).shape, (32, 184))
        empty = F.resample(np.zeros((0, 184)))
        self.assertEqual(empty.shape, (32, 184))
        self.assertFalse(empty.any())

    def test_trim_keeps_the_raised_part_with_padding(self):
        clip = np.stack([frame(False)] * 10 + [frame(True)] * 6 + [frame(False)] * 10)
        self.assertEqual(len(F.trim(clip, F.hands_up(clip))), 6 + 2 * 2)

    def test_trim_keeps_everything_when_hands_are_barely_up(self):
        clip = np.stack([frame(False)] * 10 + [frame(True)] * 3)
        self.assertEqual(len(F.trim(clip, F.hands_up(clip))), 13)

    def test_to_sequence(self):
        clip = np.stack([frame(False)] * 10 + [frame(True)] * 20 + [frame(False)] * 10)
        sequence = F.to_sequence(clip)
        self.assertEqual(sequence.shape, (32, 184))
        self.assertTrue(F.hands_up(sequence)[5:27].all())

    def test_flip_swaps_hands_and_twice_is_identity(self):
        clip = np.stack([F.frame_features(make_raw("right"))[0]] * 4)
        flipped = F.flip(clip)
        np.testing.assert_array_equal(flipped[:, F.PRESENCE_SLICE], [[1, 0]] * 4)
        np.testing.assert_allclose(flipped[:, F.HAND_SLICES[0]][:, 0::2], -clip[:, F.HAND_SLICES[1]][:, 0::2])
        np.testing.assert_allclose(F.flip(flipped), clip)
