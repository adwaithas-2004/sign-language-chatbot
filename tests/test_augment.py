import unittest

import numpy as np

from isl import features as F
from isl.augment import augment, crop_view, random_view, rotate_scale
from tests.fakes import frame, make_raw


class AugmentTests(unittest.TestCase):
    def test_output_is_a_model_sized_sequence(self):
        rng = np.random.default_rng(0)
        clip = np.stack([frame(True)] * 40)
        for _ in range(20):
            out = augment(clip, rng)
            self.assertEqual(out.shape, (32, 184))
            self.assertEqual(out.dtype, np.float32)

    def test_very_short_clips_work(self):
        rng = np.random.default_rng(0)
        for length in (1, 2, 3):
            self.assertEqual(augment(np.stack([frame(True)] * length), rng).shape, (32, 184))

    def test_missing_hand_stays_empty_and_presence_is_kept(self):
        rng = np.random.default_rng(1)
        clip = np.stack([F.frame_features(make_raw("right"))[0]] * 20)
        for _ in range(20):
            out = augment(clip, rng)
            present = out[0, F.PRESENCE_SLICE]
            self.assertEqual(sorted(present.tolist()), [0.0, 1.0])
            self.assertFalse(out[:, F.HAND_SLICES[int(np.argmin(present))]].any())

    def test_rotation_keeps_lengths_and_scale_only_changes_body_positions(self):
        seq = np.stack([frame(True)] * 3)
        out = rotate_scale(seq, 90, 2.0)
        pose, pose_out = seq[:, F.POSE_SLICE].reshape(3, 7, 2), out[:, F.POSE_SLICE].reshape(3, 7, 2)
        np.testing.assert_allclose(np.linalg.norm(pose_out, axis=2), 2 * np.linalg.norm(pose, axis=2), rtol=1e-5)
        local = slice(F.HAND_SLICES[0].start + 42, F.HAND_SLICES[0].stop)
        np.testing.assert_allclose(np.linalg.norm(out[:, local].reshape(3, 21, 2), axis=2),
                                   np.linalg.norm(seq[:, local].reshape(3, 21, 2), axis=2), rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(out[:, F.PRESENCE_SLICE], seq[:, F.PRESENCE_SLICE])


class ViewTests(unittest.TestCase):
    """Simulated closer cameras, which only see part of the body"""

    def setUp(self):
        lowered = F.frame_features(make_raw("both", wrist_y=400))[0]  # wrists 2 shoulder widths below the shoulders
        self.clip = np.stack([lowered] * 3 + [frame(True)] * 3)  # then raised: wrists 0.5 above, fingertips ~0.77

    def test_hands_below_the_view_vanish(self):
        out = crop_view(self.clip, bottom=1.0, top=-2.0, half_width=2.0)
        self.assertFalse(out[:3, F.PRESENCE_SLICE].any())
        self.assertFalse(out[:3, F.HAND_SLICES[0]].any() or out[:3, F.HAND_SLICES[1]].any())
        np.testing.assert_array_equal(out[3:], self.clip[3:])

    def test_body_points_outside_the_view_are_emptied_but_not_the_shoulders(self):
        pose = crop_view(self.clip, 1.0, -2.0, 2.0)[:3, F.POSE_SLICE].reshape(3, 7, 2)
        self.assertFalse(pose[:, 5:7].any())  # wrists
        np.testing.assert_array_equal(pose[:, :5], self.clip[:3, F.POSE_SLICE].reshape(3, 7, 2)[:, :5])

    def test_a_hand_reaching_above_the_view_vanishes(self):
        self.assertFalse(crop_view(self.clip, 3.0, -0.7, 2.0)[3:, F.PRESENCE_SLICE].any())
        self.assertTrue(crop_view(self.clip, 3.0, -0.9, 2.0)[3:, F.PRESENCE_SLICE].all())

    def test_hands_and_body_points_beside_the_view_vanish(self):
        out = crop_view(self.clip, 3.0, -2.0, 0.65)  # wrists are 0.7 from the middle, elbows 0.6
        self.assertFalse(out[:, F.PRESENCE_SLICE].any())
        pose = out[:, F.POSE_SLICE].reshape(-1, 7, 2)
        self.assertFalse(pose[:, 5:7].any())
        self.assertTrue(pose[:, 3:5].all())

    def test_the_input_is_not_changed(self):
        before = self.clip.copy()
        crop_view(self.clip, 1.0, -0.7, 0.65)
        np.testing.assert_array_equal(self.clip, before)

    def test_random_views_crop_some_clips_and_leave_others_whole(self):
        rng = np.random.default_rng(0)
        outs = [random_view(self.clip, rng) for _ in range(200)]
        whole = sum(np.array_equal(out, self.clip) for out in outs)
        self.assertTrue(40 < whole < 160, whole)
        for out in outs:
            np.testing.assert_array_equal(out[:, 2:6], self.clip[:, 2:6])  # the shoulders
