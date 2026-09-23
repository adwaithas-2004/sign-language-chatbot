import unittest

import numpy as np

from isl import features as F
from isl.augment import augment, rotate_scale
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
