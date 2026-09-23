"""Random variations of training clips, so about 14 videos per word go further"""
import numpy as np

from isl.features import HAND_SLICES, POSE_SLICE, PRESENCE_SLICE, flip, resample


def rotate_scale(sequence, angle_degrees, scale):
    """Rotate every point about the shoulder midpoint and scale body positions (hand shapes only rotate)"""
    seq = np.array(sequence, dtype=np.float32, copy=True)
    theta = np.deg2rad(angle_degrees)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)

    def transform(block, factor):
        points = seq[:, block].reshape(len(seq), -1, 2)
        seq[:, block] = (points @ rotation.T * factor).reshape(len(seq), -1)

    transform(POSE_SLICE, scale)
    for hand in HAND_SLICES:
        transform(slice(hand.start, hand.start + 42), scale)  # where the hand is, in the body frame
        transform(slice(hand.start + 42, hand.stop), 1.0)  # the hand's shape
    return seq


def augment(frames, rng):
    """A randomly varied copy of one trimmed (T, F) training clip, resampled to the model's length"""
    frames = np.asarray(frames, dtype=np.float32)
    if len(frames) >= 4:
        keep = max(2, round(len(frames) * rng.uniform(0.85, 1.0)))  # crop, which also varies the speed
        start = int(rng.integers(0, len(frames) - keep + 1))
        frames = frames[start:start + keep]
        mask = rng.random(len(frames)) > 0.1  # drop about 10 % of frames, keeping the ends
        mask[[0, -1]] = True
        frames = frames[mask]
    seq = resample(frames)
    if rng.random() < 0.5:
        seq = flip(seq)
    seq = rotate_scale(seq, rng.uniform(-10, 10), rng.uniform(0.9, 1.1))
    coords = seq[:, :PRESENCE_SLICE.start]
    coords += rng.normal(0, 0.01, coords.shape).astype(np.float32) * (coords != 0)  # missing hands stay zero
    return seq
