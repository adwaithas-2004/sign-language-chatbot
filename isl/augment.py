"""Random variations of training clips, so about 14 videos per word go further"""
import numpy as np

from isl.features import HAND_SLICES, POSE_SLICE, PRESENCE_SLICE, SHOULDER_POINTS, flip, resample

# INCLUDE's signers stand far from the camera, but a webcam at a desk often sees only down to the chest. Training
# on randomly narrowed views teaches the model to recognise signs from what such a camera still sees.
# Ranges are in shoulder widths from the shoulder midpoint, in the body frame (y grows downwards)
VIEW_PROBABILITY = 0.6
VIEW_BOTTOM = (0.2, 2.2)
VIEW_TOP = (-2.0, -0.9)
VIEW_HALF_WIDTH = (0.8, 2.0)
_MIDDLE_FINGERTIP = 12


def crop_view(frames, bottom, top, half_width):
    """A (T, F) clip as a camera seeing only y in [top, bottom] and |x| <= half_width would give it:
    hands outside the view vanish and body points outside it are left empty, as the live features do"""
    out = np.array(frames, dtype=np.float32, copy=True)
    for i, block in enumerate(HAND_SLICES):
        wrist = out[:, block.start:block.start + 2]
        tip_y = out[:, block.start + 2 * _MIDDLE_FINGERTIP + 1]
        outside = (wrist[:, 1] > bottom) | (tip_y < top) | (np.abs(wrist[:, 0]) > half_width)
        out[outside, block] = 0
        out[outside, PRESENCE_SLICE.start + i] = 0
    pose = out[:, POSE_SLICE].reshape(len(out), -1, 2)
    outside = (pose[..., 1] > bottom) | (pose[..., 1] < top) | (np.abs(pose[..., 0]) > half_width)
    outside[:, list(SHOULDER_POINTS)] = False
    pose[outside] = 0
    out[:, POSE_SLICE] = pose.reshape(len(out), -1)
    return out


def random_view(frames, rng):
    """The clip as seen by a randomly placed closer camera, or unchanged (1 - VIEW_PROBABILITY of the time)"""
    if rng.random() >= VIEW_PROBABILITY:
        return np.asarray(frames, dtype=np.float32)
    return crop_view(frames, rng.uniform(*VIEW_BOTTOM), rng.uniform(*VIEW_TOP), rng.uniform(*VIEW_HALF_WIDTH))


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
