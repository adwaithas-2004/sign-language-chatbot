"""Body-normalised landmark features, shared by the training pipeline and the live app"""
from dataclasses import dataclass

import numpy as np

FEATURE_VERSION = 2  # 2: body points out of view are left empty
SEQUENCE_LENGTH = 32  # frames per sign given to the model

# MediaPipe pose indices
NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST = 0, 11, 12, 13, 14, 15, 16
POSE_POINTS = (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST)
_POSE_MIRROR = [0, 2, 1, 4, 3, 6, 5]  # POSE_POINTS order with left and right swapped
SHOULDER_POINTS = (1, 2)  # positions of the shoulders in POSE_POINTS
MIN_VISIBILITY = 0.5
# A wrist less than this many shoulder-widths below the shoulders counts as a raised hand when trimming a sign
# to the part the model sees. Calibrated on the INCLUDE-50 train split: resting wrists sit at 1.53-1.81, so 1.25
# counts 0.4 % of them as up. The live segmenter uses its own, looser threshold to decide when signs start and end
HANDS_UP_K = 1.25

# Layout of one frame's feature vector
POSE_SLICE = slice(0, 14)  # 7 pose points (x, y) in the body frame
HAND_SLICES = (slice(14, 98), slice(98, 182))  # left, right: 21 (x, y) in the body frame + 21 (x, y) hand-local
PRESENCE_SLICE = slice(182, 184)  # 1 when the left / right hand was found
NUM_FEATURES = 184


@dataclass
class RawLandmarks:
    """MediaPipe output for one frame, in pixels of the processed image"""
    pose: np.ndarray | None  # (33, 3): x, y, visibility; None when no person was found
    hands: list  # 0-2 arrays of shape (21, 2): x, y
    width: int
    height: int


def _assign_hands(hands, pose):
    """[left hand or None, right hand or None], matching each hand to the nearest pose wrist"""
    wrists = pose[[LEFT_WRIST, RIGHT_WRIST], :2]
    if len(hands) == 0:
        return [None, None]
    if len(hands) == 1:
        slots = [None, None]
        slots[int(np.argmin(np.linalg.norm(wrists - hands[0][0], axis=1)))] = hands[0]
        return slots
    a, b = hands[:2]
    straight = np.linalg.norm(wrists[0] - a[0]) + np.linalg.norm(wrists[1] - b[0])
    swapped = np.linalg.norm(wrists[0] - b[0]) + np.linalg.norm(wrists[1] - a[0])
    return [a, b] if straight <= swapped else [b, a]


def _body_frame(pose):
    """(origin, scale) of the body frame: the shoulder midpoint and shoulder width, or None without clear shoulders"""
    if pose is None or min(pose[LEFT_SHOULDER, 2], pose[RIGHT_SHOULDER, 2]) < MIN_VISIBILITY:
        return None
    left, right = pose[LEFT_SHOULDER, :2], pose[RIGHT_SHOULDER, :2]
    scale = float(np.linalg.norm(left - right))
    return ((left + right) / 2, scale) if scale >= 1e-3 else None


def view_below_shoulders(raw):
    """How many shoulder widths below the shoulders the camera sees, or None without a body"""
    frame = _body_frame(raw.pose)
    return None if frame is None else float((raw.height - frame[0][1]) / frame[1])


def frame_features(raw):
    """(feature vector, body found) for one frame; the vector is all zeros when the shoulders aren't visible"""
    features = np.zeros(NUM_FEATURES, dtype=np.float32)
    pose = raw.pose
    frame = _body_frame(pose)
    if frame is None:
        return features, False
    origin, scale = frame

    points = pose[list(POSE_POINTS)]
    body = (points[:, :2] - origin) / scale
    # MediaPipe guesses where out-of-view arms are; leave those points empty rather than feed the guess to the model
    x, y, visibility = points[:, 0], points[:, 1], points[:, 2]
    hidden = (visibility < MIN_VISIBILITY) | (x < 0) | (x > raw.width) | (y < 0) | (y > raw.height)
    hidden[list(SHOULDER_POINTS)] = False  # the shoulders define the body frame
    body[hidden] = 0
    features[POSE_SLICE] = body.ravel()
    for i, hand in enumerate(_assign_hands(raw.hands, pose)):
        if hand is None:
            continue
        hand = np.asarray(hand, dtype=np.float32)
        size = float(np.linalg.norm(hand[9] - hand[0]))  # wrist to middle-finger knuckle
        local = (hand - hand[0]) / size if size > 1e-3 else np.zeros_like(hand)
        features[HAND_SLICES[i]] = np.concatenate([((hand - origin) / scale).ravel(), local.ravel()])
        features[PRESENCE_SLICE.start + i] = 1.0
    return features, True


def hands_up(features, k=None):
    """Whether a hand is raised (wrist less than k shoulder-widths below the shoulders, HANDS_UP_K by default):
    one bool for a frame (F,), a bool array for a sequence (T, F)"""
    k = HANDS_UP_K if k is None else k
    frames = np.atleast_2d(features)
    up = np.zeros(len(frames), dtype=bool)
    for i, block in enumerate(HAND_SLICES):
        present = frames[:, PRESENCE_SLICE.start + i] > 0.5
        wrist_y = frames[:, block.start + 1]  # body frame: y grows downwards from the shoulders
        up |= present & (wrist_y < k)
    return up if np.ndim(features) == 2 else bool(up[0])


def trim(frames, up, pad=2, min_up=4):
    """Keep the part of a clip where the hands are up, plus a little padding"""
    raised = np.flatnonzero(up)
    if len(raised) < min_up:
        return frames
    return frames[max(raised[0] - pad, 0):min(raised[-1] + pad + 1, len(frames))]


def resample(frames, length=SEQUENCE_LENGTH):
    """Linearly interpolate a (T, F) sequence to exactly `length` frames"""
    frames = np.asarray(frames, dtype=np.float32)
    if len(frames) == 0:
        return np.zeros((length, frames.shape[1]), dtype=np.float32)
    if len(frames) == 1:
        return np.repeat(frames, length, axis=0)
    positions = np.linspace(0, len(frames) - 1, length)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(frames) - 1)
    weight = (positions - lower)[:, None].astype(np.float32)
    return frames[lower] * (1 - weight) + frames[upper] * weight


def raised_part(frames):
    """The part of a sign's frames with the hands up (plus padding), as the model sees it"""
    frames = np.asarray(frames, dtype=np.float32)
    return trim(frames, hands_up(frames))


def to_sequence(frames):
    """Model input for one sign: trimmed to the raised-hands part and resampled"""
    return resample(raised_part(frames))


def flip(sequence):
    """Mirror a (T, NUM_FEATURES) sequence left-right, as if signed with the other hand"""
    seq = np.asarray(sequence, dtype=np.float32)
    out = np.empty_like(seq)
    pose = seq[:, POSE_SLICE].reshape(len(seq), 7, 2)[:, _POSE_MIRROR]
    pose[..., 0] *= -1
    out[:, POSE_SLICE] = pose.reshape(len(seq), 14)
    for dst, src in zip(HAND_SLICES, reversed(HAND_SLICES)):
        hand = seq[:, src].reshape(len(seq), 42, 2).copy()
        hand[..., 0] *= -1
        out[:, dst] = hand.reshape(len(seq), 84)
    out[:, PRESENCE_SLICE] = seq[:, PRESENCE_SLICE][:, ::-1]
    return out
