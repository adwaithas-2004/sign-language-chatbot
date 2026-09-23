"""Synthetic landmarks for tests: a signer facing the camera with shoulders 100 px apart at y = 200"""
import numpy as np

from isl.features import RawLandmarks, frame_features


def make_pose(visibility=1.0, wrist_y=150.0):
    """MediaPipe-style (33, 3) pose; the subject's left shoulder (11) is on the image's right"""
    pose = np.zeros((33, 3), dtype=np.float32)
    pose[0] = (320, 120, 1.0)  # nose
    pose[11], pose[12] = (370, 200, visibility), (270, 200, visibility)  # shoulders
    pose[13], pose[14] = (380, 260, 1.0), (260, 260, 1.0)  # elbows
    pose[15], pose[16] = (390, wrist_y, 1.0), (250, wrist_y, 1.0)  # wrists
    return pose


def make_hand(wrist_xy, size=30.0):
    """21 hand points with the wrist at wrist_xy and fingers pointing up"""
    x, y = float(wrist_xy[0]), float(wrist_xy[1])
    hand = np.array([(x + (i % 4) * 3.0, y - size * 1.5 * i / 20) for i in range(21)], dtype=np.float32)
    hand[0], hand[9] = (x, y), (x, y - size)
    return hand


def make_raw(hands="both", wrist_y=150.0, visibility=1.0):
    """hands: "both", "left", "right" or "none"; wrist_y 150 is raised, 400 is lowered"""
    pose = make_pose(visibility, wrist_y)
    left, right = make_hand(pose[15, :2]), make_hand(pose[16, :2])
    found = {"both": [right, left], "left": [left], "right": [right], "none": []}[hands]
    return RawLandmarks(pose=pose, hands=found, width=640, height=480)


def frame(up):
    """One frame's features: both hands raised (True), or no hands in view (False)"""
    return frame_features(make_raw("both" if up else "none"))[0]
