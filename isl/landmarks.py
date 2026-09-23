"""MediaPipe hand and pose landmarks for video frames"""
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from isl.features import RawLandmarks

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
                            "float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                                 "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}
PROCESS_WIDTH = 640  # Frames are downscaled to this width first, for dataset videos and the webcam alike


def ensure_model_files(models_dir=MODELS_DIR, urlopen=urllib.request.urlopen):
    """Paths of the MediaPipe model files, downloading any that are missing"""
    models_dir = Path(models_dir)
    paths = {}
    for name, url in MODEL_URLS.items():
        path = models_dir / name
        if not path.exists():
            try:
                with urlopen(url, timeout=60) as response:
                    data = response.read()
            except OSError as e:
                raise SystemExit(f"Could not download the MediaPipe model {name} ({e}).\n"
                                 f"Download it from {url}\nand save it as {path}") from e
            models_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        paths[name] = path
    return paths


class LandmarkExtractor:
    """Runs MediaPipe pose and hand detection over one video or camera stream"""

    def __init__(self, models_dir=MODELS_DIR):
        paths = ensure_model_files(models_dir)
        video = vision.RunningMode.VIDEO
        self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(paths["pose_landmarker_lite.task"])), running_mode=video))
        self._hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(paths["hand_landmarker.task"])), running_mode=video,
            num_hands=2))
        self._last_timestamp = -1

    def extract(self, frame_bgr, timestamp_ms):
        height, width = frame_bgr.shape[:2]
        if width > PROCESS_WIDTH:
            height, width = round(height * PROCESS_WIDTH / width), PROCESS_WIDTH
            frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
        # VIDEO mode needs strictly increasing timestamps
        timestamp = max(int(timestamp_ms), self._last_timestamp + 1)
        self._last_timestamp = timestamp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        pose_result = self._pose.detect_for_video(image, timestamp)
        hand_result = self._hands.detect_for_video(image, timestamp)

        pose = None
        if pose_result.pose_landmarks:
            pose = np.array([(p.x * width, p.y * height, p.visibility or 0.0) for p in pose_result.pose_landmarks[0]],
                            dtype=np.float32)
        hands = [np.array([(p.x * width, p.y * height) for p in hand], dtype=np.float32)
                 for hand in hand_result.hand_landmarks]
        return RawLandmarks(pose=pose, hands=hands, width=width, height=height)

    def close(self):
        self._pose.close()
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
