"""Splits the live frame stream into single signs: hands up, sign, hands down"""
import numpy as np

from isl.features import hands_up, to_sequence

START_SECONDS = 0.1  # hands up this long starts a sign
END_SECONDS = 0.3  # hands down this long ends it
MIN_SIGN_SECONDS = 0.3  # shorter "signs" are ignored as blips
MAX_SIGN_SECONDS = 4.0  # longer ones are cut off and recognised anyway


class Segmenter:
    def __init__(self, start_seconds=START_SECONDS, end_seconds=END_SECONDS, min_seconds=MIN_SIGN_SECONDS,
                 max_seconds=MAX_SIGN_SECONDS):
        self.start_seconds, self.end_seconds = start_seconds, end_seconds
        self.min_seconds, self.max_seconds = min_seconds, max_seconds
        self.reset()

    def reset(self):
        self.signing = False
        self._frames = []
        self._first_up = None  # when the hands went up
        self._last_up = None  # latest frame with the hands up
        self._down_since = None

    def update(self, features, now):
        """Feed one frame; returns the model input for a sign that has just ended, else None"""
        up = hands_up(features)
        if not self.signing:
            if not up:
                self.reset()
                return None
            if self._first_up is None:
                self._first_up = now
            self._frames.append(features)
            self._last_up = now
            self.signing = now - self._first_up >= self.start_seconds
            return None

        self._frames.append(features)
        if up:
            self._last_up, self._down_since = now, None
        elif self._down_since is None:
            self._down_since = now
        ended = self._down_since is not None and now - self._down_since >= self.end_seconds
        if not ended and now - self._first_up < self.max_seconds:
            return None
        frames, duration = np.stack(self._frames), self._last_up - self._first_up
        self.reset()
        return to_sequence(frames) if duration >= self.min_seconds else None
