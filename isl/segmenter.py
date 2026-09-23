"""Splits the live frame stream into single signs: hands up, sign, hands down"""
import numpy as np

from isl.features import PRESENCE_SLICE, hands_up, to_sequence

START_SECONDS = 0.1  # hands up this long starts a sign
END_SECONDS = 0.3  # hands down this long ends it
MIN_SIGN_SECONDS = 0.3  # shorter "signs" are ignored as blips
MAX_SIGN_SECONDS = 4.0  # longer ones are cut off and recognised anyway
DROPOUT_SECONDS = 0.5  # a hand MediaPipe loses for less than this doesn't count as lowered
BODY_LOST_SECONDS = 0.3  # losing the shoulders this long discards the sign in progress


class Segmenter:
    def __init__(self, start_seconds=START_SECONDS, end_seconds=END_SECONDS, min_seconds=MIN_SIGN_SECONDS,
                 max_seconds=MAX_SIGN_SECONDS, dropout_seconds=DROPOUT_SECONDS, body_lost_seconds=BODY_LOST_SECONDS):
        self.start_seconds, self.end_seconds = start_seconds, end_seconds
        self.min_seconds, self.max_seconds = min_seconds, max_seconds
        self.dropout_seconds, self.body_lost_seconds = dropout_seconds, body_lost_seconds
        self.reset()

    def reset(self):
        self.signing = False
        self._frames = []
        self._first_up = None  # when the hands went up
        self._last_up = None  # latest frame with the hands up
        self._down_since = None
        self._lost_since = None  # when the shoulders went out of view

    def update(self, features, now):
        """Feed one frame; returns the model input for a sign that has just ended, else None"""
        if not features.any():
            return self._body_lost(now)
        self._lost_since = None
        up = hands_up(features)
        # MediaPipe often loses a fast-moving hand for a few frames: that's not the same as lowering it
        bridged = (not up and not features[PRESENCE_SLICE].any() and self._last_up is not None
                   and now - self._last_up < self.dropout_seconds)
        if not self.signing:
            if not (up or bridged):
                self.reset()
                return None
            if self._first_up is None:
                self._first_up = now
            self._frames.append(features)
            if up:
                self._last_up = now
            self.signing = self._last_up - self._first_up >= self.start_seconds  # bridged frames don't count
            return None

        self._frames.append(features)
        if up:
            self._last_up, self._down_since = now, None
        elif not bridged and self._down_since is None:
            self._down_since = now
        ended = self._down_since is not None and now - self._down_since >= self.end_seconds
        if not ended and now - self._first_up < self.max_seconds:
            return None
        frames, duration = np.stack(self._frames), self._last_up - self._first_up
        self.reset()
        return to_sequence(frames) if duration >= self.min_seconds else None

    def _body_lost(self, now):
        """Shoulders out of view: a brief loss is skipped, a longer one discards the sign instead of guessing"""
        if self._first_up is not None:
            if self._lost_since is None:
                self._lost_since = now
            if now - self._lost_since >= self.body_lost_seconds:
                self.reset()
        return None
