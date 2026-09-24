import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyttsx3
from dotenv import load_dotenv
from groq import APIError, Groq

from isl.features import frame_features, view_below_shoulders
from isl.landmarks import LandmarkExtractor
from isl.model import load_recogniser
from isl.segmenter import Segmenter

BASE_DIR = Path(__file__).resolve().parent

# Read GROQ_API_KEY (and optionally GROQ_MODEL) from a .env file next to this script
load_dotenv(BASE_DIR / ".env")

# llama3-8b-8192 has been shut down by Groq; set GROQ_MODEL in .env to use another model
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM_PROMPT = ("You are a sign language interpreter for a deaf or hard-of-hearing person. "
                 "You are given the signs they made, in order, as English words. "
                 "Turn them into one short, natural sentence in the first person that says what they mean, "
                 "to be spoken aloud to a hearing person. Reply with only that sentence, in at most 12 words.")

WORD_THRESHOLD = 0.5  # Only accept a recognised word the model gives at least this probability
SEND_AFTER_SECONDS = 2.0  # Stop signing this long to send the sentence
WORD_DISPLAY_SECONDS = 3.0  # How long the last recognised word stays on screen
BACKSPACE, ESC = 8, 27
# A camera that sees less than this many shoulder widths below the shoulders is too close: recognition drops off
# (see the closer-cameras table in reports/live_path.md)
MIN_VIEW_BELOW_SHOULDERS = 0.6

WINDOW_NAME = "Sign Language Recognition"
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, YELLOW, WHITE, GREY = (0, 200, 0), (0, 220, 255), (255, 255, 255), (170, 170, 170)

# MediaPipe hand joints and pose arm points to draw on the preview
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
                    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
ARM_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


class SentenceBuilder:
    """Collects recognised words; pausing after signing finishes the sentence"""

    def __init__(self, send_after=SEND_AFTER_SECONDS):
        self.send_after = send_after
        self.words = []
        self._idle_since = None

    def add(self, word, now):
        self.words.append(word)
        self._idle_since = now

    def remove_last(self, now):
        if self.words:
            self.words.pop()
        self._idle_since = now if self.words else None

    def progress(self, now):
        """How far (0 to 1) the pause has got towards sending"""
        if not self.words or self._idle_since is None:
            return 0.0
        return min((now - self._idle_since) / self.send_after, 1.0)

    def update(self, signing, now):
        """Call every frame; returns the finished words once the signer has paused long enough, else None"""
        if signing:
            self._idle_since = None
            return None
        if not self.words:
            return None
        if self._idle_since is None:
            self._idle_since = now
        if now - self._idle_since < self.send_after:
            return None
        words, self.words, self._idle_since = self.words, [], None
        return words


def interpret(client, signs):
    """Ask the LLM to turn a list of signs into a sentence to speak (None on failure)"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Signs: " + " / ".join(signs)},
    ]
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=GROQ_MODEL,
        )
    except APIError as e:  # Bad key, no internet, rate limit... keep the camera running
        print(f"Chatbot error: {e}")
        return None
    return (response.choices[0].message.content or "").strip().strip('"') or None


def speak(text):
    """Convert chatbot text to speech"""
    # On Windows a pyttsx3 engine only speaks on its first runAndWait(),
    # so create a fresh engine for every reply
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)  # Set speech speed
    engine.setProperty("volume", 1.0)  # Set volume level
    engine.say(text)
    engine.runAndWait()
    engine.stop()


class Responder:
    """Interprets and speaks a sentence in a background thread so the video keeps running"""

    def __init__(self, client):
        self.client = client
        self.status = ""  # What the bot is doing right now, shown on screen
        self.heard = []  # Signs of the last sentence sent
        self.reply = ""  # Last sentence spoken
        self._thread = None

    @property
    def busy(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, signs):
        self.heard, self.reply, self.status = signs, "", "Thinking..."
        self._thread = threading.Thread(target=self._run, args=(signs,), daemon=True)
        self._thread.start()

    def _run(self, signs):
        try:
            reply = interpret(self.client, signs)
            if reply is None:
                self.reply = "(no reply - see the console for the error)"
                return
            print(f"Chatbot Reply: {reply}")
            self.reply = reply
            self.status = "Speaking..."
            speak(reply)
        finally:
            self.status = ""

    def wait(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)


def wrap_text(text, max_width, scale, thickness=2):
    """Split text into lines no wider than max_width pixels"""
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and cv2.getTextSize(candidate, FONT, scale, thickness)[0][0] > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def shade(frame, top, bottom):
    """Darken a horizontal band of the frame so text on it is easy to read"""
    frame[top:bottom] = (frame[top:bottom] * 0.4).astype(np.uint8)


def draw_overlay(frame, prediction_text, confident, progress, status, captions):
    """Draw the last word, a progress ring, the status line and captions onto the frame"""
    height, width = frame.shape[:2]

    shade(frame, 0, 75)
    cv2.putText(frame, prediction_text, (10, 30), FONT, 0.8, GREEN if confident else GREY, 2)
    cv2.putText(frame, status, (10, 62), FONT, 0.7, YELLOW, 2)

    # Ring that fills up while waiting to send
    if progress > 0:
        centre, radius = (width - 40, 38), 24
        cv2.circle(frame, centre, radius, GREY, 2)
        cv2.ellipse(frame, centre, (radius, radius), -90, 0, 360 * progress,
                    GREEN if progress >= 1 else YELLOW, 5)

    lines = [line for caption in captions for line in wrap_text(caption, width - 20, 0.7)]
    if lines:
        top = height - 30 * len(lines) - 15
        shade(frame, top, height)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, top + 30 * (i + 1)), FONT, 0.7, WHITE, 2)


def draw_skeleton(display, raw):
    """Draw the tracked arms and hands onto the mirrored preview"""
    height, width = display.shape[:2]
    sx, sy = width / raw.width, height / raw.height

    def point(x, y):
        return int(width - x * sx), int(y * sy)

    if raw.pose is not None:
        for a, b in ARM_CONNECTIONS:
            if min(raw.pose[a, 2], raw.pose[b, 2]) >= 0.5:
                cv2.line(display, point(*raw.pose[a, :2]), point(*raw.pose[b, :2]), GREY, 2)
    for hand in raw.hands:
        for a, b in HAND_CONNECTIONS:
            cv2.line(display, point(*hand[a]), point(*hand[b]), GREEN, 2)
        for x, y in hand:
            cv2.circle(display, point(x, y), 3, YELLOW, -1)


def camera_too_close(raw):
    """Whether the camera sees too little of the body below the shoulders to recognise signs well"""
    view = view_below_shoulders(raw)
    return view is not None and view < MIN_VIEW_BELOW_SHOULDERS


def status_text(busy_status, body_found, signing, has_words, too_close=False):
    if busy_status:
        return busy_status
    if not body_found:
        return "Move back so your shoulders are visible"
    if signing:
        return "Signing..."
    if too_close:
        return "Move back or tilt the camera down"
    if has_words:
        return "Pause to send, or sign the next word"
    return "Sign a word"


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set. Copy .env.example to .env and paste in your key "
                         "from https://console.groq.com/keys")

    recogniser = load_recogniser()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise SystemExit("Could not open the webcam")

    segmenter = Segmenter()
    builder = SentenceBuilder()
    responder = Responder(Groq(api_key=api_key))
    last_word, last_word_ok, last_word_time = "", False, -WORD_DISPLAY_SECONDS
    print("Sign a word, then lower your hands. Pause for 2 seconds to send the sentence. "
          "Backspace removes the last word, Esc quits.")

    with LandmarkExtractor() as extractor:
        start = time.monotonic()
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            now = time.monotonic()
            # The model sees the camera image un-mirrored, like the INCLUDE videos
            raw = extractor.extract(frame, (now - start) * 1000)
            features, body_found = frame_features(raw)

            if responder.busy:
                segmenter.reset()  # Ignore signing while the bot is thinking or speaking
            else:
                sequence = segmenter.update(features, now)
                if sequence is not None:
                    guesses = recogniser.predict(sequence)
                    word, probability = guesses[0]
                    last_word_ok, last_word_time = probability >= WORD_THRESHOLD, now
                    if last_word_ok:
                        builder.add(word, now)
                        last_word = f"{word} ({probability:.0%})"
                    else:
                        last_word = "? maybe: " + " / ".join(guess for guess, _ in guesses)
                    print(f"Recognized Sign: {last_word}")
                sentence = builder.update(segmenter.signing, now)
                if sentence:
                    print(f"Sending: {' / '.join(sentence)}")
                    responder.start(sentence)

            # Show a mirrored preview with what was tracked, heard and said
            display = cv2.flip(frame, 1)
            draw_skeleton(display, raw)
            captions = []
            if builder.words:
                captions.append("Signed: " + " / ".join(builder.words))
            elif responder.heard:
                captions.append("Signed: " + " / ".join(responder.heard))
                if responder.reply:
                    captions.append("Said: " + responder.reply)
            status = status_text(responder.status if responder.busy else "", body_found, segmenter.signing,
                                 bool(builder.words), camera_too_close(raw))
            recent = now - last_word_time < WORD_DISPLAY_SECONDS
            draw_overlay(display, last_word if recent else "", last_word_ok, builder.progress(now), status,
                         captions)
            cv2.imshow(WINDOW_NAME, display)

            # Press "Esc" or close the window to exit; Backspace removes the last word
            key = cv2.waitKey(1)
            if key == ESC or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == BACKSPACE:
                builder.remove_last(now)

    # Release camera and close windows
    camera.release()
    cv2.destroyAllWindows()
    responder.wait(timeout=10)  # Let a reply that's being spoken finish


if __name__ == "__main__":
    main()
