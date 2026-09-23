import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyttsx3
from dotenv import load_dotenv
from groq import APIError, Groq
# Teachable Machine exports Keras 2 models. Keras 3 (bundled with TensorFlow >= 2.16)
# fails to load them ("Error when deserializing class 'DepthwiseConv2D' ... groups"),
# so load the model with tf-keras, the Keras 2 compatibility package.
from tf_keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"

# Read GROQ_API_KEY (and optionally GROQ_MODEL) from a .env file next to this script
load_dotenv(BASE_DIR / ".env")

# llama3-8b-8192 has been shut down by Groq; set GROQ_MODEL in .env to use another model
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM_PROMPT = ("You are a sign language interpreter for a deaf or hard-of-hearing person. "
                 "You are given the signs they made, in order, as English words. "
                 "Turn them into one short, natural sentence in the first person that says what they mean, "
                 "to be spoken aloud to a hearing person. Reply with only that sentence, in at most 12 words.")

# Show the wake word sign to start a sentence, make your signs, then show it again to send.
# Label text from labels.txt, without the number
WAKE_WORD = "done"
BACKGROUND = "Background"  # "no sign" class, never sent to the chatbot

CONFIDENCE_THRESHOLD = 80  # Only accept predictions above this confidence (%)
HOLD_SECONDS = 1.0  # A sign must be confidently held this long to count
RELEASE_SECONDS = 0.5  # ...and be gone this long before the same sign can count again

# Teachable Machine mirrors webcam samples by default (its "Flip" setting), so mirror the camera
# the same way. Set to False if your model was trained with Flip off or from uploaded photos
MIRROR = True

WINDOW_NAME = "Sign Language Recognition"
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, YELLOW, WHITE, GREY = (0, 200, 0), (0, 220, 255), (255, 255, 255), (170, 170, 170)


def load_labels(path):
    """Read Teachable Machine labels ("0 I love you") without the index prefix"""
    with open(path, encoding="utf-8") as f:
        return [line.strip().split(" ", 1)[-1] for line in f if line.strip()]


def preprocess(frame):
    """Turn a webcam frame into model input the way Teachable Machine trains:
    centre square crop, RGB colour order, 224x224, scaled to [-1, 1]"""
    height, width = frame.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    square = frame[top:top + side, left:left + side]

    image = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)  # OpenCV captures BGR, the model was trained on RGB
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image_array = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    return (image_array / 127.5) - 1  # Normalize to [-1,1]


class SignTracker:
    """Turns noisy per-frame predictions into a single event per deliberately held sign"""

    def __init__(self, hold_seconds=HOLD_SECONDS, threshold=CONFIDENCE_THRESHOLD, release_seconds=RELEASE_SECONDS):
        self.hold_seconds = hold_seconds
        self.threshold = threshold
        self.release_seconds = release_seconds
        self.last_fired = None  # Blocked from firing again until it has been released
        self.gone_since = None
        self.reset()

    def reset(self):
        """Drop the current hold (the last accepted sign stays blocked until released)"""
        self.label = None
        self.since = None

    def update(self, label, confidence, now=None):
        """Return the label once it has been confidently held for hold_seconds, else None.
        A sign fires once and must then be gone for release_seconds before it can fire again,
        so holding it longer, or a brief dip in confidence, doesn't repeat it"""
        now = time.monotonic() if now is None else now

        if self.last_fired is not None:
            if label == self.last_fired:  # Still showing it, even if less confidently
                self.gone_since = None
            elif self.gone_since is None:
                self.gone_since = now
            elif now - self.gone_since >= self.release_seconds:
                self.last_fired = None

        if confidence <= self.threshold:
            self.reset()
            return None
        if label != self.label:
            self.label, self.since = label, now
        if label == self.last_fired or now - self.since < self.hold_seconds:
            return None
        self.last_fired, self.gone_since = label, None
        return label

    def progress(self, now=None):
        """How far (0 to 1) the current sign is towards being accepted"""
        if self.label is None:
            return 0.0
        if self.label == self.last_fired:
            return 1.0
        now = time.monotonic() if now is None else now
        return min((now - self.since) / self.hold_seconds, 1.0)


class SentenceBuilder:
    """Collects accepted signs into a sentence: the wake word starts it, the wake word again finishes it"""

    def __init__(self):
        self.collecting = False
        self.signs = []

    def add(self, sign):
        """Feed an accepted sign. Returns the list of signs when the sentence is finished,
        an empty list if it was cancelled (wake word twice with nothing in between), otherwise None"""
        if sign == BACKGROUND:
            return None
        if not self.collecting:
            if sign == WAKE_WORD:
                self.collecting, self.signs = True, []
            return None
        if sign != WAKE_WORD:
            self.signs.append(sign)
            return None
        sentence, self.signs, self.collecting = self.signs, [], False
        return sentence


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
    """Draw the current prediction, hold-progress ring, status line and captions onto the frame"""
    height, width = frame.shape[:2]

    shade(frame, 0, 75)
    cv2.putText(frame, prediction_text, (10, 30), FONT, 0.8, GREEN if confident else GREY, 2)
    cv2.putText(frame, status, (10, 62), FONT, 0.7, YELLOW, 2)

    # Ring that fills up while a sign is held, turning green once it's accepted
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


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set. Copy .env.example to .env and paste in your key "
                         "from https://console.groq.com/keys")

    # Load the trained sign language model and its class labels
    model = load_model(MODEL_PATH, compile=False)
    class_names = load_labels(LABELS_PATH)
    if len(class_names) != model.output_shape[-1]:
        raise SystemExit(f"labels.txt has {len(class_names)} labels but the model predicts "
                         f"{model.output_shape[-1]} classes")

    # Open webcam
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise SystemExit("Could not open the webcam")

    tracker = SignTracker()
    builder = SentenceBuilder()
    responder = Responder(Groq(api_key=api_key))
    print(f'Show "{WAKE_WORD}" to start, sign your words, then show "{WAKE_WORD}" again to send. '
          f"Press Esc to quit.")

    while True:
        # Capture webcam image
        ret, frame = camera.read()
        if not ret:
            break
        if MIRROR:
            frame = cv2.flip(frame, 1)

        # Predict the sign language gesture
        prediction = model(preprocess(frame), training=False).numpy()
        index = int(np.argmax(prediction))
        class_name = class_names[index]  # Get predicted label
        confidence_score = prediction[0][index] * 100  # Convert to percentage

        if responder.busy:
            tracker.reset()  # Ignore signs while the bot is thinking or speaking
        else:
            sign = tracker.update(class_name, confidence_score)
            if sign is not None:
                print(f"Recognized Sign: {sign} (Confidence: {confidence_score:.2f}%)")
                was_collecting = builder.collecting
                sentence = builder.add(sign)
                if sentence:
                    print(f"Sending: {' / '.join(sentence)}")
                    responder.start(sentence)
                elif sentence == []:
                    print("Nothing signed, cancelled.")
                elif builder.collecting and not was_collecting:
                    print(f'Listening... sign your words, then "{WAKE_WORD}" to send.')

        # Show the live webcam feed with what the bot sees, hears and says
        if responder.busy:
            status = responder.status
        elif builder.collecting:
            status = f'Sign your words, then "{WAKE_WORD}" to send'
        else:
            status = f'Show "{WAKE_WORD}" to start'

        signed = builder.signs if builder.collecting else responder.heard
        captions = []
        if signed or builder.collecting:
            captions.append("Signed: " + (" / ".join(signed) or "..."))
        if responder.reply and not builder.collecting:
            captions.append("Said: " + responder.reply)

        show_ring = class_name != BACKGROUND and not responder.busy
        draw_overlay(frame, f"{class_name} ({confidence_score:.0f}%)", confidence_score > CONFIDENCE_THRESHOLD,
                     tracker.progress() if show_ring else 0.0, status, captions)
        cv2.imshow(WINDOW_NAME, frame)

        # Press "Esc" (ASCII 27) or close the window to exit
        if cv2.waitKey(1) == 27 or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release camera and close windows
    camera.release()
    cv2.destroyAllWindows()
    responder.wait(timeout=10)  # Let a reply that's being spoken finish


if __name__ == "__main__":
    main()
