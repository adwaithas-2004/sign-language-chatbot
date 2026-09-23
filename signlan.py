import os
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

SYSTEM_PROMPT = ("You are a helper assisting deaf and hard-of-hearing individuals by interpreting sign language. "
                 "you have to give them short reply in 8 words.")

# Define wake word gesture (label text from labels.txt, without the number)
WAKE_WORD = "done"
BACKGROUND = "Background"  # "no sign" class, never sent to the chatbot

CONFIDENCE_THRESHOLD = 80  # Only accept predictions above this confidence (%)
HOLD_SECONDS = 1.0  # A sign must be confidently held this long to count


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

    def __init__(self, hold_seconds=HOLD_SECONDS, threshold=CONFIDENCE_THRESHOLD):
        self.hold_seconds = hold_seconds
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.label = None
        self.since = None
        self.fired = False

    def update(self, label, confidence, now=None):
        """Return the label once it has been confidently held for hold_seconds, else None.
        Fires only once per hold, so a sign kept up longer isn't repeated"""
        now = time.monotonic() if now is None else now
        if confidence <= self.threshold:
            self.reset()
            return None
        if label != self.label:
            self.label, self.since, self.fired = label, now, False
        if self.fired or now - self.since < self.hold_seconds:
            return None
        self.fired = True
        return label


def customLLMBot(client, messages, user_input):
    """Send the recognised sign to the chatbot and return its reply (None on failure)"""
    messages.append({"role": "user", "content": user_input})
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=GROQ_MODEL,
        )
    except APIError as e:  # Bad key, no internet, rate limit... keep the camera running
        messages.pop()
        print(f"Chatbot error: {e}")
        return None

    LLM_reply = (response.choices[0].message.content or "").strip()
    messages.append({"role": "assistant", "content": LLM_reply})
    return LLM_reply


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


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set. Copy .env.example to .env and paste in your key "
                         "from https://console.groq.com/keys")

    # Initialize Groq client
    client = Groq(api_key=api_key)
    messages_prmt = [{"role": "system", "content": SYSTEM_PROMPT}]

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
    wake_word_detected = False  # Flag to track if wake word has been used
    print(f'Show the "{WAKE_WORD}" sign to wake the bot, then show your message. Press Esc to quit.')

    while True:
        # Capture webcam image
        ret, frame = camera.read()
        if not ret:
            break

        # Predict the sign language gesture
        prediction = model(preprocess(frame), training=False).numpy()
        index = int(np.argmax(prediction))
        class_name = class_names[index]  # Get predicted label
        confidence_score = prediction[0][index] * 100  # Convert to percentage

        # Show the live webcam feed with the current prediction
        status = "Listening for a sign..." if wake_word_detected else f'Show "{WAKE_WORD}" to start'
        cv2.putText(frame, f"{class_name} ({confidence_score:.0f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Sign Language Recognition", frame)

        sign = tracker.update(class_name, confidence_score)
        if sign is not None:
            print(f"Recognized Sign: {sign} (Confidence: {confidence_score:.2f}%)")
            if not wake_word_detected:
                if sign == WAKE_WORD:
                    print("Wake Word Detected! Ready for command.")
                    wake_word_detected = True
            elif sign not in (WAKE_WORD, BACKGROUND):
                chatbot_response = customLLMBot(client, messages_prmt, sign)
                if chatbot_response:
                    print(f"Chatbot Reply: {chatbot_response}")

                    # Convert chatbot text to speech
                    speak(chatbot_response)

                # Reset wake word flag after responding; the next sign must be held afresh
                wake_word_detected = False
                tracker.reset()

        # Press "Esc" (ASCII 27) to exit
        if cv2.waitKey(1) == 27:
            break

    # Release camera and close windows
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
