# Sign Language Chatbot

A sign language interpreter for deaf and hard-of-hearing people. It recognises hand signs from your webcam with a
[Teachable Machine](https://teachablemachine.withgoogle.com/) model, turns them into a natural sentence with an LLM on
[Groq](https://groq.com/), and speaks that sentence out loud to a hearing person.

## How to use it

1. Show **"done"** to start a sentence.
2. Make your signs one after another (e.g. **"I love you"**). Put your hand down between signs.
3. Show **"done"** again to send. The bot says your sentence out loud and shows it on screen.

Hold each sign steady for about a second. The ring in the top-right corner fills up and turns green when the sign
is accepted. Showing "done" twice with no signs in between cancels. Press **Esc** or close the window to quit.

The video keeps running while the bot is thinking and speaking. The screen shows:

- **Top:** what the model sees right now, with its confidence, and what the bot is doing
- **Bottom:** the signs in the current sentence (`Signed:`) and the last sentence spoken (`Said:`)

## Setup (Windows)

Requires **Python 3.9 – 3.12** (TensorFlow 2.18 does not support 3.13+).

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key from https://console.groq.com/keys, then copy `.env.example` to `.env` and paste it in:

```
GROQ_API_KEY=your-key-here
```

## Run

```bash
python signlan.py
```

`Gesture.py` is a minimal webcam preview for checking that your camera works.

## Tests

```bash
python -m unittest -v
```

## Files

| File | Purpose |
|---|---|
| `signlan.py` | Main app: webcam → sign recognition → sentence → Groq interpreter → text-to-speech |
| `keras_model.h5`, `labels.txt` | Teachable Machine model and its class labels (`I love you`, `done`, `Background`) |
| `Gesture.py` | Webcam test |
| `test_signlan.py` | Unit tests |

## Notes

- Teachable Machine exports Keras 2 models, which Keras 3 can't load, so the model is loaded with `tf-keras`.
- To use a different Groq model, set `GROQ_MODEL` in `.env` (see https://console.groq.com/docs/models).
- To recognise your own signs, train an image model in Teachable Machine, export it as
  *Tensorflow → Keras*, and replace `keras_model.h5` and `labels.txt`. Update `WAKE_WORD` in `signlan.py` to match.
- The camera image is mirrored, like Teachable Machine's webcam (its **Flip** setting, on by default). If you train
  with Flip off or from uploaded photos, set `MIRROR = False` in `signlan.py`.
- Tuning knobs at the top of `signlan.py`: `CONFIDENCE_THRESHOLD` (lower it if signs are rarely accepted),
  `HOLD_SECONDS` (how long to hold a sign), and `SYSTEM_PROMPT` (how the interpreter phrases sentences).
