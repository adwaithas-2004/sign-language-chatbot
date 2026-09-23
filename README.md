# Sign Language Chatbot

Recognises hand signs from your webcam with a [Teachable Machine](https://teachablemachine.withgoogle.com/) model,
sends the recognised sign to an LLM on [Groq](https://groq.com/), and speaks the reply out loud.

**How to use it:** show the **"done"** sign to wake the bot, then show a sign (e.g. **"I love you"**).
Hold each sign steady for about a second. The bot replies in text and speech. Press **Esc** to quit.

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
| `signlan.py` | Main app: webcam → sign recognition → Groq chatbot → text-to-speech |
| `keras_model.h5`, `labels.txt` | Teachable Machine model and its class labels (`I love you`, `done`, `Background`) |
| `Gesture.py` | Webcam test |
| `test_signlan.py` | Unit tests |

## Notes

- Teachable Machine exports Keras 2 models, which Keras 3 can't load, so the model is loaded with `tf-keras`.
- To use a different Groq model, set `GROQ_MODEL` in `.env` (see https://console.groq.com/docs/models).
- To recognise your own signs, train an image model in Teachable Machine, export it as
  *Tensorflow → Keras*, and replace `keras_model.h5` and `labels.txt`. Update `WAKE_WORD` in `signlan.py` to match.
