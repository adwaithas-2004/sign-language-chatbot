# Testing the ISL interpreter

About 30 minutes. Tested on Windows 11 with Python 3.12.

## What you need

- A Windows PC with a webcam and speakers or headphones
- **Python 3.10, 3.11 or 3.12** (3.12 recommended). Python 3.13 won't work: TensorFlow 2.18 doesn't support it.
  Run `py --list` to see which versions you have.
- Internet, for installing packages, a one-time 14 MB model download and the chatbot
- A free Groq API key from https://console.groq.com/keys
- Both ZIPs: `sign-language-chatbot.zip` and `sign-language-chatbot-example-videos.zip`

## 1. Unpack

Extract both ZIPs into the **same folder**, for example `C:\isl`. Windows' "Extract All" suggests a new folder named
after each ZIP, so change the destination to `C:\isl` both times. A folder outside OneDrive is best, because setup
adds about 2 GB of Python packages. You should end up with:

```
C:\isl\sign-language-chatbot\signlan.py
C:\isl\sign-language-chatbot\data\examples\hello.MOV
```

## 2. Install

Open a terminal in `C:\isl\sign-language-chatbot` (in File Explorer, right-click the folder → Open in Terminal):

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

This takes a few minutes; TensorFlow is a large download.

## 3. Add your Groq key

```
copy .env.example .env
notepad .env
```

Replace `your-groq-api-key-here` with your key and save. Don't share this file: it holds your key.

## 4. Automated tests (1–2 minutes)

These need no webcam, key or internet:

```
.venv\Scripts\python -m unittest discover -s tests -t .
```

Expected, at the end:

```
Ran 97 tests in ...s

OK (skipped=2)
```

The 2 skipped tests need the MediaPipe model files, which download the first time you run the app (step 6).
After that, the same command shows `OK` with nothing skipped.

## 5. Webcam check (optional)

```
.venv\Scripts\python Gesture.py
```

A window shows your camera. Press Esc to close it.

## 6. Live test

### Start the app

```
.venv\Scripts\python signlan.py
```

The first run downloads the MediaPipe models (14 MB). A window called "Sign Language Recognition" opens with a mirrored
camera view.

Sit so the camera sees you from a little above your head down to at least your chest, ideally your waist, with
light from the front. At a laptop, tilting the screen changes how much of you it sees. Lines are drawn over your
arms and hands, and the top left says **Sign a word**. If it says **Move back so your shoulders are visible** or
**Move back or tilt the camera down**, the camera is too close.

### Learn a few signs

In a second terminal in the same folder:

```
.venv\Scripts\python -m isl.examples
.venv\Scripts\python -m isl.examples hello
```

The first command lists the 50 words; the second plays how "hello" is signed (Esc stops it). The list shows some
words joined up (`thankyou`, `goodmorning`, `storeorshop`), but you can type them normally: `thank you`,
`good morning`, `shop`.

Good words to start with: hello, thank you, good morning, happy, I.

### Checks

Between signs, lower your hands to your lap or out of view.

| # | Do this | Expected |
|---|---|---|
| 1 | Sign **hello**, then lower your hands | Within a second, the top left shows `hello (NN%)` in green, and the bottom shows `Signed: hello` |
| 2 | Sign 2–3 words (lowering your hands after each), then keep your hands down | A ring at the top right fills over 2 seconds. The status shows `Thinking...` then `Speaking...`, you hear a sentence, and the bottom shows `Said: ...` |
| 3 | Sign two words, then press **Backspace** before the ring fills | The last word disappears from `Signed:` |
| 4 | Make a random movement that isn't one of the 50 signs | Ideally nothing, or a grey `? maybe: ...` that isn't added to `Signed:`. If a word is added, note which |
| 5 | Rest your hands on the desk or your lap for 15 seconds | Nothing is recognised. If words appear, please report how your hands were resting |
| 6 | Start a sign, then lean out of the camera's view | No word is added, and the status asks you to move back |
| 7 | Sign while the status says `Speaking...` | Ignored |
| 8 | Lean in until the camera only sees your head and shoulders | The status says `Move back or tilt the camera down`; it goes away when you sit back |
| 9 | Press **Esc** (or close the window) | The app closes |

### Word coverage (optional, 10 minutes)

Try 10 more words from the list. Known weak spots: **priest**, **fall** and **shop** often don't register, and
**court** and **hot** are sometimes read as "shop". Your own way of signing a word may differ from the reference
video's; for example, a salute-style "hello" tends to come out as "boy".

## What to send back

1. The results of checks 1–9 (pass, or what happened instead)
2. For each word you tried:

   | Word signed | What appeared (word and %, `? maybe: ...`, or nothing) | Correct? |
   |---|---|---|
   | hello | hello (97%) | yes |

3. Your Python version (`.venv\Scripts\python --version`), webcam and lighting
4. Any errors: copy the text from the terminal running `signlan.py`

## Troubleshooting

| You see | Fix |
|---|---|
| `GROQ_API_KEY is not set` | Do step 3. The file must be called `.env` (not `.env.txt`) and sit next to `signlan.py` |
| `Could not open the webcam` | Close other apps using the camera (Teams, Zoom, Camera). In Windows Settings → Privacy & security → Camera, allow desktop apps |
| `Could not download the MediaPipe model ...` | Check your internet, or download the file from the URL shown and save it where the message says |
| `(no reply - see the console for the error)` | Read the `Chatbot error:` line in the terminal: usually a wrong key, no internet, or Groq's free rate limit (wait a minute) |
| `Move back or tilt the camera down` | The camera sees too little below your shoulders, which makes recognition worse. Tilt the screen or move back until the message goes away |
| `No example video for ...` | The videos ZIP isn't in place: check that `data\examples\` exists (step 1) |
| `pip install` fails on tensorflow | Wrong Python version: use 3.10–3.12 (step 2) |
| No sound | Check Windows' default audio output and volume |
