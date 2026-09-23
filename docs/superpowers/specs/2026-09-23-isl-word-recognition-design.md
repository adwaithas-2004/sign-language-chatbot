# ISL word recognition (INCLUDE-50) — design

**Date:** 2026-09-23
**Status:** Approved in conversation, pending written-spec review

## Goal

Replace the one-frame Teachable Machine recogniser with an **Indian Sign Language (ISL) word recogniser** that
understands signs involving movement, trained on the public **INCLUDE-50** benchmark, and feed recognised words into
the existing interpreter → speech flow.

The project is a **portfolio / GitHub showcase**: the value is a reproducible end-to-end pipeline
(dataset → landmarks → trained model → evaluation → real-time app) with honest, measured results.

### Decisions made

| Topic | Decision |
|---|---|
| Sign language | ISL |
| Vocabulary | INCLUDE-50: the paper's official 50-word subset, official train/val/test split |
| Model | Bidirectional GRU on landmark sequences, plus a logistic-regression baseline |
| Sending a sentence | Pause to send: 2 s without signing sends the sentence |
| Old model | Removed (`keras_model.h5`, `labels.txt`, `tf-keras`); remains in git history |

### Non-goals

- Full INCLUDE (263 words). The pipeline takes the split name as a parameter, so it's a later one-flag change,
  but it is not built, run or evaluated now.
- Signer-independent evaluation, recording/fine-tuning on the user's own signs, the Transformer variant,
  speech-to-text. All are possible follow-ups.
- GPU training. TensorFlow on native Windows has no GPU support; the model is sized for CPU.

## Dataset facts (verified)

- INCLUDE on Zenodo, record 4010759, licence **CC-BY-4.0**. 15 categories in 43 zips of 0.8–1.8 GB (56.8 GB total).
  Zenodo serves HTTP range requests (verified: `206 Partial Content`).
- Official split lists and label maps: GitHub `AI4Bharat/INCLUDE`, `train_test_paths/include50_{train,val,test}.txt`
  and `label_maps/label_map_include50.json`.
- INCLUDE-50: **958 videos** (all `.MOV`), 14–25 per word, spread over **all 15 categories**.
- Paths look like `Greetings/48. Hello/MVI_0089.MOV`; 15 have an extra level:
  `Places/19. House/Extra/MVI_3439.MOV`. The **second path part is always the word folder**.
- Label key = word folder without the `NN. ` prefix, lowercased, non-alphanumerics removed
  (`48. Hello` → `hello`, `T-Shirt` → `tshirt`, `you (plural)` → `youplural`). Verified: the 50 folders map exactly
  onto the 50 keys of `label_map_include50.json`.
- The split files have no trailing newline; parse with `splitlines()`.

## Architecture

The training pipeline and the live app share **one feature-extraction module**, so the model always sees landmarks
computed the same way.

```
Offline (run once)                                   Live app (signlan.py)
Zenodo zips --range fetch--> prepare_include50.py    webcam frame (un-mirrored)
      video (temp, deleted after)                           |
            |                                               v
            v                                  isl/landmarks.py + isl/features.py   (shared)
isl/landmarks.py + isl/features.py (shared)                 |
            |  data/landmarks/<split>/<key>/<video>.npz      v
            v                                        isl/segmenter.py
scripts/train.py --> models/isl50_bigru.keras --> isl/model.py (top-k)
            |          models/labels.json                    |
            v                                               v
reports/ (metrics, confusion matrix, results.md)     SentenceBuilder (pause to send)
                                                     --> Responder --> Groq interpreter --> speech (unchanged)
```

### Repository layout

```
signlan.py                      live app (entry point)
isl/
  __init__.py
  landmarks.py                  MediaPipe Tasks wrapper: frame -> raw hand + pose points
  features.py                   normalisation, hands-up test, trimming, resampling, flip
  segmenter.py                  live frame stream -> complete signs
  model.py                      build / save / load / predict top-k
  include_data.py               split + label parsing, remote zip index, single-video fetch
  examples.py                   `python -m isl.examples <word>` plays the reference video
scripts/
  prepare_include50.py          fetch -> landmarks -> delete, resumable
  train.py                      baseline + BiGRU training and evaluation
models/                         isl50_bigru.keras, labels.json (committed);
                                MediaPipe .task files (downloaded on first use, git-ignored)
reports/                        metrics.json, confusion_matrix.png, results.md (committed)
data/                           git-ignored: include_index.json, landmarks/, examples/, tmp/, failed.txt
tests/                          unit, smoke and scripted main-loop tests
docs/superpowers/specs/         this document
```

`Gesture.py` (webcam check) stays. `keras_model.h5`, `labels.txt` and `test_signlan.py` are removed or replaced;
tests move into `tests/`.

## Components

### `isl/include_data.py`

- `load_split(name) -> list[Sample]`, where `Sample = (path, key)`; `name` is `include50_train|val|test`.
  Split files and the label map are fetched once from GitHub into `data/meta/` and cached.
- `label_key(folder)` and `display_name(key)` implement the naming rules below.
- `RemoteZip(url)`: a seekable read-only file object over HTTP range requests (stdlib `urllib`), given to
  `zipfile.ZipFile` so the central directory and single members are read without downloading the whole archive.
- `build_index() -> {video_path: zip_name}`: reads the central directory of each of the 43 zips once and caches it in
  `data/include_index.json`. Members are matched by path suffix (the zips may add a top-level folder).
- `fetch_video(path, dest)`: streams one member to `dest`.

**Display names** (what the interpreter receives): the word folder without its number, lowercased except `I`, with
overrides `biglarge → big`, `smalllittle → small`, `storeorshop → shop`. Stored in `models/labels.json`.

### `isl/landmarks.py`

`LandmarkExtractor` wraps MediaPipe Tasks (`mediapipe==1.0.1`, verified to install alongside TensorFlow 2.18 with no
conflicts; the legacy `mp.solutions` API no longer exists):

- `PoseLandmarker` (lite model) and `HandLandmarker` (`num_hands=2`), both in `VIDEO` running mode with increasing
  timestamps.
- Frames are downscaled to 640 px wide before detection, the same for dataset videos and the webcam.
- `extract(frame_bgr, timestamp_ms) -> RawLandmarks` (pixel-space pose points with visibility, 0–2 hands of 21
  points each).
- The `.task` model files are downloaded to `models/` on first use from Google's `mediapipe-models` storage; if that
  fails, the error message gives the URLs to download by hand.

### `isl/features.py`

Per frame, from `RawLandmarks` (all coordinates in **pixels**, which removes the 16:9 vs 4:3 difference):

- **Pose points used (7):** nose 0, shoulders 11/12, elbows 13/14, wrists 15/16.
- **Body frame:** origin = midpoint of the shoulders, scale = shoulder width. If either shoulder has visibility < 0.5
  the frame is **invalid** (all-zero features, hands down).
- **Hand assignment:** each detected hand goes to the nearer pose wrist (subject's left = pose 15, right = 16).
  MediaPipe's handedness label isn't used.
- **Feature vector (184 floats):**
  - pose: 7 points × (x, y) in the body frame = 14
  - per hand (left, then right): 21 points × (x, y) in the body frame = 42, plus 21 × (x, y) relative to the hand's
    wrist (point 0) scaled by wrist → middle-finger knuckle (point 9) distance = 42 → 84 × 2 = 168
  - presence flag per hand = 2
  - a missing hand's block is zeros with presence 0
- **Hands up:** at least one hand present with wrist y above `shoulder_mid_y + HANDS_UP_K × shoulder_width`
  (image y grows downwards). `HANDS_UP_K` starts at 1.5 and is calibrated on real INCLUDE videos during
  implementation (target: rest poses at the start and end of clips count as down).
- `trim(frames, up)`: keep from the first to the last hands-up frame with 2 frames of padding; if fewer than 4 frames
  are up, keep the whole clip.
- `resample(frames, 32)`: linear interpolation along time to exactly `SEQUENCE_LENGTH = 32` frames.
- `flip(sequence)`: mirror the signer: negate body-frame and hand-local x, swap left/right hand blocks and presence
  flags, swap left/right pose points.

### `scripts/prepare_include50.py`

For every video in the three INCLUDE-50 splits, skipping any whose `.npz` already exists:

1. `fetch_video` into `data/tmp/`
2. run `LandmarkExtractor` over every frame (timestamps from the video's frame rate)
3. save `data/landmarks/<split>/<key>/<video-stem>.npz` with `features (T×184)`, `hands_up (T)`, `fps`, `key`
4. keep the file in `data/examples/<key>.MOV` if it's the first **test**-split video for that word, else delete it

Errors on one video are logged to `data/failed.txt` and the run continues. The run prints progress and a summary.
Peak disk use is one video plus the examples folder (about 0.6 GB).
Expected cost: ~13 GB download, ~1 hour of CPU extraction.

### `isl/model.py`

- `build_model(num_classes, sequence_length=32, num_features=184)`, Keras 3 (bundled with TensorFlow 2.18):
  `Input → TimeDistributed(Dense(128, relu)) → Dropout(0.3) → Bidirectional(GRU(128, return_sequences=True))
  → Bidirectional(GRU(64)) → Dropout(0.4) → Dense(num_classes, softmax)`, about 300k parameters.
- `load(path) -> Recogniser`; `Recogniser.predict(sequence) -> list[(key, probability)]` (top-k, default 3).
- `models/labels.json` stores the ordered keys, display names, `sequence_length`, `num_features` and a
  `feature_version`. The app refuses to start if these don't match what `isl/features.py` produces.

### `scripts/train.py`

- Loads the `.npz` files, applies `trim` and `resample` (the same functions the live app uses).
- **Baseline:** per-feature mean, std, min and max over time → `StandardScaler` + `LogisticRegression`
  (scikit-learn).
- **BiGRU:** Adam (lr 1e-3), batch 32, up to 150 epochs, early stopping on the official **val** split
  (patience 25, restore best weights), fixed seeds.
- **Augmentation** (training batches only): random flip (p = 0.5), scale ±10 %, rotation ±10°, speed change and
  crop ±15 % before resampling, random frame dropout, Gaussian noise.
- **Evaluation** on the official **test** split for both models: top-1, top-3, macro-F1, per-word accuracy,
  inference time per sign.
- **Writes:** `models/isl50_bigru.keras`, `models/labels.json`, `reports/metrics.json`,
  `reports/confusion_matrix.png`, `reports/results.md` (baseline vs BiGRU vs the paper's 94.5 % on INCLUDE-50).
- `--smoke` runs on a tiny synthetic dataset for 2 epochs (used by tests).

### `isl/segmenter.py`

`Segmenter.update(features, hands_up, now) -> Sign | None`, where a `Sign` holds that sign's frames.

- **Start:** hands up for `START_SECONDS = 0.1`.
- **End:** hands down for `END_SECONDS = 0.3`; the sign is the frames from its start to the last hands-up frame.
- Signs shorter than `MIN_SIGN_SECONDS = 0.3` are dropped. At `MAX_SIGN_SECONDS = 4.0` the sign is emitted anyway.
- The emitted frames go through the same `trim` / `resample` as training.
- `signing` property for the on-screen status.

### `signlan.py` (live app)

Loop: read frame → `LandmarkExtractor` on the **un-mirrored** frame → features → `Segmenter` → on a completed sign,
`Recogniser.predict` → words → sentence → `Responder`.

- **Word acceptance:** the top-1 probability must be ≥ `WORD_THRESHOLD = 0.5`, otherwise the screen shows
  "? maybe: a / b / c" for 2 s and nothing is added.
- **SentenceBuilder (pause to send):** accepted words are appended. After `SEND_AFTER_SECONDS = 2.0` with no signing
  and a non-empty sentence, it's sent. **Backspace** removes the last word. Signing is ignored while the `Responder`
  is busy.
- **Keys:** Esc, or closing the window, quits (as now).
- **Screen** (preview mirrored for the user):
  - status: waiting / ● signing / Thinking… / Speaking…
  - the last word with its confidence, or the "maybe" list
  - a ring counting down to send
  - hand and arm skeleton overlay
  - `Signed:` / `Said:` captions
- **Kept unchanged:** `Responder`, `interpret()`, `speak()`, the Groq setup and `.env` handling, `wrap_text`,
  `shade`.
- **Removed:** `SignTracker`, `preprocess`, `MIRROR` for model input, the Teachable Machine loading code.
- The interpreter prompt keeps its role; the user message stays `Signs: w1 / w2 / ...`, using display names.

## Error handling

| Situation | Behaviour |
|---|---|
| `models/isl50_bigru.keras` or `labels.json` missing | Exit with "run `scripts/train.py` (or pull the repo)" |
| `labels.json` feature settings don't match `isl/features.py` | Exit with a mismatch message naming both values |
| MediaPipe `.task` download fails | Exit with the URLs and where to put the files |
| Shoulders not visible | Frame invalid; screen says "Move back so your shoulders are visible" |
| Groq / speech errors | As today: logged, `(no reply…)` caption, camera keeps running |
| One dataset video fails during prepare | Logged to `data/failed.txt`, run continues; train reports how many samples it used |

## Testing

- **Unit:**
  - `features`: translating or scaling the whole body leaves features unchanged; flip swaps hand blocks and
    applying it twice is the identity; resample gives 32 frames and keeps the endpoints; trim behaviour; an invalid
    frame gives zeros
  - `segmenter`: synthetic hands-up/down timelines with an injected clock, including blips, over-long signs and
    back-to-back signs
  - `SentenceBuilder`: pause-to-send timing, Backspace, nothing sent while empty
  - `include_data`: all 958 split paths map to one of the 50 keys; display-name overrides; `RemoteZip` reads a
    member from a local zip through a fake range reader
  - `model`: output shape, save/load round-trip, top-k ordering
- **Smoke:** `scripts/train.py --smoke` end to end. A real `prepare_include50.py` run on 2 videos, inspected by hand
  before the full run.
- **Scripted main loop:** fake camera, extractor and model drive `main()` through sign → pause → send → speak.
- **Live check:** Claude verifies tracking and segmentation on the webcam; the user signs a few words, using
  `python -m isl.examples <word>` to learn them.
- The existing tests for `interpret`, `Responder`, `wrap_text` and the overlay are kept (updated where signatures
  change).

## Dependencies

- **Add:** `mediapipe==1.0.1`, `scikit-learn`, `matplotlib` (already pulled in by MediaPipe).
- **Remove:** `tf-keras`.
- **Keep:** `tensorflow==2.18.0` (provides Keras 3), `opencv-python`, `numpy`, `pyttsx3`, `groq`, `python-dotenv`.
- MediaPipe 1.0.1 depends on `opencv-contrib-python`, which installs a second copy of the `cv2` module next to
  `opencv-python`. The implementation must check that the two don't clash and keep only one if they do.

## Success criteria

1. `prepare_include50.py` produces landmarks for ≥ 95 % of the 958 videos without filling the disk.
2. The BiGRU clearly beats the baseline and reaches **≥ 85 % top-1** on the official INCLUDE-50 test split.
   This number is reported honestly whatever it turns out to be.
3. The live app turns a signed word into a caption and spoken sentence, and the video stays live throughout.
4. All tests pass, and the README documents setup, the pipeline, results, limitations and the INCLUDE citation.

## Risks

- **Seen-signer test set:** INCLUDE has 7 signers shared across the splits, so test accuracy overstates accuracy on
  a new signer. The README says so. Fine-tuning on the user's own recordings is the follow-up.
- **Hands-up threshold:** it may need tuning per camera setup. It's a named constant, calibrated on INCLUDE and
  checked live.
- **Download reliability:** Zenodo throttling or timeouts are handled by retries in `RemoteZip` and the resumable
  prepare step.
- **Startup time:** TensorFlow import stays slow (10–60 s). Accepted for now.

## Attribution

INCLUDE: A. Sridhar, R. G. Ganesan, P. Kumar, M. Khapra, "INCLUDE: A Large Scale Dataset for Indian Sign Language
Recognition", ACM Multimedia 2020. Data under CC-BY-4.0 (Zenodo record 4010759). The README cites this and links the
record.
