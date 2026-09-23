# ISL word recognition (INCLUDE-50) — design

**Date:** 2026-09-23
**Status:** Implemented

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

- INCLUDE on Zenodo, record 4010759, licence **CC-BY-4.0**. 15 categories in **44 zips** of 0.8–1.8 GB (56.8 GB
  total). Zenodo serves HTTP range requests (verified: `206 Partial Content`). Zip members are named like the
  split paths except that some are `.MP4` where the split says `.MOV`; they are deflate-compressed and average
  15.7 MB per video.
- Official split lists and label maps: GitHub `AI4Bharat/INCLUDE` (MIT licence, branch `master`),
  `train_test_paths/include50_{train,val,test}.txt` and `label_maps/label_map_include50.json`. These four small files
  are **committed** into `isl/include50/` with a `SOURCE.md` credit, so nothing needs fetching to read the splits.
- INCLUDE-50: **958 videos** (all `.MOV`), 14–25 per word, spread over **all 15 categories**. Splits: train 689,
  val 77, test 192, with no overlap. The **test split covers 49 of the 50 words**.
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
  include50/                    official split lists + label map (from AI4Bharat/INCLUDE, MIT) + SOURCE.md
  landmarks.py                  MediaPipe Tasks wrapper: frame -> raw hand + pose points
  features.py                   RawLandmarks, normalisation, hands-up test, trimming, resampling, flip
  augment.py                    training-time random variations
  segmenter.py                  live frame stream -> complete signs
  model.py                      build / save / load / predict top-k
  include_data.py               split + label parsing, remote zip index, single-video fetch
  examples.py                   `python -m isl.examples <word>` plays the reference video
scripts/
  prepare_include50.py          fetch -> landmarks -> delete, resumable
  train.py                      baseline + BiGRU training and evaluation (5 seeds, mean ± std)
  evaluate_live.py              replays the videos through the live segmenter: live-path accuracy, trigger rates
models/                         isl50_bigru.keras, labels.json (committed);
                                MediaPipe .task files (downloaded on first use, git-ignored)
reports/                        metrics.json, confusion_matrix.png, results.md (committed)
data/                           git-ignored: include_index.json, landmarks/, examples/, tmp/, failed.txt
tests/fakes.py                  synthetic landmarks shared by the tests
tests/                          unit, smoke and scripted main-loop tests
docs/superpowers/specs/         this document
```

`Gesture.py` (webcam check) stays. `keras_model.h5`, `labels.txt` and `test_signlan.py` are removed or replaced;
tests move into `tests/`.

## Components

### `isl/include_data.py`

- `load_split(split) -> list[Sample]` for `"train" | "val" | "test"`, where
  `Sample = (path, key, display, split)`, read from the committed files in `isl/include50/`.
  `load_label_keys()` gives the 50 keys in the official class order.
- `label_key(folder)` and `display_name(folder)` implement the naming rules below.
- `HttpSource(url)`: byte ranges of a remote file (stdlib `urllib`), with retries. It refuses a response that
  isn't `206 Partial Content`, so a server ignoring the range can't start a 1.3 GB download.
- `list_members(source, zip_name)`: reads a zip's central directory through a seekable, read-ahead wrapper handed to
  `zipfile.ZipFile` (about 3 requests per zip).
- `extract_member(source, member, dest)`: fetches one member's compressed bytes in **a single range request**,
  inflates them while streaming to `dest.part`, checks the CRC, then renames to `dest`. Going through `zipfile` for
  this would make thousands of tiny requests.
- `build_index(paths) -> {path: Member}`: reads the directories of the zips in the needed categories once and caches
  the member records (zip, offset, sizes, CRC, method) in `data/include_index.json`. Paths are matched **without
  their extension, ignoring case**: the split lists call every video `.MOV`, but some are `.MP4` inside the zips
  (7 of the 29 Seasons videos, for example).
- `fetch_video(member, dest)`: `extract_member` from the right Zenodo zip.

**Display names** (what the interpreter receives): the word folder without its number, lowercased except `I`, with
overrides `biglarge → big`, `smalllittle → small`, `storeorshop → shop`. Stored in `models/labels.json`.

### `isl/landmarks.py`

`LandmarkExtractor` wraps MediaPipe Tasks (`mediapipe==1.0.1`, verified to install alongside TensorFlow 2.18 with no
conflicts; the legacy `mp.solutions` API no longer exists):

- `PoseLandmarker` (lite model) and `HandLandmarker` (`num_hands=2`), both in `VIDEO` running mode with increasing
  timestamps.
- Frames are downscaled to 640 px wide before detection, the same for dataset videos and the webcam.
- `extract(frame_bgr, timestamp_ms) -> RawLandmarks` (pixel-space pose points with visibility, 0–2 hands of 21
  points each). `RawLandmarks` is a plain dataclass defined in `isl/features.py`, so the feature code and its tests
  don't depend on MediaPipe.
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
- `frame_features(raw) -> (features, body_found)`.
- **Hands up:** `hands_up(features)` is true when at least one hand is present with its wrist above
  `shoulder_mid_y + HANDS_UP_K × shoulder_width` (image y grows downwards). It is computed **from the feature vector**
  (for one frame or a whole sequence), so changing `HANDS_UP_K` never requires re-extracting videos.
  `HANDS_UP_K = 1.25` decides **trimming** (which frames the model sees) and is calibrated per frame on the train
  split. The live segmenter uses its own looser `SEGMENT_HANDS_UP_K = 1.5` to decide when signs **start and end**,
  chosen with `DROPOUT_SECONDS` by **live-path accuracy**: `scripts/evaluate_live.py` replays every prepared video
  through the segmenter and saved model (one shared threshold: 91.1 %; split thresholds: 93.2 %).
- `trim(frames, up)`: keep from the first to the last hands-up frame with 2 frames of padding; if fewer than 4 frames
  are up, keep the whole clip.
- `resample(frames, 32)`: linear interpolation along time to exactly `SEQUENCE_LENGTH = 32` frames.
- `flip(sequence)`: mirror the signer: negate body-frame and hand-local x, swap left/right hand blocks and presence
  flags, swap left/right pose points.

### `scripts/prepare_include50.py`

For every video in the three INCLUDE-50 splits (processed test first, then val, then train), skipping any whose
`.npz` already exists:

1. `fetch_video` into `data/tmp/`
2. run `LandmarkExtractor` over every frame (timestamps from the video's frame rate)
3. save `data/landmarks/<split>/<key>/<video-stem>.npz` with `features (T×184)`, `fps`, `key`
4. keep the file as `data/examples/<key>.MOV` if that word has no example yet, else delete it. Because test videos
   come first, examples are test videos; the one word missing from the test split gets a val video.

Errors on one video are logged to `data/failed.txt` and the run continues. The run prints progress and a summary.
Zenodo gives each connection only about 0.3 MB/s (measured; 8 parallel connections reached 4.2 MB/s), so videos
are **downloaded 8 at a time**, at most 16 ahead of the one being processed, while extraction stays sequential and in
order. Peak disk use is about 16 videos (~250 MB) plus the examples folder (about 0.8 GB).
Expected cost: ~15 GB download, ~1–1.5 hours in total.

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
- **Augmentation** (`isl/augment.py`, training batches only): random flip (p = 0.5), scale ±10 %, rotation ±10°,
  speed change and crop ±15 % before resampling, random frame dropout, Gaussian noise.
- **Evaluation** on the official **test** split for both models: top-1, top-3, macro-F1 (over the words present in
  the test split), per-word accuracy, inference time per sign.
- **Writes:** `models/isl50_bigru.keras`, `models/labels.json`, `reports/metrics.json`,
  `reports/confusion_matrix.png`, `reports/results.md` (baseline vs BiGRU vs the paper's 94.5 % on INCLUDE-50).
- `--smoke` runs on a tiny synthetic dataset for 2 epochs (used by tests).

### `isl/segmenter.py`

`Segmenter.update(features, now) -> sequence | None`: returns the model-ready `(32, 184)` sequence for a sign that
has just ended (hands up is derived from the features).

- **Start:** real hands-up frames spanning `START_SECONDS = 0.1`.
- **End:** hands down for `END_SECONDS = 0.3`; the sign is the frames from its start to the last hands-up frame.
- **Tracking dropouts:** a frame where MediaPipe finds no hand (but the body is visible) within
  `DROPOUT_SECONDS = 0.5` of the last raised hand counts as still signing, since MediaPipe often loses fast-moving
  hands mid-sign. A hand seen below the threshold still counts as lowered straight away.
- **Body lost:** frames without shoulders are skipped; if they last `BODY_LOST_SECONDS = 0.3`, the sign in progress
  is discarded rather than guessed at.
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
- **Replace:** `opencv-python` with `opencv-contrib-python`. MediaPipe 1.0.1 requires the contrib package, and both
  install the same `cv2` module, so only the contrib one is kept (it's a superset).
- **Remove:** `tf-keras` (once the Teachable Machine code is gone).
- **Keep:** `tensorflow==2.18.0` (provides Keras 3), `numpy<2.1`, `pyttsx3`, `groq`, `python-dotenv`.

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
