# ISL Word Recognition (INCLUDE-50) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the one-frame Teachable Machine recogniser with an Indian Sign Language word recogniser trained on INCLUDE-50 (MediaPipe landmarks → BiGRU), with a reproducible data pipeline, measured results, and the live app sending sentences after a pause.

**Architecture:** One shared feature module (`isl/features.py`) turns MediaPipe hand + pose landmarks into body-normalised 184-number frames, used identically by the offline pipeline (Zenodo range-fetch → landmarks → `.npz` → training) and the live app (webcam → segmenter → model → sentence → existing Groq interpreter + speech). The model is a small Keras BiGRU over 32-frame sequences, compared against a logistic-regression baseline on the official test split.

**Tech Stack:** Python 3.12 (Windows), TensorFlow 2.18 / Keras 3, MediaPipe 1.0.1 Tasks API, OpenCV (contrib), NumPy, scikit-learn, matplotlib, unittest. Existing: groq, pyttsx3, python-dotenv.

**Spec:** `docs/superpowers/specs/2026-09-23-isl-word-recognition-design.md`

## Global Constraints

- Project root: `C:\Users\Adwaith A S\OneDrive\Desktop\Gesture Workshop\Gesture Workshop`. All commands run from there.
- Python is always the project venv: `.venv/Scripts/python`. Tests: `.venv/Scripts/python -m unittest discover -s tests -t . -v`.
- `tensorflow==2.18.0` (Keras 3 bundled), `mediapipe==1.0.1`, `opencv-contrib-python` only (never also `opencv-python`), `numpy<2.1`.
- `FEATURE_VERSION = 1`, `SEQUENCE_LENGTH = 32`, `NUM_FEATURES = 184`.
- INCLUDE-50 official splits only: 958 videos (train 689, val 77, test 192), 50 label keys from `label_map_include50.json`.
- Git: never commit `data/`, `.env` or `models/*.task`; do commit `models/isl50_bigru.keras`, `models/labels.json`, `reports/`. Every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. **Never push** without the user's explicit OK.
- Downloads need the user's explicit OK before they start (Task 0): MediaPipe model files (13.6 MB, storage.googleapis.com) and INCLUDE-50 videos (~15 GB streamed from zenodo.org, deleted after processing).
- Match the existing code style: short docstrings, sparse comments, 120-character lines, plain functions and small classes.

## Review Focus

- **Signer too close to the camera (shoulders out of frame):** the app must say "Move back so your shoulders are visible" and add no words, not crash or guess. Pinned by `test_hidden_shoulders_give_an_empty_invalid_frame` (Task 4) and `StatusTextTests` (Task 12).
- **Zenodo ignores the Range header, or the connection drops mid-video:** no 1.3 GB response read into memory, and no truncated video treated as complete. Pinned by `test_server_ignoring_ranges_is_an_error` and `test_corrupt_data_leaves_no_file` (Task 3).
- **`prepare_include50.py` interrupted (Ctrl+C) and run again:** finished videos are skipped, and a half-written video or `.npz` (both written as `.part` and renamed when complete) never counts as done. Pinned by `test_pending_skips_finished_videos` (Task 6) and the `.part` assertions in Task 3.
- **A dataset video OpenCV can't decode:** logged to `data/failed.txt`, and the run continues. Pinned by `test_unreadable_video_is_an_error` (Task 6).
- **Backspace on an empty sentence, or a "sign" whose frames are all empty:** no crash and no stray send. Pinned by `test_backspace_on_an_empty_sentence_does_nothing` (Task 12) and the all-zeros prediction in `test_save_load_and_predict_top_k` (Task 9).

---

### Task 0: Confirm downloads with the user

No code. This is a gate required by the Global Constraints.

- [ ] **Step 1: Ask the user, in chat, to approve these downloads, and wait for a clear yes**

  - Python packages from PyPI into `.venv`: `mediapipe==1.0.1` (~54 MB), `opencv-contrib-python` (~50 MB), `scikit-learn` (~11 MB), `matplotlib` (~8 MB)
  - 4 small metadata files (~30 KB) from github.com/AI4Bharat/INCLUDE (MIT)
  - MediaPipe models `hand_landmarker.task` (7.8 MB) and `pose_landmarker_lite.task` (5.8 MB) from storage.googleapis.com/mediapipe-models
  - INCLUDE-50: 958 videos, ~15 GB in total, streamed one at a time from zenodo.org record 4010759 (CC-BY-4.0) and deleted after processing. One video per word (~0.8 GB) is kept in `data/examples/`. Needs about 1 GB of free disk at peak; C: has 61 GB free.

  If the user declines the dataset download, stop after Task 13 and report.

---

### Task 1: Dependencies, package skeleton, test layout

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Create: `isl/__init__.py`, `tests/__init__.py`
- Move: `test_signlan.py` → `tests/test_signlan.py`

**Interfaces:**
- Produces: the `isl` package, and the test command `.venv/Scripts/python -m unittest discover -s tests -t . -v` used by every later task.

- [ ] **Step 1: Replace `requirements.txt`** (`tf-keras` stays until Task 12 removes the old model)

```
# Needs Python 3.9 - 3.12 (TensorFlow 2.18 has no wheels for 3.13+)
tensorflow==2.18.0
tf-keras==2.18.0
mediapipe==1.0.1
opencv-contrib-python>=4.8
numpy<2.1
scikit-learn>=1.5
matplotlib>=3.8
pyttsx3>=2.98
groq>=1.0
python-dotenv>=1.0
```

- [ ] **Step 2: Swap OpenCV and install**

Both OpenCV packages install into the same `cv2/` folder, so removing `opencv-python` deletes files the contrib package needs. That's why contrib is force-reinstalled afterwards.

```bash
.venv/Scripts/python -m pip uninstall -y opencv-python
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install --force-reinstall --no-deps "opencv-contrib-python>=4.8"
.venv/Scripts/python -m pip check
```

Expected: `No broken requirements found.`, and `pip list` shows `opencv-contrib-python` but not `opencv-python`.

- [ ] **Step 3: Verify cv2 (including the GUI) and MediaPipe import**

```bash
.venv/Scripts/python -c "import cv2, mediapipe, sklearn; print(cv2.__version__, mediapipe.__version__, sklearn.__version__); cv2.namedWindow('check'); cv2.destroyAllWindows()"
```

Expected: three version numbers (MediaPipe `1.0.1`) and no error.

- [ ] **Step 4: Create the package and test folders, move the existing tests**

`isl/__init__.py`:

```python
"""Indian Sign Language word recognition: landmarks, features, dataset and model"""
```

`tests/__init__.py`: an empty file.

```bash
git mv test_signlan.py tests/test_signlan.py
```

- [ ] **Step 5: Update `.gitignore`**

```
# Secrets
.env

# Python
__pycache__/
*.pyc
.venv/
venv/

# IDE
.idea/

# Dataset, landmarks and downloaded MediaPipe models (recreated by the scripts)
data/
models/*.task
```

- [ ] **Step 6: Run the existing tests from their new place**

Run: `.venv/Scripts/python -m unittest discover -s tests -t . -v`
Expected: `Ran 30 tests` … `OK`

- [ ] **Step 7: Commit** (includes the spec amendments and this plan)

```bash
git add requirements.txt .gitignore isl/__init__.py tests/__init__.py tests/test_signlan.py docs/superpowers
git commit -m "Set up the isl package and tests folder, add MediaPipe and scikit-learn" -m "Swap opencv-python for opencv-contrib-python, which MediaPipe 1.0.1 requires; both install the same cv2 module." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: INCLUDE-50 splits and labels

**Files:**
- Create: `isl/include50/include50_train.txt`, `include50_val.txt`, `include50_test.txt`, `label_map_include50.json`, `SOURCE.md`
- Create: `isl/include_data.py` (labels part)
- Test: `tests/test_include_data.py`

**Interfaces:**
- Produces:
  - `Sample(path: str, key: str, display: str, split: str)`, a frozen dataclass
  - `label_key(folder: str) -> str`, `display_name(folder: str) -> str`
  - `parse_split(text: str, split: str) -> list[Sample]`, `load_split(split: str) -> list[Sample]` for `"train" | "val" | "test"`
  - `load_label_keys() -> list[str]` (50 keys, official order)
  - `ZIP_NAMES: list[str]` (44 names), `DATA_DIR: Path`, `INDEX_PATH: Path`, `DISPLAY_OVERRIDES: dict[str, str]`

- [ ] **Step 1: Vendor the official files**

```bash
mkdir -p isl/include50
for f in train_test_paths/include50_train.txt train_test_paths/include50_val.txt train_test_paths/include50_test.txt label_maps/label_map_include50.json; do gh api "repos/AI4Bharat/INCLUDE/contents/$f" -H "Accept: application/vnd.github.raw" > "isl/include50/$(basename $f)"; done
```

`isl/include50/SOURCE.md`:

```markdown
# INCLUDE-50 split lists and label map

Copied unchanged from https://github.com/AI4Bharat/INCLUDE (`train_test_paths/` and `label_maps/`),
MIT License, Copyright (c) AI4Bharat.

The videos themselves are the INCLUDE dataset (CC-BY-4.0), https://zenodo.org/record/4010759:
A. Sridhar, R. G. Ganesan, P. Kumar, M. Khapra. "INCLUDE: A Large Scale Dataset for Indian Sign Language
Recognition". ACM Multimedia 2020.
```

- [ ] **Step 2: Write the failing tests** in `tests/test_include_data.py`

```python
import unittest

from isl import include_data


class LabelTests(unittest.TestCase):
    def test_label_keys_match_the_official_label_map_style(self):
        cases = {"48. Hello": "hello", "12. T-Shirt": "tshirt", "5. you (plural)": "youplural",
                 "28. Store or Shop": "storeorshop", "3. Good Morning": "goodmorning"}
        for folder, key in cases.items():
            self.assertEqual(include_data.label_key(folder), key)

    def test_display_names(self):
        self.assertEqual(include_data.display_name("48. Hello"), "hello")
        self.assertEqual(include_data.display_name("40. I"), "I")
        self.assertEqual(include_data.display_name("1. big large"), "big")
        self.assertEqual(include_data.display_name("28. Store or Shop"), "shop")
        self.assertEqual(include_data.display_name("3. Good Morning"), "good morning")

    def test_extra_folder_paths_use_the_word_folder(self):
        [sample] = include_data.parse_split("Places/19. House/Extra/MVI_3439.MOV", "train")
        self.assertEqual((sample.key, sample.display, sample.split), ("house", "house", "train"))


class OfficialSplitTests(unittest.TestCase):
    def test_all_958_videos_map_to_the_50_labels(self):
        splits = {split: include_data.load_split(split) for split in ("train", "val", "test")}
        self.assertEqual({split: len(samples) for split, samples in splits.items()},
                         {"train": 689, "val": 77, "test": 192})
        keys = include_data.load_label_keys()
        self.assertEqual(len(keys), 50)
        self.assertEqual({s.key for samples in splits.values() for s in samples}, set(keys))

    def test_splits_do_not_overlap(self):
        paths = {split: {s.path for s in include_data.load_split(split)} for split in ("train", "val", "test")}
        self.assertFalse(paths["train"] & paths["test"])
        self.assertFalse(paths["train"] & paths["val"])
        self.assertFalse(paths["val"] & paths["test"])

    def test_zip_names(self):
        self.assertEqual(len(include_data.ZIP_NAMES), 44)
        self.assertIn("Days_and_Time_3of3.zip", include_data.ZIP_NAMES)
        self.assertIn("Seasons_1of1.zip", include_data.ZIP_NAMES)
```

- [ ] **Step 3: Run to see them fail**

Run: `.venv/Scripts/python -m unittest tests.test_include_data -v`
Expected: `ImportError`/`ModuleNotFoundError` for `isl.include_data`.

- [ ] **Step 4: Implement** `isl/include_data.py`

```python
"""INCLUDE-50: the official splits and labels, and fetching single videos from the Zenodo zips"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_PATH = DATA_DIR / "include_index.json"
META_DIR = Path(__file__).resolve().parent / "include50"  # split lists + label map from AI4Bharat/INCLUDE (MIT)

ZENODO_URL = "https://zenodo.org/records/4010759/files/{name}?download=1"
ZIP_PARTS = {"Adjectives": 8, "Animals": 2, "Clothes": 2, "Colours": 2, "Days_and_Time": 3, "Electronics": 2,
             "Greetings": 2, "Home": 4, "Jobs": 2, "Means_of_Transportation": 2, "People": 5, "Places": 4,
             "Pronouns": 2, "Seasons": 1, "Society": 3}
ZIP_NAMES = [f"{category}_{part}of{parts}.zip" for category, parts in ZIP_PARTS.items()
             for part in range(1, parts + 1)]

# Clearer words for the interpreter than the dataset's folder names
DISPLAY_OVERRIDES = {"biglarge": "big", "smalllittle": "small", "storeorshop": "shop"}


@dataclass(frozen=True)
class Sample:
    path: str  # e.g. "Greetings/48. Hello/MVI_0089.MOV"
    key: str  # label key, e.g. "hello"
    display: str  # word given to the interpreter, e.g. "hello"
    split: str  # "train", "val" or "test"


def _word_name(folder):
    return re.sub(r"^\d+\.\s*", "", folder)


def label_key(folder):
    """ "48. Hello" -> "hello", "T-Shirt" -> "tshirt", matching label_map_include50.json"""
    return re.sub(r"[^a-z0-9]", "", _word_name(folder).lower())


def display_name(folder):
    key = label_key(folder)
    if key in DISPLAY_OVERRIDES:
        return DISPLAY_OVERRIDES[key]
    name = _word_name(folder).lower()
    return "I" if name == "i" else name


def parse_split(text, split):
    samples = []
    for line in text.splitlines():  # the files have no trailing newline
        path = line.strip()
        if path:
            folder = path.split("/")[1]  # the word folder, also for ".../Extra/..." paths
            samples.append(Sample(path, label_key(folder), display_name(folder), split))
    return samples


def load_split(split, meta_dir=META_DIR):
    """The official INCLUDE-50 "train", "val" or "test" samples"""
    return parse_split((Path(meta_dir) / f"include50_{split}.txt").read_text(encoding="utf-8"), split)


def load_label_keys(meta_dir=META_DIR):
    """The 50 label keys in the official class order"""
    label_map = json.loads((Path(meta_dir) / "label_map_include50.json").read_text(encoding="utf-8"))
    return sorted(label_map, key=label_map.get)
```

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python -m unittest tests.test_include_data -v`
Expected: 6 tests, `OK`.

- [ ] **Step 6: Commit**

```bash
git add isl/include50 isl/include_data.py tests/test_include_data.py
git commit -m "Add INCLUDE-50 split lists, label map and label parsing" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Fetching single videos from the Zenodo zips

**Files:**
- Modify: `isl/include_data.py` (append the remote-zip part and add imports)
- Test: `tests/test_include_data.py` (append)

**Interfaces:**
- Consumes: `ZIP_NAMES`, `INDEX_PATH`, `ZENODO_URL` (Task 2).
- Produces:
  - `HttpSource(url, retries=4, timeout=60, urlopen=urllib.request.urlopen)` with `.size`, `.read_range(start, length) -> bytes` and `.stream_range(start, length, chunk_size=1 << 20) -> Iterator[bytes]`
  - `Member(zip_name, header_offset, compress_size, file_size, crc, compress_type)`, a frozen dataclass
  - `list_members(source, zip_name) -> dict[str, Member]`
  - `extract_member(source, member, dest: Path) -> None`
  - `build_index(paths, index_path=INDEX_PATH, open_source=<Zenodo>, progress=print) -> dict[str, Member]`
  - `fetch_video(member, dest, open_source=<Zenodo>) -> None`
  - A "source" is any object with `size`, `read_range` and `stream_range`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_include_data.py`, and add these imports at the top of the file:

```python
import io
import tempfile
import urllib.error
import zipfile
from pathlib import Path
from unittest import mock
```

Then append:

```python
def make_zip(files, method=zipfile.ZIP_DEFLATED):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=method) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


class BytesSource:
    """Stands in for HttpSource: serves byte ranges from memory and counts requests"""

    def __init__(self, data):
        self.data, self.size, self.requests = data, len(data), 0

    def read_range(self, start, length):
        self.requests += 1
        return self.data[start:start + length]

    def stream_range(self, start, length, chunk_size=1 << 20):
        self.requests += 1
        for offset in range(start, start + length, chunk_size):
            yield self.data[offset:min(offset + chunk_size, start + length)]


class RemoteZipTests(unittest.TestCase):
    VIDEO = bytes(range(256)) * 4000  # about 1 MB

    def test_members_are_listed_with_few_requests(self):
        source = BytesSource(make_zip({"Seasons/61. Summer/a.MOV": self.VIDEO, "Seasons/61. Summer/b.MOV": b"x"}))
        members = include_data.list_members(source, "Seasons_1of1.zip")
        self.assertEqual(set(members), {"Seasons/61. Summer/a.MOV", "Seasons/61. Summer/b.MOV"})
        self.assertEqual(members["Seasons/61. Summer/a.MOV"].zip_name, "Seasons_1of1.zip")
        self.assertLessEqual(source.requests, 4)

    def test_deflated_and_stored_members_extract_exactly_with_one_data_request(self):
        for method in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED):
            source = BytesSource(make_zip({"W/1. X/a.MOV": self.VIDEO}, method))
            member = include_data.list_members(source, "z.zip")["W/1. X/a.MOV"]
            source.requests = 0
            with tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp) / "a.MOV"
                include_data.extract_member(source, member, dest)
                self.assertEqual(dest.read_bytes(), self.VIDEO)
                self.assertFalse((Path(tmp) / "a.MOV.part").exists())
            self.assertEqual(source.requests, 2)  # local header + the data itself

    def test_corrupt_data_leaves_no_file(self):
        data = bytearray(make_zip({"W/1. X/a.MOV": self.VIDEO}, zipfile.ZIP_STORED))
        member = include_data.list_members(BytesSource(bytes(data)), "z.zip")["W/1. X/a.MOV"]
        data[member.header_offset + 100] ^= 0xFF
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(zipfile.BadZipFile):
                include_data.extract_member(BytesSource(bytes(data)), member, Path(tmp) / "a.MOV")
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_index_is_built_once_and_cached(self):
        zip_bytes = make_zip({"Seasons/61. Summer/a.MOV": b"a", "Seasons/61. Summer/b.MOV": b"b"})
        opened = []

        def open_source(name):
            opened.append(name)
            return BytesSource(zip_bytes)

        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "index.json"
            index = include_data.build_index(["Seasons/61. Summer/a.MOV"], index_path, open_source,
                                             progress=lambda _: None)
            self.assertEqual(opened, ["Seasons_1of1.zip"])
            self.assertEqual(index["Seasons/61. Summer/a.MOV"].zip_name, "Seasons_1of1.zip")
            again = include_data.build_index(["Seasons/61. Summer/b.MOV"], index_path, open_source,
                                             progress=lambda _: None)
            self.assertEqual(opened, ["Seasons_1of1.zip"])  # answered from the cache
            self.assertIn("Seasons/61. Summer/b.MOV", again)

    def test_fetch_video_uses_the_members_zip(self):
        zip_bytes = make_zip({"W/1. X/a.MOV": self.VIDEO})
        member = include_data.list_members(BytesSource(zip_bytes), "W_1of1.zip")["W/1. X/a.MOV"]
        opened = []
        with tempfile.TemporaryDirectory() as tmp:
            include_data.fetch_video(member, Path(tmp) / "a.MOV",
                                     open_source=lambda name: opened.append(name) or BytesSource(zip_bytes))
            self.assertEqual((Path(tmp) / "a.MOV").read_bytes(), self.VIDEO)
        self.assertEqual(opened, ["W_1of1.zip"])


class FakeResponse(io.BytesIO):
    def __init__(self, data, status, headers=None):
        super().__init__(data)
        self.status, self.headers = status, headers or {}


class HttpSourceTests(unittest.TestCase):
    DATA = b"0123456789"

    def fake_urlopen(self, honour_range=True, failures=0):
        calls = {"n": 0}

        def urlopen(request, timeout=None):
            calls["n"] += 1
            if calls["n"] <= failures:
                raise urllib.error.URLError("temporary failure")
            if request.get_method() == "HEAD":
                return FakeResponse(b"", 200, {"Content-Length": str(len(self.DATA))})
            byte_range = request.get_header("Range")
            if byte_range and honour_range:
                start, end = map(int, byte_range.split("=")[1].split("-"))
                return FakeResponse(self.DATA[start:end + 1], 206)
            return FakeResponse(self.DATA, 200)

        return urlopen, calls

    def test_reads_ranges(self):
        urlopen, _ = self.fake_urlopen()
        source = include_data.HttpSource("http://example", urlopen=urlopen)
        self.assertEqual(source.size, 10)
        self.assertEqual(source.read_range(2, 3), b"234")
        self.assertEqual(b"".join(source.stream_range(5, 5, chunk_size=2)), b"56789")

    def test_server_ignoring_ranges_is_an_error(self):
        urlopen, _ = self.fake_urlopen(honour_range=False)
        source = include_data.HttpSource("http://example", urlopen=urlopen)
        with self.assertRaises(OSError):
            source.read_range(2, 3)

    def test_transient_errors_are_retried(self):
        urlopen, calls = self.fake_urlopen(failures=2)
        with mock.patch.object(include_data.time, "sleep"):
            source = include_data.HttpSource("http://example", urlopen=urlopen)
        self.assertEqual(source.size, 10)
        self.assertEqual(calls["n"], 3)
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python -m unittest tests.test_include_data -v`
Expected: the new tests error with `AttributeError: module 'isl.include_data' has no attribute 'list_members'` (or `HttpSource`).

- [ ] **Step 3: Implement.** In `isl/include_data.py`, replace the import block with:

```python
import io
import json
import re
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
```

and append:

```python
class HttpSource:
    """Byte ranges of a remote file, retrying network errors"""

    def __init__(self, url, retries=4, timeout=60, urlopen=urllib.request.urlopen):
        self.url, self.retries, self.timeout, self._urlopen = url, retries, timeout, urlopen
        with self._open(method="HEAD") as response:
            self.size = int(response.headers["Content-Length"])

    def _open(self, start=None, length=None, method="GET"):
        headers = {} if start is None else {"Range": f"bytes={start}-{start + length - 1}"}
        for attempt in range(self.retries):
            try:
                response = self._urlopen(urllib.request.Request(self.url, headers=headers, method=method),
                                         timeout=self.timeout)
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
        if start is not None and response.status != 206:
            response.close()  # A plain 200 would stream the whole 1+ GB zip
            raise OSError(f"server ignored the byte range request for {self.url} (HTTP {response.status})")
        return response

    def read_range(self, start, length):
        with self._open(start, length) as response:
            return response.read()

    def stream_range(self, start, length, chunk_size=1 << 20):
        with self._open(start, length) as response:
            while chunk := response.read(chunk_size):
                yield chunk


class _RangeReader(io.RawIOBase):
    """Seekable file over a range source, reading ahead in blocks so zipfile needs only a few requests"""

    BLOCK = 256 * 1024

    def __init__(self, source):
        self.source, self.pos = source, 0
        self._cache_start, self._cache = 0, b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.source.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.source.size - self.pos
        n = min(n, self.source.size - self.pos)
        if n <= 0:
            return b""
        offset = self.pos - self._cache_start
        if not (0 <= offset and offset + n <= len(self._cache)):
            self._cache_start, offset = self.pos, 0
            self._cache = self.source.read_range(self.pos, min(max(n, self.BLOCK), self.source.size - self.pos))
        data = self._cache[offset:offset + n]
        self.pos += len(data)
        return data


@dataclass(frozen=True)
class Member:
    """Where one file lives inside a zip: enough to fetch it without reading the zip's directory again"""
    zip_name: str
    header_offset: int
    compress_size: int
    file_size: int
    crc: int
    compress_type: int


def list_members(source, zip_name):
    with zipfile.ZipFile(_RangeReader(source)) as archive:
        return {info.filename: Member(zip_name, info.header_offset, info.compress_size, info.file_size, info.CRC,
                                      info.compress_type)
                for info in archive.infolist() if not info.is_dir()}


def extract_member(source, member, dest):
    """Write one zip member to dest with a single range request for its data, checking the CRC"""
    header = source.read_range(member.header_offset, 30)
    if header[:4] != b"PK\x03\x04":
        raise zipfile.BadZipFile(f"no local file header at offset {member.header_offset}")
    name_length, extra_length = struct.unpack("<HH", header[26:30])
    start = member.header_offset + 30 + name_length + extra_length
    if member.compress_type == zipfile.ZIP_DEFLATED:
        inflater = zlib.decompressobj(-zlib.MAX_WBITS)
    elif member.compress_type == zipfile.ZIP_STORED:
        inflater = None
    else:
        raise zipfile.BadZipFile(f"unsupported compression method {member.compress_type}")

    dest = Path(dest)
    partial = dest.with_name(dest.name + ".part")
    crc = 0
    with open(partial, "wb") as out:
        for chunk in source.stream_range(start, member.compress_size):
            data = inflater.decompress(chunk) if inflater else chunk
            crc = zlib.crc32(data, crc)
            out.write(data)
        if inflater:
            data = inflater.flush()
            crc = zlib.crc32(data, crc)
            out.write(data)
    if crc != member.crc:
        partial.unlink()
        raise zipfile.BadZipFile(f"CRC mismatch for {dest.name}")
    partial.replace(dest)


def _zenodo(zip_name):
    return HttpSource(ZENODO_URL.format(name=zip_name))


def build_index(paths, index_path=INDEX_PATH, open_source=_zenodo, progress=print):
    """{video path: Member} for the given paths, reading each needed zip's directory once and caching it"""
    index_path = Path(index_path)
    if index_path.exists():
        cache = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        cache = {"scanned": [], "members": {}}
    categories = {path.split("/")[0] for path in paths}
    for zip_name in ZIP_NAMES:
        if zip_name.rsplit("_", 1)[0] not in categories or zip_name in cache["scanned"]:
            continue
        progress(f"Reading the file list of {zip_name}")
        members = list_members(open_source(zip_name), zip_name)
        cache["members"].update({name: asdict(member) for name, member in members.items()})
        cache["scanned"].append(zip_name)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(cache), encoding="utf-8")
    return {path: Member(**cache["members"][path]) for path in paths if path in cache["members"]}


def fetch_video(member, dest, open_source=_zenodo):
    extract_member(open_source(member.zip_name), member, dest)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest tests.test_include_data -v`
Expected: 14 tests, `OK`.

- [ ] **Step 5: Check against the real Zenodo** (index only, a few KB: this reads one zip's file list, no videos)

```bash
.venv/Scripts/python -c "from isl import include_data as d; m = d.list_members(d.HttpSource(d.ZENODO_URL.format(name='Seasons_1of1.zip')), 'Seasons_1of1.zip'); print(len(m), next(iter(m)))"
```

Expected: `66 Seasons/61. Summer/MVI_4565.MOV` (66 videos; the first name may differ).

- [ ] **Step 6: Commit**

```bash
git add isl/include_data.py tests/test_include_data.py
git commit -m "Fetch single INCLUDE videos from the Zenodo zips with range requests" -m "Reads each zip's directory in a few requests, then streams one member's data in a single request, inflating it and checking the CRC before it counts as downloaded." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Landmark features

**Files:**
- Create: `isl/features.py`
- Create: `tests/fakes.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces:
  - `RawLandmarks(pose: np.ndarray | None, hands: list[np.ndarray], width: int, height: int)`. `pose` is `(33, 3)` holding x, y (pixels) and visibility; each hand is `(21, 2)` pixels.
  - `frame_features(raw) -> tuple[np.ndarray (184,) float32, bool body_found]`
  - `hands_up(features) -> bool | np.ndarray[bool]`, for one frame `(184,)` or a sequence `(T, 184)`
  - `trim(frames, up, pad=2, min_up=4) -> np.ndarray`, `resample(frames, length=32) -> np.ndarray (32, F)`,
    `to_sequence(frames) -> np.ndarray (32, 184)`, `flip(sequence) -> np.ndarray`
  - Constants: `FEATURE_VERSION = 1`, `SEQUENCE_LENGTH = 32`, `NUM_FEATURES = 184`, `HANDS_UP_K = 1.5`,
    `POSE_SLICE`, `HAND_SLICES`, `PRESENCE_SLICE`
  - `tests/fakes.py`: `make_pose(...)`, `make_hand(wrist_xy, size=30.0)`,
    `make_raw(hands="both", wrist_y=150.0, visibility=1.0)`, `frame(up: bool) -> (184,)`

- [ ] **Step 1: Write the test fakes** in `tests/fakes.py`

```python
"""Synthetic landmarks for tests: a signer facing the camera with shoulders 100 px apart at y = 200"""
import numpy as np

from isl.features import RawLandmarks, frame_features


def make_pose(visibility=1.0, wrist_y=150.0):
    """MediaPipe-style (33, 3) pose; the subject's left shoulder (11) is on the image's right"""
    pose = np.zeros((33, 3), dtype=np.float32)
    pose[0] = (320, 120, 1.0)  # nose
    pose[11], pose[12] = (370, 200, visibility), (270, 200, visibility)  # shoulders
    pose[13], pose[14] = (380, 260, 1.0), (260, 260, 1.0)  # elbows
    pose[15], pose[16] = (390, wrist_y, 1.0), (250, wrist_y, 1.0)  # wrists
    return pose


def make_hand(wrist_xy, size=30.0):
    """21 hand points with the wrist at wrist_xy and fingers pointing up"""
    x, y = float(wrist_xy[0]), float(wrist_xy[1])
    hand = np.array([(x + (i % 4) * 3.0, y - size * 1.5 * i / 20) for i in range(21)], dtype=np.float32)
    hand[0], hand[9] = (x, y), (x, y - size)
    return hand


def make_raw(hands="both", wrist_y=150.0, visibility=1.0):
    """hands: "both", "left", "right" or "none"; wrist_y 150 is raised, 400 is lowered"""
    pose = make_pose(visibility, wrist_y)
    left, right = make_hand(pose[15, :2]), make_hand(pose[16, :2])
    found = {"both": [right, left], "left": [left], "right": [right], "none": []}[hands]
    return RawLandmarks(pose=pose, hands=found, width=640, height=480)


def frame(up):
    """One frame's features: both hands raised (True), or no hands in view (False)"""
    return frame_features(make_raw("both" if up else "none"))[0]
```

- [ ] **Step 2: Write the failing tests** in `tests/test_features.py`

```python
import unittest

import numpy as np

from isl import features as F
from isl.features import RawLandmarks
from tests.fakes import frame, make_hand, make_raw


class FrameFeatureTests(unittest.TestCase):
    def test_vector_layout(self):
        vector, found = F.frame_features(make_raw("both"))
        self.assertTrue(found)
        self.assertEqual(F.NUM_FEATURES, 184)
        self.assertEqual(vector.shape, (184,))
        self.assertEqual(vector.dtype, np.float32)
        np.testing.assert_array_equal(vector[F.PRESENCE_SLICE], [1, 1])
        # the shoulders are pose points 1 and 2 of 7, at +-0.5 shoulder widths from the origin
        np.testing.assert_allclose(vector[F.POSE_SLICE].reshape(7, 2)[1:3], [[0.5, 0.0], [-0.5, 0.0]])

    def test_moving_or_scaling_the_body_changes_nothing(self):
        raw = make_raw("both")
        shift, factor = np.array([123.0, -45.0], dtype=np.float32), 1.7
        pose = raw.pose.copy()
        pose[:, :2] = raw.pose[:, :2] * factor + shift
        moved = RawLandmarks(pose=pose, hands=[h * factor + shift for h in raw.hands], width=1280, height=720)
        np.testing.assert_allclose(F.frame_features(moved)[0], F.frame_features(raw)[0], atol=1e-5)

    def test_hands_are_matched_to_the_nearest_wrist(self):
        vector = F.frame_features(make_raw("right"))[0]
        np.testing.assert_array_equal(vector[F.PRESENCE_SLICE], [0, 1])
        self.assertFalse(vector[F.HAND_SLICES[0]].any())
        both = F.frame_features(make_raw("both"))[0]  # listed right-then-left, must still land left-then-right
        np.testing.assert_allclose(both[F.HAND_SLICES[1]], vector[F.HAND_SLICES[1]])

    def test_hidden_shoulders_give_an_empty_invalid_frame(self):
        vector, found = F.frame_features(make_raw("both", visibility=0.2))
        self.assertFalse(found)
        self.assertFalse(vector.any())
        vector, found = F.frame_features(RawLandmarks(pose=None, hands=[make_hand((300, 150))], width=640, height=480))
        self.assertFalse(found)
        self.assertFalse(F.hands_up(vector))


class HandsUpTests(unittest.TestCase):
    def test_raised_hand(self):
        self.assertTrue(F.hands_up(F.frame_features(make_raw("left", wrist_y=150))[0]))

    def test_lowered_or_missing_hands(self):
        self.assertFalse(F.hands_up(F.frame_features(make_raw("both", wrist_y=400))[0]))
        self.assertFalse(F.hands_up(F.frame_features(make_raw("none"))[0]))

    def test_sequence_gives_one_flag_per_frame(self):
        clip = np.stack([frame(True), frame(False), frame(True)])
        np.testing.assert_array_equal(F.hands_up(clip), [True, False, True])


class SequenceTests(unittest.TestCase):
    def test_resample_length_and_endpoints(self):
        clip = np.arange(10 * 184, dtype=np.float32).reshape(10, 184)
        out = F.resample(clip)
        self.assertEqual(out.shape, (32, 184))
        np.testing.assert_allclose(out[0], clip[0])
        np.testing.assert_allclose(out[-1], clip[-1])

    def test_resample_single_frame_and_empty(self):
        self.assertEqual(F.resample(np.ones((1, 184))).shape, (32, 184))
        empty = F.resample(np.zeros((0, 184)))
        self.assertEqual(empty.shape, (32, 184))
        self.assertFalse(empty.any())

    def test_trim_keeps_the_raised_part_with_padding(self):
        clip = np.stack([frame(False)] * 10 + [frame(True)] * 6 + [frame(False)] * 10)
        self.assertEqual(len(F.trim(clip, F.hands_up(clip))), 6 + 2 * 2)

    def test_trim_keeps_everything_when_hands_are_barely_up(self):
        clip = np.stack([frame(False)] * 10 + [frame(True)] * 3)
        self.assertEqual(len(F.trim(clip, F.hands_up(clip))), 13)

    def test_to_sequence(self):
        clip = np.stack([frame(False)] * 10 + [frame(True)] * 20 + [frame(False)] * 10)
        sequence = F.to_sequence(clip)
        self.assertEqual(sequence.shape, (32, 184))
        self.assertTrue(F.hands_up(sequence)[5:27].all())

    def test_flip_swaps_hands_and_twice_is_identity(self):
        clip = np.stack([F.frame_features(make_raw("right"))[0]] * 4)
        flipped = F.flip(clip)
        np.testing.assert_array_equal(flipped[:, F.PRESENCE_SLICE], [[1, 0]] * 4)
        np.testing.assert_allclose(flipped[:, F.HAND_SLICES[0]][:, 0::2], -clip[:, F.HAND_SLICES[1]][:, 0::2])
        np.testing.assert_allclose(F.flip(flipped), clip)
```

- [ ] **Step 3: Run to see them fail**

Run: `.venv/Scripts/python -m unittest tests.test_features -v`
Expected: `ModuleNotFoundError: No module named 'isl.features'`.

- [ ] **Step 4: Implement** `isl/features.py`

```python
"""Body-normalised landmark features, shared by the training pipeline and the live app"""
from dataclasses import dataclass

import numpy as np

FEATURE_VERSION = 1
SEQUENCE_LENGTH = 32  # frames per sign given to the model

# MediaPipe pose indices
NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST = 0, 11, 12, 13, 14, 15, 16
POSE_POINTS = (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST)
_POSE_MIRROR = [0, 2, 1, 4, 3, 6, 5]  # POSE_POINTS order with left and right swapped
MIN_VISIBILITY = 0.5
# A wrist less than this many shoulder-widths below the shoulders counts as a raised hand
HANDS_UP_K = 1.5

# Layout of one frame's feature vector
POSE_SLICE = slice(0, 14)  # 7 pose points (x, y) in the body frame
HAND_SLICES = (slice(14, 98), slice(98, 182))  # left, right: 21 (x, y) in the body frame + 21 (x, y) hand-local
PRESENCE_SLICE = slice(182, 184)  # 1 when the left / right hand was found
NUM_FEATURES = 184


@dataclass
class RawLandmarks:
    """MediaPipe output for one frame, in pixels of the processed image"""
    pose: np.ndarray | None  # (33, 3): x, y, visibility; None when no person was found
    hands: list  # 0-2 arrays of shape (21, 2): x, y
    width: int
    height: int


def _assign_hands(hands, pose):
    """[left hand or None, right hand or None], matching each hand to the nearest pose wrist"""
    wrists = pose[[LEFT_WRIST, RIGHT_WRIST], :2]
    if len(hands) == 0:
        return [None, None]
    if len(hands) == 1:
        slots = [None, None]
        slots[int(np.argmin(np.linalg.norm(wrists - hands[0][0], axis=1)))] = hands[0]
        return slots
    a, b = hands[:2]
    straight = np.linalg.norm(wrists[0] - a[0]) + np.linalg.norm(wrists[1] - b[0])
    swapped = np.linalg.norm(wrists[0] - b[0]) + np.linalg.norm(wrists[1] - a[0])
    return [a, b] if straight <= swapped else [b, a]


def frame_features(raw):
    """(feature vector, body found) for one frame; the vector is all zeros when the shoulders aren't visible"""
    features = np.zeros(NUM_FEATURES, dtype=np.float32)
    pose = raw.pose
    if pose is None or min(pose[LEFT_SHOULDER, 2], pose[RIGHT_SHOULDER, 2]) < MIN_VISIBILITY:
        return features, False
    left, right = pose[LEFT_SHOULDER, :2], pose[RIGHT_SHOULDER, :2]
    origin, scale = (left + right) / 2, float(np.linalg.norm(left - right))
    if scale < 1e-3:
        return features, False

    features[POSE_SLICE] = ((pose[list(POSE_POINTS), :2] - origin) / scale).ravel()
    for i, hand in enumerate(_assign_hands(raw.hands, pose)):
        if hand is None:
            continue
        hand = np.asarray(hand, dtype=np.float32)
        size = float(np.linalg.norm(hand[9] - hand[0]))  # wrist to middle-finger knuckle
        local = (hand - hand[0]) / size if size > 1e-3 else np.zeros_like(hand)
        features[HAND_SLICES[i]] = np.concatenate([((hand - origin) / scale).ravel(), local.ravel()])
        features[PRESENCE_SLICE.start + i] = 1.0
    return features, True


def hands_up(features):
    """Whether a hand is raised: one bool for a frame (F,), a bool array for a sequence (T, F)"""
    frames = np.atleast_2d(features)
    up = np.zeros(len(frames), dtype=bool)
    for i, block in enumerate(HAND_SLICES):
        present = frames[:, PRESENCE_SLICE.start + i] > 0.5
        wrist_y = frames[:, block.start + 1]  # body frame: y grows downwards from the shoulders
        up |= present & (wrist_y < HANDS_UP_K)
    return up if np.ndim(features) == 2 else bool(up[0])


def trim(frames, up, pad=2, min_up=4):
    """Keep the part of a clip where the hands are up, plus a little padding"""
    raised = np.flatnonzero(up)
    if len(raised) < min_up:
        return frames
    return frames[max(raised[0] - pad, 0):min(raised[-1] + pad + 1, len(frames))]


def resample(frames, length=SEQUENCE_LENGTH):
    """Linearly interpolate a (T, F) sequence to exactly `length` frames"""
    frames = np.asarray(frames, dtype=np.float32)
    if len(frames) == 0:
        return np.zeros((length, frames.shape[1]), dtype=np.float32)
    if len(frames) == 1:
        return np.repeat(frames, length, axis=0)
    positions = np.linspace(0, len(frames) - 1, length)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(frames) - 1)
    weight = (positions - lower)[:, None].astype(np.float32)
    return frames[lower] * (1 - weight) + frames[upper] * weight


def to_sequence(frames):
    """Model input for one sign: trimmed to the raised-hands part and resampled"""
    frames = np.asarray(frames, dtype=np.float32)
    return resample(trim(frames, hands_up(frames)))


def flip(sequence):
    """Mirror a (T, NUM_FEATURES) sequence left-right, as if signed with the other hand"""
    seq = np.asarray(sequence, dtype=np.float32)
    out = np.empty_like(seq)
    pose = seq[:, POSE_SLICE].reshape(len(seq), 7, 2)[:, _POSE_MIRROR]
    pose[..., 0] *= -1
    out[:, POSE_SLICE] = pose.reshape(len(seq), 14)
    for dst, src in zip(HAND_SLICES, reversed(HAND_SLICES)):
        hand = seq[:, src].reshape(len(seq), 42, 2).copy()
        hand[..., 0] *= -1
        out[:, dst] = hand.reshape(len(seq), 84)
    out[:, PRESENCE_SLICE] = seq[:, PRESENCE_SLICE][:, ::-1]
    return out
```

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python -m unittest tests.test_features -v`
Expected: 14 tests, `OK`.

- [ ] **Step 6: Commit**

```bash
git add isl/features.py tests/fakes.py tests/test_features.py
git commit -m "Add body-normalised landmark features shared by training and the live app" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: MediaPipe landmark extractor

**Files:**
- Create: `isl/landmarks.py`
- Test: `tests/test_landmarks.py`

**Interfaces:**
- Consumes: `RawLandmarks` (Task 4).
- Produces:
  - `MODELS_DIR: Path` (`<root>/models`), `MODEL_URLS: dict[str, str]`, `PROCESS_WIDTH = 640`
  - `ensure_model_files(models_dir=MODELS_DIR, urlopen=urllib.request.urlopen) -> dict[str, Path]`
  - `LandmarkExtractor(models_dir=MODELS_DIR)`, usable as a context manager, with
    `.extract(frame_bgr: np.ndarray, timestamp_ms: float) -> RawLandmarks` and `.close()`

- [ ] **Step 1: Write the failing tests** in `tests/test_landmarks.py`

```python
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from isl import landmarks


class ModelFileTests(unittest.TestCase):
    def test_missing_files_are_downloaded_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            urlopen = mock.Mock(side_effect=lambda url, timeout: io.BytesIO(b"model:" + url.encode()))
            paths = landmarks.ensure_model_files(Path(tmp), urlopen=urlopen)
            self.assertEqual(set(paths), set(landmarks.MODEL_URLS))
            self.assertTrue(all(path.read_bytes().startswith(b"model:") for path in paths.values()))
            landmarks.ensure_model_files(Path(tmp), urlopen=urlopen)
            self.assertEqual(urlopen.call_count, 2)

    def test_failed_download_says_where_to_get_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as caught:
                landmarks.ensure_model_files(Path(tmp), urlopen=mock.Mock(side_effect=OSError("offline")))
            self.assertIn("storage.googleapis.com", str(caught.exception))


@unittest.skipUnless(all((landmarks.MODELS_DIR / name).exists() for name in landmarks.MODEL_URLS),
                     "MediaPipe model files not downloaded yet")
class ExtractorTests(unittest.TestCase):
    def test_empty_frame_has_no_person_and_is_downscaled(self):
        with landmarks.LandmarkExtractor() as extractor:
            for i in range(3):
                raw = extractor.extract(np.zeros((720, 1280, 3), np.uint8), i * 33)
        self.assertIsNone(raw.pose)
        self.assertEqual(raw.hands, [])
        self.assertEqual((raw.width, raw.height), (640, 360))

    def test_repeated_timestamps_are_accepted(self):
        with landmarks.LandmarkExtractor() as extractor:
            extractor.extract(np.zeros((480, 640, 3), np.uint8), 100)
            extractor.extract(np.zeros((480, 640, 3), np.uint8), 100)
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python -m unittest tests.test_landmarks -v`
Expected: `ModuleNotFoundError: No module named 'isl.landmarks'`.

- [ ] **Step 3: Implement** `isl/landmarks.py`

```python
"""MediaPipe hand and pose landmarks for video frames"""
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from isl.features import RawLandmarks

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
                            "float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                                 "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}
PROCESS_WIDTH = 640  # Frames are downscaled to this width first, for dataset videos and the webcam alike


def ensure_model_files(models_dir=MODELS_DIR, urlopen=urllib.request.urlopen):
    """Paths of the MediaPipe model files, downloading any that are missing"""
    models_dir = Path(models_dir)
    paths = {}
    for name, url in MODEL_URLS.items():
        path = models_dir / name
        if not path.exists():
            try:
                with urlopen(url, timeout=60) as response:
                    data = response.read()
            except OSError as e:
                raise SystemExit(f"Could not download the MediaPipe model {name} ({e}).\n"
                                 f"Download it from {url}\nand save it as {path}") from e
            models_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        paths[name] = path
    return paths


class LandmarkExtractor:
    """Runs MediaPipe pose and hand detection over one video or camera stream"""

    def __init__(self, models_dir=MODELS_DIR):
        paths = ensure_model_files(models_dir)
        video = vision.RunningMode.VIDEO
        self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(paths["pose_landmarker_lite.task"])), running_mode=video))
        self._hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(paths["hand_landmarker.task"])), running_mode=video,
            num_hands=2))
        self._last_timestamp = -1

    def extract(self, frame_bgr, timestamp_ms):
        height, width = frame_bgr.shape[:2]
        if width > PROCESS_WIDTH:
            height, width = round(height * PROCESS_WIDTH / width), PROCESS_WIDTH
            frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
        # VIDEO mode needs strictly increasing timestamps
        timestamp = max(int(timestamp_ms), self._last_timestamp + 1)
        self._last_timestamp = timestamp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        pose_result = self._pose.detect_for_video(image, timestamp)
        hand_result = self._hands.detect_for_video(image, timestamp)

        pose = None
        if pose_result.pose_landmarks:
            pose = np.array([(p.x * width, p.y * height, p.visibility or 0.0) for p in pose_result.pose_landmarks[0]],
                            dtype=np.float32)
        hands = [np.array([(p.x * width, p.y * height) for p in hand], dtype=np.float32)
                 for hand in hand_result.hand_landmarks]
        return RawLandmarks(pose=pose, hands=hands, width=width, height=height)

    def close(self):
        self._pose.close()
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
```

- [ ] **Step 4: Run the tests** (the extractor tests skip because the models aren't downloaded yet)

Run: `.venv/Scripts/python -m unittest tests.test_landmarks -v`
Expected: 2 pass, 2 skipped, `OK (skipped=2)`.

- [ ] **Step 5: Download the MediaPipe models** (approved in Task 0) **and re-run**

```bash
.venv/Scripts/python -c "from isl.landmarks import ensure_model_files; print(ensure_model_files())"
.venv/Scripts/python -m unittest tests.test_landmarks -v
```

Expected: two paths under `models/`; then 4 tests, `OK` (no skips).

- [ ] **Step 6: Commit** (the `.task` files are git-ignored)

```bash
git add isl/landmarks.py tests/test_landmarks.py
git commit -m "Add the MediaPipe hand and pose landmark extractor" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Dataset preparation script

**Files:**
- Create: `scripts/prepare_include50.py`
- Test: `tests/test_prepare.py`

**Interfaces:**
- Consumes: `include_data.load_split`, `build_index`, `fetch_video`, `DATA_DIR`, `Sample` (Tasks 2–3); `frame_features` (Task 4); `LandmarkExtractor` (Task 5).
- Produces:
  - `data/landmarks/<split>/<key>/<video-stem>.npz`, containing `features (T, 184) float32`, `fps`, `key`
  - `data/examples/<key>.MOV`, `data/failed.txt`
  - Functions: `all_samples()`, `landmarks_path(sample)`, `pending(samples)`, `video_landmarks(path) -> (features, fps)`, `main(argv=None)`

- [ ] **Step 1: Write the failing tests** in `tests/test_prepare.py`

```python
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_script("prepare_include50")


class PrepareTests(unittest.TestCase):
    def test_test_videos_come_first_so_examples_are_test_videos(self):
        samples = prepare.all_samples()
        self.assertEqual(len(samples), 958)
        self.assertEqual((samples[0].split, samples[-1].split), ("test", "train"))

    def test_pending_skips_finished_videos(self):
        samples = prepare.all_samples()[:3]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(prepare, "LANDMARKS_DIR", Path(tmp)):
            done = prepare.landmarks_path(samples[0])
            done.parent.mkdir(parents=True)
            done.write_bytes(b"")
            half_written = prepare.landmarks_path(samples[1])  # a run stopped while saving this one
            half_written.parent.mkdir(parents=True, exist_ok=True)
            half_written.with_name(half_written.name + ".part").write_bytes(b"")
            self.assertEqual(prepare.pending(samples), samples[1:])

    def test_landmarks_path_layout(self):
        sample = prepare.all_samples()[0]
        path = prepare.landmarks_path(sample)
        self.assertEqual(path.parts[-3:-1], (sample.split, sample.key))
        self.assertEqual(path.suffix, ".npz")

    def test_unreadable_video_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.MOV"
            bad.write_bytes(b"not a video")
            with mock.patch.object(prepare, "LandmarkExtractor"), self.assertRaises(ValueError):
                prepare.video_landmarks(bad)
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python -m unittest tests.test_prepare -v`
Expected: `FileNotFoundError` for `scripts/prepare_include50.py`.

- [ ] **Step 3: Implement** `scripts/prepare_include50.py`

```python
"""Download the INCLUDE-50 videos one at a time, turn each into landmarks, and delete it.

Run from the project folder:  python scripts/prepare_include50.py [--limit N]
Safe to stop and run again: finished videos are skipped.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isl import include_data  # noqa: E402
from isl.features import frame_features  # noqa: E402
from isl.landmarks import LandmarkExtractor  # noqa: E402

LANDMARKS_DIR = include_data.DATA_DIR / "landmarks"
EXAMPLES_DIR = include_data.DATA_DIR / "examples"
TMP_DIR = include_data.DATA_DIR / "tmp"
FAILED_PATH = include_data.DATA_DIR / "failed.txt"


def all_samples():
    """Every INCLUDE-50 video, test split first so each word's example video comes from it"""
    return [sample for split in ("test", "val", "train") for sample in include_data.load_split(split)]


def landmarks_path(sample):
    return LANDMARKS_DIR / sample.split / sample.key / (Path(sample.path).stem + ".npz")


def pending(samples):
    return [sample for sample in samples if not landmarks_path(sample).exists()]


def video_landmarks(video_path):
    """(features (T, 184), fps) for every frame of a video file"""
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    features = []
    with LandmarkExtractor() as extractor:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            features.append(frame_features(extractor.extract(frame, len(features) * 1000 / fps))[0])
    capture.release()
    if not features:
        raise ValueError(f"no frames could be read from {Path(video_path).name}")
    return np.stack(features), float(fps)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare INCLUDE-50 landmarks from the Zenodo videos.")
    parser.add_argument("--limit", type=int, help="only process this many videos (for a quick check)")
    args = parser.parse_args(argv)

    samples = all_samples()[:args.limit] if args.limit else all_samples()
    todo = pending(samples)
    print(f"{len(samples)} videos: {len(samples) - len(todo)} already done, {len(todo)} to go")
    if not todo:
        return
    index = include_data.build_index([sample.path for sample in todo])
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    failed, start = 0, time.time()
    for n, sample in enumerate(todo, 1):
        video = TMP_DIR / Path(sample.path).name
        try:
            if sample.path not in index:
                raise KeyError("not found in any INCLUDE zip")
            include_data.fetch_video(index[sample.path], video)
            features, fps = video_landmarks(video)
            out = landmarks_path(sample)
            out.parent.mkdir(parents=True, exist_ok=True)
            partial = out.with_name(out.name + ".part")
            with open(partial, "wb") as f:  # A run stopped mid-write must not leave a "finished" file
                np.savez_compressed(f, features=features, fps=fps, key=sample.key)
            partial.replace(out)
            example = EXAMPLES_DIR / f"{sample.key}{video.suffix}"
            if not example.exists():
                EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
                video.replace(example)
        except Exception as e:  # One bad video shouldn't stop an hour-long run
            failed += 1
            with open(FAILED_PATH, "a", encoding="utf-8") as log:
                log.write(f"{sample.path}\t{type(e).__name__}: {e}\n")
            print(f"  FAILED {sample.path}: {e}")
        finally:
            video.unlink(missing_ok=True)
        remaining = (time.time() - start) / n * (len(todo) - n) / 60
        print(f"[{n}/{len(todo)}] {sample.split:5s} {sample.key:12s} about {remaining:.0f} min left", flush=True)
    print(f"Done: {len(todo) - failed} prepared, {failed} failed" + (f" (see {FAILED_PATH})" if failed else ""))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest tests.test_prepare -v`
Expected: 4 tests, `OK`.

- [ ] **Step 5: Real run on 2 videos** (approved download of ~30 MB)

```bash
.venv/Scripts/python scripts/prepare_include50.py --limit 2
```

Expected: the file lists of the needed category's zips are read, then `[1/2] test ...` and `[2/2] test ...`, then `Done: 2 prepared, 0 failed`.

- [ ] **Step 6: Inspect the 2 results by hand**

```bash
.venv/Scripts/python -c "
import glob, numpy as np
from isl.features import hands_up, PRESENCE_SLICE
for p in glob.glob('data/landmarks/*/*/*.npz'):
    f = np.load(p)['features']
    body = f.any(axis=1)
    print(p, f.shape, f'body found {body.mean():.0%}', f'any hand {f[:, PRESENCE_SLICE].max(1).mean():.0%}',
          f'hands up {hands_up(f).mean():.0%}', 'up flags:', ''.join('^' if u else '.' for u in hands_up(f)))
"
```

Expected: roughly 50–90 frames per video, body found in ≥ 95 % of frames, and an up-flag strip shaped like `....^^^^^^^^^^^^^^^^....`, with dots at the start and end where the signer rests. If the body is found in < 80 % of frames, **stop and report**: the pose model or the 640 px downscale isn't working for these videos. The exact `HANDS_UP_K` gets calibrated in Task 14.

- [ ] **Step 7: Commit**

```bash
git add scripts/prepare_include50.py tests/test_prepare.py
git commit -m "Add the INCLUDE-50 preparation script (fetch, landmarks, delete, resume)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Full dataset preparation (runs in the background)

**Files:** none (produces `data/`, which is git-ignored)

- [ ] **Step 1: Start the full run in the background** (approved ~15 GB streamed download)

```bash
.venv/Scripts/python -u scripts/prepare_include50.py > data/prepare.log 2>&1
```

Run it with `run_in_background`. Expect about 1–2 hours. **Continue with Tasks 8–13 while it runs.** It's safe to stop and start again: finished videos are skipped.

- [ ] **Step 2: When it finishes, check the result**

```bash
tail -n 3 data/prepare.log
find data/landmarks -name "*.npz" | wc -l
ls data/examples | wc -l
```

Expected: `Done: … prepared, N failed` with **at least 910 `.npz` files** (≥ 95 % of 958, per the spec's success criteria) and 50 examples. If failures are network errors, run the same command again; it only redoes the missing ones. If more than 5 % still fail, report the `data/failed.txt` reasons to the user before continuing.

---

### Task 8: Training augmentation

**Files:**
- Create: `isl/augment.py`
- Test: `tests/test_augment.py`

**Interfaces:**
- Consumes: `flip`, `resample`, `POSE_SLICE`, `HAND_SLICES`, `PRESENCE_SLICE` (Task 4).
- Produces: `rotate_scale(sequence, angle_degrees, scale) -> np.ndarray` and `augment(frames (T, F), rng: np.random.Generator) -> np.ndarray (32, 184) float32`.

- [ ] **Step 1: Write the failing tests** in `tests/test_augment.py`

```python
import unittest

import numpy as np

from isl import features as F
from isl.augment import augment, rotate_scale
from tests.fakes import frame, make_raw


class AugmentTests(unittest.TestCase):
    def test_output_is_a_model_sized_sequence(self):
        rng = np.random.default_rng(0)
        clip = np.stack([frame(True)] * 40)
        for _ in range(20):
            out = augment(clip, rng)
            self.assertEqual(out.shape, (32, 184))
            self.assertEqual(out.dtype, np.float32)

    def test_very_short_clips_work(self):
        rng = np.random.default_rng(0)
        for length in (1, 2, 3):
            self.assertEqual(augment(np.stack([frame(True)] * length), rng).shape, (32, 184))

    def test_missing_hand_stays_empty_and_presence_is_kept(self):
        rng = np.random.default_rng(1)
        clip = np.stack([F.frame_features(make_raw("right"))[0]] * 20)
        for _ in range(20):
            out = augment(clip, rng)
            present = out[0, F.PRESENCE_SLICE]
            self.assertEqual(sorted(present.tolist()), [0.0, 1.0])
            self.assertFalse(out[:, F.HAND_SLICES[int(np.argmin(present))]].any())

    def test_rotation_keeps_lengths_and_scale_only_changes_body_positions(self):
        seq = np.stack([frame(True)] * 3)
        out = rotate_scale(seq, 90, 2.0)
        pose, pose_out = seq[:, F.POSE_SLICE].reshape(3, 7, 2), out[:, F.POSE_SLICE].reshape(3, 7, 2)
        np.testing.assert_allclose(np.linalg.norm(pose_out, axis=2), 2 * np.linalg.norm(pose, axis=2), rtol=1e-5)
        local = slice(F.HAND_SLICES[0].start + 42, F.HAND_SLICES[0].stop)
        np.testing.assert_allclose(np.linalg.norm(out[:, local].reshape(3, 21, 2), axis=2),
                                   np.linalg.norm(seq[:, local].reshape(3, 21, 2), axis=2), rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(out[:, F.PRESENCE_SLICE], seq[:, F.PRESENCE_SLICE])
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python -m unittest tests.test_augment -v`
Expected: `ModuleNotFoundError: No module named 'isl.augment'`.

- [ ] **Step 3: Implement** `isl/augment.py`

```python
"""Random variations of training clips, so about 14 videos per word go further"""
import numpy as np

from isl.features import HAND_SLICES, POSE_SLICE, PRESENCE_SLICE, flip, resample


def rotate_scale(sequence, angle_degrees, scale):
    """Rotate every point about the shoulder midpoint and scale body positions (hand shapes only rotate)"""
    seq = np.array(sequence, dtype=np.float32, copy=True)
    theta = np.deg2rad(angle_degrees)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)

    def transform(block, factor):
        points = seq[:, block].reshape(len(seq), -1, 2)
        seq[:, block] = (points @ rotation.T * factor).reshape(len(seq), -1)

    transform(POSE_SLICE, scale)
    for hand in HAND_SLICES:
        transform(slice(hand.start, hand.start + 42), scale)  # where the hand is, in the body frame
        transform(slice(hand.start + 42, hand.stop), 1.0)  # the hand's shape
    return seq


def augment(frames, rng):
    """A randomly varied copy of one trimmed (T, F) training clip, resampled to the model's length"""
    frames = np.asarray(frames, dtype=np.float32)
    if len(frames) >= 4:
        keep = max(2, round(len(frames) * rng.uniform(0.85, 1.0)))  # crop, which also varies the speed
        start = int(rng.integers(0, len(frames) - keep + 1))
        frames = frames[start:start + keep]
        mask = rng.random(len(frames)) > 0.1  # drop about 10 % of frames, keeping the ends
        mask[[0, -1]] = True
        frames = frames[mask]
    seq = resample(frames)
    if rng.random() < 0.5:
        seq = flip(seq)
    seq = rotate_scale(seq, rng.uniform(-10, 10), rng.uniform(0.9, 1.1))
    coords = seq[:, :PRESENCE_SLICE.start]
    coords += rng.normal(0, 0.01, coords.shape).astype(np.float32) * (coords != 0)  # missing hands stay zero
    return seq
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest tests.test_augment -v`
Expected: 4 tests, `OK`.

- [ ] **Step 5: Commit**

```bash
git add isl/augment.py tests/test_augment.py
git commit -m "Add training augmentation: flip, rotate, scale, crop, frame drop, noise" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Model module

**Files:**
- Create: `isl/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `FEATURE_VERSION`, `NUM_FEATURES`, `SEQUENCE_LENGTH` (Task 4).
- Produces:
  - `MODEL_PATH` (`models/isl50_bigru.keras`), `LABELS_PATH` (`models/labels.json`)
  - `build_model(num_classes, sequence_length=32, num_features=184) -> keras.Model` (compiled)
  - `save_labels(path, keys: list[str], displays: list[str])`
  - `Recogniser(model, keys, display)` with `.predict(sequence (32, 184), top_k=3) -> list[tuple[str display, float probability]]`, most likely first
  - `load_recogniser(model_path=MODEL_PATH, labels_path=LABELS_PATH) -> Recogniser`, which raises `SystemExit` with a clear message when files are missing or don't match

- [ ] **Step 1: Write the failing tests** in `tests/test_model.py`

```python
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from isl.model import build_model, load_recogniser, save_labels


class ModelTests(unittest.TestCase):
    def test_output_is_a_probability_per_class(self):
        model = build_model(5)
        out = np.asarray(model(np.zeros((2, 32, 184), np.float32), training=False))
        self.assertEqual(out.shape, (2, 5))
        np.testing.assert_allclose(out.sum(axis=1), 1, rtol=1e-5)

    def test_save_load_and_predict_top_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            build_model(4).save(tmp / "m.keras")
            save_labels(tmp / "labels.json", ["a", "b", "c", "d"], ["A", "B", "C", "D"])
            recogniser = load_recogniser(tmp / "m.keras", tmp / "labels.json")
            guesses = recogniser.predict(np.zeros((32, 184), np.float32))  # an empty sign must not crash
        self.assertEqual(len(guesses), 3)
        probabilities = [p for _, p in guesses]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))
        self.assertTrue({word for word, _ in guesses} <= {"A", "B", "C", "D"})

    def test_missing_model_explains_how_to_get_one(self):
        with self.assertRaises(SystemExit) as caught:
            load_recogniser(Path("missing.keras"), Path("missing.json"))
        self.assertIn("scripts/train.py", str(caught.exception))

    def test_feature_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            build_model(2).save(tmp / "m.keras")
            (tmp / "labels.json").write_text(json.dumps({"feature_version": 0, "sequence_length": 32,
                                                          "num_features": 184, "keys": ["a", "b"],
                                                          "display": ["a", "b"]}))
            with self.assertRaises(SystemExit) as caught:
                load_recogniser(tmp / "m.keras", tmp / "labels.json")
        self.assertIn("feature_version", str(caught.exception))
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python -m unittest tests.test_model -v`
Expected: `ModuleNotFoundError: No module named 'isl.model'`.

- [ ] **Step 3: Implement** `isl/model.py`

```python
"""The BiGRU sign classifier: building, saving and loading it, and turning a sign into words"""
import json
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np

from isl.features import FEATURE_VERSION, NUM_FEATURES, SEQUENCE_LENGTH

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "isl50_bigru.keras"
LABELS_PATH = MODELS_DIR / "labels.json"


def build_model(num_classes, sequence_length=SEQUENCE_LENGTH, num_features=NUM_FEATURES):
    layers = keras.layers
    model = keras.Sequential([
        keras.Input((sequence_length, num_features)),
        layers.TimeDistributed(layers.Dense(128, activation="relu")),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.GRU(128, return_sequences=True)),
        layers.Bidirectional(layers.GRU(64)),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def save_labels(path, keys, displays):
    Path(path).write_text(json.dumps({"feature_version": FEATURE_VERSION, "sequence_length": SEQUENCE_LENGTH,
                                      "num_features": NUM_FEATURES, "keys": list(keys),
                                      "display": list(displays)}, indent=2), encoding="utf-8")


@dataclass
class Recogniser:
    model: keras.Model
    keys: list
    display: list  # word for the interpreter, same order as keys

    def predict(self, sequence, top_k=3):
        """[(word, probability)] for one (32, F) sign, most likely first"""
        probabilities = np.asarray(self.model(np.asarray(sequence, np.float32)[None], training=False))[0]
        best = np.argsort(probabilities)[::-1][:top_k]
        return [(self.display[i], float(probabilities[i])) for i in best]


def load_recogniser(model_path=MODEL_PATH, labels_path=LABELS_PATH):
    model_path, labels_path = Path(model_path), Path(labels_path)
    if not model_path.exists() or not labels_path.exists():
        raise SystemExit(f"No trained model found ({model_path.name}). Run: python scripts/train.py "
                         "(or pull the trained model from GitHub)")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    expected = {"feature_version": FEATURE_VERSION, "sequence_length": SEQUENCE_LENGTH, "num_features": NUM_FEATURES}
    for name, value in expected.items():
        if labels.get(name) != value:
            raise SystemExit(f"{labels_path.name} was made with {name}={labels.get(name)}, but isl/features.py "
                             f"uses {value}. Retrain with: python scripts/train.py")
    return Recogniser(keras.models.load_model(model_path), labels["keys"], labels["display"])
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest tests.test_model -v`
Expected: 4 tests, `OK`.

- [ ] **Step 5: Commit**

```bash
git add isl/model.py tests/test_model.py
git commit -m "Add the BiGRU model with save/load checks and top-k prediction" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Training and evaluation script

**Files:**
- Create: `scripts/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `include_data.load_label_keys`, `load_split`, `DATA_DIR` (Task 2); `hands_up`, `trim`, `resample`, `NUM_FEATURES` (Task 4); `augment` (Task 8); `build_model`, `save_labels`, `load_recogniser` (Task 9).
- Produces:
  - `main(argv=None)` with options `--smoke`, `--out DIR` and `--epochs N`
  - Writes `<out>/models/isl50_bigru.keras`, `<out>/models/labels.json`, `<out>/reports/metrics.json` (keys `counts`, `baseline`, `bigru`, `inference_ms`, `paper_top1`, `epochs_trained`), `<out>/reports/confusion_matrix.png` and `<out>/reports/results.md`
  - `metrics(probabilities, y, keys) -> {"top1", "top3", "macro_f1", "per_word"}`

- [ ] **Step 1: Write the failing tests** in `tests/test_train.py`

```python
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from isl.model import load_recogniser
from tests.test_prepare import load_script

train = load_script("train")


class MetricsTests(unittest.TestCase):
    def test_metrics_only_score_words_in_the_test_set(self):
        probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.5, 0.4, 0.1]])
        result = train.metrics(probabilities, np.array([0, 1, 1]), ["a", "b", "c"])
        self.assertAlmostEqual(result["top1"], 1 / 3)
        self.assertAlmostEqual(result["top3"], 1.0)
        self.assertEqual(result["per_word"], {"a": 1.0, "b": 0.0})
        self.assertAlmostEqual(result["macro_f1"], 1 / 3, places=3)  # F1 a = 2/3, b = 0; "c" isn't averaged in


class TrainSmokeTests(unittest.TestCase):
    def test_smoke_run_writes_model_labels_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("builtins.print"):
                train.main(["--smoke", "--out", tmp])
            out = Path(tmp)
            for relative in ("models/isl50_bigru.keras", "models/labels.json", "reports/metrics.json",
                             "reports/confusion_matrix.png", "reports/results.md"):
                self.assertTrue((out / relative).exists(), relative)
            report = json.loads((out / "reports/metrics.json").read_text())
            self.assertEqual(set(report), {"counts", "baseline", "bigru", "inference_ms", "paper_top1",
                                           "epochs_trained"})
            self.assertIn("| Model | Top-1 |", (out / "reports/results.md").read_text(encoding="utf-8"))
            recogniser = load_recogniser(out / "models/isl50_bigru.keras", out / "models/labels.json")
            self.assertEqual(len(recogniser.predict(np.zeros((32, 184), np.float32))), 3)
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python -m unittest tests.test_train -v`
Expected: `FileNotFoundError` for `scripts/train.py`.

- [ ] **Step 3: Implement** `scripts/train.py`

```python
"""Train and evaluate the INCLUDE-50 sign classifiers, then write the model and reports.

Run from the project folder after scripts/prepare_include50.py:  python scripts/train.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import keras  # noqa: E402

from isl import include_data  # noqa: E402
from isl.augment import augment  # noqa: E402
from isl.features import NUM_FEATURES, hands_up, resample, trim  # noqa: E402
from isl.model import build_model, save_labels  # noqa: E402

LANDMARKS_DIR = include_data.DATA_DIR / "landmarks"
PAPER_TOP1 = 0.945  # best INCLUDE-50 result reported in the INCLUDE paper
SEED = 42
BATCH_SIZE = 32
PATIENCE = 25


def load_clips(landmarks_dir, keys):
    """{split: (list of trimmed (T, F) clips, labels)} from the prepared .npz files"""
    index = {key: i for i, key in enumerate(keys)}
    clips = {}
    for split in ("train", "val", "test"):
        frames, labels = [], []
        for path in sorted((Path(landmarks_dir) / split).glob("*/*.npz")):
            with np.load(path) as data:
                features = data["features"]
            frames.append(trim(features, hands_up(features)))
            labels.append(index[path.parent.name])
        clips[split] = (frames, np.array(labels, dtype=np.int64))
    return clips


def synthetic_clips(rng, num_classes=3, per_split=(12, 4, 4)):
    """A tiny made-up dataset where each class has its own movement, for --smoke runs"""
    clips = {}
    for split, count in zip(("train", "val", "test"), per_split):
        frames, labels = [], []
        for label in range(num_classes):
            for _ in range(count):
                length = int(rng.integers(20, 40))
                clip = rng.normal(0, 0.05, (length, NUM_FEATURES)).astype(np.float32)
                clip[:, label] += np.linspace(0, 1, length, dtype=np.float32)
                frames.append(clip)
                labels.append(label)
        clips[split] = (frames, np.array(labels, dtype=np.int64))
    return clips


def summary_features(X):
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1)], axis=1)


def train_baseline(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(summary_features(X), y)


def baseline_probabilities(baseline, X, num_classes):
    probabilities = np.zeros((len(X), num_classes))
    probabilities[:, baseline.classes_] = baseline.predict_proba(summary_features(X))
    return probabilities


def predict(model, X):
    return np.asarray(model.predict(X, batch_size=64, verbose=0))


def train_bigru(train_clips, y_train, X_val, y_val, num_classes, epochs, rng):
    """Fresh augmentations every epoch; keeps the weights with the best validation accuracy"""
    model = build_model(num_classes)
    best_accuracy, best_weights, best_epoch, since_best = -1.0, model.get_weights(), 0, 0
    for epoch in range(1, epochs + 1):
        X = np.stack([augment(clip, rng) for clip in train_clips])
        history = model.fit(X, y_train, batch_size=BATCH_SIZE, epochs=1, shuffle=True, verbose=0)
        val_accuracy = float(np.mean(predict(model, X_val).argmax(1) == y_val))
        if val_accuracy > best_accuracy:
            best_accuracy, best_weights, best_epoch, since_best = val_accuracy, model.get_weights(), epoch, 0
        else:
            since_best += 1
        print(f"epoch {epoch:3d}  loss {history.history['loss'][0]:.3f}  val acc {val_accuracy:.3f}  "
              f"best {best_accuracy:.3f}", flush=True)
        if since_best >= PATIENCE:
            break
    model.set_weights(best_weights)
    return model, best_epoch


def metrics(probabilities, y, keys):
    """Top-1/top-3 accuracy, macro-F1 and per-word accuracy over the words present in y"""
    from sklearn.metrics import f1_score
    predicted = probabilities.argmax(1)
    top3 = np.argsort(probabilities, axis=1)[:, ::-1][:, :3]
    present = np.unique(y)
    return {
        "top1": float(np.mean(predicted == y)),
        "top3": float(np.mean([label in row for label, row in zip(y, top3)])),
        "macro_f1": float(f1_score(y, predicted, labels=present, average="macro", zero_division=0)),
        "per_word": {keys[i]: float(np.mean(predicted[y == i] == i)) for i in present},
    }


def save_confusion_matrix(probabilities, y, words, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y, probabilities.argmax(1), labels=list(range(len(words))))
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(words)), words, rotation=90, fontsize=7)
    ax.set_yticks(range(len(words)), words, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("INCLUDE-50 test split: BiGRU confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def results_markdown(report):
    baseline, bigru, counts = report["baseline"], report["bigru"], report["counts"]
    hardest = sorted(bigru["per_word"].items(), key=lambda item: item[1])[:5]
    return "\n".join([
        f"Official INCLUDE-50 test split: {counts['test']} videos "
        f"(trained on {counts['train']}, validated on {counts['val']}).",
        "",
        "| Model | Top-1 | Top-3 | Macro-F1 |",
        "|---|---|---|---|",
        f"| Baseline: logistic regression on summary features | {baseline['top1']:.1%} | {baseline['top3']:.1%} "
        f"| {baseline['macro_f1']:.3f} |",
        f"| **BiGRU on landmark sequences (this project)** | **{bigru['top1']:.1%}** | {bigru['top3']:.1%} "
        f"| {bigru['macro_f1']:.3f} |",
        f"| INCLUDE paper, best model on INCLUDE-50 | {report['paper_top1']:.1%} | – | – |",
        "",
        f"BiGRU inference: {report['inference_ms']:.1f} ms per sign on CPU. "
        f"Hardest words: " + ", ".join(f"{word} ({accuracy:.0%})" for word, accuracy in hardest) + ".",
        "",
        "INCLUDE's 7 signers appear in every split, so these are *seen-signer* results. "
        "Accuracy for a new signer and camera will be lower.",
        "",
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate the INCLUDE-50 sign classifiers.")
    parser.add_argument("--smoke", action="store_true", help="quick run on a tiny synthetic dataset")
    parser.add_argument("--out", type=Path, default=ROOT, help="folder that gets models/ and reports/")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args(argv)

    keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)
    if args.smoke:
        keys = displays = ["a", "b", "c"]
        clips, epochs = synthetic_clips(rng), 2
    else:
        keys = include_data.load_label_keys()
        names = {s.key: s.display for split in ("train", "val", "test") for s in include_data.load_split(split)}
        displays = [names[key] for key in keys]
        clips, epochs = load_clips(LANDMARKS_DIR, keys), args.epochs
    counts = {split: len(clips[split][1]) for split in clips}
    print("clips per split:", counts)
    if not all(counts.values()):
        raise SystemExit("No prepared landmarks found. Run: python scripts/prepare_include50.py")

    X = {split: np.stack([resample(clip) for clip in clips[split][0]]) for split in clips}
    y = {split: clips[split][1] for split in clips}

    baseline = train_baseline(X["train"], y["train"])
    bigru, best_epoch = train_bigru(clips["train"][0], y["train"], X["val"], y["val"], len(keys), epochs, rng)

    bigru_test = predict(bigru, X["test"])
    start = time.perf_counter()
    for sequence in X["test"][:20]:
        bigru(sequence[None], training=False)
    inference_ms = (time.perf_counter() - start) / min(20, len(X["test"])) * 1000

    report = {"counts": counts,
              "baseline": metrics(baseline_probabilities(baseline, X["test"], len(keys)), y["test"], keys),
              "bigru": metrics(bigru_test, y["test"], keys),
              "inference_ms": inference_ms, "paper_top1": PAPER_TOP1, "epochs_trained": best_epoch}

    models_dir, reports_dir = args.out / "models", args.out / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    bigru.save(models_dir / "isl50_bigru.keras")
    save_labels(models_dir / "labels.json", keys, displays)
    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_confusion_matrix(bigru_test, y["test"], displays, reports_dir / "confusion_matrix.png")
    (reports_dir / "results.md").write_text(results_markdown(report), encoding="utf-8")
    print(results_markdown(report))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest tests.test_train -v`
Expected: 2 tests, `OK` (the smoke run takes well under a minute).

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py tests/test_train.py
git commit -m "Add the training script: baseline + BiGRU, official test-split evaluation, reports" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Live segmenter

**Files:**
- Create: `isl/segmenter.py`
- Test: `tests/test_segmenter.py`

**Interfaces:**
- Consumes: `hands_up`, `to_sequence` (Task 4).
- Produces: `Segmenter(start_seconds=0.1, end_seconds=0.3, min_seconds=0.3, max_seconds=4.0)` with `.update(features (184,), now: float) -> np.ndarray (32, 184) | None`, `.reset()` and `.signing: bool`.

- [ ] **Step 1: Write the failing tests** in `tests/test_segmenter.py`

```python
import unittest

from isl.segmenter import Segmenter
from tests.fakes import frame


class SegmenterTests(unittest.TestCase):
    def run_timeline(self, timeline, fps=30):
        """timeline: [(hands up?, seconds)]; returns [(time, sequence)] for each sign emitted"""
        segmenter, now, emitted = Segmenter(), 0.0, []
        for up, seconds in timeline:
            for _ in range(round(seconds * fps)):
                result = segmenter.update(frame(up), now)
                if result is not None:
                    emitted.append((now, result))
                now += 1 / fps
        return emitted

    def test_one_sign_ends_after_the_hands_are_down(self):
        emitted = self.run_timeline([(False, 0.5), (True, 1.0), (False, 1.0)])
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0][1].shape, (32, 184))
        self.assertAlmostEqual(emitted[0][0], 1.5 + 0.3, delta=0.1)

    def test_blips_are_ignored(self):
        self.assertEqual(self.run_timeline([(True, 0.05), (False, 0.5), (True, 0.2), (False, 1.0)]), [])

    def test_back_to_back_signs(self):
        self.assertEqual(len(self.run_timeline([(True, 0.8), (False, 0.4), (True, 0.8), (False, 0.5)])), 2)

    def test_a_brief_drop_does_not_split_a_sign(self):
        self.assertEqual(len(self.run_timeline([(True, 0.6), (False, 0.1), (True, 0.6), (False, 0.5)])), 1)

    def test_long_signs_are_cut_off(self):
        emitted = self.run_timeline([(True, 5.0)])
        self.assertEqual(len(emitted), 1)
        self.assertAlmostEqual(emitted[0][0], 4.0, delta=0.1)

    def test_signing_flag_and_reset(self):
        segmenter = Segmenter()
        for i in range(6):
            segmenter.update(frame(True), i / 30)
        self.assertTrue(segmenter.signing)
        segmenter.reset()
        self.assertFalse(segmenter.signing)
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python -m unittest tests.test_segmenter -v`
Expected: `ModuleNotFoundError: No module named 'isl.segmenter'`.

- [ ] **Step 3: Implement** `isl/segmenter.py`

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest tests.test_segmenter -v`
Expected: 6 tests, `OK`.

- [ ] **Step 5: Commit**

```bash
git add isl/segmenter.py tests/test_segmenter.py
git commit -m "Add the live segmenter: hands up, sign, hands down" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Live app on the ISL model

**Files:**
- Replace: `signlan.py`
- Replace: `tests/test_signlan.py`
- Modify: `requirements.txt` (remove the `tf-keras` line)
- Delete: `keras_model.h5`, `labels.txt`

**Interfaces:**
- Consumes: `frame_features`, `RawLandmarks` (Task 4); `LandmarkExtractor` (Task 5); `load_recogniser`, whose `Recogniser.predict` returns `[(word, probability)]` (Task 9); `Segmenter` (Task 11); `tests/fakes.make_raw`, `make_hand` (Task 4).
- Produces:
  - `SentenceBuilder(send_after=2.0)` with `.add(word, now)`, `.remove_last(now)`, `.progress(now) -> float`, `.update(signing, now) -> list[str] | None` and `.words`
  - `status_text(busy_status: str, body_found: bool, signing: bool, has_words: bool) -> str`
  - `draw_skeleton(display, raw)` and `main()`
  - Kept unchanged: `interpret`, `speak`, `Responder`, `wrap_text`, `shade`, `draw_overlay`

- [ ] **Step 1: Write the new tests.** Replace `tests/test_signlan.py` completely:

```python
import os
import threading
import time
import unittest
from unittest import mock

import cv2
import numpy as np
from groq import APIConnectionError

import signlan
from isl.features import RawLandmarks
from tests.fakes import make_hand, make_raw


def fake_client(reply="I love you so much!"):
    client = mock.Mock()
    client.chat.completions.create.return_value.choices = [mock.Mock(message=mock.Mock(content=reply))]
    return client


class SentenceBuilderTests(unittest.TestCase):
    def test_pause_after_signing_sends_the_words(self):
        builder = signlan.SentenceBuilder(send_after=2.0)
        builder.add("hello", now=10.0)
        self.assertIsNone(builder.update(signing=False, now=11.0))
        self.assertAlmostEqual(builder.progress(11.0), 0.5)
        self.assertEqual(builder.update(signing=False, now=12.0), ["hello"])
        self.assertEqual(builder.words, [])

    def test_signing_again_restarts_the_pause(self):
        builder = signlan.SentenceBuilder(send_after=2.0)
        builder.add("hello", now=0.0)
        builder.update(signing=True, now=1.5)
        self.assertIsNone(builder.update(signing=False, now=2.5))
        builder.add("teacher", now=3.0)
        self.assertEqual(builder.update(signing=False, now=5.0), ["hello", "teacher"])

    def test_nothing_is_sent_without_words(self):
        builder = signlan.SentenceBuilder(send_after=2.0)
        self.assertIsNone(builder.update(signing=False, now=100.0))
        self.assertEqual(builder.progress(100.0), 0.0)

    def test_backspace_removes_the_last_word_and_gives_more_time(self):
        builder = signlan.SentenceBuilder(send_after=2.0)
        builder.add("hello", now=0.0)
        builder.add("dog", now=1.0)
        builder.remove_last(now=2.5)
        self.assertEqual(builder.words, ["hello"])
        self.assertIsNone(builder.update(signing=False, now=4.0))
        self.assertEqual(builder.update(signing=False, now=4.5), ["hello"])

    def test_backspace_on_an_empty_sentence_does_nothing(self):
        builder = signlan.SentenceBuilder()
        builder.remove_last(now=1.0)
        self.assertEqual(builder.words, [])
        self.assertIsNone(builder.update(signing=False, now=10.0))


class StatusTextTests(unittest.TestCase):
    def test_priorities(self):
        self.assertEqual(signlan.status_text("Thinking...", False, True, True), "Thinking...")
        self.assertEqual(signlan.status_text("", False, False, False), "Move back so your shoulders are visible")
        self.assertEqual(signlan.status_text("", True, True, False), "Signing...")
        self.assertIn("Pause to send", signlan.status_text("", True, False, True))
        self.assertEqual(signlan.status_text("", True, False, False), "Sign a word")


class SkeletonTests(unittest.TestCase):
    def test_skeleton_is_drawn_mirrored(self):
        display = np.zeros((480, 640, 3), np.uint8)
        raw = RawLandmarks(pose=None, hands=[make_hand((500.0, 200.0))], width=640, height=480)
        signlan.draw_skeleton(display, raw)
        columns = np.flatnonzero(display.any(axis=(0, 2)))
        self.assertTrue(columns.size)
        self.assertLess(columns.max(), 320)  # a hand on the camera image's right is on the preview's left


class InterpretTests(unittest.TestCase):
    def test_signs_are_sent_in_order_with_the_interpreter_prompt(self):
        client = fake_client()
        signlan.interpret(client, ["hello", "teacher"])
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], signlan.GROQ_MODEL)
        self.assertEqual(kwargs["messages"], [
            {"role": "system", "content": signlan.SYSTEM_PROMPT},
            {"role": "user", "content": "Signs: hello / teacher"},
        ])

    def test_reply_is_cleaned_up(self):
        self.assertEqual(signlan.interpret(fake_client(' "Hello, teacher!" '), ["hello"]), "Hello, teacher!")

    def test_api_error_returns_none(self):
        client = mock.Mock()
        client.chat.completions.create.side_effect = APIConnectionError(request=mock.Mock())
        with mock.patch("builtins.print"):
            self.assertIsNone(signlan.interpret(client, ["hello"]))

    def test_empty_reply_returns_none(self):
        self.assertIsNone(signlan.interpret(fake_client(""), ["hello"]))


class ResponderTests(unittest.TestCase):
    def test_replies_in_the_background_and_speaks(self):
        speaking, finish_speaking = threading.Event(), threading.Event()

        def slow_speak(text):
            speaking.set()
            finish_speaking.wait(5)

        responder = signlan.Responder(fake_client("Hello there!"))
        with mock.patch.object(signlan, "speak", side_effect=slow_speak) as speak, mock.patch("builtins.print"):
            responder.start(["hello"])
            self.assertTrue(speaking.wait(5))
            self.assertTrue(responder.busy)
            self.assertEqual(responder.status, "Speaking...")
            finish_speaking.set()
            responder.wait(5)
        self.assertFalse(responder.busy)
        self.assertEqual((responder.status, responder.heard, responder.reply), ("", ["hello"], "Hello there!"))
        speak.assert_called_once_with("Hello there!")

    def test_failed_reply_is_shown_and_not_spoken(self):
        client = mock.Mock()
        client.chat.completions.create.side_effect = APIConnectionError(request=mock.Mock())
        responder = signlan.Responder(client)
        with mock.patch.object(signlan, "speak") as speak, mock.patch("builtins.print"):
            responder.start(["hello"])
            responder.wait(5)
        speak.assert_not_called()
        self.assertIn("no reply", responder.reply)


class OverlayTests(unittest.TestCase):
    def test_long_text_is_wrapped_to_fit(self):
        text = "Said: " + "this is a fairly long sentence that will not fit on one line " * 2
        lines = signlan.wrap_text(text, 300, 0.7)
        self.assertGreater(len(lines), 1)
        self.assertEqual(" ".join(lines), " ".join(text.split()))
        for line in lines:
            self.assertLessEqual(cv2.getTextSize(line, signlan.FONT, 0.7, 2)[0][0], 300)

    def test_overlay_draws_on_the_frame(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        signlan.draw_overlay(frame, "hello (91%)", True, 0.5, "Thinking...", ["Signed: hello", "Said: Hello!"])
        self.assertTrue((frame != 128).any())


class MainLoopTests(unittest.TestCase):
    """Drive main() with scripted landmarks: sign, lower the hands, pause, hear the reply"""

    def test_sign_pause_send_speak(self):
        script = ["none"] * 5 + ["both"] * 20 + ["none"] * 90
        position = {"i": 0}

        class FakeCamera:
            def isOpened(self):
                return True

            def read(self):
                time.sleep(0.03)
                position["i"] += 1
                return True, np.zeros((480, 640, 3), dtype=np.uint8)

            def release(self):
                pass

        class FakeExtractor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

            def extract(self, frame, timestamp_ms):
                return make_raw(script[min(position["i"], len(script)) - 1])

        recogniser = mock.Mock()
        recogniser.predict.return_value = [("hello", 0.9), ("good", 0.05), ("bank", 0.01)]
        client = fake_client("Hello there!")
        spoken = []
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "test"}), \
                mock.patch.object(signlan, "load_recogniser", return_value=recogniser), \
                mock.patch.object(signlan, "LandmarkExtractor", FakeExtractor), \
                mock.patch.object(signlan, "Groq", return_value=client), \
                mock.patch.object(signlan, "speak", side_effect=spoken.append), \
                mock.patch.object(signlan.cv2, "VideoCapture", return_value=FakeCamera()), \
                mock.patch.object(signlan.cv2, "imshow"), \
                mock.patch.object(signlan.cv2, "destroyAllWindows"), \
                mock.patch.object(signlan.cv2, "getWindowProperty", return_value=1), \
                mock.patch.object(signlan.cv2, "waitKey",
                                  side_effect=lambda _: 27 if position["i"] >= len(script) else -1), \
                mock.patch("builtins.print"):
            signlan.main()

        recogniser.predict.assert_called_once()
        self.assertEqual(recogniser.predict.call_args.args[0].shape, (32, 184))
        self.assertEqual(client.chat.completions.create.call_args.kwargs["messages"][-1]["content"], "Signs: hello")
        self.assertEqual(spoken, ["Hello there!"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python -m unittest tests.test_signlan -v`
Expected: errors such as `AttributeError: module 'signlan' has no attribute 'status_text'`, plus failures in the sentence-builder tests.

- [ ] **Step 3: Replace `signlan.py`**

```python
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyttsx3
from dotenv import load_dotenv
from groq import APIError, Groq

from isl.features import frame_features
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


def status_text(busy_status, body_found, signing, has_words):
    if busy_status:
        return busy_status
    if not body_found:
        return "Move back so your shoulders are visible"
    if signing:
        return "Signing..."
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
                                 bool(builder.words))
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
```

- [ ] **Step 4: Remove the old model and `tf-keras`**

```bash
git rm keras_model.h5 labels.txt
.venv/Scripts/python -m pip uninstall -y tf-keras
```

Then delete the line `tf-keras==2.18.0` from `requirements.txt`.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python -m unittest discover -s tests -t . -v`
Expected: every test passes (`OK`). The `ExtractorTests` run too, now that the models are downloaded.

- [ ] **Step 6: Commit**

```bash
git add signlan.py tests/test_signlan.py requirements.txt
git commit -m "Switch the live app to ISL word recognition with pause-to-send" -m "Signs are cut out of the landmark stream by the segmenter, recognised by the INCLUDE-50 BiGRU, and a 2 second pause sends the sentence to the interpreter. Backspace removes the last word. The Teachable Machine model and tf-keras are removed." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Example video player

**Files:**
- Create: `isl/examples.py`
- Test: `tests/test_examples.py`

**Interfaces:**
- Consumes: `DATA_DIR`, `label_key`, `DISPLAY_OVERRIDES` (Task 2); `data/examples/<key>.MOV` (Tasks 6–7).
- Produces: `example_path(word, examples_dir=EXAMPLES_DIR) -> Path | None`, `play(path)` and `main(argv=None)`, run as `python -m isl.examples <word>`.

- [ ] **Step 1: Write the failing tests** in `tests/test_examples.py`

```python
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from isl import examples


class ExampleTests(unittest.TestCase):
    def test_words_and_display_names_find_their_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            for key in ("thankyou", "biglarge", "hello"):
                (Path(tmp) / f"{key}.MOV").write_bytes(b"")
            self.assertEqual(examples.example_path("thank you", Path(tmp)).name, "thankyou.MOV")
            self.assertEqual(examples.example_path("Big", Path(tmp)).name, "biglarge.MOV")
            self.assertEqual(examples.example_path("HELLO", Path(tmp)).name, "hello.MOV")
            self.assertIsNone(examples.example_path("penguin", Path(tmp)))

    def test_unknown_word_explains(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(examples, "EXAMPLES_DIR", Path(tmp)):
            with self.assertRaises(SystemExit) as caught:
                examples.main(["penguin"])
        self.assertIn("penguin", str(caught.exception))
```

- [ ] **Step 2: Run to see them fail**

Run: `.venv/Scripts/python -m unittest tests.test_examples -v`
Expected: `ModuleNotFoundError: No module named 'isl.examples'`.

- [ ] **Step 3: Implement** `isl/examples.py`

```python
"""Play the INCLUDE reference video for a word, to learn how it's signed:  python -m isl.examples <word>"""
import sys

import cv2

from isl.include_data import DATA_DIR, DISPLAY_OVERRIDES, label_key

EXAMPLES_DIR = DATA_DIR / "examples"
_KEYS_BY_DISPLAY = {display: key for key, display in DISPLAY_OVERRIDES.items()}


def example_path(word, examples_dir=EXAMPLES_DIR):
    key = _KEYS_BY_DISPLAY.get(word.strip().lower(), label_key(word))
    matches = sorted(examples_dir.glob(f"{key}.*"))
    return matches[0] if matches else None


def play(path):
    capture = cv2.VideoCapture(str(path))
    delay = int(1000 / (capture.get(cv2.CAP_PROP_FPS) or 25))
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        if width > 960:
            frame = cv2.resize(frame, (960, round(height * 960 / width)))
        cv2.imshow(f"ISL example: {path.stem}", frame)
        if cv2.waitKey(delay) == 27:
            break
    capture.release()
    cv2.destroyAllWindows()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        words = sorted(path.stem for path in EXAMPLES_DIR.glob("*.*"))
        print("Usage: python -m isl.examples <word>\nAvailable: "
              + (", ".join(words) or "none yet - run scripts/prepare_include50.py"))
        return
    word = " ".join(argv)
    path = example_path(word, EXAMPLES_DIR)
    if path is None:
        raise SystemExit(f"No example video for {word!r}. Run python -m isl.examples to list the words.")
    play(path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m unittest tests.test_examples -v`
Expected: 2 tests, `OK`.

- [ ] **Step 5: Commit**

```bash
git add isl/examples.py tests/test_examples.py
git commit -m "Add an example player to learn how each ISL word is signed" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: Calibrate hands-up, train on INCLUDE-50, check the results

Requires Task 7 finished (≥ 910 `.npz` files).

**Files:**
- Modify: `isl/features.py` (`HANDS_UP_K`, only if calibration says so)
- Create: `models/isl50_bigru.keras`, `models/labels.json`, `reports/metrics.json`, `reports/confusion_matrix.png`, `reports/results.md`

- [ ] **Step 1: Measure where resting and signing wrists sit**

```bash
.venv/Scripts/python -c "
import glob, numpy as np
from isl.features import HAND_SLICES, PRESENCE_SLICE
rest, signing = [], []
for path in glob.glob('data/landmarks/train/*/*.npz'):
    f = np.load(path)['features']
    ys = [np.where(f[:, PRESENCE_SLICE.start + i] > 0.5, f[:, block.start + 1], np.inf) for i, block in enumerate(HAND_SLICES)]
    wrist = np.min(ys, axis=0)  # highest visible wrist per frame; inf = no hand seen
    n = len(wrist)
    rest += list(wrist[:3]) + list(wrist[-3:])
    signing += list(wrist[n // 4: 3 * n // 4])
rest, signing = np.array(rest), np.array(signing)
for k in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5):
    print(f'K={k}: rest frames counted as up {np.mean(rest < k):5.1%}   signing frames counted as up {np.mean(signing < k):5.1%}')
"
```

- [ ] **Step 2: Choose `HANDS_UP_K`.** Pick the K that maximises (signing counted as up) − (rest counted as up), preferring rest ≤ 10 % and signing ≥ 90 %. If that K isn't 1.5, change `HANDS_UP_K` in `isl/features.py` and re-run the suite (`.venv/Scripts/python -m unittest discover -s tests -t . -v`, expected `OK`). Keep the printed table for the commit message and the README.

- [ ] **Step 3: Train on the real data**

```bash
.venv/Scripts/python -u scripts/train.py > data/train.log 2>&1
tail -n 15 data/train.log
```

Run it with `run_in_background` (about 5–10 minutes). Expected: the log ends with the results table.

- [ ] **Step 4: Check against the success criteria**

```bash
.venv/Scripts/python -c "import json; r = json.load(open('reports/metrics.json')); print('counts', r['counts']); print('baseline top1 %.3f' % r['baseline']['top1'], '| bigru top1 %.3f top3 %.3f' % (r['bigru']['top1'], r['bigru']['top3']), '| best epoch', r['epochs_trained'])"
```

Criteria: the BiGRU's top-1 is clearly above the baseline's, and **≥ 0.85**. **If either fails, stop and report the numbers to the user**, together with the spec's next options (more augmentation, or the Transformer variant). Don't change the design without their approval. Either way, the numbers go into the README exactly as measured.

- [ ] **Step 5: Look at the confusion matrix**

Open `reports/confusion_matrix.png` (Read tool). Note which pairs of words get confused; they go in the README's limitations.

- [ ] **Step 6: Check the app loads the real model**

```bash
.venv/Scripts/python -c "import numpy as np; from isl.model import load_recogniser; r = load_recogniser(); print(len(r.keys), r.predict(np.zeros((32, 184), np.float32)))"
```

Expected: `50 [(word, p), (word, p), (word, p)]`.

- [ ] **Step 7: Commit the model and reports**

```bash
git add isl/features.py models/isl50_bigru.keras models/labels.json reports
git commit -m "Train the INCLUDE-50 BiGRU and add the evaluation reports" -m "<paste: the HANDS_UP_K table from Step 1 and the chosen value, plus baseline and BiGRU top-1/top-3 from reports/results.md>" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(The middle `-m` must contain the real measured table and numbers, not the angle-bracket text.)

---

### Task 15: README, live check, final verification

**Files:**
- Replace: `README.md`
- Modify: `docs/superpowers/specs/2026-09-23-isl-word-recognition-design.md` (Status line)

- [ ] **Step 1: Replace `README.md`**

````markdown
# ISL Sign Language Interpreter

Recognises **Indian Sign Language (ISL) word signs** from a webcam, turns them into a natural sentence with an LLM on
[Groq](https://groq.com/), and speaks it out loud, so a deaf or hard-of-hearing signer can talk to a hearing person.

- **Vocabulary:** the 50 words of the INCLUDE-50 benchmark (hello, thank you, good morning, I, you (plural), happy,
  teacher, father, brother, time, Monday, ...).
- **How it works:** MediaPipe tracks the hands and upper body → body-normalised landmark sequences → a bidirectional
  GRU classifies each sign → a Groq LLM phrases the sentence → text-to-speech.
- **Everything is reproducible:** scripts download the dataset, extract landmarks, train and evaluate.

## Results

<!-- results -->

![Confusion matrix](reports/confusion_matrix.png)

## How to use it

1. Sign a word, then lower your hands. The word appears at the bottom (`Signed:`).
2. Sign more words the same way.
3. Keep your hands down for 2 seconds. The ring fills up, the sentence is sent, and the bot says it out loud (`Said:`).

**Backspace** removes the last word; **Esc** or closing the window quits. Sit so your shoulders are in view.
Words the model isn't sure about show as "? maybe: ..." and aren't added.

To learn how a word is signed, play its reference video: `python -m isl.examples thank you`
(`python -m isl.examples` lists them).

## Setup (Windows)

Requires **Python 3.9 – 3.12**.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and paste your free key from https://console.groq.com/keys. The MediaPipe models
(14 MB) download automatically the first time you run the app.

```bash
python signlan.py
```

The trained model is in `models/`, so the app works without the dataset.

## Reproducing the model

```bash
python scripts/prepare_include50.py   # ~15 GB streamed from Zenodo, ~1-2 hours; safe to stop and re-run
python scripts/train.py               # ~10 minutes on CPU; writes models/ and reports/
```

`prepare_include50.py` reads only the 958 INCLUDE-50 videos out of the 57 GB dataset. It uses HTTP range requests
to fetch single files from inside the Zenodo zips, turns each video into landmarks, then deletes it.

## Project layout

| Path | Purpose |
|---|---|
| `signlan.py` | Live app: webcam → landmarks → segmenter → model → sentence → Groq → speech |
| `isl/features.py` | Landmarks → 184 body-normalised numbers per frame (shared by training and the app) |
| `isl/landmarks.py` | MediaPipe hand + pose tracking |
| `isl/segmenter.py` | Cuts the live stream into single signs (hands up → sign → hands down) |
| `isl/model.py` | The BiGRU: build, save, load, predict |
| `isl/augment.py` | Training-time variations (flip, rotate, scale, speed, noise) |
| `isl/include_data.py` | INCLUDE-50 splits and labels; range-fetching videos from Zenodo |
| `scripts/` | Dataset preparation and training |
| `reports/` | Measured results and confusion matrix |
| `tests/` | Unit, smoke and end-to-end tests: `python -m unittest discover -s tests -t . -v` |

## Limitations

- INCLUDE has 7 signers, and the same people appear in every split. The results above measure *seen signers*, so
  expect lower accuracy for a new signer, camera and room.
- Only the 50 INCLUDE-50 words are recognised. The pipeline supports the full 263-word INCLUDE set with a split
  change, but that hasn't been trained or evaluated.
- One sign at a time: lower your hands between signs.

## Credits

Dataset: **INCLUDE** by A. Sridhar, R. G. Ganesan, P. Kumar and M. Khapra, "INCLUDE: A Large Scale Dataset for
Indian Sign Language Recognition", ACM Multimedia 2020. Videos under CC-BY-4.0 from
[Zenodo record 4010759](https://zenodo.org/record/4010759); split lists from
[AI4Bharat/INCLUDE](https://github.com/AI4Bharat/INCLUDE) (MIT), see `isl/include50/SOURCE.md`.
Hand and pose tracking: [MediaPipe](https://ai.google.dev/edge/mediapipe).
````

- [ ] **Step 2: Put the measured results into the README**

```bash
.venv/Scripts/python -c "from pathlib import Path; r = Path('README.md'); r.write_text(r.read_text(encoding='utf-8').replace('<!-- results -->', Path('reports/results.md').read_text(encoding='utf-8').strip()), encoding='utf-8')"
grep -c "<!-- results -->" README.md
```

Expected: `0`. Then add the confusing word pairs noted in Task 14 Step 5 as a bullet under **Limitations**.

- [ ] **Step 3: Mark the spec implemented.** In the spec, change `**Status:** Approved in conversation, pending written-spec review` to `**Status:** Implemented`.

- [ ] **Step 4: Full suite**

Run: `.venv/Scripts/python -m unittest discover -s tests -t . -v`
Expected: all tests pass, none skipped.

- [ ] **Step 5: Live check, part 1 (Claude, no signing needed).** Run the app for about 60 seconds with the real webcam and a placeholder key, and count frames:

```bash
GROQ_API_KEY=placeholder timeout 90 .venv/Scripts/python -u -c "
import time, signlan, cv2
n = [0]; t0 = time.time(); orig = cv2.imshow
def imshow(*a):
    n[0] += 1
    if n[0] % 100 == 0: print(f'{n[0]} frames, {n[0] / (time.time() - t0):.1f} fps', flush=True)
    return orig(*a)
signlan.cv2.imshow = imshow
signlan.main()
"
```

Expected: at least 8 fps after startup, and no traceback (exit code 124 from `timeout` is fine).

- [ ] **Step 6: Live check, part 2 (with the user).** Ask the user to run `.venv/Scripts/python signlan.py` and sign 3–5 words they've learned from `python -m isl.examples <word>`. Record which were recognised. If most are rejected as "? maybe", ask the user to paste the console output, and report before changing any threshold.

- [ ] **Step 7: Commit, then ask about pushing**

```bash
git add README.md docs/superpowers/specs/2026-09-23-isl-word-recognition-design.md
git commit -m "Document the ISL pipeline, results and limitations" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git log --oneline -20
```

Then **ask the user** whether to push to `origin main`. Don't push without a yes.
