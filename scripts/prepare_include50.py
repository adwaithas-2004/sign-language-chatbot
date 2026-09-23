"""Download the INCLUDE-50 videos one at a time, turn each into landmarks, and delete it.

Run from the project folder:  python scripts/prepare_include50.py [--limit N]
Safe to stop and run again: finished videos are skipped.
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
DOWNLOAD_WORKERS = 8  # Zenodo gives each connection ~0.3 MB/s, but parallel downloads add up


def all_samples():
    """Every INCLUDE-50 video, test split first so each word's example video comes from it"""
    return [sample for split in ("test", "val", "train") for sample in include_data.load_split(split)]


def landmarks_path(sample):
    return LANDMARKS_DIR / sample.split / sample.key / (Path(sample.path).stem + ".npz")


def pending(samples):
    return [sample for sample in samples if not landmarks_path(sample).exists()]


def temp_video_path(sample):
    return TMP_DIR / f"{sample.key}_{Path(sample.path).name}"

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


def _download(sample, index):
    if sample.path not in index:
        raise KeyError("not found in any INCLUDE zip")
    video = temp_video_path(sample)
    include_data.fetch_video(index[sample.path], video)
    return video


def _save(sample, video):
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
    for leftover in TMP_DIR.iterdir():  # from a run that was stopped
        leftover.unlink()

    # Download a few videos ahead while the current one is processed, without piling up the whole dataset
    ahead = 2 * DOWNLOAD_WORKERS
    pool = ThreadPoolExecutor(DOWNLOAD_WORKERS)
    downloads = [pool.submit(_download, sample, index) for sample in todo[:ahead]]
    failed, start = 0, time.time()
    try:
        for n, sample in enumerate(todo, 1):
            if n - 1 + ahead < len(todo):
                downloads.append(pool.submit(_download, todo[n - 1 + ahead], index))
            try:
                _save(sample, downloads[n - 1].result())
            except Exception as e:  # One bad video shouldn't stop an hour-long run
                failed += 1
                with open(FAILED_PATH, "a", encoding="utf-8") as log:
                    log.write(f"{sample.path}\t{type(e).__name__}: {e}\n")
                print(f"  FAILED {sample.path}: {e}")
            finally:
                temp_video_path(sample).unlink(missing_ok=True)
            remaining = (time.time() - start) / n * (len(todo) - n) / 60
            print(f"[{n}/{len(todo)}] {sample.split:5s} {sample.key:12s} about {remaining:.0f} min left", flush=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    print(f"Done: {len(todo) - failed} prepared, {failed} failed" + (f" (see {FAILED_PATH})" if failed else ""))


if __name__ == "__main__":
    main()
