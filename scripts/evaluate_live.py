"""Replay the prepared INCLUDE-50 videos through the live app's segmenter and the saved model.

Offline accuracy scores each whole (trimmed) clip. This checks what the app actually does with the same videos:
whether each sign starts and ends a segment, and whether that segment is recognised.

Run from the project folder after scripts/train.py:  python scripts/evaluate_live.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isl import include_data  # noqa: E402
from isl.augment import crop_view  # noqa: E402
from isl.features import HAND_SLICES, PRESENCE_SLICE  # noqa: E402
from isl.model import load_recogniser  # noqa: E402
from isl.segmenter import DROPOUT_SECONDS, END_SECONDS, Segmenter  # noqa: E402

LANDMARKS_DIR = include_data.DATA_DIR / "landmarks"
REPORTS_DIR = ROOT / "reports"
# Hands-down time added after each clip, like a signer lowering their hands; long enough for any sign to end
REST_SECONDS = DROPOUT_SECONDS + END_SECONDS + 0.5
# Closer cameras to test: how many shoulder widths below the shoulders each one sees (a desk webcam often sees
# only 0.5-1). Each also sees VIEW_TOP above the shoulders and VIEW_HALF_WIDTH to each side
CLOSER_VIEWS = (1.4, 0.9, 0.6, 0.3)
VIEW_TOP, VIEW_HALF_WIDTH = -1.5, 1.5
RARE_TRIGGER = 0.8  # words that start a sign in fewer videos than this are listed in the report


def replay(features, fps):
    """The sequences the live segmenter emits for one clip followed by a short rest with the hands out of view"""
    rest = np.array(features[-1], dtype=np.float32, copy=True)
    rest[HAND_SLICES[0].start:PRESENCE_SLICE.stop] = 0
    frames = list(features) + [rest] * round(REST_SECONDS * fps)
    segmenter, signs = Segmenter(), []
    for i, frame in enumerate(frames):
        sign = segmenter.update(frame, i / fps)
        if sign is not None:
            signs.append(sign)
    return signs


def evaluate(landmarks_dir, recogniser):
    """Per-word trigger rate over every prepared video, and live-path accuracy on the test split"""
    display = dict(zip(recogniser.keys, recogniser.display))
    triggered, total = defaultdict(int), defaultdict(int)
    test_clips = correct = no_sign = several = 0
    for path in sorted(Path(landmarks_dir).glob("*/*/*.npz")):
        split, key = path.parent.parent.name, path.parent.name
        with np.load(path) as data:
            signs = replay(data["features"], float(data["fps"]))
        total[key] += 1
        triggered[key] += bool(signs)
        if split == "test":
            test_clips += 1
            no_sign += not signs
            several += len(signs) > 1
            correct += len(signs) == 1 and recogniser.predict(signs[0])[0][0] == display[key]
    return {"test_clips": test_clips, "top1": correct / test_clips if test_clips else 0.0, "no_sign": no_sign,
            "several_signs": several, "trigger_rate": {key: triggered[key] / total[key] for key in sorted(total)}}


def view_accuracy(landmarks_dir, recogniser, bottoms=CLOSER_VIEWS):
    """Live-path top-1 on the test split for the full view and for each closer camera in bottoms"""
    display = dict(zip(recogniser.keys, recogniser.display))
    clips = []
    for path in sorted((Path(landmarks_dir) / "test").glob("*/*.npz")):
        with np.load(path) as data:
            clips.append((path.parent.name, data["features"], float(data["fps"])))
    results = []
    for bottom in (None, *bottoms):
        correct = 0
        for key, features, fps in clips:
            if bottom is not None:
                features = crop_view(features, bottom, VIEW_TOP, VIEW_HALF_WIDTH)
            signs = replay(features, fps)
            correct += len(signs) == 1 and recogniser.predict(signs[0])[0][0] == display[key]
        results.append({"view_below_shoulders": bottom, "top1": correct / len(clips) if clips else 0.0})
    return results


def live_markdown(result, recogniser):
    display = dict(zip(recogniser.keys, recogniser.display))
    rare = sorted((rate, display.get(key, key)) for key, rate in result["trigger_rate"].items() if rate < RARE_TRIGGER)
    return "\n".join([
        f"**Live path:** replaying the {result['test_clips']} test videos through the app's own segmenter and model "
        f"gives **{result['top1']:.1%}** top-1. {result['no_sign']} videos never started a sign and "
        f"{result['several_signs']} were split into several.",
        "",
        f"Words that start a sign in under {RARE_TRIGGER:.0%} of their videos (all splits): "
        + (", ".join(f"{word} ({rate:.0%})" for rate, word in rare) or "none") + ".",
        "",
    ] + ([
        f"**Closer cameras:** the same test videos, cropped to what a closer camera sees (down to the given number "
        f"of shoulder widths below the shoulders, and at least {-VIEW_TOP:g} above and {VIEW_HALF_WIDTH:g} to each "
        "side). A desk webcam often sees only 0.5-1 below the shoulders.",
        "",
        "| Camera sees below the shoulders | Live top-1 |",
        "|---|---|",
    ] + [f"| {'full view' if view['view_below_shoulders'] is None else view['view_below_shoulders']} "
         f"| {view['top1']:.1%} |" for view in result["closer_views"]] + [""] if "closer_views" in result else []))


def main():
    recogniser = load_recogniser()
    result = evaluate(LANDMARKS_DIR, recogniser)
    if not result["test_clips"]:
        raise SystemExit("No prepared landmarks found. Run: python scripts/prepare_include50.py")
    result["closer_views"] = view_accuracy(LANDMARKS_DIR, recogniser)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "live_path.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (REPORTS_DIR / "live_path.md").write_text(live_markdown(result, recogniser), encoding="utf-8")
    print(live_markdown(result, recogniser))


if __name__ == "__main__":
    main()
