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
