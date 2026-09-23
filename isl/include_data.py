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
