"""INCLUDE-50: the official splits and labels, and fetching single videos from the Zenodo zips"""
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


def _match_key(path):
    """The split lists call every video .MOV, but some are .MP4 inside the zips: match without the extension"""
    return path.rsplit(".", 1)[0].lower()


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
        cache["members"].update({_match_key(name): asdict(member) for name, member in members.items()})
        cache["scanned"].append(zip_name)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(cache), encoding="utf-8")
    return {path: Member(**cache["members"][_match_key(path)]) for path in paths
            if _match_key(path) in cache["members"]}


def fetch_video(member, dest, open_source=_zenodo):
    extract_member(open_source(member.zip_name), member, dest)
