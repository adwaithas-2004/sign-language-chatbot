import io
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest import mock

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

    def test_split_paths_find_videos_stored_under_another_extension(self):
        # The split lists call every video .MOV, but some are .MP4 inside the Zenodo zips
        zip_bytes = make_zip({"Seasons/61. Summer/MVI_9122.MP4": b"a", "Seasons/61. Summer/MVI_4565.MOV": b"b"})
        with tempfile.TemporaryDirectory() as tmp:
            index = include_data.build_index(["Seasons/61. Summer/MVI_9122.MOV", "Seasons/61. Summer/MVI_4565.MOV"],
                                             Path(tmp) / "index.json", lambda name: BytesSource(zip_bytes),
                                             progress=lambda _: None)
        self.assertEqual(set(index), {"Seasons/61. Summer/MVI_9122.MOV", "Seasons/61. Summer/MVI_4565.MOV"})

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
