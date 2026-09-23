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
