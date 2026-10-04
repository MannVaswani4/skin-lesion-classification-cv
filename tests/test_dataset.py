"""
Dataset loading & split integrity unit tests.
Verifies class mapping, directory counts, image shapes, and channel formats.
"""
import os
import glob
import unittest
from PIL import Image

DATA_ROOT = "data/ISIC_2018_128X128_6705"
EXPECTED_CLASSES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]

EXPECTED_COUNTS = {
    "train": {cls: 4693 for cls in EXPECTED_CLASSES},
    "validation": {cls: 671 for cls in EXPECTED_CLASSES},
    "test": {cls: 1341 for cls in EXPECTED_CLASSES}
}

class TestDatasetIntegrity(unittest.TestCase):

    def test_directory_structure_and_counts(self):
        """Verify that all splits and class subdirectories exist with exact file counts."""
        self.assertTrue(os.path.exists(DATA_ROOT), f"Dataset root {DATA_ROOT} does not exist.")
        
        for split, class_dict in EXPECTED_COUNTS.items():
            split_dir = os.path.join(DATA_ROOT, split)
            self.assertTrue(os.path.exists(split_dir), f"Split directory {split_dir} missing.")
            
            for cls_name, expected_count in class_dict.items():
                cls_dir = os.path.join(split_dir, cls_name)
                self.assertTrue(os.path.exists(cls_dir), f"Class directory {cls_dir} missing.")
                files = glob.glob(os.path.join(cls_dir, "*.jpg"))
                self.assertEqual(
                    len(files), expected_count,
                    f"Mismatch in {split}/{cls_name}: expected {expected_count}, found {len(files)}"
                )

    def test_sample_image_properties(self):
        """Inspect sample images to ensure 128x128 resolution, RGB color mode, and JPEG format."""
        sample_file = os.path.join(DATA_ROOT, "train", "AKIEC", "ISIC_0024329.jpg")
        self.assertTrue(os.path.exists(sample_file), f"Sample file {sample_file} not found.")
        
        with Image.open(sample_file) as img:
            self.assertEqual(img.size, (128, 128), f"Expected image size (128, 128), got {img.size}")
            self.assertEqual(img.mode, "RGB", f"Expected RGB mode, got {img.mode}")
            self.assertEqual(img.format, "JPEG", f"Expected JPEG format, got {img.format}")

if __name__ == "__main__":
    unittest.main()
