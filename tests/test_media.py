import tempfile
import unittest
from pathlib import Path
import numpy as np
from en424.media import load, save


class MediaTests(unittest.TestCase):
    def test_npy_png_round_trip_and_limit(self):
        frames = np.random.default_rng(4).integers(0, 256, (3, 9, 11, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ['frames.npy', 'png_frames']:
                target = Path(tmp) / name
                save(target, frames)
                actual, fps = load(target)
                np.testing.assert_array_equal(actual, frames)
                self.assertIsNone(fps)
                limited, _ = load(target, limit=1)
                np.testing.assert_array_equal(limited, frames[:1])
                gray, _ = load(target, gray=True)
                self.assertEqual(gray.shape, frames.shape[:3])

    def test_grayscale_npy_and_png(self):
        frames = np.arange(128, dtype=np.uint8).reshape(2, 8, 8)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ['frames.npy', 'png_frames']:
                target = Path(tmp) / name
                save(target, frames)
                actual, _ = load(target, gray=True)
                np.testing.assert_array_equal(actual, frames)
