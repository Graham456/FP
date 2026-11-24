import tempfile
import unittest
from pathlib import Path

import numpy as np

from en424.codec import Config, decode, encode, quality_step
from en424.motion import half_prediction
from en424.media import load, save
from en424.transform import Reader, read_block, write_block


class LuminanceTests(unittest.TestCase):
    def test_half_pixel_prediction_and_quality(self):
        ramp = np.array([[0, 2], [4, 6]], dtype=np.uint8)
        self.assertEqual(half_prediction(ramp, 1, 1, -1, -1, 1).item(), 3)
        self.assertEqual(half_prediction(ramp, 0, 0, 0, 1, 1).item(), 1)
        self.assertEqual([quality_step(q) for q in (10, 30, 50, 70, 90)], [65, 42, 29, 21, 10])
        with self.assertRaises(ValueError):
            quality_step(101)

    def test_raw_round_trip_multiple_sizes(self):
        rng = np.random.default_rng(424)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'clip.y'
            for height, width in [(1, 1), (57, 93), (240, 320)]:
                frames = rng.integers(0, 256, (3, height, width), dtype=np.uint8)
                save(path, frames)
                source, fps = load(path, limit=2, width=width, height=height)
                self.assertIsNone(fps)
                np.testing.assert_array_equal(source, frames[:2])
                data, local, stats = encode(source, Config(gop=2, compact=True))
                decoded, meta = decode(data)
                self.assertEqual((meta['width'], meta['height']), (width, height))
                self.assertEqual([row['type'] for row in stats], ['I', 'P'])
                np.testing.assert_array_equal(decoded, local)
                save(path, decoded)
                self.assertEqual(path.read_bytes(), decoded.tobytes())

    def test_raw_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.y'
            path.write_bytes(bytes(9))
            for dimensions in ({}, {'width': 3}, {'width': 2, 'height': 3}):
                with self.assertRaises(ValueError):
                    load(path, **dimensions)

    def test_compact_coefficients_and_escapes(self):
        coeff = np.arange(-32,32,dtype=np.int32).reshape(8,8)
        coeff[1:4] = 0
        for compact in (False, True):
            raw = bytearray()
            write_block(raw, coeff, compact)
            reader = Reader(raw)
            np.testing.assert_array_equal(read_block(reader, compact), coeff)
            self.assertEqual(reader.pos, len(raw))

