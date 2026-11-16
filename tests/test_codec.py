import math
import unittest
import numpy as np
from en424 import Config, encode, decode
from en424 import entropy
from en424.codec import HEADER, PACKET
from en424.transform import dct, idct, ZIGZAG, Reader, write_block, read_block
from en424.motion import estimate
from en424.metrics import measure, bd_rate


class TransformTests(unittest.TestCase):
    def test_transform_inverse_and_dc(self):
        block = np.random.default_rng(1).normal(size=(8, 8))
        np.testing.assert_allclose(idct(dct(block)), block, atol=1e-12)
        coeff = dct(np.full((8, 8), 10))
        self.assertAlmostEqual(coeff[0, 0], 80)
        coeff[0, 0] = 0
        np.testing.assert_allclose(coeff, 0, atol=1e-12)

    def test_zigzag_rle_edges(self):
        self.assertEqual(ZIGZAG[:6], [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2)])
        self.assertEqual(len(set(ZIGZAG)), 64)
        for block in [np.zeros((8, 8), dtype=int), np.arange(-32, 32).reshape(8, 8),
                      np.pad(np.array([[-2040]]), ((7, 0), (7, 0)))]:
            raw = bytearray()
            write_block(raw, block)
            reader = Reader(raw)
            np.testing.assert_array_equal(block, read_block(reader))
            self.assertEqual(reader.pos, len(raw))


class EntropyTests(unittest.TestCase):
    def test_round_trip_and_rescaling(self):
        for raw in [b'', b'\0', bytes(range(256)), b'\xff' * 20000,
                    np.random.default_rng(42).integers(0, 256, 20000, dtype=np.uint8).tobytes()]:
            self.assertEqual(entropy.decompress(entropy.compress(raw), len(raw)), raw)
        self.assertLess(len(entropy.compress(b'\0' * 1000)), 100)


class MotionTests(unittest.TestCase):
    def test_known_translation_and_tie(self):
        reference = np.random.default_rng(3).integers(0, 256, (32, 32), dtype=np.uint8)
        current = reference.copy()
        current[8:16, 8:16] = reference[10:18, 5:13]
        self.assertEqual(estimate(current, reference, 8, 8, 8, 4), (2, -3))
        flat = np.zeros((32, 32))
        self.assertEqual(estimate(flat, flat, 0, 0, 16, 4), (0, 0))


class CodecTests(unittest.TestCase):
    def test_closed_loop_rgb_gray_padding_gop(self):
        rng = np.random.default_rng(424)
        for shape in [(7, 19, 23), (5, 17, 25, 3), (2, 1, 1)]:
            frames = rng.integers(0, 256, shape, dtype=np.uint8)
            for intra in [False, True]:
                data, recon, stats = encode(frames, Config(qstep=23, search_radius=2, gop=3, intra_only=intra))
                decoded, meta = decode(data)
                np.testing.assert_array_equal(decoded, recon)
                self.assertEqual(decoded.shape, frames.shape)
                self.assertEqual(meta['frames'], len(frames))
                self.assertEqual(stats[0]['type'], 'I')
                self.assertEqual(len(data), HEADER.size + sum(row['bytes'] for row in stats))

    def test_block_sizes_and_skip_off(self):
        frames = np.full((3, 17, 19), 124, dtype=np.uint8)
        for size in [8, 16, 32]:
            data, recon, stats = encode(frames, Config(block_size=size, skip=False))
            np.testing.assert_array_equal(decode(data)[0], recon)
            self.assertEqual(sum(s['skip'] for s in stats), 0)

    def test_static_sequence_saves_bits_and_skips(self):
        frame = np.random.default_rng(8).integers(0, 256, (32, 32), dtype=np.uint8)
        frames = np.repeat(frame[None], 8, axis=0)
        intra, _, _ = encode(frames, Config(intra_only=True))
        hybrid, recon, stats = encode(frames)
        self.assertLess(len(hybrid), len(intra) / 2)
        np.testing.assert_array_equal(decode(hybrid)[0], recon)
        # Lossy rounding can leave nonzero residuals even for repeated originals.
        # A constant exactly represented by the quantizer exercises skip itself.
        _, _, skip_stats = encode(np.full((3, 32, 32), 128, dtype=np.uint8))
        self.assertEqual(sum(s['skip'] for s in skip_stats), 8)

    def test_corruption_truncation_trailing_and_limits(self):
        data, _, _ = encode(np.full((2, 16, 16), 100, dtype=np.uint8))
        bad = bytearray(data)
        bad[HEADER.size + PACKET.size] ^= 1
        for invalid in [b'', data[:10], data[:-1], bytes(bad), data + b'x', b'BAD!' + data[4:]]:
            with self.assertRaises(ValueError):
                decode(invalid)
        with self.assertRaises(ValueError):
            decode(data, max_samples=1)

    def test_invalid_input(self):
        for config in [Config(qstep=0), Config(qstep=1.5), Config(fps=0), Config(gop=0),
                       Config(search_radius=-1), Config(block_size=7)]:
            with self.assertRaises(ValueError):
                encode(np.zeros((1, 8, 8), dtype=np.uint8), config)
        with self.assertRaises(ValueError):
            encode(np.zeros((1, 8, 8)))


class MetricTests(unittest.TestCase):
    def test_metrics(self):
        a = np.zeros((2, 8, 8), dtype=np.uint8)
        self.assertTrue(math.isinf(measure(a, a, 100, 30)['psnr_db']))
        values = measure(a, a + 10, 100, 30)
        self.assertEqual(values['mse'], 100)
        self.assertEqual(values['bitrate_bps'], 12000)

    def test_bd_rate_known_offset_and_invalid(self):
        points = [(100, 30), (200, 35), (400, 40), (800, 45)]
        self.assertAlmostEqual(bd_rate(points, points), 0, places=8)
        self.assertAlmostEqual(bd_rate(points, [(r * .8, p) for r, p in points]), -20, places=8)
        with self.assertRaises(ValueError):
            bd_rate(points, [(r, p + 20) for r, p in points])
        with self.assertRaises(ValueError):
            bd_rate(points[:3], points)


if __name__ == '__main__':
    unittest.main()
