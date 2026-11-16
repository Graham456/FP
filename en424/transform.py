"""Orthonormal 8x8 DCT-II, scalar quantization and zigzag/RLE coding."""
import numpy as np

N = 8
C = np.cos(np.pi * np.arange(N)[:, None] * (2 * np.arange(N) + 1) / (2 * N))
C *= np.sqrt(2 / N)
C[0] /= np.sqrt(2)
ZIGZAG = [(y, s - y) for s in range(15)
          for y in (range(7, -1, -1) if s % 2 == 0 else range(8))
          if 0 <= s - y < 8]


def dct(block):
    return C @ np.asarray(block, dtype=np.float64) @ C.T


def idct(coefficients):
    return C.T @ np.asarray(coefficients, dtype=np.float64) @ C


def quantize(block, step):
    return np.rint(dct(block) / step).astype(np.int32)


def reconstruct(coefficients, step):
    return idct(coefficients * step)


def put_uint(out, value):
    value = int(value)
    if value < 0:
        raise ValueError("Unsigned integer cannot be negative")
    while value >= 128:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)


def put_signed(out, value):
    value = int(value)
    put_uint(out, 2 * value if value >= 0 else -2 * value - 1)


class Reader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def uint(self):
        value = 0
        for shift in range(0, 35, 7):
            if self.pos >= len(self.data):
                raise ValueError("Truncated integer")
            byte = self.data[self.pos]
            self.pos += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
        raise ValueError("Integer too large")

    def signed(self):
        value = self.uint()
        return value // 2 if value % 2 == 0 else -(value // 2) - 1


def write_block(out, coefficients):
    """Count, then (preceding zero run, signed level); trailing zeros implicit."""
    values = [int(coefficients[y, x]) for y, x in ZIGZAG]
    put_uint(out, sum(v != 0 for v in values))
    run = 0
    for value in values:
        if value == 0:
            run += 1
        else:
            put_uint(out, run)
            put_signed(out, value)
            run = 0


def read_block(reader):
    coefficients = np.zeros((8, 8), dtype=np.int32)
    count, pos = reader.uint(), -1
    if count > 64:
        raise ValueError("Invalid nonzero coefficient count")
    for _ in range(count):
        pos += reader.uint() + 1
        level = reader.signed()
        if pos >= 64 or level == 0 or abs(level) > 65535:
            raise ValueError("Invalid run/level")
        coefficients[ZIGZAG[pos]] = level
    return coefficients
