"""Hybrid I/P encoder and independent decoder with a self-contained bitstream."""
from dataclasses import dataclass
import math
import struct
import zlib
import numpy as np
from . import entropy
from .transform import quantize, reconstruct, write_block, read_block, Reader, put_signed
from .motion import estimate, compensate

# Magic/version, width, height, channels, frames, fps, qstep, MB size, radius, GOP.
HEADER = struct.Struct('<4sBIIBIdHHHH')
PACKET = struct.Struct('<BIIII')  # frame type, raw bytes, coded bytes, raw CRC, coded CRC


@dataclass(frozen=True)
class Config:
    qstep: int = 16
    block_size: int = 16
    search_radius: int = 4
    gop: int = 12
    fps: float = 30.0
    intra_only: bool = False
    skip: bool = True

    def validate(self):
        for name in ('qstep', 'block_size', 'search_radius', 'gop'):
            if type(getattr(self, name)) is not int:
                raise ValueError(name + ' must be an integer')
        if not 1 <= self.qstep <= 255:
            raise ValueError('qstep must be 1..255')
        if self.block_size not in (8, 16, 32):
            raise ValueError('block_size must be 8, 16, or 32')
        if not 0 <= self.search_radius <= 64 or not 1 <= self.gop <= 65535:
            raise ValueError('Invalid search radius or GOP')
        if not math.isfinite(self.fps) or not 0 < self.fps <= 1000:
            raise ValueError('fps must be positive and at most 1000')


def normalize_frames(frames):
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        raise ValueError('Frames must be uint8 (0..255)')
    if frames.ndim == 3:
        frames = frames[..., None]
    if frames.ndim != 4 or frames.shape[-1] not in (1, 3) or min(frames.shape) < 1:
        raise ValueError('Expected (T,H,W) grayscale or (T,H,W,1|3) RGB')
    return frames


def luma(frame):
    if frame.shape[-1] == 1:
        return frame[..., 0].astype(np.float64)
    return frame @ np.array([0.299, 0.587, 0.114])


def encode_residual(residual, qstep):
    size, _, channels = residual.shape
    out = bytearray()
    restored = np.empty_like(residual)
    all_zero = True
    for c in range(channels):
        for y in range(0, size, 8):
            for x in range(0, size, 8):
                coeff = quantize(residual[y:y + 8, x:x + 8, c], qstep)
                all_zero = all_zero and not np.any(coeff)
                write_block(out, coeff)
                restored[y:y + 8, x:x + 8, c] = reconstruct(coeff, qstep)
    return out, restored, all_zero


def pixels(values):
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def encode(frames, config=Config()):
    """Return (bitstream bytes, encoder reconstruction, per-frame statistics).

    Color is coded as three full-resolution RGB planes. Shared MVs are chosen
    using luma. This intentionally avoids chroma subsampling/color-conversion
    losses while establishing the core codec. Memory use scales with clip size.
    """
    config.validate()
    frames = normalize_frames(frames)
    count, height, width, channels = frames.shape
    b = config.block_size
    ph, pw = (-height) % b, (-width) % b
    stream = bytearray(HEADER.pack(b'E424', 1, width, height, channels, count,
                                  config.fps, config.qstep, b, config.search_radius, config.gop))
    reference = None
    reconstructions, stats = [], []
    for index, frame in enumerate(frames):
        source = np.pad(frame, ((0, ph), (0, pw), (0, 0)), mode='edge')
        reconstructed = np.empty_like(source)
        is_i = config.intra_only or index % config.gop == 0
        current_y = luma(source)
        reference_y = None if is_i else luma(reference)
        raw = bytearray()
        modes = [0, 0, 0]
        for y in range(0, source.shape[0], b):
            for x in range(0, source.shape[1], b):
                block = source[y:y + b, x:x + b].astype(np.float64)
                intra_data, intra_residual, _ = encode_residual(block - 128.0, config.qstep)
                candidate = bytearray([0]) + intra_data
                result = intra_residual + 128.0
                mode = 0
                if not is_i:
                    dy, dx = estimate(current_y, reference_y, y, x, b, config.search_radius)
                    prediction = compensate(reference, y, x, dy, dx, b)
                    inter_data, inter_residual, zero = encode_residual(block - prediction, config.qstep)
                    inter = bytearray([1])
                    put_signed(inter, dy)
                    put_signed(inter, dx)
                    inter.extend(inter_data)
                    inter_mode = 1
                    if config.skip and zero and dy == 0 and dx == 0:
                        inter, inter_mode = bytearray([2]), 2
                    # Transparent low-cost heuristic: serialized syntax bytes.
                    # This is not full rate-distortion optimized mode selection.
                    if len(inter) <= len(candidate):
                        candidate, result, mode = inter, prediction + inter_residual, inter_mode
                raw.extend(candidate)
                modes[mode] += 1
                reconstructed[y:y + b, x:x + b] = pixels(result)
        coded = entropy.compress(raw)
        stream.extend(PACKET.pack(0 if is_i else 1, len(raw), len(coded),
                                  zlib.crc32(raw), zlib.crc32(coded)))
        stream.extend(coded)
        reference = reconstructed  # Closed loop: NEVER use original source as reference.
        reconstructions.append(reconstructed[:height, :width])
        stats.append(dict(frame=index, type='I' if is_i else 'P',
                          bytes=PACKET.size + len(coded), intra=modes[0],
                          inter=modes[1], skip=modes[2]))
    result = np.stack(reconstructions)
    return bytes(stream), result[..., 0] if channels == 1 else result, stats


def decode(data, max_samples=100_000_000):
    """Decode only bytes; no encoder state needed. Return (uint8 frames, metadata).

    max_samples bounds total padded output samples before allocation. CRCs detect
    accidental corruption; they are not authentication for hostile bitstreams.
    """
    if len(data) < HEADER.size:
        raise ValueError('Truncated header')
    magic, version, w, h, channels, count, fps, q, b, radius, gop = HEADER.unpack_from(data)
    if magic != b'E424' or version != 1:
        raise ValueError('Unknown codec or version')
    config = Config(q, b, radius, gop, fps)
    config.validate()
    if min(w, h, count) == 0 or channels not in (1, 3):
        raise ValueError('Invalid dimensions')
    hp, wp = ((h + b - 1) // b) * b, ((w + b - 1) // b) * b
    if hp * wp * channels * count > max_samples:
        raise ValueError('Decoded sample limit exceeded; use a shorter/smaller clip')
    blocks = hp * wp // (b * b)
    # Every transform block: count + 64 * (run + up to 3 bytes of level).
    max_raw = blocks * (11 + channels * (b // 8) ** 2 * 257)
    offset = HEADER.size
    frames, reference = [], None
    for index in range(count):
        if offset + PACKET.size > len(data):
            raise ValueError('Truncated frame packet')
        kind, raw_size, coded_size, raw_crc, coded_crc = PACKET.unpack_from(data, offset)
        offset += PACKET.size
        if kind not in (0, 1) or (index == 0 and kind != 0):
            raise ValueError('Invalid frame type')
        if not blocks <= raw_size <= max_raw or coded_size < 1 or offset + coded_size > len(data):
            raise ValueError('Invalid frame payload size')
        coded = data[offset:offset + coded_size]
        offset += coded_size
        if zlib.crc32(coded) != coded_crc:
            raise ValueError('Compressed payload checksum mismatch')
        raw = entropy.decompress(coded, raw_size)
        if zlib.crc32(raw) != raw_crc:
            raise ValueError('Decoded syntax checksum mismatch')
        reader = Reader(raw)
        reconstructed = np.empty((hp, wp, channels), dtype=np.uint8)
        for y in range(0, hp, b):
            for x in range(0, wp, b):
                mode = reader.uint()
                if mode not in (0, 1, 2) or (kind == 0 and mode != 0):
                    raise ValueError('Invalid block mode')
                if mode == 0:
                    prediction = np.full((b, b, channels), 128.0)
                else:
                    dy, dx = (reader.signed(), reader.signed()) if mode == 1 else (0, 0)
                    if abs(dy) > radius or abs(dx) > radius:
                        raise ValueError('Motion vector exceeds search range')
                    prediction = compensate(reference, y, x, dy, dx, b)
                if mode != 2:
                    for c in range(channels):
                        for by in range(0, b, 8):
                            for bx in range(0, b, 8):
                                coeff = read_block(reader)
                                prediction[by:by + 8, bx:bx + 8, c] += reconstruct(coeff, q)
                reconstructed[y:y + b, x:x + b] = pixels(prediction)
        if reader.pos != len(raw):
            raise ValueError('Trailing syntax bytes')
        reference = reconstructed
        frames.append(reconstructed[:h, :w])
    if offset != len(data):
        raise ValueError('Trailing bitstream bytes')
    result = np.stack(frames)
    metadata = dict(width=w, height=h, channels=channels, frames=count, fps=fps,
                    qstep=q, block_size=b, search_radius=radius, gop=gop)
    return (result[..., 0] if channels == 1 else result), metadata
