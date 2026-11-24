"""Raw .y (luma-only) video file I/O (from motion.py + me_utils.py)."""
import os

import numpy as np


def count_frames(infile, height, width):
    """Number of whole height x width 8-bit frames in a raw .y file."""
    return os.path.getsize(infile) // (height * width)


def read_Yframe(infile, height, width, frame_index):
    """Read frame number frame_index (counted from 1) as a float64 height x width array."""
    with open(infile, 'rb') as f:
        f.seek(height * width * (frame_index - 1))
        raw = np.frombuffer(f.read(height * width), dtype=np.uint8)
    return raw.astype(np.float64).reshape(height, width)


def write_Yframe(frameY, outfile):
    """Append one frame to a raw .y file (so delete the file before the first frame)."""
    with open(outfile, 'ab') as f:
        f.write(uint8(frameY.ravel()))


def uint8(x):
    """MATLAB uint8(x): round halves away from zero, clip to 0..255, NaN -> 0.

    (x.astype(np.uint8) would truncate instead, and np.round rounds halves to even.)
    """
    v = np.asarray(x, dtype=np.float64)
    v = np.copysign(np.floor(np.abs(v) + 0.5), v)
    v = np.where(np.isnan(v), 0, v)
    return np.clip(v, 0, 255).astype(np.uint8)
