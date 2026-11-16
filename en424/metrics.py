"""Full-clip MSE/PSNR, actual bitstream bitrate, classical cubic BD-Rate."""
import math
import numpy as np


def measure(original, reconstructed, byte_count, fps):
    if original.shape != reconstructed.shape or original.size == 0:
        raise ValueError('Nonempty matching frame arrays are required')
    mse = float(np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2))
    return dict(mse=mse, psnr_db=math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse),
                bytes=byte_count, bitrate_bps=byte_count * 8 * fps / len(original),
                bits_per_pixel=byte_count * 8 / np.prod(original.shape[:3]),
                raw_to_coded_ratio=original.nbytes / byte_count)


def bd_rate(anchor, test):
    """Classical cubic fit to ln(rate) versus PSNR over shared quality range.

    Inputs are (bitrate, PSNR) pairs. Requires >=4 distinct finite quality
    points and monotonic rates. Negative percent means test saves bits.
    Cubic interpolation may overshoot: inspect RD points before reporting it.
    """
    def curve(points):
        arr = np.array(points, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 4:
            raise ValueError('BD-Rate requires at least four points per curve')
        if not np.isfinite(arr).all() or np.any(arr[:, 0] <= 0):
            raise ValueError('BD-Rate needs finite PSNR and positive finite rates')
        arr = arr[np.argsort(arr[:, 1])]
        if np.any(np.diff(arr[:, 1]) <= 0) or np.any(np.diff(arr[:, 0]) <= 0):
            raise ValueError('RD points must have distinct PSNR and strictly increasing rate')
        return arr
    a, t = curve(anchor), curve(test)
    lo, hi = max(a[0, 1], t[0, 1]), min(a[-1, 1], t[-1, 1])
    if hi <= lo:
        raise ValueError('No shared PSNR interval for BD-Rate')
    def average(arr):
        # Polynomial.fit scales the domain for better numerical conditioning.
        p = np.polynomial.Polynomial.fit(arr[:, 1], np.log(arr[:, 0]), 3).integ()
        return float((p(hi) - p(lo)) / (hi - lo))
    return 100 * math.expm1(average(t) - average(a))
