"""PSNR, bitrate and Bjøntegaard delta metrics (from the notebook's BD cell)."""
import numpy as np


def psnr(original, reconstructed):
    """Peak signal-to-noise ratio in dB for 8-bit images."""
    return 10 * np.log10(255 ** 2 / np.mean((original - reconstructed) ** 2))


def to_kbps(bits, n_frames, fps):
    """Total bits for n_frames frames -> kilobits per second at fps."""
    return bits / n_frames * fps / 1000


def bd_metrics(r1, p1, r2, p2):
    """Bjøntegaard (VCEG-M33): (BD-rate %, BD-PSNR dB) of curve 2 against anchor curve 1.

    r = bitrates, p = PSNRs. A cubic is fitted on a log10-rate axis and the
    gap between the two fits is averaged over the range both curves cover.
    BD-rate < 0: curve 2 needs fewer bits for the same PSNR.
    BD-PSNR > 0: curve 2 has higher quality at the same rate.
    """
    l1, l2 = np.log10(r1), np.log10(r2)

    # BD-PSNR: integrate PSNR(log R) over the common log-rate range
    a1, a2 = np.polyfit(l1, p1, 3), np.polyfit(l2, p2, 3)
    lo, hi = max(l1.min(), l2.min()), min(l1.max(), l2.max())
    i1, i2 = np.polyint(a1), np.polyint(a2)
    bd_psnr = ((np.polyval(i2, hi) - np.polyval(i2, lo))
               - (np.polyval(i1, hi) - np.polyval(i1, lo))) / (hi - lo)

    # BD-rate: integrate log R(PSNR) over the common PSNR range
    b1, b2 = np.polyfit(p1, l1, 3), np.polyfit(p2, l2, 3)
    lo, hi = max(min(p1), min(p2)), min(max(p1), max(p2))
    j1, j2 = np.polyint(b1), np.polyint(b2)
    avg = ((np.polyval(j2, hi) - np.polyval(j2, lo))
           - (np.polyval(j1, hi) - np.polyval(j1, lo))) / (hi - lo)
    return (10 ** avg - 1) * 100, bd_psnr
