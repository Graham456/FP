"""
me_utils.py  —  ENSC 424 Lab 6 Helper Module
==============================================
Python equivalents of the MATLAB functions that Lab 6 relies on:

    uint8  →  uint8(x)
    mean   →  mean(x)
    log10  →  log10(x)

This file is provided to students; you do NOT need to modify it.
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# Conversion to 8-bit integers
# ---------------------------------------------------------------------------

def uint8(x):
    """
    Convert to 8-bit unsigned integers.

    Equivalent to MATLAB's  uint8(x): values are rounded to the nearest
    integer, with halves rounded *away from zero* (so 100.5 becomes 101), and
    values outside [0, 255] are set to 0 or 255 (NaN becomes 0).  (numpy's  x.astype(np.uint8)
    truncates instead, and np.round rounds halves to the nearest even number.)
    """
    v = np.asarray(x, dtype=np.float64)
    v = np.copysign(np.floor(np.abs(v) + 0.5), v)
    v = np.where(np.isnan(v), 0, v)             # MATLAB: uint8(NaN) is 0
    return np.clip(v, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Mean, added in MATLAB's order
# ---------------------------------------------------------------------------

def mean(x):
    """
    Mean of all the elements of a vector or a 2-D array, as a number.

    For a vector (a 1-D array, or a 2-D array with one row or one column),
    this is MATLAB's  mean(x).  For any other 2-D array it is MATLAB's
    mean(mean(x)): the mean of each column first, then the mean of those.
    Either way the result agrees with MATLAB to the last bit.

    Floating-point addition is not associative, so the last digit of a long
    sum depends on the order in which its terms are added.  np.mean gives the
    same value to about 15 significant digits; this function adds in
    MATLAB's order.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 0:
        return np.float64(a)
    if a.ndim == 1 or 1 in a.shape:
        v = a.ravel()
        return np.float64(_matlab_sum(v) / v.size)
    if a.ndim != 2:
        raise ValueError("mean: x must be a vector or a 2-D array")
    return mean(_column_sums(a) / a.shape[0])


# MATLAB's summation order, measured against MATLAB R2026b in MATLAB Online:
#   * a vector of up to 43 elements is added left to right;
#   * from 44 elements to somewhere between 88,500 and 89,000, in four
#     interleaved partial sums (element k goes to accumulator k mod 4) that are
#     combined as ((s0 + s1) + s2) + s3;
#   * longer vectors are cut into blocks of 4,096 elements, each block is
#     summed as above, each of the 8 threads adds up its run of consecutive
#     blocks, and the 8 thread totals are added left to right (pinned down for
#     262,144 elements in Lab 4; other long lengths share the blocks out among
#     the threads as evenly as possible);
#   * each column of a matrix is added up by the same rules as a vector.

def _lanes4(v):
    n = v.size
    m = n - n % 4
    acc = np.add.reduce(v[:m].reshape(-1, 4), axis=0).copy() if m else np.zeros(4)
    acc[:n - m] += v[m:]
    s0, s1, s2, s3 = acc.tolist()
    return ((s0 + s1) + s2) + s3


def _matlab_sum(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    n = v.size
    if n <= 43:
        s = 0.0
        for a in v.tolist():
            s += a
        return s
    if n <= 88500:
        return _lanes4(v)
    blocks = [_lanes4(v[i:i + 4096]) for i in range(0, n, 4096)]
    per, extra = divmod(len(blocks), 8)
    total, start = 0.0, 0
    for t in range(8):
        count = per + (1 if t < extra else 0)
        if count == 0:
            continue
        part = blocks[start]
        for b in blocks[start + 1:start + count]:
            part += b
        start += count
        total = part if t == 0 else total + part
    return total


def _column_sums(a):
    return np.array([_matlab_sum(a[:, j]) for j in range(a.shape[1])])


# ---------------------------------------------------------------------------
# Base-10 logarithm
# ---------------------------------------------------------------------------

def log10(x):
    """
    Base-10 logarithm, element by element.

    Equivalent to MATLAB's  log10(x), to the last bit: log10(0) is -Inf.
    (np.log10 gives the same values to about 15 significant digits, but its
    last bit differs from MATLAB's for a few percent of all inputs.)
    """
    def one(t):
        t = float(t)
        if t == 0:
            return -math.inf
        if t < 0:
            raise ValueError("log10: negative input (MATLAB would return a complex number)")
        if math.isnan(t) or math.isinf(t):
            return t
        return math.log10(t)
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 0:
        return np.float64(one(a))
    return np.array([one(t) for t in a.ravel().tolist()]).reshape(a.shape)
