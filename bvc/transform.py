"""8x8 DCT, rounding and JPEG quantization table (trimmed from dct_utils.py).

Kept: init_jpeg (intra anchor's quantization table and zig-zag scan),
dct2 / idct2 (bit-exact MATLAB 8x8 DCT), matlab_round (MATLAB round).
The image I/O, im2col, save/load and MATLAB-sum helpers were unused and removed.
"""
import numpy as np


# ---------------------------------------------------------------------------
# JPEG-like quantization table and zig-zag scan (used by the intra anchor)
# ---------------------------------------------------------------------------

def init_jpeg(quality):
    """
    Quantization step sizes and zig-zag scan for JPEG-like image coding.

    Equivalent to the function init_jpeg.m supplied with the 2011 lab:
    [qt, zag] = init_jpeg(quality).

    Parameters
    ----------
    quality : number
        The quality of the quantization, 1 to 100 (values outside this range
        are clipped to it).  The higher the quality, the smaller the step
        sizes.  With quality = 50 the table is the standard JPEG luminance
        table.

    Returns
    -------
    qt : numpy.ndarray of float64, shape (64,)
        The quantization step sizes of the 64 DCT coefficients of an 8x8
        block, in zig-zag order: qt[0] is the step size of the DC
        coefficient, qt[1] that of the next coefficient in the zig-zag scan,
        and so on.
    zag : numpy.ndarray of int, shape (8, 8)
        The zig-zag scan.  zag[i, j] is the position (counted from 1, as in
        MATLAB) of coefficient (i, j) of the block in the zig-zag scan, so
        zag[0, 0] = 1 (the DC coefficient comes first), zag[0, 1] = 2,
        zag[1, 0] = 3, and so on up to zag[7, 7] = 64.
    """
    qt = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 58, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 59, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float64)
    zag = np.array([[0, 1, 5, 6, 14, 15, 27, 28],
                    [2, 4, 7, 13, 16, 26, 29, 42],
                    [3, 8, 12, 17, 25, 30, 41, 43],
                    [9, 11, 18, 24, 31, 40, 44, 53],
                    [10, 19, 23, 32, 39, 45, 52, 54],
                    [20, 22, 33, 38, 46, 51, 55, 60],
                    [21, 34, 37, 47, 50, 56, 59, 61],
                    [35, 36, 48, 49, 57, 58, 62, 63]]) + 1

    # zig-zag scan of qt
    x = np.zeros(64)
    for i in range(8):
        for j in range(8):
            x[zag[i, j] - 1] = qt[i, j]
    qt = x

    quality = float(quality)
    if quality <= 0:
        quality = 1.0
    elif quality > 100:
        quality = 100.0
    if quality < 50:
        quality = 5000 / quality
    else:
        quality = 200 - 2 * quality

    # Scale the table according to quality (quality = 50 leaves it unchanged).
    for i in range(64):
        temp = (qt[i] * quality + 50) / 100
        if temp <= 0:
            temp = 1.0
        elif temp > 255:
            temp = 255.0
        qt[i] = np.floor(temp)
    return qt, zag




# ---------------------------------------------------------------------------
# 2-D DCT of 8x8 blocks
# ---------------------------------------------------------------------------
# MATLAB's dct2 transforms the columns of the block, then its rows, with a
# one-dimensional DCT that is computed through an 8-point FFT (FFTW).  The
# functions below carry out exactly the same floating-point operations, in
# the same order, so that the results agree with MATLAB to the last bit.
# (scipy.fft.dctn(blk, norm='ortho') computes the same transform, and agrees
# with MATLAB to about 15 significant digits.)

_K = 0.707106781186547524400844362104849039284835938      # 1/sqrt(2)
_n = 8
_W_DCT = 2 * np.exp(-1j * (np.arange(_n) * np.pi / (2 * _n))) / np.sqrt(2 * _n)
_W_DCT[0] = _W_DCT[0] / np.sqrt(2)
_W_IDCT = np.sqrt(2 * _n) * np.exp(1j * (np.pi * np.arange(_n) / (2 * _n)))
_W_IDCT[0] = _W_IDCT[0] / np.sqrt(2)


def _fft8_real(x):
    """8-point DFT of real data along axis 0 (FFTW's r2cf_8), returned as
    separate real and imaginary parts."""
    x0, x1, x2, x3, x4, x5, x6, x7 = x
    Tj = x0 + x4; T3 = x0 - x4
    Tk = x2 + x6; Te = x2 - x6
    To = x1 + x5; T6 = x1 - x5
    Tn = x7 + x3; T9 = x7 - x3
    Ta = _K * (T6 + T9); Tf = _K * (T6 - T9)
    Tm = Tj + Tk; Tp = To + Tn
    zero = np.zeros_like(Tm)
    re = [Tm + Tp, T3 + Ta, Tj - Tk, T3 - Ta, Tm - Tp, None, None, None]
    im = [zero, -(Te + Tf), Tn - To, Te - Tf, zero, None, None, None]
    for k in (5, 6, 7):
        re[k] = re[8 - k]
        im[k] = -im[8 - k]
    return np.stack(re), np.stack(im)


def _ifft8_complex(zr, zi):
    """8-point inverse DFT, without the 1/8 factor, of complex data along
    axis 0 (FFTW's n1bv_8), given and returned as real and imaginary parts."""
    def add(a, b): return a[0] + b[0], a[1] + b[1]
    def sub(a, b): return a[0] - b[0], a[1] - b[1]
    def kmul(a): return _K * a[0], _K * a[1]
    def byi(a): return -a[1], a[0]
    x = [(zr[k], zi[k]) for k in range(8)]
    Tj = add(x[0], x[4]); T3 = sub(x[0], x[4])
    Tk = add(x[2], x[6]); Te = sub(x[2], x[6])
    To = add(x[1], x[5]); T6 = sub(x[1], x[5])
    Tn = add(x[7], x[3]); T9 = sub(x[7], x[3])
    Ta = add(T6, T9); Tf = sub(T6, T9)
    X = [None] * 8
    Tm = add(Tj, Tk); Tp = add(Tn, To)
    X[0] = add(Tm, Tp); X[4] = sub(Tm, Tp)
    Tl = sub(Tj, Tk); Tq = byi(sub(To, Tn))
    X[2] = add(Tl, Tq); X[6] = sub(Tl, Tq)
    Tb = add(T3, kmul(Ta)); Tg = byi(add(Te, kmul(Tf)))
    X[1] = add(Tb, Tg); X[7] = sub(Tb, Tg)
    Th = sub(T3, kmul(Ta)); Ti = byi(sub(Te, kmul(Tf)))
    X[5] = add(Th, Ti); X[3] = sub(Th, Ti)
    return np.stack([v[0] for v in X]), np.stack([v[1] for v in X])


def _dct8(a, axis):
    """MATLAB's one-dimensional DCT (as used by dct2) along one axis of length 8."""
    a = np.moveaxis(np.asarray(a, dtype=np.float64), axis, 0)
    y = np.concatenate([a[0:_n:2], a[_n - 1:0:-2]])            # even, then odd in reverse
    yr, yi = _fft8_real(y)
    shape = (_n,) + (1,) * (a.ndim - 1)
    wr, wi = _W_DCT.real.reshape(shape), _W_DCT.imag.reshape(shape)
    b = wr * yr - wi * yi                                       # real part of W .* fft(y)
    return np.moveaxis(b, 0, axis)


def _idct8(b, axis):
    """MATLAB's one-dimensional inverse DCT (as used by idct2) along one axis of length 8."""
    b = np.moveaxis(np.asarray(b, dtype=np.float64), axis, 0)
    shape = (_n,) + (1,) * (b.ndim - 1)
    wr, wi = _W_IDCT.real.reshape(shape), _W_IDCT.imag.reshape(shape)
    yr, _ = _ifft8_complex(wr * b, wi * b)
    yr = yr / _n
    a = np.empty_like(b)
    a[0:_n:2] = yr[0:_n // 2]
    a[1:_n:2] = yr[_n - 1:_n // 2 - 1:-1]
    return np.moveaxis(a, 0, axis)


def dct2(blk):
    """
    Two-dimensional DCT of an 8x8 block.

    Equivalent to MATLAB's  dct2(blk)  (and  dct2(blk, 8, 8)), to the last
    bit: the orthonormal DCT-II of the columns of the block, then of its rows.

    Parameters
    ----------
    blk : array-like, shape (8, 8), or (..., 8, 8) for a stack of blocks
        The block.  A stack of blocks (for example an array of shape
        (K, 8, 8)) is also accepted: each 8x8 block in it is transformed
        separately, exactly as dct2 would transform it on its own.

    Returns
    -------
    numpy.ndarray of float64, the same shape as blk
        The DCT coefficients.  Element [0, 0] is the DC coefficient;
        frequencies increase down the rows and along the columns.
    """
    blk = np.asarray(blk, dtype=np.float64)
    if blk.ndim < 2 or blk.shape[-2:] != (8, 8):
        raise ValueError(f"dct2: expected an 8x8 block (or a stack of them), got shape {blk.shape}")
    return _dct8(_dct8(blk, -2), -1)


def idct2(blk):
    """
    Two-dimensional inverse DCT of an 8x8 block.

    Equivalent to MATLAB's  idct2(blk)  (and  idct2(blk, 8, 8)), to the
    last bit: the inverse of dct2.

    Parameters
    ----------
    blk : array-like, shape (8, 8), or (..., 8, 8) for a stack of blocks
        The DCT coefficients of the block.  A stack of blocks is also
        accepted, as for dct2.

    Returns
    -------
    numpy.ndarray of float64, the same shape as blk
    """
    blk = np.asarray(blk, dtype=np.float64)
    if blk.ndim < 2 or blk.shape[-2:] != (8, 8):
        raise ValueError(f"idct2: expected an 8x8 block (or a stack of them), got shape {blk.shape}")
    return _idct8(_idct8(blk, -2), -1)


def matlab_round(x):
    """
    Round to the nearest integer, with halves rounded away from zero.

    Equivalent to MATLAB's  round(x)  (so 2.5 becomes 3 and -2.5 becomes -3).
    np.round(x) rounds halves to the nearest even number instead (2.5
    becomes 2).  The result is floating point, as in MATLAB.
    """
    v = np.asarray(x, dtype=np.float64)
    r = _round_half_away(v)
    return r if r.ndim else float(r)


def _round_half_away(v):
    t = np.trunc(v)
    with np.errstate(invalid='ignore'):
        return np.where(np.abs(v - t) >= 0.5, t + np.sign(v), t)

