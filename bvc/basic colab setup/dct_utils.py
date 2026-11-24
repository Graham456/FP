"""
dct_utils.py  —  ENSC 424 Lab 5 Helper Module
===============================================
Python equivalents of the MATLAB functions that Lab 5 relies on:

    init_jpeg  →  qt, zag = init_jpeg(quality)
    dct2       →  dct2(blk)
    idct2      →  idct2(blk)
    round      →  matlab_round(x)
    imread     →  img, map = imread(filename)
    imwrite    →  imwrite(A, map, filename, 'BMP')   or   imwrite(A, filename, 'JPG', quality=q)
    uint8      →  uint8(x)
    im2col     →  im2col(A, (m, n), 'distinct')
    col2im     →  col2im(B, (m, n), (mm, nn), 'distinct')
    sum        →  matlab_sum(v)
    mean       →  mean(x)
    log10      →  log10(x)
    save       →  save(filename, var1=..., var2=...)
    load       →  load(filename)

This file is provided to students; you do NOT need to modify it.
"""

import io
import math
import os
import struct
import time
import zlib

import numpy as np
from scipy.io import loadmat


# ---------------------------------------------------------------------------
# JPEG-like quantization table and zig-zag scan
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


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def imread(filename):
    """
    Read an image file: an 8-bit indexed (palette) BMP, or a JPEG.

    Equivalent to MATLAB's  [img, map] = imread(filename)  for these files.

    Parameters
    ----------
    filename : str
        Path to the image file.  The format is taken from the file's
        contents (a BMP file starts with 'BM', a JPEG file with the bytes
        FF D8).

    Returns
    -------
    img : numpy.ndarray of uint8, shape (rows, columns)
        The pixel values.  For a BMP file these are indices into the color
        map; for a (grayscale) JPEG file they are the gray levels.  Row 0 is
        the top row of the image, as in MATLAB.
    map : numpy.ndarray of float64, shape (256, 3) for a BMP file
        The color map: one row per index, holding the red, green and blue
        intensities scaled to [0, 1], as in MATLAB.  A JPEG file has no color
        map: map is then an empty array of shape (0, 3), as MATLAB returns
        map = [].
    """
    with open(filename, 'rb') as f:
        data = f.read()
    if data[:2] == b'\xff\xd8':
        return _jpeg_read(data, filename), np.zeros((0, 3))
    if data[:2] != b'BM':
        raise ValueError(f"{filename}: not a BMP or JPEG file")
    offset, = struct.unpack_from('<I', data, 10)
    (hdr_size, width, height, planes, bpp, compression, _, _, _,
     n_colors, _) = struct.unpack_from('<IiiHHIIiiII', data, 14)
    if bpp != 8 or compression != 0:
        raise ValueError(f"{filename}: expected an uncompressed 8-bit indexed BMP")
    if n_colors == 0:
        n_colors = 256
    pal = np.frombuffer(data, dtype=np.uint8, count=4 * n_colors, offset=14 + hdr_size)
    pal = pal.reshape(n_colors, 4)
    cmap = pal[:, [2, 1, 0]].astype(np.float64) / 255.0       # stored as B, G, R, 0
    rows = abs(height)
    stride = (width + 3) // 4 * 4                              # rows are padded to 4 bytes
    pix = np.frombuffer(data, dtype=np.uint8, count=stride * rows, offset=offset)
    img = pix.reshape(rows, stride)[:, :width]
    if height > 0:                                             # bottom-up row order
        img = img[::-1]
    return img.copy(), cmap


def imwrite(A, *args, quality=75):
    """
    Write an image to a BMP or JPEG file.

    Equivalent to MATLAB's imwrite, in these two forms:

        imwrite(A, map, filename, 'BMP')            MATLAB: imwrite(A, map, filename, 'BMP')
        imwrite(A, filename, 'JPG', quality=q)      MATLAB: imwrite(A, filename, 'JPG', 'Quality', q)

    Both files are written byte for byte as MATLAB writes them.

    Parameters
    ----------
    A : array-like of uint8, shape (rows, columns)
        The pixel values: indices into map for a BMP file, gray levels for a
        JPEG file.  Use `uint8` (below) to convert an array of doubles first,
        as you would in MATLAB.
    map : array-like, shape (K, 3), K <= 256
        (BMP only) The color map, with intensities in [0, 1].
    filename : str
        Path of the file to write.
    fmt : str
        'BMP', or 'JPG' (also 'JPEG').
    quality : int
        (JPEG only) The quality, 0 to 100, as MATLAB's 'Quality' option
        (default 75).  Higher quality means better image quality and a
        larger file.
    """
    if len(args) >= 1 and isinstance(args[0], (str, os.PathLike)):
        map, filename = None, args[0]
        fmt = args[1] if len(args) > 1 else os.path.splitext(str(filename))[1].lstrip('.')
    elif len(args) >= 2:
        map, filename = args[0], args[1]
        fmt = args[2] if len(args) > 2 else os.path.splitext(str(filename))[1].lstrip('.')
    else:
        raise TypeError("imwrite: use imwrite(A, map, filename, 'BMP') or imwrite(A, filename, 'JPG', quality=q)")
    A = np.asarray(A)
    if A.dtype != np.uint8 or A.ndim != 2:
        raise TypeError("imwrite: A must be a 2-D uint8 array (see uint8)")
    fmt = str(fmt).upper()
    if fmt in ('JPG', 'JPEG'):
        if map is not None:
            raise ValueError("imwrite: a JPEG file is written without a color map")
        _jpeg_write(A, filename, quality)
        return
    if fmt != 'BMP':
        raise ValueError("imwrite: only the 'BMP' and 'JPG' formats are supported")
    if map is None:
        raise ValueError("imwrite: a BMP file needs a color map: imwrite(A, map, filename, 'BMP')")
    cmap = np.asarray(map, dtype=np.float64)
    rows, width = A.shape
    pal = np.zeros((256, 4), dtype=np.uint8)
    pal[:len(cmap), :3] = uint8(cmap[:, [2, 1, 0]] * 255)      # B, G, R, 0
    stride = (width + 3) // 4 * 4
    body = np.zeros((rows, stride), dtype=np.uint8)
    body[:, :width] = A[::-1]                                  # bottom-up row order
    offset = 14 + 40 + 1024
    image_size = stride * rows
    header = (b'BM' + struct.pack('<IHHI', offset + image_size, 0, 0, offset)
              + struct.pack('<IiiHHIIiiII', 40, width, rows, 1, 8, 0,
                            image_size, 0, 0, 256, 0))
    with open(filename, 'wb') as f:
        f.write(header + pal.tobytes() + body.tobytes())


def _jpeg_write(A, filename, quality):
    # Baseline JPEG with the standard tables scaled by quality, as MATLAB's
    # imwrite (both use the Independent JPEG Group's library: here through Pillow).
    from PIL import Image
    q = int(np.clip(np.round(quality), 0, 100))
    buf = io.BytesIO()
    Image.fromarray(A).save(buf, format='JPEG', quality=q, optimize=False, progressive=False)
    with open(filename, 'wb') as f:
        f.write(buf.getvalue())


def _jpeg_read(data, filename):
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    if im.mode != 'L':
        raise ValueError(f"{filename}: expected a grayscale JPEG file")
    return np.asarray(im, dtype=np.uint8).copy()


def uint8(x):
    """
    Convert to 8-bit unsigned integers.

    Equivalent to MATLAB's  uint8(x): values are rounded to the nearest
    integer, with halves rounded *away from zero* (so 100.5 becomes 101), and
    values outside [0, 255] are set to 0 or 255.  (numpy's  x.astype(np.uint8)
    truncates instead, and np.round rounds halves to the nearest even number.)
    """
    v = _round_half_away(np.asarray(x, dtype=np.float64))
    return np.clip(np.nan_to_num(v, nan=0.0), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Block <-> column rearrangement
# ---------------------------------------------------------------------------

def im2col(A, block, kind='distinct'):
    """
    Rearrange non-overlapping image blocks into columns.

    Equivalent to MATLAB's  B = im2col(A, [m n], 'distinct').

    The image is split into m-by-n blocks, starting at the top-left corner.
    Each block becomes one column of B, its pixels taken column by column
    (down the first column of the block, then down the second, ...).  The
    blocks themselves are also taken column by column: down the first column
    of blocks, then down the second, and so on.  If the image size is not a
    multiple of the block size, it is padded with zeros at the bottom and
    right, as in MATLAB.

    Parameters
    ----------
    A : array-like, shape (rows, columns)
        The image.
    block : (m, n)
        The block size.
    kind : str
        Only 'distinct' is supported.

    Returns
    -------
    B : numpy.ndarray, shape (m*n, number of blocks)
    """
    if kind != 'distinct':
        raise ValueError("im2col: only 'distinct' blocks are supported")
    A = np.asarray(A)
    m, n = block
    R, C = A.shape
    Rp, Cp = -(-R // m) * m, -(-C // n) * n
    if (Rp, Cp) != (R, C):
        P = np.zeros((Rp, Cp), dtype=A.dtype)
        P[:R, :C] = A
        A = P
    br, bc = Rp // m, Cp // n
    # axes after reshape: (block row, row in block, block column, column in block)
    T = A.reshape(br, m, bc, n)
    # column index of B = block row + br * block column; row index = row in block + m * column in block
    return T.transpose(3, 1, 2, 0).reshape(m * n, bc * br)


def col2im(B, block, size, kind='distinct'):
    """
    Rearrange columns into non-overlapping image blocks.

    Equivalent to MATLAB's  A = col2im(B, [m n], [mm nn], 'distinct'), the
    inverse of im2col: each column of B becomes one m-by-n block of an
    mm-by-nn image, in the same order im2col uses.

    Parameters
    ----------
    B : array-like, shape (m*n, number of blocks)
    block : (m, n)
        The block size.
    size : (mm, nn)
        The size of the image to build.
    kind : str
        Only 'distinct' is supported.

    Returns
    -------
    A : numpy.ndarray, shape (mm, nn)
    """
    if kind != 'distinct':
        raise ValueError("col2im: only 'distinct' blocks are supported")
    B = np.asarray(B)
    m, n = block
    mm, nn = size
    br, bc = -(-mm // m), -(-nn // n)
    if B.shape != (m * n, br * bc):
        raise ValueError(f"col2im: expected B of shape {(m * n, br * bc)}, got {B.shape}")
    T = B.reshape(n, m, bc, br).transpose(3, 1, 2, 0)
    return T.reshape(br * m, bc * n)[:mm, :nn]


# ---------------------------------------------------------------------------
# Sums, added in MATLAB's order
# ---------------------------------------------------------------------------

def matlab_sum(v):
    """
    Sum of the elements of a vector, added in the same order MATLAB adds them.

    Equivalent to MATLAB's  sum(v)  for a vector.

    Floating-point addition is not associative, so the last digit of a long
    sum depends on the order in which its terms are added.  np.sum(v) gives
    the same answer to about 15 significant digits; matlab_sum(v) adds in
    MATLAB's order, so that results agree with MATLAB to the last bit.

    The result is a NumPy float, so that, as in MATLAB, dividing it by zero
    gives inf or nan (with a warning) rather than raising an error.
    """
    return np.float64(_matlab_sum(v))


def mean(x):
    """
    Mean, as MATLAB computes it.

    Equivalent to MATLAB's  mean(x), to the last bit (see matlab_sum):

    * for a vector (a 1-D array, or a 2-D array with one row or one column),
      the mean of its elements;
    * for a matrix, a 1-D array with the mean of each column.

    So MATLAB's  mean(mean(A))  of a matrix A is  mean(mean(A))  here too.
    np.mean gives the same values to about 15 significant digits.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim <= 1 or 1 in x.shape:
        x = x.ravel()
        return np.float64(_matlab_sum(x) / x.size)
    if x.ndim != 2:
        raise ValueError("mean: expected a vector or a matrix")
    return np.array([_matlab_sum(x[:, j]) for j in range(x.shape[1])]) / x.shape[0]


def log10(x):
    """
    Base-10 logarithm.

    Equivalent to MATLAB's  log10(x), to the last bit.  np.log10 gives the
    same value to about 15 significant digits, but its last bit sometimes
    differs from MATLAB's.  Works on a number or, element by element, on an
    array.
    """
    v = np.asarray(x, dtype=np.float64)
    if v.ndim == 0:
        return np.float64(_log10(float(v)))
    return np.vectorize(_log10, otypes=[np.float64])(v)


def _log10(t):
    if t > 0:
        return math.log10(t) if t != math.inf else math.inf
    if t == 0:
        return -math.inf
    return math.nan            # MATLAB returns a complex number here


# ---------------------------------------------------------------------------
# MAT-files
# ---------------------------------------------------------------------------

def save(filename, **variables):
    """
    Save variables to a MATLAB MAT-file.

    Equivalent to MATLAB's  save(filename, 'var1', 'var2', ...).  The variables
    are passed by name, so MATLAB's

        save vq_codebook codebook

    is written

        save('vq_codebook', codebook=codebook)

    As in MATLAB, '.mat' is appended if `filename` has no extension.

    The file is written the way MATLAB's save writes it by default (a
    compressed MAT-file, version 7), so it has the same size as the file MATLAB
    would write for the same data, and can be opened in MATLAB.  Numbers are
    saved as MATLAB doubles (MATLAB's default numeric type), a 1-D array or
    list is saved as a row vector, and a 2-D array keeps its shape.

    Parameters
    ----------
    filename : str
        Name of the file to write.
    **variables : array-like
        The variables to save, each given as  name=value.  Values must be real
        numbers: scalars, lists or numpy arrays.
    """
    if not os.path.splitext(filename)[1]:
        filename += '.mat'
    text = ('MATLAB 5.0 MAT-file, Platform: Python, Created on: '
            + time.strftime('%a %b %d %H:%M:%S %Y')).encode('ascii')
    data = text.ljust(116, b' ') + bytes(8) + struct.pack('<H', 0x0100) + b'IM'
    for name, value in variables.items():
        packed = _mat_compress(_mat_variable(name, value))
        data += struct.pack('<II', 15, len(packed)) + packed      # miCOMPRESSED
    with open(filename, 'wb') as f:
        f.write(data)


def load(filename):
    """
    Load the variables stored in a MATLAB MAT-file.

    Equivalent to MATLAB's  S = load(filename).  The variables are returned in
    a dictionary, so MATLAB's  S.codebook  is  load(filename)['codebook'].
    As in MATLAB, '.mat' is appended if `filename` has no extension.

    A row or column vector is returned as a 1-D numpy array, a 1-by-1 value
    as a scalar, and any other matrix as a 2-D array.  MATLAB doubles (all
    numbers saved with `save`) are returned as floating point.
    """
    if not os.path.splitext(filename)[1]:
        filename += '.mat'
    contents = loadmat(filename, mat_dtype=True)     # numbers come back in their MATLAB class
    variables = {}
    for name, value in contents.items():
        if name.startswith('__'):
            continue
        if isinstance(value, np.ndarray) and value.ndim == 2 and 1 in value.shape:
            value = value.item() if value.size == 1 else value.ravel()
        variables[name] = value
    return variables


# MATLAB's save stores a double array whose values are all whole numbers in the
# smallest of these integer types that holds them (it is still read back as
# double).  The order was established by saving test vectors in MATLAB R2025b
# and comparing the files byte for byte (Lab 3).
_MAT_COMPACT = ('uint8', 'uint16', 'int16', 'int32')
_MI_TYPE = {'uint8': 2, 'int16': 3, 'uint16': 4, 'int32': 5, 'double': 9}
_MX_DOUBLE = 6


def _mat_compress(body):
    """zlib-compress one variable as MATLAB's save does: at the default level,
    fed to the compressor 8,192 bytes at a time with a full flush after each
    piece but the last.  (A variable of up to 8 KB is therefore compressed in
    one piece; larger ones, such as a 64-by-16 codebook, come out a few
    hundred bytes larger than a one-piece compression would.  Established by
    comparing files with MATLAB R2026b byte for byte.)"""
    co = zlib.compressobj(6)
    out = []
    for i in range(0, len(body), 8192):
        out.append(co.compress(body[i:i + 8192]))
        if i + 8192 < len(body):
            out.append(co.flush(zlib.Z_FULL_FLUSH))
    out.append(co.flush())
    return b''.join(out)


def _mat_element(mi_type, payload, small_ok=True):
    """One MAT-file data element: tag + payload, padded to 8 bytes.  Payloads
    of up to 4 bytes use the 'small data element' form, as MATLAB does."""
    if small_ok and 0 < len(payload) <= 4:
        return struct.pack('<HH', mi_type, len(payload)) + payload.ljust(4, b'\0')
    return (struct.pack('<II', mi_type, len(payload)) + payload
            + bytes((-len(payload)) % 8))


def _mat_variable(name, value):
    a = np.asarray(value)
    if a.dtype.kind not in 'biuf':
        raise TypeError(f"save: variable '{name}' must hold real numbers")
    # Python numbers are saved as MATLAB doubles, MATLAB's default numeric class
    # (a histogram built from Python ints is then saved exactly as MATLAB saves
    # one built with zeros()).
    a = a.astype(np.float64)
    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.ndim == 1:
        a = a.reshape(1, -1)
    elif a.ndim > 2:
        raise ValueError(f"save: variable '{name}' has more than 2 dimensions")
    stored = 'double'
    if a.size and np.all(np.isfinite(a)) and np.all(a == np.round(a)) \
            and not np.any(np.signbit(a) & (a == 0)):
        lo, hi = a.min(), a.max()
        for t in _MAT_COMPACT:
            if np.iinfo(t).min <= lo and hi <= np.iinfo(t).max:
                stored = t
                break
    body = (_mat_element(6, struct.pack('<BBxxI', _MX_DOUBLE, 0, 0), False)   # array flags
            + _mat_element(5, struct.pack(f'<{a.ndim}i', *a.shape), False)          # dimensions
            + _mat_element(1, name.encode('ascii'))                                 # name
            + _mat_element(_MI_TYPE[stored],                                        # real part
                           a.astype(np.dtype(stored).newbyteorder('<')).tobytes(order='F')))
    return struct.pack('<II', 14, len(body)) + body                                # miMATRIX


# ---------------------------------------------------------------------------
# MATLAB's summation order
# ---------------------------------------------------------------------------
# The order was established by measurement against MATLAB R2026b in MATLAB
# Online (8 computational threads):
#   * up to 43 elements: a plain left-to-right sum;
#   * from 44 elements to somewhere between 88,500 and 89,000: four interleaved
#     partial sums (element k goes to accumulator k mod 4) that are combined as
#     ((s0 + s1) + s2) + s3;
#   * above that, the vector is cut into blocks of 4,096 elements, each block
#     is summed as above, each of the 8 threads adds up its run of consecutive
#     blocks left to right, and the 8 thread totals are added left to right.
#     This was pinned down for 262,144 elements (the Lab 4 image, 512 x 512,
#     whose 64 blocks split evenly over the 8 threads).  For other long
#     lengths the split of the blocks among threads has not been measured, so
#     the same rule is applied with the blocks shared out as evenly as
#     possible; the result may then differ from MATLAB's in the last bit.

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
