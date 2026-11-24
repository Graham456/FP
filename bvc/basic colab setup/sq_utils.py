"""
sq_utils.py  —  ENSC 424 Lab 3 Helper Module
==============================================
Python equivalents of the MATLAB built-in functions that Lab 3 relies on:

    wavread    →  wavread(filename)
    wavwrite   →  wavwrite(y, Fs, filename)
    quantiz    →  quantiz(sig, partition, codebook)
    lloyds     →  lloyds(training_set, ini_codebook, tol)
    save       →  save(filename, var1=..., var2=...)
    load       →  load(filename)

This file is provided to students; you do NOT need to modify it.
"""

import os
import struct
import time
import zlib

import numpy as np
from scipy.io import wavfile, loadmat


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def wavread(filename):
    """
    Read a 16-bit PCM WAV file and return its samples as floating point.

    Equivalent to MATLAB's  y = wavread(filename).

    The 16-bit integer samples are scaled to the range [-1, +1) by dividing by
    32768, exactly as MATLAB does, so a sample stored as -32768 reads as -1.0
    and one stored as 32767 reads as 32767/32768.

    Parameters
    ----------
    filename : str
        Path to the WAV file.

    Returns
    -------
    y : numpy.ndarray of float64
        The audio samples, one per entry (a mono signal is returned as a 1-D
        array).
    """
    _, data = wavfile.read(filename)
    if data.dtype != np.int16:
        raise ValueError(f"{filename}: expected 16-bit PCM, got {data.dtype}")
    return data.astype(np.float64) / 32768.0


def wavwrite(y, Fs, filename):
    """
    Write a floating-point signal to a 16-bit PCM WAV file.

    Equivalent to MATLAB's  wavwrite(y, Fs, filename).

    The samples are scaled by 32768, rounded to the nearest integer and
    clipped to [-32768, 32767], exactly as MATLAB does.  Rounding is *half away
    from zero* (MATLAB's round), not numpy's round-half-to-even: a sample that
    lands exactly on a half-integer, such as 0.5/32768, is written as 1.

    Parameters
    ----------
    y : array-like
        The samples, nominally in the range [-1, +1).
    Fs : int
        The sampling frequency in Hz.
    filename : str
        Path of the WAV file to write.
    """
    v = np.asarray(y, dtype=np.float64).ravel() * 32768.0
    v = np.copysign(np.floor(np.abs(v) + 0.5), v)          # MATLAB round()
    v = np.clip(v, -32768, 32767).astype(np.int16)
    wavfile.write(filename, int(Fs), v)


# ---------------------------------------------------------------------------
# Scalar quantization
# ---------------------------------------------------------------------------

def quantiz(sig, partition, codebook):
    """
    Produce a quantization index and a quantized output value.

    Equivalent to MATLAB's  [index, quants, distor] = quantiz(sig, partition, codebook).

    `partition` is a vector of strictly increasing values that splits the real
    line into length(partition)+1 bins.  The index of a sample tells which bin
    it falls into:

        index = 0            if  sig <= partition[0]
        index = m            if  partition[m-1] < sig <= partition[m]
        index = N-1          if  partition[N-2] < sig

    where N = len(partition) + 1.  Note that the bins are closed on the
    *right* and that, as in MATLAB, the lowest index is 0.

    `codebook` gives the quantized value for each bin and must have exactly
    one more entry than `partition`.

    Parameters
    ----------
    sig : array-like
        The input signal.
    partition : array-like, length N-1
        The bin boundaries.
    codebook : array-like, length N
        The quantized value for each bin.

    Returns
    -------
    index : numpy.ndarray of int
        The bin index of each sample, from 0 to N-1.
    quants : numpy.ndarray of float64
        The quantized signal, quants[i] = codebook[index[i]].
    distor : float
        The mean squared error between sig and quants.
    """
    sig       = np.asarray(sig, dtype=np.float64).ravel()
    partition = np.asarray(partition, dtype=np.float64).ravel()
    codebook  = np.asarray(codebook, dtype=np.float64).ravel()
    if len(codebook) != len(partition) + 1:
        raise ValueError("codebook must have exactly one more entry than partition")

    # MATLAB:  for i = 1:length(partition), indx = indx + (sig > partition(i)); end
    index = np.zeros(len(sig), dtype=np.int64)
    for p in partition:
        index += sig > p

    quants = codebook[index]

    # MATLAB accumulates the distortion bin by bin, in index order:
    #   for i = 0:length(codebook)-1
    #       distor = distor + sum((sig(indx==i) - codebook(i+1)).^2);
    #   end
    #   distor = distor / length(sig);
    distor = 0.0
    for i in range(len(codebook)):
        distor = distor + _matlab_sum((sig[index == i] - codebook[i]) ** 2)
    distor = distor / len(sig)
    return index, quants, distor


def lloyds(training_set, ini_codebook, tol=None, return_distortion=False):
    """
    Optimize quantization parameters using the Lloyd algorithm.

    Equivalent to MATLAB's  [partition, codebook] = lloyds(training_set, ini_codebook, tol).

    Parameters
    ----------
    training_set : array-like
        Training data: samples typical of the source to be quantized.
    ini_codebook : array-like or int
        The initial guess of the codebook values.  The optimized codebook has
        the same length.  If an integer is given instead, it is the length of
        the codebook, and the initial codebook is spread uniformly over the
        range of the training data.
    tol : float, optional
        The optimization stops when the relative change in distortion falls
        below tol.  Default 10**(-7).
    return_distortion : bool, optional
        If True, also return MATLAB's third and fourth outputs: the final
        mean squared distortion and the final relative change in distortion.

    Returns
    -------
    partition : numpy.ndarray
        The optimized bin boundaries (one fewer than the codebook).
    codebook : numpy.ndarray
        The optimized quantized value for each bin.

    Notes
    -----
    A cell that receives no training samples is handled as MATLAB does: its
    codebook value is moved to the middle of the cell (for the outermost cells,
    halfway between the boundary and the extreme of the training data), so the
    codebook always keeps its full length.
    """
    training_set = np.asarray(training_set, dtype=np.float64).ravel()
    ini = np.atleast_1d(np.asarray(ini_codebook, dtype=np.float64)).ravel()
    if training_set.size < 1:
        raise ValueError("training_set is empty")
    if ini.size < 1:
        raise ValueError("ini_codebook is empty")
    min_training = np.min(training_set)
    max_training = np.max(training_set)
    if ini.size < 2:
        n = int(np.floor(ini[0]))
        if n < 1:
            raise ValueError("ini_codebook must be a positive integer")
        int_training = (max_training - min_training) / n
        if int_training <= 0:
            raise ValueError("training_set must contain at least two distinct values")
        codebook = _matlab_linspace(min_training + int_training / 2,
                                    max_training - int_training / 2, n)
    else:
        codebook = np.sort(ini)
        n = codebook.size
    if tol is None:
        tol = 0.0000001

    # initial partition and distortion
    partition = (codebook[1:n] + codebook[0:n - 1]) / 2
    index, _, distor = quantiz(training_set, partition, codebook)
    last_distor = 0.0
    ter_cond2 = np.finfo(np.float64).eps * max_training
    rel_distor = abs(distor - last_distor) / distor if distor > ter_cond2 else distor

    while (rel_distor > tol) and (rel_distor > ter_cond2):
        # centroid condition
        for i in range(n):
            members = np.nonzero(index == i)[0]
            if members.size > 0:
                codebook[i] = _matlab_mean(training_set[members])
            elif i == 0:
                tmp = training_set[training_set <= partition[0]]
                codebook[0] = ((partition[0] + min_training) / 2 if tmp.size == 0
                               else _matlab_mean(tmp))
            elif i == n - 1:
                tmp = training_set[training_set >= partition[i - 1]]
                codebook[i] = ((max_training + partition[i - 1]) / 2 if tmp.size == 0
                               else _matlab_mean(tmp))
            else:
                tmp = training_set[training_set >= partition[i - 1]]
                tmp = tmp[tmp <= partition[i]]
                codebook[i] = ((partition[i] + partition[i - 1]) / 2 if tmp.size == 0
                               else _matlab_mean(tmp))
        # nearest-neighbour condition
        partition = np.sort((codebook[1:n] + codebook[0:n - 1]) / 2)
        # stopping test
        last_distor = distor
        index, _, distor = quantiz(training_set, partition, codebook)
        rel_distor = abs(distor - last_distor) / distor if distor > ter_cond2 else distor

    if return_distortion:
        return partition, codebook, distor, rel_distor
    return partition, codebook


# ---------------------------------------------------------------------------
# MAT-files
# ---------------------------------------------------------------------------

def save(filename, **variables):
    """
    Save variables to a MATLAB MAT-file.

    Equivalent to MATLAB's  save(filename, 'var1', 'var2', ...).  The variables
    are passed by name, so MATLAB's

        save uq_codebook codebook

    is written

        save('uq_codebook', codebook=codebook)

    As in MATLAB, '.mat' is appended if `filename` has no extension.

    The file is written the way MATLAB's save writes it by default (a
    compressed MAT-file, version 7), so it has the same size as the file MATLAB
    would write for the same data, and can be opened in MATLAB.  Numbers are
    saved as MATLAB doubles (MATLAB's default numeric type), and a 1-D array or
    list is saved as a row vector.

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
        packed = zlib.compress(_mat_variable(name, value))
        data += struct.pack('<II', 15, len(packed)) + packed      # miCOMPRESSED
    with open(filename, 'wb') as f:
        f.write(data)


def load(filename):
    """
    Load the variables stored in a MATLAB MAT-file.

    Equivalent to MATLAB's  S = load(filename).  The variables are returned in
    a dictionary, so MATLAB's  S.codebook  is  load(filename)['codebook'].
    As in MATLAB, '.mat' is appended if `filename` has no extension.

    A row or column vector is returned as a 1-D numpy array, and a 1-by-1 value
    as a scalar.  MATLAB doubles (all numbers saved with `save`) are returned as
    floating point.
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
# and comparing the files byte for byte.
_MAT_COMPACT = ('uint8', 'uint16', 'int16', 'int32')
_MI_TYPE = {'uint8': 2, 'int16': 3, 'uint16': 4, 'int32': 5, 'double': 9}
_MX_DOUBLE = 6


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
# MATLAB arithmetic, reproduced exactly
# ---------------------------------------------------------------------------
# Floating-point addition is not associative, so a sum's last bit depends on
# the order in which its terms are added.  These helpers add in the order
# MATLAB does, so that results agree with MATLAB to the last bit.  The order
# was established by measurement against MATLAB R2025b (MATLAB Online): a plain
# left-to-right sum for vectors of up to 43 elements, and for longer vectors
# four interleaved partial sums (element k goes to accumulator k mod 4) that
# are combined as ((s0 + s1) + s2) + s3.  It was verified on vectors of up to
# 64,010 elements, the length of the Lab 3 audio signal; MATLAB switches to a
# different scheme somewhere below 100,000 elements.

def _matlab_sum(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    n = v.size
    if n <= 43:
        s = 0.0
        for a in v.tolist():
            s += a
        return s
    m = n - n % 4
    acc = np.add.reduce(v[:m].reshape(-1, 4), axis=0).copy()   # each lane in order
    acc[:n - m] += v[m:]
    s0, s1, s2, s3 = acc.tolist()
    return ((s0 + s1) + s2) + s3


def _matlab_mean(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return _matlab_sum(v) / v.size


def _matlab_linspace(d1, d2, n):
    """MATLAB's linspace(d1, d2, n)."""
    n1 = n - 1
    if n1 < 1:
        return np.array([d2], dtype=np.float64)
    k = np.arange(n, dtype=np.float64)
    y = d1 + (k * (d2 - d1)) / n1
    y[0] = d1
    y[-1] = d2
    return y
