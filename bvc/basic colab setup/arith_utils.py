"""
arith_utils.py  —  ENSC 424 Lab 2 Helper Module
=================================================
Provides Python equivalents of MATLAB's arithmetic coding toolbox functions:
    arithenco  →  arith_encode(message_int, counts)
    arithdeco  →  arith_decode(bitstream, counts, n_symbols)

This file is provided to students; you do NOT need to modify it.

Algorithm
---------
Implements finite-precision arithmetic coding with E1/E2/E3 interval-rescaling
(the standard integer-precision arithmetic coder), using MATLAB's two sizing
conventions so that the output is **bit-for-bit identical to arithenco**:
working precision N = ceil(log2(total_count)) + 2, and a terminating flush that
writes the whole N-bit `low` register.  See the notes on each below.

Input convention
----------------
Both functions assume the input is a sequence of **positive integers**, exactly
as MATLAB's arithenco/arithdeco do.  Let M be the maximum number in the input
sequence.  Both need the histogram in the form of a vector [c1, c2, ..., cM],
where ck is the number of occurrences of the number (symbol) k in the input
sequence; symbol k corresponds to counts[k-1].  Counts are occurrence counts,
not probabilities -- the coder normalises them itself.  If the source sequence
is a sequence of characters or some other symbols, it must first be converted
to positive integers 1, 2, ..., M prior to arithmetic encoding.
"""

from __future__ import annotations
from typing import List


# ---------------------------------------------------------------------------
# Working precision
# ---------------------------------------------------------------------------
# The precision is not a fixed constant: it is derived from the histogram, as
# MATLAB's arithenco.m does.  That file sizes its register with
#
#     N = ceil(log2(total_count)) + 2;
#
# and this module uses the same rule, so the two coders agree bit for bit.
#
# Why the rule works.  After a rescaling step the interval width always exceeds
# 2^(N-2), so the narrowest symbol receives a subinterval of at least
# floor(2^(N-2) * min_count / total) ticks.  That floors to zero -- the interval
# collapses and the encoder spins -- only once total/min_count exceeds 2^(N-2).
# With N = ceil(log2(total)) + 2 we have 2^(N-2) >= total >= total/min_count,
# so the collapse can never happen, whatever the skew.  (A fixed 16 bits, the
# classic textbook choice, does collapse: it cannot code a 168490:1 histogram.)
#
# ceil(log2(n)) is computed exactly in integer arithmetic as (n-1).bit_length().
# Encoder and decoder each derive N from `counts`, so they always agree and
# nothing extra has to be stored in the bitstream.


def _precision_for(total):
    """Working bits, and the derived interval constants, for one histogram."""
    prec = (total - 1).bit_length() + 2        # ceil(log2(total)) + 2
    full = 1 << prec
    half = full >> 1
    quarter = half >> 1
    return prec, full, half, quarter, half + quarter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def arith_encode(message_int: List[int], counts: List[int]) -> List[int]:
    """
    Encode a sequence of positive integers using arithmetic coding.

    Analogous to MATLAB's:
        bitstream = arithenco(message_int, counts)

    Parameters
    ----------
    message_int : list of int
        Sequence of symbols to encode.  Each symbol must be in [1, M]
        where M = len(counts).
    counts : list of int
        Histogram of symbol occurrences.  counts[k-1] is the count for
        symbol k.  All entries must be >= 1.

    Returns
    -------
    bitstream : list of int
        Flat list of 0s and 1s (the arithmetic-coded bit output).
    """
    # Build cumulative frequency table (0-indexed at positions 0..M)
    cum, total = _build_cumulative(counts)
    _PRECISION, _FULL, _HALF, _QUARTER, _3QUARTER = _precision_for(total)

    low  = 0
    high = _FULL
    pending = 0          # pending bits count (E3 rescaling)
    bits: List[int] = []
    bits_append = bits.append   # local alias for speed

    for sym in message_int:
        # Update interval
        c_lo = cum[sym - 1]
        c_hi = cum[sym]
        rng  = high - low
        high = low + (rng * c_hi) // total
        low  = low + (rng * c_lo) // total

        # Rescaling loop
        while True:
            if high <= _HALF:                            # E1
                bits_append(0)
                for _ in range(pending):
                    bits_append(1)
                pending = 0
                low  <<= 1
                high <<= 1
            elif low >= _HALF:                           # E2
                bits_append(1)
                for _ in range(pending):
                    bits_append(0)
                pending = 0
                low  = (low  - _HALF) << 1
                high = (high - _HALF) << 1
            elif low >= _QUARTER and high <= _3QUARTER:  # E3
                pending += 1
                low  = (low  - _QUARTER) << 1
                high = (high - _QUARTER) << 1
            else:
                break

    # Flush.
    # Two bits would be enough to single out a number inside [low, high), and a
    # minimal coder emits just those.  MATLAB writes out the entire N-bit `low`
    # register instead: its leading bit, then the pending bits (which are its
    # complement), then the remaining N-1 bits of `low`.  That is N-2 bits more
    # than strictly necessary -- about 0.005% on a 330k-bit stream -- and it is
    # the whole of the difference in length between the two coders.  We follow
    # MATLAB so that the streams are identical.
    msb = (low >> (_PRECISION - 1)) & 1
    bits_append(msb)
    for _ in range(pending):
        bits_append(1 - msb)
    for shift in range(_PRECISION - 2, -1, -1):
        bits_append((low >> shift) & 1)

    return bits


def arith_decode(bitstream: List[int], counts: List[int], n_symbols: int) -> List[int]:
    """
    Decode an arithmetic-coded bitstream back to the original symbol sequence.

    Analogous to MATLAB's:
        message_dec_int = arithdeco(bitstream, counts, n_symbols)

    Parameters
    ----------
    bitstream : list of int
        Binary sequence (0s and 1s) produced by arith_encode.
    counts : list of int
        The same histogram used for encoding.
    n_symbols : int
        Exact number of symbols in the original message.

    Returns
    -------
    decoded : list of int
        Reconstructed list of positive integers.
    """
    cum, total = _build_cumulative(counts)
    _PRECISION, _FULL, _HALF, _QUARTER, _3QUARTER = _precision_for(total)
    M = len(counts)

    # Convert bitstream to a list for indexed access (pad with zeros at end)
    bits = list(bitstream)
    n_bits = len(bits)

    # Prime the decoder register with _PRECISION bits
    bit_idx = 0
    value   = 0
    for _ in range(_PRECISION):
        b = bits[bit_idx] if bit_idx < n_bits else 0
        bit_idx += 1
        value = (value << 1) | b

    low  = 0
    high = _FULL
    decoded: List[int] = []
    decoded_append = decoded.append

    for _ in range(n_symbols):
        # Find symbol: binary search in cumulative table
        rng    = high - low
        scaled = ((value - low + 1) * total - 1) // rng
        sym    = _find_symbol(cum, scaled, M)

        decoded_append(sym)

        # Update interval
        high = low + (rng * cum[sym]) // total
        low  = low + (rng * cum[sym - 1]) // total

        # Rescaling loop
        while True:
            if high <= _HALF:
                low   <<= 1
                high  <<= 1
                b = bits[bit_idx] if bit_idx < n_bits else 0
                bit_idx += 1
                value = (value << 1) | b
            elif low >= _HALF:
                low   = (low   - _HALF) << 1
                high  = (high  - _HALF) << 1
                b = bits[bit_idx] if bit_idx < n_bits else 0
                bit_idx += 1
                value = ((value - _HALF) << 1) | b
            elif low >= _QUARTER and high <= _3QUARTER:
                low   = (low   - _QUARTER) << 1
                high  = (high  - _QUARTER) << 1
                b = bits[bit_idx] if bit_idx < n_bits else 0
                bit_idx += 1
                value = ((value - _QUARTER) << 1) | b
            else:
                break

    return decoded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cumulative(counts):
    """Cumulative table cum[k] = sum(counts[0:k]), and the total count
    (the working precision is derived from the total)."""
    # Treat zero counts as 1 to avoid degeneracy
    adj = [max(c, 1) for c in counts]
    cum = [0] * (len(adj) + 1)
    for i, c in enumerate(adj):
        cum[i + 1] = cum[i] + c
    total = cum[-1]
    return cum, total


def _find_symbol(cum, scaled, M):
    """Binary search: find 1-indexed symbol k s.t. cum[k-1] <= scaled < cum[k]."""
    lo, hi = 1, M
    while lo < hi:
        mid = (lo + hi) >> 1
        if cum[mid] <= scaled:
            lo = mid + 1
        else:
            hi = mid
    return lo
