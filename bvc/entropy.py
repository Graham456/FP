"""Static arithmetic coder (from arith_utils.py + lab2_arith.py).

arith_encode / arith_decode (MATLAB arithenco / arithdeco), enc_arith (write
the coded bits to a file), entropy_code (the notebook's "Entropy coding" block:
shift values to symbols >= 1, save their histogram, arithmetic-code them), and
write_chunk / read_chunk, which store the same thing inside one .bvc file.
"""
import os
import pickle
import struct

import numpy as np


# ---------------------------------------------------------------------------
# Arithmetic encoder (arith_utils.py)
# ---------------------------------------------------------------------------

def _precision_for(total):
    """Register size N = ceil(log2(total)) + 2 (MATLAB's rule) and the interval constants.

    With this N the narrowest symbol can never get an empty sub-interval, so the
    coder works for any histogram skew. Encoder and decoder both derive N from
    the counts, so nothing extra is stored in the bitstream.
    """
    prec = (total - 1).bit_length() + 2        # ceil(log2(total)) + 2
    full = 1 << prec
    half = full >> 1
    quarter = half >> 1
    return prec, full, half, quarter, half + quarter


def _build_cumulative(counts):
    """cum[k] = counts[0] + ... + counts[k-1]; zero counts are treated as 1."""
    adj = [max(c, 1) for c in counts]
    cum = [0] * (len(adj) + 1)
    for i, c in enumerate(adj):
        cum[i + 1] = cum[i] + c
    return cum, cum[-1]


def arith_encode(message_int, counts):
    """Arithmetic-code positive integer symbols (symbol k has count counts[k-1]).

    Equivalent to MATLAB's arithenco, bit for bit. Returns a list of 0/1 bits.
    """
    cum, total = _build_cumulative(counts)
    prec, full, half, quarter, three_quarter = _precision_for(total)

    low, high = 0, full
    pending = 0                  # E3 (middle-half) rescalings waiting for their bits
    bits = []
    emit = bits.append

    for sym in message_int:
        # Narrow [low, high) to this symbol's share of the interval
        rng = high - low
        high = low + (rng * cum[sym]) // total
        low = low + (rng * cum[sym - 1]) // total

        # Rescale while the interval sits in one half (E1/E2) or the middle half (E3)
        while True:
            if high <= half:                                 # E1: lower half -> 0
                emit(0)
                bits.extend([1] * pending)
                pending = 0
                low <<= 1
                high <<= 1
            elif low >= half:                                # E2: upper half -> 1
                emit(1)
                bits.extend([0] * pending)
                pending = 0
                low = (low - half) << 1
                high = (high - half) << 1
            elif low >= quarter and high <= three_quarter:   # E3: middle half
                pending += 1
                low = (low - quarter) << 1
                high = (high - quarter) << 1
            else:
                break

    # Flush the whole N-bit `low` register, as MATLAB does (MSB, pending bits, rest)
    msb = (low >> (prec - 1)) & 1
    emit(msb)
    bits.extend([1 - msb] * pending)
    for shift in range(prec - 2, -1, -1):
        emit((low >> shift) & 1)
    return bits


def _find_symbol(cum, scaled, m):
    """Binary search: 1-indexed symbol k with cum[k-1] <= scaled < cum[k]."""
    lo, hi = 1, m
    while lo < hi:
        mid = (lo + hi) >> 1
        if cum[mid] <= scaled:
            lo = mid + 1
        else:
            hi = mid
    return lo


def arith_decode(bitstream, counts, n_symbols):
    """Decode n_symbols symbols from arith_encode's bits (MATLAB arithdeco), from arith_utils.py.

    Bits past the end of the stream are read as 0, so byte padding is harmless.
    """
    cum, total = _build_cumulative(counts)
    prec, full, half, quarter, three_quarter = _precision_for(total)
    m = len(counts)
    bits, n_bits = list(bitstream), len(bitstream)

    def next_bit(idx):
        return bits[idx] if idx < n_bits else 0

    # Prime the register with the first prec bits
    value, bit_idx = 0, 0
    for _ in range(prec):
        value = (value << 1) | next_bit(bit_idx)
        bit_idx += 1

    low, high = 0, full
    decoded = []
    for _ in range(n_symbols):
        # Which symbol's slice of [low, high) does value fall in?
        rng = high - low
        sym = _find_symbol(cum, ((value - low + 1) * total - 1) // rng, m)
        decoded.append(sym)
        high = low + (rng * cum[sym]) // total
        low = low + (rng * cum[sym - 1]) // total

        # Same E1/E2/E3 rescalings as the encoder, shifting in one new bit each time
        while True:
            if high <= half:
                low, high, value = low << 1, high << 1, value << 1
            elif low >= half:
                low, high, value = (low - half) << 1, (high - half) << 1, (value - half) << 1
            elif low >= quarter and high <= three_quarter:
                low, high, value = (low - quarter) << 1, (high - quarter) << 1, (value - quarter) << 1
            else:
                break
            value |= next_bit(bit_idx)
            bit_idx += 1
    return decoded


# ---------------------------------------------------------------------------
# File writer (lab2_arith.py)
# ---------------------------------------------------------------------------

def fwrite_ubit1(fid, bits):
    """Pack bits into bytes, LSB first, zero-padding the last byte (MATLAB 'ubit1')."""
    padded = list(bits) + [0] * ((8 - len(bits) % 8) % 8)
    out = bytearray()
    for i in range(0, len(padded), 8):
        byte_val = 0
        for j in range(8):
            byte_val |= padded[i + j] << j
        out.append(byte_val)
    fid.write(bytes(out))


def enc_arith(message_int, histfile, outfile):
    """Arithmetic-code message_int with the histogram saved in histfile.

    outfile gets a 4-byte symbol count followed by the packed bits.
    Returns the number of arithmetic-coded bits (the rate the codec reports).
    """
    with open(histfile, 'rb') as f:
        counts = pickle.load(f)
    bitstream = arith_encode(message_int, counts)
    with open(outfile, 'wb') as f:
        f.write(struct.pack('<I', len(message_int)))
        fwrite_ubit1(f, bitstream)
    return len(bitstream)


# ---------------------------------------------------------------------------
# Entropy-coding block of the video coder (notebook's entropy_code)
# ---------------------------------------------------------------------------

def entropy_code(data, name, out_dir):
    """Arithmetic-code an integer array; writes <name>_hist.pkl and <name>.bit in out_dir.

    The values are shifted so the smallest becomes symbol 1 (the coder needs
    symbols >= 1), and their histogram is saved for the decoder. Returns bits used.
    """
    data = np.asarray(data).astype(np.int64).ravel()
    message_int = data - data.min() + 1
    counts = np.maximum(np.bincount(message_int)[1:], 1).tolist()
    os.makedirs(out_dir, exist_ok=True)
    histfile = os.path.join(out_dir, f'{name}_hist.pkl')
    with open(histfile, 'wb') as f:
        pickle.dump(counts, f)
    return enc_arith(message_int.tolist(), histfile, os.path.join(out_dir, f'{name}.bit'))


# ---------------------------------------------------------------------------
# Chunks inside a .bvc file: everything the decoder needs for one integer array
# ---------------------------------------------------------------------------
#   int32  offset       data.min(), to undo the shift to symbols >= 1
#   uint32 n            histogram size
#   uint32 counts[n]    histogram (zero counts raised to 1)
#   uint32 nbytes       length of the packed bitstream
#   bytes  bits         arithmetic-coded bits, packed MSB first (np.packbits)
# The symbol count is not stored: the decoder knows it from the frame size.
CHUNK_HEAD = struct.Struct('<iI')
UINT32 = struct.Struct('<I')


def write_chunk(f, data):
    """Arithmetic-code an integer array into the open file f; returns the bytes written."""
    data = np.asarray(data).astype(np.int64).ravel()
    offset = int(data.min())
    message_int = data - offset + 1                              # symbols must be >= 1
    counts = np.maximum(np.bincount(message_int)[1:], 1).tolist()
    bits = arith_encode(message_int.tolist(), counts)
    packed = np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()
    f.write(CHUNK_HEAD.pack(offset, len(counts)))
    f.write(struct.pack(f'<{len(counts)}I', *counts))
    f.write(UINT32.pack(len(packed)))
    f.write(packed)
    return CHUNK_HEAD.size + 4 * len(counts) + UINT32.size + len(packed)


def _read_exact(f, n):
    data = f.read(n)
    if len(data) != n:
        raise ValueError('Truncated .bvc file')
    return data


def read_chunk(f, n_symbols):
    """Read one chunk written by write_chunk and decode n_symbols values (int64 array)."""
    offset, n = CHUNK_HEAD.unpack(_read_exact(f, CHUNK_HEAD.size))
    counts = list(struct.unpack(f'<{n}I', _read_exact(f, 4 * n)))
    (nbytes,) = UINT32.unpack(_read_exact(f, UINT32.size))
    bits = np.unpackbits(np.frombuffer(_read_exact(f, nbytes), dtype=np.uint8)).tolist()
    return np.asarray(arith_decode(bits, counts, n_symbols), dtype=np.int64) + offset - 1
