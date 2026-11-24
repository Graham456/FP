import os
import math
import pickle
import struct

from arith_utils import arith_encode, arith_decode

def fread_ubit1(fid):
    """Read the rest of an open binary file as a list of bits (0/1).

    The inverse of fwrite_ubit1, and the equivalent of MATLAB's
    fread(fid, inf, 'ubit1').  The trailing zero padding of the last byte comes
    back too; that is harmless, because the decoder stops after n_symbols
    symbols and never reaches it.  (provided)
    """
    bits = []
    for byte in fid.read():
        for j in range(8):
            bits.append((byte >> j) & 1)
    return bits


def dec_arith(histfile, infile):
    """
    Decode an arithmetic-coded binary file back to the original symbol sequence.

    Parameters
    ----------
    histfile : str
        Path to the file holding the histogram vector `counts` used for encoding.
    infile : str
        Path to the binary file produced by enc_arith.

    Returns
    -------
    message_dec_int : list of int
        Reconstructed sequence of 1-indexed symbols.
    """
    # Step 1: Load the histogram vector `counts` from histfile
    # YOUR CODE HERE
    with open(histfile, 'rb') as f:
        counts = pickle.load(f)
    # Steps 2-5: Open infile, read n_symbols, read the bitstream, close
    # Hint: struct.unpack('<I', fid.read(4))[0]
    #       fread_ubit1(fid)
    # YOUR CODE HERE
    with open(infile, 'rb') as fid:
        n_symbols = struct.unpack('<I', fid.read(4))[0]
        bitstream = fread_ubit1(fid)
    # Step 6: Decode
    # Hint: arith_decode(bitstream, counts, n_symbols)
    # YOUR CODE HERE
    message_dec_int = arith_decode(bitstream, counts, n_symbols)
    # Step 7: Release everything except message_dec_int
    # YOUR CODE HERE
    del counts, bitstream, n_symbols
    return message_dec_int

def fwrite_ubit1(fid, bits):
    """Write a list of bits (0/1) to an open binary file, one bit at a time.

    The equivalent of MATLAB's fwrite(fid, bits, 'ubit1'): bits are packed into
    bytes starting at each byte's least-significant bit, and the last byte is
    zero-padded.  (provided)
    """
    pad = (8 - len(bits) % 8) % 8
    padded = list(bits) + [0] * pad
    out = bytearray()
    for i in range(0, len(padded), 8):
        byte_val = 0
        for j in range(8):
            byte_val |= padded[i + j] << j
        out.append(byte_val)
    fid.write(bytes(out))


def enc_arith(message_int, histfile, outfile):
    """
    Arithmetic-encode message_int and write the compressed data to outfile.

    Parameters
    ----------
    message_int : list of int
        Sequence of 1-indexed symbols to encode.
    histfile : str
        Path to the file holding the histogram vector `counts`.
    outfile : str
        Path to the output binary file.

    Returns
    -------
    n_bits : int
        Total number of bits in the arithmetic bitstream.
    """
    # Step 1: Load the histogram vector `counts` from histfile
    # Hint: with open(histfile, 'rb') as f: counts = pickle.load(f)
    # YOUR CODE HERE
    with open(histfile,'rb') as f:
        counts=pickle.load(f)
    # Step 2: Get the number of symbols
    # YOUR CODE HERE
    n_symbols=len(message_int)
    # Step 3: Arithmetic encode → list of bits
    # Hint: arith_encode(message_int, counts)
    # YOUR CODE HERE
    bitstream= arith_encode(message_int,counts)
    # Step 4: Get n_bits
    # YOUR CODE HERE
    n_bits=len(bitstream)
    # Steps 5-7: Open outfile, write n_symbols, then write the bitstream
    # Hint: struct.pack('<I', value)  →  4-byte little-endian uint32
    #       fwrite_ubit1(fid, bitstream)
    # YOUR CODE HERE
    with open(outfile,'wb') as f:
        f.write(struct.pack('<I',n_symbols))
        fwrite_ubit1(f,bitstream)
    # Step 8: Release everything except n_bits
    # YOUR CODE HERE
    del counts,bitstream,n_symbols
    return n_bits