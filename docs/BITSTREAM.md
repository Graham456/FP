# E424 version 1 bitstream

All fixed-width fields are little-endian. The decoder requires the entire file;
it rejects trailing bytes. There is no external model, reference file, or pickle.

## Sequence header: Python struct `<4sBIIBIdHHHH` (34 bytes)

In order: magic `E424` (4 bytes), version 1 (uint8), width (uint32), height
(uint32), channels 1 or 3 (uint8), frame count (uint32), FPS (float64),
quantization step (uint16), prediction block size (uint16), search radius
(uint16), GOP length (uint16). Dimensions describe the cropped image.
Padding is edge replication to the next multiple of the prediction block size.

## Frame packet: `<BIIII` header (17 bytes), then payload

Fields: kind (uint8: 0=I, 1=P), uncompressed syntax byte count (uint32),
arithmetic-coded payload byte count (uint32), syntax CRC32 (uint32), payload
CRC32 (uint32). The first packet must be I. CRCs cover the corresponding bytes,
not the sequence header; they detect accidental errors, not malicious edits.

The arithmetic coder resets its eight `(1,1)` frequency models for each packet.
Each byte is processed most-significant bit first, with context equal to the bit
position 0..7. Increment the observed count; when the context total reaches
16384, replace each count with `(count+1)//2`. A 32-bit inclusive low/high
interval uses E1/E2/E3 normalization. Termination emits the final interval bit
and deferred opposite bits; unused final byte bits and decoder lookahead are
zero-filled. The syntax length determines how many bytes to decode.

## Block syntax

Visit prediction blocks in raster order. First emit a one-byte mode:

- 0: intra, prediction is constant 128, followed by residual syntax.
- 1: inter, followed by signed dy, signed dx, then residual syntax.
- 2: skip, no additional syntax; copy the colocated reconstructed reference.

I packets allow only mode 0. P packets may mix all three modes. Motion vectors
are backward reference offsets `(reference_y-current_y, reference_x-current_x)`.
They must lie within the signaled search radius and padded reference bounds.

Residual syntax visits channels in order (gray or R,G,B), then each 8x8
transform block in raster order within the prediction block. Each transform
block stores nonzero count followed by that many `(preceding zero run, signed
level)` pairs in zigzag order. Levels must be nonzero; all remaining coefficients
are zero. There is no DC differential predictor in version 1.

Unsigned integers use base-128 little-endian groups: low 7 bits per byte, high
bit means continuation. Signed integers map `v>=0` to `2*v`, and `v<0` to
`-2*v-1`, then use the same unsigned representation.

The format is intended for course experiments, not untrusted network services.
The implementation validates dimensions, lengths, modes, coefficient positions,
motion bounds and CRCs, with an output-sample allocation limit. It does not claim
cross-platform bit-exactness across all floating-point implementations.
