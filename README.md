# ENSC 424 Python video codec

A working educational hybrid codec covering the proposal's milestones through
November 13 and the integration work in progress on November 15. The actual
compressed format is `.e424`; it is not H.264, HEVC, or an MP4 file.

A second, separate codec, `bvc` (block-based video coder built from the lab
code and `Codec.ipynb`), writes `.bvc` files; see [bvc](#bvc-block-based-hybrid-video-coder) below.

## Quick start (PowerShell)

Install Python 3.10 or newer if needed, then open PowerShell in this folder:

```powershell
cd 'C:\Users\Graham McRobbie\Courses\ENSC424\FP'
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m en424 demo
.\.venv\Scripts\python.exe -m en424 encode output/demo.npy output/demo.e424 --qstep 16
.\.venv\Scripts\python.exe -m en424 decode output/demo.e424 output/reconstructed
.\.venv\Scripts\python.exe -m en424 evaluate output/demo.npy output/evaluation
```

`decode` creates numbered PNG frames, or a NumPy array when the output ends in
`.npy`. It runs independently, using only the encoded file. `evaluate` writes
eight compressed streams, `results.csv`, and `summary.json`. It also verifies
that each decoded sequence exactly matches the encoder's local reconstruction.
The CSV can be plotted in Excel or another plotting tool as bitrate versus PSNR.

To use an MP4/AVI/MOV input, install the optional video adapter:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-video.txt
.\.venv\Scripts\python.exe -m en424 encode clip.mp4 output/clip.e424 --frames 30 --gray --qstep 16
.\.venv\Scripts\python.exe -m en424 decode output/clip.e424 output/clip_decoded.npy
.\.venv\Scripts\python.exe -m en424 evaluate clip.mp4 output/clip_results --frames 30 --gray --qsteps 8 16 32 64
```

OpenCV only reads the source video; compression is implemented in this project.
Audio is not processed. Video FPS is read from the source; NPY/PNG input defaults
to 30 FPS. Use `--fps 24` (or the appropriate value) to override. Start with a
short, low-resolution clip: exhaustive search in Python is not real-time.
Decoded PNGs/NPY are the authoritative output, avoiding another lossy codec when
measuring reconstruction quality. The optional OpenCV adapter has not been run
in the supplied environment, where OpenCV is unavailable.

## Inputs and settings

- NPY: `uint8` arrays shaped `(frames,height,width)` or `(frames,height,width,3)`.
  A singleton final grayscale channel is also accepted and squeezed on output.
- PNG directory: names sorted lexicographically; use zero-padded names such as
  `00000.png`. Frames must have the same size. Default is RGB; `--gray` selects luma.
- RGB is coded as three full-resolution planes with shared luma-selected motion
  vectors. Grayscale is useful for faster initial experiments. No YUV 4:2:0 yet.
- `--qstep`: uniform scalar DCT quantization step, 1..255; larger is coarser.
  It is not an H.264 QP. Step 1 is still not guaranteed lossless.
- `--intra-only`: disable temporal prediction to produce the baseline.
- `--block-size`: 8, 16 (default), or 32 pixels for prediction. DCT stays 8x8.
- `--search-radius`: full-search radius in pixels, default 4, range 0..64.
- `--gop`: periodic I-frame interval, default 12. Other frames are P-frames.
- `--no-skip`: disable the optional zero-vector, zero-residual skip optimization.

Both API and CLI operate on clips in memory. The decoder defaults to a total
limit of 100 million padded samples; use short clips, or explicitly adjust the
API's `max_samples` after considering memory usage.

## Code map

| Module | Responsibility |
| --- | --- |
| `en424/transform.py` | Orthonormal DCT/IDCT, quantization, zigzag, run/level syntax |
| `en424/entropy.py` | Actual adaptive binary arithmetic encoder and decoder |
| `en424/motion.py` | Exhaustive SAD search and bounded motion compensation |
| `en424/codec.py` | I/P pipeline, local reconstruction, packet format, decoder |
| `en424/media.py` | NPY, PNG, optional video input; synthetic demo generator |
| `en424/metrics.py` | MSE, PSNR, full-file bitrate, classical cubic BD-Rate |
| `en424/__main__.py` | Command-line entry point and evaluation runner |
| `tests/test_codec.py` | Transform, entropy, motion, codec and metric tests |

## Design choices

1. Edge-pad each frame to prediction-block boundaries; crop the decoded output
   back to the original dimensions. Padding is included in compressed bytes but
   excluded from distortion measurements.
2. I-block prediction is a constant 128 (level shift). This is the proposal's
   transform baseline, not H.264 directional intra prediction.
3. P-block motion is an integer `(dy,dx)` offset into the previous **reconstructed**
   frame. Exhaustive SAD is computed on luma; ties favor shorter displacement.
   References remain within the padded frame. This closed loop avoids drift.
4. Apply 8x8 DCT to prediction residuals, divide by a uniform scalar step, and
   round to nearest with NumPy's ties-to-even rounding. Zigzag scan and encode
   nonzero count plus `(zero run, signed level)` pairs. Trailing zeros are implicit.
5. Compare serialized intra and inter syntax lengths, retaining the shorter.
   This is a rate proxy, not full rate-distortion optimization or exact
   arithmetic-coded bit cost. Scene changes can fall back to intra blocks.
6. Skip applies only when the chosen vector is zero and all residual coefficients
   quantize to zero. The decoder copies the colocated reference block.
7. Arithmetic-code all syntax bytes with eight adaptive binary contexts (one per
   byte bit position). This is an original simple coder, not H.264 CABAC, and
   does not use zlib compression. zlib is used solely for CRC32 checksums.
8. Inverse-quantize, inverse-transform, add prediction, round/clip to uint8, and
   use that reconstructed frame as the next reference in both encoder and decoder.

The H.264 slides' closed-loop architecture and intra/inter distinction, HEVC's
similar hybrid framework, and Motion-1's exhaustive block matching inform the
implementation. Motion-2's hierarchical search and the standards' fractional
motion, variable partitions, deblocking, and advanced entropy models are future
options, not necessary for the November 15 baseline.

## Metrics and experimental interpretation

PSNR uses MSE over every original sample in the clip (all RGB planes if color),
not an average of frame PSNR values. Zero error gives infinite PSNR. Bitrate is
`8 * complete_file_bytes * fps / frame_count`, including headers, vectors,
mode flags, checksums, and entropy-coded residuals. Bits/pixel counts spatial
pixels, even for RGB; raw compression ratio counts every source channel byte.

BD-Rate uses a classical cubic fit to log bitrate versus PSNR, integrated only
over the shared quality interval. It requires at least four finite, distinct,
monotonic RD points in each curve. Invalid curves are reported as unavailable,
not extrapolated. Cubic interpolation can overshoot; inspect RD plots and
consider a validated monotone-interpolation implementation for the final report.

The supplied deterministic demo is a correctness smoke test, not a benchmark
dataset. It contains a mostly static textured background and a moving square,
which favors inter coding. No general compression claim follows from this clip.

See `docs/STATUS_2026-11-15.md` for milestone mapping and outstanding evaluation.
See `docs/BITSTREAM.md` for the binary format.

## Final evaluation

[Final report](experiments/Final-Report.md) and [architecture and block diagrams](experiments/Architecture.md). The report links the measured results, rate-distortion plots, scene-cut analysis and reproduction commands.

## Instructor luminance benchmark

[Raw Y commands and version-2 design](docs/LUMINANCE.md) · [First 100 Foreman frames, results](experiments/Final-Report.md#instructor-iteration-0-comparison)

## bvc: block-based hybrid video coder

`bvc/` is a self-contained package that does not use `en424`. It implements the
hybrid coder from the 09_Video_coding slides and `bvc/Codec.ipynb`, built from the
lab helper files in `bvc/basic colab setup/` (`motion.py`, `dct_utils.py`,
`arith_utils.py`, `lab2_arith.py`, `DCT.py`). Input is raw 8-bit luma (`.y`),
QCIF 176x144 by default; `input/foreman_qcif.y` (400 frames) is included.

Per frame:

    Current frame -(+)-> MCPR -> 8x8 DCT -> Q (uniform step) -> arithmetic coding -> .bvc
    Q -> Q^-1 -> IDCT -(+ prediction)-> reconstructed frame (next frame's reference)
    previous reconstructed + current -> motion estimation (full search SAD) -> MV -> motion compensation
    MV -> arithmetic coding -> .bvc

The first frame is predicted as flat 128 (effectively intra); every later frame is
predicted from the previous *reconstructed* frame, so encoder and decoder never drift.

### Commands (run from this folder)

```powershell
python -m bvc encode input/foreman_qcif.y output/foreman.bvc --step 16
python -m bvc decode output/foreman.bvc output/foreman_dec.y --ref input/foreman_qcif.y
python -m bvc benchmark input/foreman_qcif.y --frames 30
```

- `encode`: options `--step` (Q step, default 16), `--block-size` (16), `--search-range` (7),
  `--frames` (default all), `--height/--width` (144/176), `--fps` (30). Writes the `.bvc`
  file plus `<name>_enc_recon.y` (encoder reconstruction) and `<name>_frames.csv`
  (bytes and PSNR per frame). Rate reported = 8 x the whole `.bvc` file size.
- `decode`: uses only the `.bvc` file (size, step and block size are in its header) and
  writes a raw `.y` clip. `--ref` prints PSNR against the original.
- `benchmark`: codes the clip with the hybrid coder (`--steps`, default 48 32 20 12 8)
  and with an intra-only DCT anchor (`DCT.py`'s `image_dct_enc`, `--qualities`,
  default 10 25 50 75 90), prints BD-rate / BD-PSNR and saves `output/bd_rd_curve.png`.
  Both codecs are charged for their headers and histograms.

From a notebook: `from bvc.__main__ import main; main(['benchmark', 'input/foreman_qcif.y', '--show'])`.

### .bvc file format

    header   '<4sHHHHH'  magic b'BVC1', height, width, n_frames, step, block_size
    per frame: MV chunk (mvx then mvy), then residual chunk (quantized coefficients)
    chunk:   int32 offset (min value), uint32 n, uint32 counts[n] (histogram),
             uint32 nbytes, nbytes of arithmetic-coded bits (packed MSB first)

### Checking a decode

The decoded clip must equal the encoder's reconstruction byte for byte:

```powershell
python -c "import numpy as np; print(np.array_equal(np.fromfile('output/foreman_enc_recon.y',np.uint8), np.fromfile('output/foreman_dec.y',np.uint8)))"
```

To view `.y` files in YUView, copy to a `.yuv` name with the size in it (for example
`foreman_dec_176x144.yuv`) and set the format to 4:0:0 (luma only), 8-bit.

### Results (foreman QCIF)

| Run | Result |
| --- | --- |
| Full clip, step 16 | 1,271,693 bytes, 763 kbps @ 30 fps, 1.00 bpp, avg PSNR 35.58 dB; decode identical to encoder |
| Benchmark, 30 frames | BD-rate -55.6 %, BD-PSNR +5.46 dB (hybrid vs intra DCT anchor) |

### Module map

| Module | Responsibility |
| --- | --- |
| `bvc/codec.py` | Hybrid encoder/decoder, `.bvc` format, intra DCT anchor, RD points |
| `bvc/motion.py` | Full-search block motion estimation and motion compensation |
| `bvc/transform.py` | 8x8 DCT/IDCT, MATLAB-style rounding, JPEG tables (from `dct_utils.py`) |
| `bvc/entropy.py` | Static arithmetic encoder/decoder, `.bvc` chunk read/write |
| `bvc/media.py` | Raw `.y` frame read/write |
| `bvc/metrics.py` | PSNR, kbps, BD-rate / BD-PSNR |
| `bvc/__main__.py` | Command line: `encode`, `decode`, `benchmark` |

Not implemented yet: per-block intra/inter mode decision, periodic I-frames (GOP),
deadzone quantization.
