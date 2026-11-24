"""Block-based hybrid video coder from Codec.ipynb (09_Video_coding slide 8), plus the intra anchor.

Hybrid loop, per frame:
    Current frame -(+)-> MCPR -> DCT -> Q -> entropy coding -> bitstream
    Q -> Q^-1 -> IDCT -(+ prediction)-> reconstructed frame (next frame's reference)
    previous reconstructed + current -> motion estimation -> MV -> motion compensation
    MV -> entropy coding

Intra anchor: every frame coded on its own like JPEG (DCT, quality-scaled
quantization table, DPCM on DC), from DCT.py's image_dct_enc.

.bvc file layout (everything the decoder needs, nothing else):
    header   '<4sHHHHH'  magic b'BVC1', height, width, n_frames, step, block_size
    frame 1  MV chunk, residual chunk      (chunk format: see entropy.write_chunk)
    frame 2  MV chunk, residual chunk
    ...
"""
from dataclasses import dataclass
import csv
import io
import os
import pickle
import struct
import tempfile

import numpy as np

from .entropy import enc_arith, read_chunk, write_chunk
from .media import count_frames, read_Yframe, write_Yframe
from .metrics import psnr
from .motion import motion_estimation, mc_prediction
from .transform import dct2, idct2, init_jpeg, matlab_round


@dataclass(frozen=True)
class Config:
    height: int = 144            # QCIF luma
    width: int = 176
    step: int = 16               # Q step size (bigger = fewer bits, lower quality)
    block_size: int = 16         # motion estimation block size
    search_range: int = 7        # motion search range, in pixels

    def validate(self):
        if self.height % self.block_size or self.width % self.block_size or self.block_size % 8:
            raise ValueError('height and width must be multiples of block_size, '
                             'and block_size a multiple of 8')
        if not 1 <= self.step <= 65535:
            raise ValueError('step must be 1..65535')


MAGIC = b'BVC1'
HEADER = struct.Struct('<4sHHHHH')   # magic, height, width, n_frames, step, block_size


# ---------------------------------------------------------------------------
# Hybrid (ME/MC + DCT) coder
# ---------------------------------------------------------------------------

def reconstruct_frame(q, pred, step):
    """Q^-1 -> IDCT on 8x8 blocks, + prediction, clip: the decoder's view of a frame.

    The encoder and the decoder both call this, so their reconstructions are
    identical and the next frame's reference never drifts.
    """
    h, w = q.shape
    res_rec = np.zeros((h, w))
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            res_rec[i:i + 8, j:j + 8] = idct2(q[i:i + 8, j:j + 8] * step)
    return np.clip(np.round(pred + res_rec), 0, 255)


def encode_frame(curr, prev_recon, config):
    """Code one frame against the previous reconstructed frame (None for the first frame).

    Returns a dict with the quantized coefficients q, the motion vectors
    mvx/mvy, the prediction, the residual mcpr and the reconstruction recon.
    """
    h, w, blk, step = config.height, config.width, config.block_size, config.step

    # Motion estimation -> MV -> motion compensation -> prediction; MCPR = current - prediction
    if prev_recon is None:       # first frame: no reference yet, predict mid-gray
        mvx = np.zeros((h // blk, w // blk), dtype=np.int32)
        mvy = np.zeros_like(mvx)
        pred = np.full((h, w), 128.0)
        mcpr = curr - pred
    else:
        mvx, mvy = motion_estimation(prev_recon, curr, blk, blk, config.search_range)
        mcpr, pred = mc_prediction(prev_recon, curr, mvx, mvy)

    # DCT -> Q on 8x8 blocks
    q = np.zeros((h, w))
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            q[i:i + 8, j:j + 8] = matlab_round(dct2(mcpr[i:i + 8, j:j + 8]) / step)

    # Q^-1 -> IDCT -> + prediction = reconstructed frame (closed loop)
    recon = reconstruct_frame(q, pred, step)
    return dict(q=q, mvx=mvx, mvy=mvy, pred=pred, mcpr=mcpr, recon=recon)


def write_frame(f, coded):
    """Entropy-code a frame's motion vectors then residual into f; returns (bytes_mv, bytes_res)."""
    mv = np.concatenate([coded['mvx'].ravel(), coded['mvy'].ravel()])
    return write_chunk(f, mv), write_chunk(f, coded['q'])


def encode(infile, outfile, config=Config(), n_frames=None, verbose=True):
    """Hybrid-code a raw .y clip into one .bvc file (the notebook's block-based video coder cell).

    Also writes, next to outfile: <name>_enc_recon.y (the encoder's own
    reconstruction, which decode() must reproduce exactly) and <name>_frames.csv.
    Returns dict(bytes_mv, bytes_res, psnr) per frame and total_bytes (file size).
    """
    config.validate()
    total_frames = count_frames(infile, config.height, config.width)
    n_frames = total_frames if n_frames is None else min(n_frames, total_frames)
    if n_frames < 1:
        raise ValueError('No frames to encode')
    stem = os.path.splitext(outfile)[0]
    os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
    recon_file = stem + '_enc_recon.y'
    if os.path.exists(recon_file):
        os.remove(recon_file)    # write_Yframe appends

    prev_recon = None            # "Previous reconstructed frame(s)"
    bytes_mv, bytes_res, psnrs = [], [], []
    with open(outfile, 'wb') as f:
        f.write(HEADER.pack(MAGIC, config.height, config.width, n_frames,
                            config.step, config.block_size))
        for k in range(1, n_frames + 1):
            curr = read_Yframe(infile, config.height, config.width, k)
            coded = encode_frame(curr, prev_recon, config)
            m, r = write_frame(f, coded)
            bytes_mv.append(m)
            bytes_res.append(r)
            write_Yframe(coded['recon'], recon_file)
            prev_recon = coded['recon']
            psnrs.append(psnr(curr, prev_recon))
            if verbose and (k % 50 == 0 or k == n_frames):
                print(f'frame {k}/{n_frames}: {8 * r} + {8 * m} bits, PSNR {psnrs[-1]:.2f} dB',
                      flush=True)

    with open(stem + '_frames.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['frame', 'bytes_residual', 'bytes_mv', 'psnr_db'])
        wr.writerows(zip(range(1, n_frames + 1), bytes_res, bytes_mv, psnrs))
    return dict(bytes_mv=bytes_mv, bytes_res=bytes_res, psnr=psnrs,
                total_bytes=os.path.getsize(outfile))


def decode(infile, outfile, verbose=True):
    """Decode a .bvc file into a raw .y clip, using only the .bvc file. Returns (height, width, n_frames)."""
    if os.path.exists(outfile):
        os.remove(outfile)       # write_Yframe appends
    with open(infile, 'rb') as f:
        head = f.read(HEADER.size)
        if len(head) != HEADER.size:
            raise ValueError('Not a .bvc file (too short)')
        magic, h, w, n_frames, step, blk = HEADER.unpack(head)
        if magic != MAGIC:
            raise ValueError('Not a .bvc file (bad magic)')
        Config(h, w, step, blk).validate()
        nby, nbx = h // blk, w // blk

        prev_recon = None
        for k in range(1, n_frames + 1):
            # Same order the encoder wrote: motion vectors, then quantized residual
            mv = read_chunk(f, 2 * nby * nbx)
            mvx = mv[:nby * nbx].reshape(nby, nbx)
            mvy = mv[nby * nbx:].reshape(nby, nbx)
            q = read_chunk(f, h * w).reshape(h, w).astype(np.float64)

            # Prediction: mid-gray for the first frame, else motion compensation of the
            # previous reconstruction (mc_prediction's pred does not depend on curr)
            if prev_recon is None:
                pred = np.full((h, w), 128.0)
            else:
                _, pred = mc_prediction(prev_recon, np.zeros((h, w)), mvx, mvy)

            prev_recon = reconstruct_frame(q, pred, step)
            write_Yframe(prev_recon, outfile)
            if verbose and (k % 50 == 0 or k == n_frames):
                print(f'decoded frame {k}/{n_frames}', flush=True)
        if f.read(1):
            raise ValueError('Extra data after the last frame')
    return h, w, n_frames


def hybrid_point(frames, config):
    """One RD point for the hybrid coder: (total bits, mean PSNR) over frames.

    Same coder as encode(), written to memory; bits = 8 * the .bvc file size,
    so the header and every histogram are counted.
    """
    config.validate()
    buf, psnrs, prev = io.BytesIO(), [], None
    buf.write(HEADER.pack(MAGIC, config.height, config.width, len(frames),
                          config.step, config.block_size))
    for curr in frames:
        coded = encode_frame(curr, prev, config)
        write_frame(buf, coded)
        prev = coded['recon']
        psnrs.append(psnr(curr, prev))
    return 8 * buf.tell(), float(np.mean(psnrs))


# ---------------------------------------------------------------------------
# Intra DCT anchor (DCT.py's image_dct_enc)
# ---------------------------------------------------------------------------

def image_dct_enc(img, out_dir, quality):
    """JPEG-like intra coding of one frame; returns (bits, reconstructed frame).

    Writes img_header.hdr, dct_hist.pkl and frame.bit in out_dir (the original
    wrote them into the working folder).
    """
    # Quantization table and zig-zag scan for this quality
    qt, zag = init_jpeg(quality)
    qt = np.maximum(np.asarray(qt).flatten(), 1)
    zag_f = np.asarray(zag).flatten(order='F').astype(int) - 1

    img = np.asarray(img, dtype=np.float64) - 128          # level shift to around zero
    h, w = img.shape
    imgq = np.zeros(((h // 8) * (w // 8), 64))
    rec = np.zeros((h, w))
    block_index = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            # DCT, reorder into zig-zag (MATLAB: scan(zag(:)) = coeff(:)), quantize
            scan = np.empty(64)
            scan[zag_f] = dct2(img[i:i + 8, j:j + 8]).flatten(order='F')
            imgq[block_index] = matlab_round(scan / qt)
            # Decoder's view of this block: dequantize, undo zig-zag, IDCT
            rec[i:i + 8, j:j + 8] = idct2((imgq[block_index] * qt)[zag_f].reshape((8, 8), order='F'))
            block_index += 1
    recon = np.clip(np.rint(rec + 128), 0, 255)

    # DPCM on the DC coefficients (each DC sent as the difference from the previous one)
    imgq[1:, 0] = np.diff(imgq[:, 0])

    # Shift to symbols >= 1 for the arithmetic coder
    min_index = int(np.min(imgq))
    symbols = [int(x) for x in (imgq - min_index + 1).astype(np.int64).flatten(order='C')]

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'img_header.hdr'), 'wb') as f:
        f.write(struct.pack('<HHBh', h, w, int(quality), min_index))

    # Histogram over 1..max symbol (at least 2 symbols), zero counts raised to 1
    alpha_size = max(2, max(symbols))
    hist = np.bincount(np.asarray(symbols, dtype=np.int64), minlength=alpha_size + 1)[1:alpha_size + 1]
    histfile = os.path.join(out_dir, 'dct_hist.pkl')
    with open(histfile, 'wb') as f:
        pickle.dump(np.maximum(hist, 1).astype(int).tolist(), f)

    bits = enc_arith(symbols, histfile, os.path.join(out_dir, 'frame.bit'))
    # Count the side information the decoder needs as well, sized like a .bvc chunk
    # (7-byte header, histogram size + counts as uint32, bitstream length), so the
    # anchor and the hybrid coder are charged for the same things in the BD-rate.
    side_bits = 8 * (struct.calcsize('<HHBh') + 4 + 4 * len(hist) + 4)
    return bits + side_bits, recon


def intra_point(frames, quality):
    """One RD point for the intra anchor: (total bits, mean PSNR) over frames."""
    bits, psnrs = 0, []
    with tempfile.TemporaryDirectory() as tmp:
        for f in frames:
            b, recon = image_dct_enc(f, tmp, quality)
            bits += b
            psnrs.append(psnr(f, recon))
    return bits, float(np.mean(psnrs))
