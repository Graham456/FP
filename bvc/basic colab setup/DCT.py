import os
import pickle
import struct

import numpy as np

from arith_utils import arith_encode, arith_decode
from dct_utils import (init_jpeg, dct2, idct2, matlab_round, imread, imwrite, uint8,
                       im2col, col2im, matlab_sum, mean, log10, save, load)

from lab2_arith import enc_arith, dec_arith

def video_dct_enc(in_frames, bitfile, quality, iframe_interval=10):
    """
    Encodes a sequence of video frames using 8x8 DCT, quantization, and arithmetic coding.
    Supports both Intra-frames (I-frames) and Predicted frames (P-frames) using temporal difference.
    
    Parameters:
    - in_frames: List of 2D numpy arrays representing grayscale video frames.
    - bitfile: Destination path for the compressed bitstream.
    - quality: JPEG-like quality parameter (1-100).
    - iframe_interval: Keyframe (I-frame) interval.
    """
    # 1. Initialize JPEG quantization table and zigzag scan
    qt, zag = init_jpeg(quality)
    qt = np.maximum(np.asarray(qt).flatten(), 1)
    zag = np.asarray(zag)
    zag_f = zag.flatten(order='F').astype(int) - 1

    num_frames = len(in_frames)
    if num_frames == 0:
        return 0

    h, w = in_frames[0].shape
    num_blocks = (h // 8) * (w // 8)

    # All block-level symbols for the entire video will be concatenated here
    video_symbols = []
    frame_types = []  # Record frame types: 0 for I-frame, 1 for P-frame

    prev_frame = None

    for idx, frame in enumerate(in_frames):
        is_iframe = (idx % iframe_interval == 0)
        frame_types.append(0 if is_iframe else 1)

        # Frame preparation (Subtraction/Residual coding)
        curr_img = frame.astype(np.float64)
        if is_iframe:
            # Center pixel values around zero for I-frames
            proc_img = curr_img - 128
        else:
            # Temporal prediction error (Residual) for P-frames
            proc_img = curr_img - prev_frame

        prev_frame = curr_img.copy()

        # Segment and process 8x8 blocks
        imgq = np.zeros((num_blocks, 64))
        block_index = 0

        for i in range(0, h, 8):
            for j in range(0, w, 8):
                blk = proc_img[i:i+8, j:j+8]
                blk_dct = dct2(blk)
                
                scan = np.empty(64)
                scan[zag_f] = blk_dct.flatten(order='F')
                
                blk_quant = matlab_round(scan / qt)
                imgq[block_index, :] = blk_quant
                block_index += 1

        # Apply DPCM on DC coefficients (only for I-frames)
        if is_iframe:
            dc_coeffs = imgq[:, 0].copy()
            imgq[1:, 0] = np.diff(dc_coeffs)

        video_symbols.append(imgq)

    # Concatenate all frame quantized matrices
    all_imgq = np.vstack(video_symbols)

    # Shift symbols to positive integers for arithmetic coding
    min_index = int(np.min(all_imgq))
    all_imgq = all_imgq - min_index + 1
    all_imgq = all_imgq.astype(np.int64).flatten(order='C')
    symbol_list = [int(x) for x in all_imgq]

    # Write video-specific header
    with open('video_header.hdr', 'wb') as f:
        f.write(struct.pack(
            '<HHHBh',
            int(h),
            int(w),
            int(num_frames),
            int(quality),
            min_index
        ))
        # Write frame configuration payload
        f.write(struct.pack(f'<{num_frames}B', *frame_types))

    # Generate and save histogram
    alpha_size = max(2, max(symbol_list))
    hist = np.bincount(
        np.asarray(symbol_list, dtype=np.int64),
        minlength=alpha_size + 1
    )[1:alpha_size + 1]
    hist = np.maximum(hist, 1).astype(int).tolist()

    with open('dct_hist.pkl', 'wb') as f:
        pickle.dump(hist, f)

    # Arithmetic encoding
    Nbits = enc_arith(symbol_list, 'dct_hist.pkl', bitfile)
    return Nbits

def video_dct_dec(bitfile, out_video_path=None):
    """
    Decodes a compressed video bitstream from bitfile.
    
    Parameters:
    - bitfile: Path to the compressed bitstream.
    - out_video_path: (Optional) Path to save the decoded frames (e.g. as a pickle or directory of BMPs).
    
    Returns:
    - out_frames: List of 2D numpy arrays representing decoded grayscale frames.
    """
    # 1. Read video-specific header
    with open('video_header.hdr', 'rb') as f:
        h, w, num_frames, quality, min_index = struct.unpack('<HHHBh', f.read(9))
        frame_types = list(struct.unpack(f'<{num_frames}B', f.read(num_frames)))

    # 2. Arithmetic decoding
    decoded_symbols = dec_arith('dct_hist.pkl', bitfile)
    all_imgq = np.asarray(decoded_symbols, dtype=np.int64)

    # 3. Reshape and shift back
    num_blocks = (h // 8) * (w // 8)
    all_imgq = all_imgq.reshape((num_frames * num_blocks, 64), order='C')
    all_imgq = all_imgq + min_index - 1

    # 4. Initialize JPEG structures
    qt, zag = init_jpeg(quality)
    qt = np.maximum(np.asarray(qt).flatten(), 1)
    zag = np.asarray(zag)
    zag_f = zag.flatten(order='F').astype(int) - 1

    out_frames = []
    prev_frame = None

    for idx in range(num_frames):
        is_iframe = (frame_types[idx] == 0)
        start_row = idx * num_blocks
        end_row = start_row + num_blocks
        imgq = all_imgq[start_row:end_row, :].copy()

        # Reverse DPCM on DC coefficients (only for I-frames)
        if is_iframe:
            imgq[:, 0] = np.cumsum(imgq[:, 0])

        # Reconstruct spatial domain residual/frame
        proc_img = np.zeros((h, w), dtype=np.float64)
        block_index = 0

        for i in range(0, h, 8):
            for j in range(0, w, 8):
                blk_zigzag = imgq[block_index, :]
                scan = blk_zigzag * qt
                
                blk_dct = scan[zag_f].reshape((8, 8), order='F')
                blk_idct = idct2(blk_dct)
                
                proc_img[i:i+8, j:j+8] = blk_idct
                block_index += 1

        # Reconstruct final pixel values based on frame type
        if is_iframe:
            curr_frame = proc_img + 128
        else:
            curr_frame = proc_img + prev_frame

        # Store reconstructed frame and clip boundaries
        curr_frame_clipped = np.clip(np.rint(curr_frame), 0, 255).astype(np.uint8)
        out_frames.append(curr_frame_clipped)
        prev_frame = curr_frame_clipped.astype(np.float64)

    # Optionally save output video frames
    if out_video_path:
        with open(out_video_path, 'wb') as f:
            pickle.dump(out_frames, f)

    return out_frames

def image_dct_enc(infile, bitfile, quality):

    # 1. Initialize JPEG quantization table and zigzag scan
    qt, zag = init_jpeg(quality)

    qt = np.maximum(np.asarray(qt).flatten(), 1)
    zag = np.asarray(zag)

    zag_f = zag.flatten(order='F').astype(int) - 1

    # 2. Read image (or take an array directly) and convert to float64
    if isinstance(infile, np.ndarray):
        img = infile.astype(np.float64) - 128
    else:
        img, colormap = imread(infile)
        img = img.astype(np.float64) - 128

    h, w = img.shape

    # 3. Extract non-overlapping 8x8 blocks
    num_blocks = (h // 8) * (w // 8)

    imgq = np.zeros((num_blocks, 64))

    block_index = 0

    for i in range(0, h, 8):
        for j in range(0, w, 8):

            # Extract 8x8 block
            blk = img[i:i+8, j:j+8]

            # Apply 2D DCT
            blk_dct = dct2(blk)

            # MATLAB equivalent:
            # scan(zag(:)) = coeff(:)
            scan = np.empty(64)
            scan[zag_f] = blk_dct.flatten(order='F')

            # Quantization
            blk_quant = matlab_round(scan / qt)

            # Store coefficients
            imgq[block_index, :] = blk_quant

            block_index += 1

    # 4. DPCM on DC coefficients
    dc_coeffs = imgq[:, 0].copy()

    imgq[1:, 0] = np.diff(dc_coeffs)

    # 5. Shift coefficients to positive integers
    min_index = int(np.min(imgq))

    imgq = imgq - min_index + 1

    imgq = imgq.astype(np.int64)

    imgq = imgq.flatten(order='C')

    imgq = [int(x) for x in imgq]

    # 6. Write header
    with open('img_header.hdr', 'wb') as f:

        f.write(struct.pack(
            '<HHBh',
            int(h),
            int(w),
            int(quality),
            min_index
        ))

    # 7. Generate histogram
    alpha_size = max(2, max(imgq))

    hist = np.bincount(
        np.asarray(imgq, dtype=np.int64),
        minlength=alpha_size + 1
    )[1:alpha_size + 1]

    hist = np.maximum(hist, 1).astype(int).tolist()

    # Save histogram
    with open('dct_hist.pkl', 'wb') as f:
        pickle.dump(hist, f)

    # 8. Arithmetic encoding
    Nbits = enc_arith(imgq, 'dct_hist.pkl', bitfile)

    return Nbits

def image_dct_dec(bitfile, outfile):

    # 1. Read header information
    with open('img_header.hdr', 'rb') as f:
        header = struct.unpack('<HHBh', f.read(7))

    h = header[0]
    w = header[1]
    quality = header[2]
    min_index = header[3]

    # 2. Arithmetic decoding
    imgq = dec_arith('dct_hist.pkl', bitfile)

    imgq = np.asarray(imgq, dtype=np.int64)

    # 3. Reshape decoded symbols
    num_blocks = (h // 8) * (w // 8)

    imgq = imgq.reshape((num_blocks, 64), order='C')

    # 4. Reverse positive integer shift
    imgq = imgq + min_index - 1

    # 5. Reverse DPCM
    imgq[:, 0] = np.cumsum(imgq[:, 0])

    # 6. Initialize JPEG quantization and zigzag
    qt, zag = init_jpeg(quality)

    qt = np.maximum(np.asarray(qt).flatten(), 1)
    zag = np.asarray(zag)

    zag_f = zag.flatten(order='F').astype(int) - 1

    # 7. Reconstruct image blocks
    img_dec = np.zeros((h, w), dtype=np.float64)

    block_index = 0

    for i in range(0, h, 8):
        for j in range(0, w, 8):

            # Get quantized zigzag coefficients
            blk_zigzag = imgq[block_index, :]

            # Reverse quantization
            scan = blk_zigzag * qt

            blk_dct = scan[zag_f].reshape((8, 8), order='F')

            # Debug first block
            # if block_index == 0:
            #     print("First block coefficients:")
            #     print(blk_dct)

            # Inverse DCT
            blk_idct = idct2(blk_dct)

            img_dec[i:i+8, j:j+8] = blk_idct

            block_index += 1

    # 8. Restore pixel values
    img_dec = img_dec + 128

    img_dec = np.clip(
        np.rint(img_dec), 0, 255
    ).astype(np.uint8)

    # 9. Generate grayscale colormap
    x = np.linspace(0, 1, 256)

    map_gray = np.column_stack([x, x, x])

    # 10. Save decoded image
    imwrite(img_dec, map_gray, outfile, 'BMP')

    return img_dec