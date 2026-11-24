import os

import numpy as np
import matplotlib.pyplot as plt

from me_utils import uint8, mean, log10
def read_Yframe(infile, height, width, frame_index):
    with open(infile, "rb") as f:
        # Seek to the start of the target frame
        f.seek(height * width * (frame_index - 1))
        # Read raw bytes as 8-bit unsigned integers (1 byte per pixel)
        raw_data = np.frombuffer(f.read(height * width), dtype=np.uint8)
        # Convert to double (float64) and reshape to height x width
        frameY = raw_data.astype(np.float64).reshape(height, width)
    return frameY

def write_Yframe(frameY, outfile):
    # YOUR CODE HERE
    frameY = frameY.ravel()
    with open(outfile, 'ab') as f:
      f.write(uint8(frameY))
    return



def motion_estimation(prev, curr, blkx, blky, search_range):

    # 1). Determine frame dimensions
    h, w = curr.shape
    prev_h, prev_w = prev.shape

    # 2). Determine number of blocks
    num_blocks_y = h // blky
    num_blocks_x = w // blkx

    mvx = np.zeros((num_blocks_y, num_blocks_x), dtype=np.int32)
    mvy = np.zeros((num_blocks_y, num_blocks_x), dtype=np.int32)

    # 3). Process each block
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):

            # a). Extract current block
            y = i * blky
            x = j * blkx

            block_curr = curr[y:y+blky, x:x+blkx]

            # b). Search for the best matching block
            min_SAD = float('inf')
            best_dx = 0
            best_dy = 0

            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):

                    # Check that previous block is within frame boundaries
                    py = y + dy
                    px = x + dx

                    if (py < 0 or px < 0 or
                        py + blky > prev_h or
                        px + blkx > prev_w):
                        continue

                    block_prev = prev[py:py+blky, px:px+blkx]

                    # Calculate SAD
                    SAD = np.sum(np.abs(
                        block_curr.astype(np.int16) -
                        block_prev.astype(np.int16)
                    ))

                    # Keep the best matching block
                    if SAD < min_SAD:
                        min_SAD = SAD
                        best_dx = dx
                        best_dy = dy

            # c). Store motion vectors
            mvx[i, j] = best_dx
            mvy[i, j] = best_dy

    return mvx, mvy


def mc_prediction(prev, curr, mvx, mvy):
    # 1). Determine block size from frame and motion vector dimensions
    h, w = curr.shape
    num_blocks_y, num_blocks_x = mvx.shape
    blky = h // num_blocks_y
    blkx = w // num_blocks_x

    # 2). Initialize pred and mcpr
    mcpr = np.zeros((h, w))
    pred = np.zeros((h, w))

    # 3). Perform compensation block by block
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            # a). Current block coordinates
            y = i * blky
            x = j * blkx
            block_curr = curr[y:y+blky, x:x+blkx]

            # b). Retrieve motion vector displacement
            dx = mvx[i, j]
            dy = mvy[i, j]

            # Source location in previous frame
            py = y + dy
            px = x + dx

            # Extract the matching block from previous frame
            block_prev = prev[py:py+blky, px:px+blkx]

            # c). Store prediction
            pred[y:y+blky, x:x+blkx] = block_prev

            # d). Store residual
            mcpr[y:y+blky, x:x+blkx] = block_curr - block_prev

    return mcpr, pred

# Where to Get the "Previous Reconstructed Frames"?

# In video coding/compression standards (like H.264, HEVC, etc.), the previous reconstructed frame is used as the reference (prev) instead of the original uncompressed frame to avoid drift error between the encoder and decoder.

# In your current laboratory setup:

#     Currently: Your loop in cell 2a0c20ec reads the original previous frame directly from the source video file using read_Yframe("foreman_qcif.y", ..., f_idx) to use as reference.
#     To use reconstructed frames instead: You would load your previously saved prediction foreman_qcif_pred.y (or add the residual mcpr back to the prediction pred to simulate a reconstructed frame if there's no quantization, meaning reconstructed = pred + mcpr, which mathematically equals curr in this lossless scenario). If you introduce quantization to the residual (lossy compression), you must reconstruct the frame on the fly:

#     # Reconstructed frame in encoder loop:
#     reconstructed_frame = pred + quantized_residual

#     And then use this reconstructed_frame as the prev reference in the next loop iteration.
