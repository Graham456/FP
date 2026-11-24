"""Full-search block motion estimation and motion compensation (from motion.py)."""
import numpy as np


def motion_estimation(prev, curr, blkx, blky, search_range):
    """Find one motion vector per blkx x blky block of curr by exhaustive SAD search.

    prev is the previous *reconstructed* frame (what the decoder will also have),
    so encoder and decoder predictions never drift apart. Candidates that would
    reach outside prev are skipped. Returns (mvx, mvy), one entry per block.
    """
    h, w = curr.shape
    prev_h, prev_w = prev.shape
    num_blocks_y, num_blocks_x = h // blky, w // blkx
    mvx = np.zeros((num_blocks_y, num_blocks_x), dtype=np.int32)
    mvy = np.zeros((num_blocks_y, num_blocks_x), dtype=np.int32)

    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            y, x = i * blky, j * blkx
            block_curr = curr[y:y + blky, x:x + blkx]
            min_sad, best_dx, best_dy = float('inf'), 0, 0

            # Try every displacement in [-search_range, search_range]^2
            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    py, px = y + dy, x + dx
                    if py < 0 or px < 0 or py + blky > prev_h or px + blkx > prev_w:
                        continue                     # candidate block leaves the frame
                    block_prev = prev[py:py + blky, px:px + blkx]
                    sad = np.sum(np.abs(block_curr.astype(np.int16) - block_prev.astype(np.int16)))
                    if sad < min_sad:                # first minimum wins ties
                        min_sad, best_dx, best_dy = sad, dx, dy

            mvx[i, j], mvy[i, j] = best_dx, best_dy
    return mvx, mvy


def mc_prediction(prev, curr, mvx, mvy):
    """Build the motion-compensated prediction and the residual MCPR = curr - prediction.

    The block size is inferred from the frame size and the number of vectors.
    Returns (mcpr, pred).
    """
    h, w = curr.shape
    num_blocks_y, num_blocks_x = mvx.shape
    blky, blkx = h // num_blocks_y, w // num_blocks_x
    mcpr = np.zeros((h, w))
    pred = np.zeros((h, w))

    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            y, x = i * blky, j * blkx
            py, px = y + mvy[i, j], x + mvx[i, j]           # matching block in prev
            block_prev = prev[py:py + blky, px:px + blkx]
            pred[y:y + blky, x:x + blkx] = block_prev
            mcpr[y:y + blky, x:x + blkx] = curr[y:y + blky, x:x + blkx] - block_prev
    return mcpr, pred
