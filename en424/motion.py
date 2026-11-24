"""Integer-pixel exhaustive block matching with SAD and bounded references."""
import numpy as np


def estimate(current, reference, y, x, block_size=16, radius=4):
    """Return (dy, dx), where prediction is reference[y+dy, x+dx].

    Both inputs are 2D luma arrays. Ties prefer the shortest displacement,
    then dy/dx, ensuring static/flat regions use the zero vector.
    """
    h, w = reference.shape
    target = current[y:y + block_size, x:x + block_size].astype(np.float64)
    best = None
    for dy in range(max(-radius, -y), min(radius, h - block_size - y) + 1):
        for dx in range(max(-radius, -x), min(radius, w - block_size - x) + 1):
            candidate = reference[y + dy:y + dy + block_size, x + dx:x + dx + block_size]
            sad = float(np.abs(target - candidate.astype(np.float64)).sum())
            score = (sad, abs(dy) + abs(dx), dy, dx)
            if best is None or score < best:
                best = score
    return best[2], best[3]


def compensate(reference, y, x, dy, dx, block_size):
    h, w = reference.shape[:2]
    if not (0 <= y + dy <= h - block_size and 0 <= x + dx <= w - block_size):
        raise ValueError("Motion vector points outside reference")
    return reference[y + dy:y + dy + block_size, x + dx:x + dx + block_size].astype(np.float64)


def half_prediction(reference, y, x, dy, dx, size):
    """Bilinear half-pixel predictor. Signed vectors use half-sample units."""
    iy, fy = divmod(dy, 2)
    ix, fx = divmod(dx, 2)
    block = compensate(reference, y, x, iy, ix, size)
    if fx:
        block += compensate(reference, y, x, iy, ix + 1, size)
    if fy:
        block += compensate(reference, y, x, iy + 1, ix, size)
        if fx:
            block += compensate(reference, y, x, iy + 1, ix + 1, size)
    return block / ((1 + fx) * (1 + fy))


def refine(current, reference, y, x, size, radius, vector):
    """Search the eight half-pixel neighbors of the winning integer vector."""
    h, w = reference.shape
    target = current[y:y + size, x:x + size]
    cy, cx = (2 * v for v in vector)
    best = None
    for dy in range(max(cy-1, -2*radius, -2*y), min(cy+1, 2*radius, 2*(h-size-y))+1):
        for dx in range(max(cx-1, -2*radius, -2*x), min(cx+1, 2*radius, 2*(w-size-x))+1):
            prediction = half_prediction(reference, y, x, dy, dx, size)
            score = (np.abs(target-prediction).sum(), abs(dy)+abs(dx), dy, dx)
            if best is None or score < best:
                best = score
    return best[2], best[3]
