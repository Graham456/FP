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
