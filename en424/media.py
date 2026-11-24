"""NPY arrays, PNG sequences and optional OpenCV video adapters."""
from pathlib import Path
import numpy as np
from PIL import Image
from .codec import normalize_frames


def load(path, limit=None, gray=False, width=None, height=None):
    path = Path(path)
    fps = None
    if limit is not None and limit < 1:
        raise ValueError('Frame limit must be positive')
    if path.suffix.lower() == '.y':
        if width is None or height is None or min(width, height) < 1:
            raise ValueError('Raw .y input requires positive --width and --height')
        frame_size = width * height
        count, remainder = divmod(path.stat().st_size, frame_size)
        if count == 0 or remainder:
            raise ValueError('Raw .y size must contain complete 8-bit luminance frames')
        samples = min(limit or count, count) * frame_size
        return np.fromfile(path, dtype=np.uint8, count=samples).reshape(-1, height, width), None
    if width is not None or height is not None:
        raise ValueError('Input dimensions apply only to headerless .y files')
    if path.suffix.lower() == '.npy':
        frames = np.load(path, allow_pickle=False)
        frames = normalize_frames(frames)
        if limit:
            frames = frames[:limit]
    elif path.is_dir():
        names = sorted(path.glob('*.png'))
        if limit:
            names = names[:limit]
        if not names:
            raise ValueError('No PNG frames found')
        items = []
        for name in names:
            with Image.open(name) as im:
                items.append(np.asarray(im.convert('L' if gray else 'RGB')))
        frames = normalize_frames(np.stack(items))
    else:
        try:
            import cv2
        except ImportError as exc:
            raise ValueError('Video input requires: pip install -r requirements-video.txt') from exc
        cap = cv2.VideoCapture(str(path))
        items = []
        try:
            if not cap.isOpened():
                raise ValueError('Cannot open video: ' + str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            while limit is None or len(items) < limit:
                ok, frame = cap.read()
                if not ok:
                    break
                items.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        if not items:
            raise ValueError('Video contains no readable frames')
        frames = np.stack(items)
    if gray and frames.shape[-1] == 3:
        frames = np.clip(np.rint(frames @ np.array([.299, .587, .114])), 0, 255).astype(np.uint8)
    elif frames.shape[-1] == 1:
        frames = frames[..., 0]
    return frames, fps


def save(path, frames):
    path = Path(path)
    if path.suffix.lower() == '.y':
        if frames.ndim != 3 or frames.dtype != np.uint8:
            raise ValueError('Raw .y output requires 8-bit grayscale frames')
        path.parent.mkdir(parents=True, exist_ok=True)
        frames.tofile(path)
        return
    if path.suffix.lower() == '.npy':
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, frames, allow_pickle=False)
    else:
        path.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(path / f'{i:05d}.png')


def demo_frames(count=12, height=64, width=80):
    """Deterministic textured background plus moving object (smoke test only)."""
    rng = np.random.default_rng(424)
    y, x = np.indices((height, width))
    background = np.clip(40 + x + y / 2 + rng.normal(0, 5, (height, width)), 0, 255)
    frames = []
    for i in range(count):
        frame = background.copy()
        left = 4 + (2 * i) % (width - 24)
        frame[20:36, left:left + 16] = 220
        frames.append(frame.astype(np.uint8))
    return np.stack(frames)
