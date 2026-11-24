"""Create a nearest-neighbor detail comparison for the final report."""
from pathlib import Path

import cv2
from PIL import Image, ImageDraw
from en424.codec import Config, decode, encode
from en424.media import load
import numpy as np

source = Path(r'C:\Program Files\MATLAB\R2025b\toolbox\matlab\audiovideo\xylophone.mp4')
frames, fps = load(source, limit=48, gray=True)
height, width = frames.shape[1:]
clip = np.stack([cv2.resize(f, (128, round(128*height/width)), interpolation=cv2.INTER_AREA)
                 for f in frames[::3]])
samples = [('Source', clip[-1])]
for step in (8, 64):
    stream, local, _ = encode(clip, Config(qstep=step, fps=fps/3))
    decoded, _ = decode(stream)
    np.testing.assert_array_equal(local, decoded)
    samples.append((f'Hybrid, step {step}', decoded[-1]))
canvas = Image.new('RGB', (3*272, clip.shape[1]*2+50), 'white')
draw = ImageDraw.Draw(canvas)
for index, (label, pixels) in enumerate(samples):
    draw.text((index*272+8, 8), label, fill='black')
    tile = Image.fromarray(pixels).resize((256, 2*len(pixels)), Image.Resampling.NEAREST)
    canvas.paste(tile, (index*272+8, 32))
canvas.save(Path(__file__).with_name('detail.png'))

