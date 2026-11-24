"""Check recovery across a hard cut inside a GOP using two recorded sequences."""
from pathlib import Path
import csv
import cv2
import numpy as np
from en424.media import load
from en424.codec import Config, encode, decode

root = Path(r'C:\Program Files\MATLAB\R2025b\toolbox\matlab')
parts = []
for source in (root/'demos/shuttle.avi', root/'audiovideo/xylophone.mp4'):
    frames, _ = load(source, limit=8, gray=True)
    parts.extend(cv2.resize(f, (128, 96), interpolation=cv2.INTER_AREA) for f in frames)
clip = np.stack(parts)
stream, local, stats = encode(clip, Config(qstep=16))
decoded, _ = decode(stream)
np.testing.assert_array_equal(local, decoded)
with Path(__file__).with_name('cut.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=stats[0])
    writer.writeheader()
    writer.writerows(stats)
print('Verified hard cut at frame 8:', stats[7:10])

