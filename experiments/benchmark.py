"""Run from the FP root with .venv/Scripts/python.exe -m experiments.benchmark."""
import csv
import os
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from en424.codec import Config, encode, decode
from en424.media import load
from en424.metrics import measure, bd_rate

DEFAULT = Path(r'C:\Program Files\Wondershare\Wondershare DemoCreator 8\UIImages\ugMedia')
OUT = Path(__file__).resolve().parent / os.environ.get('EN424_RESULTS', 'results')

def run(source):
    frames, fps = load(source, limit=48, gray=True)
    if len(frames) < 48 or not fps:
        raise ValueError(f'Expected 48 frames and a valid frame rate in {source}')
    height, width = frames.shape[1:]
    clip = np.stack([cv2.resize(frame, (128, round(height*128/width)), interpolation=cv2.INTER_AREA)
                     for frame in frames[::3]])
    rows = []
    for mode in ('intra', 'hybrid', 'no_skip', 'radius2', 'radius8'):
        for q in (8, 16, 32, 64):
            config = Config(qstep=q, fps=fps/3, intra_only=mode=='intra', skip=mode!='no_skip',
                            search_radius={'radius2':2,'radius8':8}.get(mode,4))
            start = perf_counter()
            data, local, stats = encode(clip, config)
            seconds = perf_counter()-start
            decoded, _ = decode(data)
            if not np.array_equal(local, decoded):
                raise RuntimeError('Decoder drift detected')
            rows.append(dict(clip=source.stem,mode=mode,q=q,frames=len(clip),width=128,height=clip.shape[1],
                             fps=config.fps,seconds=seconds,skip_blocks=sum(f['skip'] for f in stats),
                             **measure(clip, decoded, len(data),config.fps)))
    return rows

def main():
    sources = list(map(Path,sys.argv[1:])) or [DEFAULT/name for name in ('PanZoom.mp4','Template.mp4','textEditing.mp4')]
    OUT.mkdir(exist_ok=True)
    rows = []
    report = ['# Codec evaluation','', 'Source videos are decoded by OpenCV. Each trial uses source frames 0, 3, …, 45, converted to grayscale and area-resized to 128 pixels wide. The playback rate is one third of the original. All modes see identical pixels. Each sequence spans two GOPs (I frames at indices 0 and 12).', '', '| Clip | Configuration | BD-Rate vs intra | Mean encode time (s/clip) | Skip blocks across sweep |','|---|---|---:|---:|---:|']
    for source in sources:
        trial = run(source)
        rows.extend(trial)
        for mode in ('intra','hybrid','no_skip','radius2','radius8'):
            points = [r for r in trial if r['mode']==mode]
            curve = [(r['bitrate_bps'],r['psnr_db']) for r in points]
            if mode=='intra':
                anchor = curve
            try:
                delta = f'{bd_rate(anchor,curve):.2f}%'
            except ValueError as e:
                delta = str(e)
            report.append(f"| {source.name} | {mode} | {delta} | {np.mean([r['seconds'] for r in points]):.3f} | {sum(r['skip_blocks'] for r in points)} |")
        print(source.name, 'verified', flush=True)
    with (OUT/'measurements.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    report += ['',f'All {len(rows)} independently decoded streams matched the encoder reconstruction pixel for pixel. Timing is one encode per operating point and excludes preprocessing and decode. BD-Rate integrates a cubic fit over overlapping PSNR only. Short, downscaled grayscale excerpts limit extrapolation to full-resolution color footage.','', '![Measured bitrate and PSNR](curves.svg)']
    (OUT/'Evaluation.md').write_text('\n'.join(report),encoding='utf8')
    from xml.sax.saxutils import escape
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="{len(sources)*245}"><rect width="100%" height="100%" fill="white"/>']
    for i,source in enumerate(sources):
        base = i*245
        group = [r for r in rows if r['clip']==source.stem and r['mode'] in ('intra','hybrid')]
        limit = np.ceil(max(r['bitrate_bps'] for r in group)/100000)*100
        lo = int(min(r['psnr_db'] for r in group))-1
        hi = int(max(r['psnr_db'] for r in group))+1
        def label(x,y,t):
            svg.append(f'<text x="{x}" y="{base+y}" font-family="Arial" font-size="13">{escape(str(t))}</text>')
        label(15,19,source.name+'   •   blue: intra-only   •   orange: hybrid')
        label(12,42,'PSNR (dB)')
        for j in range(5):
            y = 65+30*j
            label(15,y,round(hi-(hi-lo)*j/4,1))
            label(55+130*j,209,round(limit*j/4))
            svg.append(f'<path d="M55 {base+y}h520" stroke="#ddd"/>')
        label(265,232,'Bitrate (kb/s)')
        for mode,color in [('intra','#285b97'),('hybrid','#c75c16')]:
            xy = [(55+520*r['bitrate_bps']/1000/limit,base+65+120*(hi-r['psnr_db'])/(hi-lo)) for r in group if r['mode']==mode]
            pts = ' '.join(f'{x},{y}' for x,y in xy)
            svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
            svg.extend(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>' for x,y in xy)
    (OUT/'curves.svg').write_text(''.join(svg)+'</svg>',encoding='utf8')

if __name__ == '__main__':
    main()
