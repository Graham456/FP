"""Compare the first 100 QCIF Foreman frames to the supplied iteration-0 curve."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from en424.codec import Config, encode, decode, quality_step
from en424.media import load
from en424.metrics import measure, bd_rate

REFERENCE = [(10,.38,29.12),(30,.61,31.27),(50,.80,32.64),(70,1.07,34.10),(90,1.93,38.78)]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    frames, _ = load(args.source, limit=100, width=176, height=144)
    if len(frames) != 100:
        parser.error('Foreman comparison requires the first 100 complete frames')
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for quality, _, _ in REFERENCE:
        step = quality_step(quality)
        data, reconstruction, stats = encode(frames, Config(qstep=step, gop=100, compact=True))
        path = args.output / f'foreman_q{step}.e424'
        path.write_bytes(data)
        decoded, _ = decode(path.read_bytes())
        np.testing.assert_array_equal(decoded, reconstruction)
        assert [f['type'] for f in stats] == ['I'] + ['P'] * 99
        result = measure(frames, decoded, path.stat().st_size, 30)
        rows.append(dict(qstep=step, bytes=result['bytes'], bpp=float(result['bits_per_pixel']), psnr=result['psnr_db']))
        print(rows[-1], flush=True)
    with (args.output/'measured.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    curve = [(r['bpp'],r['psnr']) for r in rows]
    comparison = dict(reference_qf_bpp_psnr=REFERENCE, measured=rows,
                      bd_rate_percent=bd_rate([(b,p) for _,b,p in REFERENCE], curve),
                      psnr_overlap=[max(REFERENCE[0][2], rows[0]['psnr']), min(REFERENCE[-1][2],rows[-1]['psnr'])])
    (args.output/'comparison.json').write_text(json.dumps(comparison, indent=2))
    print(comparison['bd_rate_percent'])

if __name__ == '__main__':
    main()

