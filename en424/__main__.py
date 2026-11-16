"""Run with python -m en424 --help."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from .codec import Config, encode, decode
from .media import load, save, demo_frames
from .metrics import measure, bd_rate


def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description='ENSC 424 educational video codec')
    subs = parser.add_subparsers(dest='command', required=True)
    demo = subs.add_parser('demo', help='Generate a deterministic grayscale clip')
    demo.add_argument('output', nargs='?', default='output/demo.npy')
    enc = subs.add_parser('encode', help='Encode NPY, PNG directory, or video')
    dec = subs.add_parser('decode', help='Decode custom E424 to NPY or PNG directory')
    evaluate = subs.add_parser('evaluate', help='Compare intra and hybrid across quantizers')
    for sub in (enc, evaluate):
        sub.add_argument('input')
        sub.add_argument('output')
        sub.add_argument('--fps', type=float, default=None)
        sub.add_argument('--frames', type=int, default=None)
        sub.add_argument('--gray', action='store_true')
        sub.add_argument('--block-size', type=int, default=16, choices=[8, 16, 32])
        sub.add_argument('--search-radius', type=int, default=4)
        sub.add_argument('--gop', type=int, default=12)
        sub.add_argument('--no-skip', action='store_true')
    enc.add_argument('--qstep', type=int, default=16)
    enc.add_argument('--intra-only', action='store_true')
    evaluate.add_argument('--qsteps', type=int, nargs='+', default=[8, 16, 32, 64])
    dec.add_argument('input')
    dec.add_argument('output')
    args = parser.parse_args()
    try:
        if args.command == 'demo':
            save(args.output, demo_frames())
            print('Created ' + args.output)
            return
        if args.command == 'decode':
            frames, meta = decode(Path(args.input).read_bytes())
            save(args.output, frames)
            print(json.dumps(meta, indent=2))
            return
        frames, source_fps = load(args.input, args.frames, args.gray)
        fps = args.fps if args.fps is not None else (source_fps if source_fps else 30.0)
        options = dict(fps=fps, block_size=args.block_size, search_radius=args.search_radius,
                       gop=args.gop, skip=not args.no_skip)
        if args.command == 'encode':
            data, reconstruction, stats = encode(frames, Config(qstep=args.qstep,
                                                               intra_only=args.intra_only, **options))
            write_bytes(args.output, data)
            result = measure(frames, reconstruction, len(data), fps)
            print(json.dumps(dict(result, frame_stats=stats), indent=2))
            return
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        rows, curves = [], {}
        for name, intra in [('intra', True), ('hybrid', False)]:
            curves[name] = []
            for q in args.qsteps:
                start = time.perf_counter()
                data, encoder_recon, _ = encode(frames, Config(qstep=q, intra_only=intra, **options))
                encode_s = time.perf_counter() - start
                start = time.perf_counter()
                decoded, _ = decode(data)
                decode_s = time.perf_counter() - start
                if not np.array_equal(encoder_recon, decoded):
                    raise RuntimeError('Encoder/decoder prediction drift detected')
                write_bytes(out / f'{name}_q{q}.e424', data)
                result = measure(frames, decoded, len(data), fps)
                rows.append(dict(mode=name, qstep=q, encode_seconds=encode_s,
                                 decode_seconds=decode_s, **result))
                curves[name].append((result['bitrate_bps'], result['psnr_db']))
                print(f"{name:6} q={q:3}: {len(data):7} bytes, {result['psnr_db']:.2f} dB", flush=True)
        with (out / 'results.csv').open('w', newline='', encoding='utf8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = dict(input=str(Path(args.input).resolve()), shape=list(frames.shape),
                       fps=fps, options=options, qsteps=args.qsteps,
                       metric='Classical cubic log-rate BD-Rate; whole-clip PSNR; all container bytes included')
        try:
            summary['bd_rate_percent'] = bd_rate(curves['intra'], curves['hybrid'])
        except ValueError as exc:
            summary['bd_rate_percent'] = None
            summary['bd_rate_unavailable_reason'] = str(exc)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
        print(json.dumps(summary, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, 'Error: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
