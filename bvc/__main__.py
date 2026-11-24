"""Command line for the bvc codec. Run from the FP folder: python -m bvc --help

    python -m bvc encode    <clip.y> [out.bvc]          hybrid-code a raw .y clip into one .bvc file
    python -m bvc decode    <in.bvc> [out.y] [--ref clip.y]   decode it (and report PSNR vs the original)
    python -m bvc benchmark <clip.y>                    BD-rate / BD-PSNR, hybrid vs intra DCT

From a notebook, call main() with the arguments as a list, e.g.
    main(['benchmark', 'input/foreman_qcif.y', '--show'])
"""
import argparse
import json
import os

import numpy as np

from .codec import Config, decode, encode, hybrid_point, intra_point
from .media import count_frames, read_Yframe
from .metrics import bd_metrics, psnr, to_kbps

ANCHOR, TEST = 'Intra DCT (anchor)', 'Hybrid ME/MC + DCT'


def run_benchmark(args):
    """Sweep rate points for both codecs, compute BD metrics, save (and maybe show) the RD plot."""
    n = min(args.frames, count_frames(args.input, args.height, args.width))
    frames = [read_Yframe(args.input, args.height, args.width, k) for k in range(1, n + 1)]

    # One (kbps, PSNR) point per quality (anchor) and per Q step (hybrid)
    rd = {ANCHOR: [], TEST: []}
    for qv in args.qualities:
        b, p = intra_point(frames, qv)
        rd[ANCHOR].append((to_kbps(b, n, args.fps), p))
        print(f'intra  quality {qv:3d}: {rd[ANCHOR][-1][0]:8.1f} kbps  {p:6.2f} dB', flush=True)
    for s in args.steps:
        cfg = Config(args.height, args.width, s, args.block_size, args.search_range)
        b, p = hybrid_point(frames, cfg)
        rd[TEST].append((to_kbps(b, n, args.fps), p))
        print(f'hybrid step    {s:3d}: {rd[TEST][-1][0]:8.1f} kbps  {p:6.2f} dB', flush=True)

    # Sort each curve by rate, then split into rate and PSNR arrays
    (ra, pa), (rt, pt) = [np.array(sorted(v)).T for v in rd.values()]
    bd_rate, bd_psnr = bd_metrics(ra, pa, rt, pt)
    print(f'\nBD-rate : {bd_rate:+.2f} %   (negative = hybrid needs fewer bits for the same PSNR)')
    print(f'BD-PSNR : {bd_psnr:+.2f} dB  (positive = hybrid is better at the same rate)')

    if args.plot:
        plot_rd(ra, pa, rt, pt, bd_rate, bd_psnr, n, args.fps, args.plot, args.show)
    return dict(frames=n, rd=rd, bd_rate_percent=bd_rate, bd_psnr_db=bd_psnr)


def plot_rd(ra, pa, rt, pt, bd_rate, bd_psnr, n, fps, path, show):
    """RD curves on a linear rate axis, cubic fits dashed, BD overlap shaded (the notebook's plot)."""
    import matplotlib
    if not show:
        matplotlib.use('Agg')    # just save the file; no window needed from the command line
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.set_facecolor('#fcfcfb'); fig.patch.set_facecolor('#fcfcfb')
    colors = {ANCHOR: '#2a78d6', TEST: '#eb6834'}
    # Fit only over the log-rate range both curves share (where BD is measured)
    lo = max(np.log10(ra).min(), np.log10(rt).min())
    hi = min(np.log10(ra).max(), np.log10(rt).max())
    xs = np.linspace(lo, hi, 200)
    fits = {}
    for (label, r, p), mk in zip([(ANCHOR, ra, pa), (TEST, rt, pt)], ['o', 's']):
        c = colors[label]
        fits[label] = np.polyval(np.polyfit(np.log10(r), p, 3), xs)
        ax.plot(r, p, color=c, lw=2, marker=mk, ms=8, mec='#fcfcfb', mew=2, label=label, zorder=3)
        ax.plot(10 ** xs, fits[label], color=c, lw=1, ls='--', alpha=0.7, zorder=2)
        ax.annotate(label, (r[-1], p[-1]), xytext=(8, 0), textcoords='offset points',
                    va='center', fontsize=9, color='#52514e')
    ax.fill_between(10 ** xs, fits[ANCHOR], fits[TEST],
                    color='#52514e', alpha=0.10, lw=0, label='BD area (overlap)')
    ax.set_xlabel(f'Bitrate (kbps @ {fps:g} fps)', color='#52514e')
    ax.set_ylabel('Average Y-PSNR (dB)', color='#52514e')
    ax.set_title(f'Rate–distortion, foreman QCIF ({n} frames)\n'
                 f'BD-rate {bd_rate:+.1f} %   ·   BD-PSNR {bd_psnr:+.2f} dB',
                 loc='left', fontsize=11, color='#0b0b0b')
    ax.grid(True, which='both', color='#e6e5e1', lw=0.6)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color('#c3c2b7')
    ax.tick_params(colors='#52514e')
    ax.legend(frameon=False, loc='lower right', fontsize=9)
    ax.set_xlim(0, max(ra.max(), rt.max()) * 1.3)       # room for the end labels
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.savefig(path, dpi=150)
    print('Saved ' + path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(argv=None):
    """Parse argv (default: the command line), run the command, return its results."""
    parser = argparse.ArgumentParser(prog='python -m bvc', description='ENSC 424 block-based video coder')
    subs = parser.add_subparsers(dest='command', required=True)
    enc = subs.add_parser('encode', help='Hybrid-code a raw .y clip into a .bvc file')
    dec = subs.add_parser('decode', help='Decode a .bvc file into a raw .y clip')
    bench = subs.add_parser('benchmark', help='BD-rate/BD-PSNR of hybrid vs intra DCT, with RD plot')

    # decode needs no size or codec options: they are all in the .bvc header
    dec.add_argument('input', help='.bvc file from encode')
    dec.add_argument('output', nargs='?', default=None, help='decoded .y file (default: <input>_dec.y)')
    dec.add_argument('--ref', default=None, help='original .y clip, to report PSNR of the decoded clip')
    dec.add_argument('--fps', type=float, default=30)

    # Options both commands share
    for sub in (enc, bench):
        sub.add_argument('input', help='raw 8-bit luma (.y) file')
        sub.add_argument('--height', type=int, default=144)
        sub.add_argument('--width', type=int, default=176)
        sub.add_argument('--block-size', type=int, default=16, help='motion estimation block size')
        sub.add_argument('--search-range', type=int, default=7, help='motion search range (pixels)')

    enc.add_argument('output', nargs='?', default='output/foreman.bvc', help='.bvc file to write')
    enc.add_argument('--step', type=int, default=16, help='Q step size')
    enc.add_argument('--frames', type=int, default=None, help='frames to code (default: all)')
    enc.add_argument('--fps', type=float, default=30)

    bench.add_argument('--frames', type=int, default=30,
                       help='frames per rate point (300 = whole foreman clip)')
    bench.add_argument('--qualities', type=int, nargs='+', default=[10, 25, 50, 75, 90],
                       help='intra DCT qualities (anchor)')
    bench.add_argument('--steps', type=int, nargs='+', default=[48, 32, 20, 12, 8],
                       help='hybrid Q steps (test)')
    bench.add_argument('--fps', type=float, default=30)
    bench.add_argument('--plot', default='output/bd_rd_curve.png', help="RD plot file ('' = no plot)")
    bench.add_argument('--show', action='store_true', help='also display the plot (e.g. in a notebook)')

    args = parser.parse_args(argv)
    if args.command == 'encode':
        cfg = Config(args.height, args.width, args.step, args.block_size, args.search_range)
        result = encode(args.input, args.output, cfg, args.frames)
        n = len(result['psnr'])
        total = 8 * result['total_bytes']        # the whole .bvc file, header and histograms included
        print(f"Wrote {args.output}: {result['total_bytes']} bytes = {total} bits "
              f"({8 * sum(result['bytes_mv'])} for MVs), avg {total / n:.0f} bits/frame, "
              f"{total / (n * args.height * args.width):.3f} bpp, "
              f"{to_kbps(total, n, args.fps):.1f} kbps @ {args.fps:g} fps, "
              f"avg PSNR {np.mean(result['psnr']):.2f} dB, STEP {args.step}")
        return result
    if args.command == 'decode':
        out = args.output or os.path.splitext(args.input)[0] + '_dec.y'
        h, w, n = decode(args.input, out)
        print(f'Decoded {n} frames of {w}x{h} into {out}')
        result = dict(height=h, width=w, frames=n, output=out)
        if args.ref:
            psnrs = [psnr(read_Yframe(args.ref, h, w, k), read_Yframe(out, h, w, k))
                     for k in range(1, n + 1)]
            bits = 8 * os.path.getsize(args.input)
            print(f'avg PSNR vs {args.ref}: {np.mean(psnrs):.2f} dB, '
                  f'{to_kbps(bits, n, args.fps):.1f} kbps @ {args.fps:g} fps')
            result['psnr'] = psnrs
        return result
    result = run_benchmark(args)
    if argv is None:             # command line: also print a JSON summary
        print(json.dumps(dict(frames=result['frames'], bd_rate_percent=result['bd_rate_percent'],
                              bd_psnr_db=result['bd_psnr_db']), indent=2))
    return result


if __name__ == '__main__':
    main()
