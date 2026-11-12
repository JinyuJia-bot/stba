#!/usr/bin/env python3
"""Plot recovered offsets and one reconstructed 3D trajectory."""
import argparse
import csv
from pathlib import Path


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--input', type=Path, help='Optional input dataset with truth files')
    parser.add_argument('--track', type=int, default=0)
    parser.add_argument('--output', type=Path, help='Defaults to RESULT/summary.png')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cameras = sorted(read(args.result / 'cameras.csv'), key=lambda r: int(r['camera_id']))
    observations = [r for r in read(args.result / 'trajectories.csv') if int(r['track_id']) == args.track]
    if not observations:
        parser.error(f'No reconstructed observations for track {args.track}')
    figure = plt.figure(figsize=(12, 5))
    ax = figure.add_subplot(121)
    origin = float(cameras[0]['beta']) / float(cameras[0]['fps'])
    ax.plot([int(r['camera_id']) for r in cameras],
            [1000 * (float(r['beta']) / float(r['fps']) - origin) for r in cameras], 'o-', label='estimated')
    if args.input and (args.input / 'true_cameras.csv').exists():
        truth = sorted(read(args.input / 'true_cameras.csv'), key=lambda r: int(r['camera_id']))
        origin = float(truth[0]['beta']) / float(truth[0]['fps'])
        ax.plot([int(r['camera_id']) for r in truth],
                [1000 * (float(r['beta']) / float(r['fps']) - origin) for r in truth], 'x--', label='truth')
    ax.set(xlabel='Camera id', ylabel='Relative beta/fps (ms)', title='Clock offsets')
    ax.set_xticks([int(r['camera_id']) for r in cameras])
    ax.legend()
    ax.grid(alpha=0.2)
    ax = figure.add_subplot(122, projection='3d')
    ax.scatter(*[[float(r[a]) for r in observations] for a in ['x', 'y', 'z']], s=5, color='tab:blue', label='STBA samples')
    dct_path = args.result / 'dct.csv'
    if dct_path.exists():
        dct = sorted((r for r in read(dct_path) if int(r['track_id']) == args.track), key=lambda r: float(r['time']))
        if dct:
            ax.plot(*[[float(r[a]) for r in dct] for a in ['x', 'y', 'z']], color='tab:red', label='DCT')
    if args.input and (args.input / 'truth.csv').exists():
        truth = [r for r in read(args.input / 'truth.csv') if int(r['track_id']) == args.track]
        if truth:
            ax.scatter(*[[float(r[a]) for r in truth] for a in ['x', 'y', 'z']], s=3, alpha=0.5, color='black', label='truth')
    ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title=f'Track {args.track}')
    ax.legend()
    figure.tight_layout()
    output = args.output or args.result / 'summary.png'
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(output)


if __name__ == '__main__':
    main()
