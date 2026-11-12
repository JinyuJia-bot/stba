#!/usr/bin/env python3
"""Run a small explicit synthetic experiment matrix and save comparable metrics."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from evaluate_alignment import evaluate


def numbers(text, cast):
    return [cast(value) for value in text.split(',')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path('build/stba'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--cameras', default='3,5')
    parser.add_argument('--noise', default='0,0.5')
    parser.add_argument('--seeds', default='1,2,3')
    parser.add_argument('--modes', default='full', help='Comma-separated fixed,incremental,full')
    parser.add_argument('--frames', type=int, default=30)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f'Executable not found: {binary}')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('--out must be empty, so results from different runs cannot mix')
    args.out.mkdir(parents=True, exist_ok=True)
    generator = Path(__file__).with_name('generate_synthetic.py')
    records = []
    for cameras in numbers(args.cameras, int):
        for noise in numbers(args.noise, float):
            for seed in numbers(args.seeds, int):
                name = f'cams{cameras}_noise{noise:g}_seed{seed}'
                case = args.out / name
                data = case / 'data'
                subprocess.run([sys.executable, str(generator), '--cameras', str(cameras), '--frames', str(args.frames),
                                '--noise', str(noise), '--seed', str(seed), '--out', str(data)], check=True)
                for mode in args.modes.split(','):
                    if mode not in ['fixed', 'incremental', 'full']:
                        parser.error(f'Unknown mode: {mode}')
                    result = case / mode
                    command = [str(binary), '--input', str(data), '--output', str(result), '--mode', mode]
                    if args.config:
                        command += ['--config', str(args.config)]
                    completed = subprocess.run(command, text=True, capture_output=True)
                    (case / f'{mode}.log').write_text(completed.stdout + completed.stderr)
                    row = {'case': name, 'mode': mode, 'camera_count': cameras, 'noise_px': noise,
                           'seed': seed, 'success': completed.returncode == 0}
                    if completed.returncode == 0:
                        report = evaluate(data, result)
                        (case / f'{mode}_evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
                        for key in ['observation_coverage', 'offset_mae_frames', 'offset_max_error_frames',
                                    'dynamic_trajectory_rmse_m', 'static_trajectory_rmse_m', 'dynamic_dct_interpolated_rmse_m']:
                            row[key] = report.get(key, '')
                    records.append(row)
                    print(name, mode, 'PASS' if row['success'] else 'FAILED; see log', flush=True)
    fields = ['case', 'mode', 'camera_count', 'noise_px', 'seed', 'success', 'observation_coverage',
              'offset_mae_frames', 'offset_max_error_frames', 'dynamic_trajectory_rmse_m', 'static_trajectory_rmse_m', 'dynamic_dct_interpolated_rmse_m']
    with (args.out / 'summary.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(records)
    if not all(row['success'] for row in records):
        raise SystemExit('At least one experiment failed; failed cases are retained in summary.csv.')


if __name__ == '__main__':
    main()
