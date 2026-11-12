#!/usr/bin/env python3
"""Report observed residuals and, only when provided, independent ground-truth errors."""
import argparse
from bisect import bisect_right
import csv
import json
import math
from pathlib import Path


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def point_key(row):
    return tuple(int(row[k]) for k in ['track_id', 'camera_id', 'frame'])


def evaluate(data, result, align_similarity=False):
    report = {'stages': rows(result / 'metrics.csv')}
    trajectory = rows(result / 'trajectories.csv')
    report['reconstructed_observations'] = len(trajectory)
    if data is None:
        report['ground_truth'] = 'not provided'
        return report
    input_rows = rows(data / 'observations.csv')
    report['input_observations'] = len(input_rows)
    report['observation_coverage'] = len(trajectory) / len(input_rows) if input_rows else 0.0
    truth_path = data / 'truth.csv'
    if truth_path.exists():
        truth = {point_key(row): row for row in rows(truth_path)}
        kinds = {point_key(row): int(row['is_static']) for row in input_rows}
        errors = {'all': [], 'dynamic': [], 'static': []}
        matched = []
        for row in trajectory:
            key = point_key(row)
            if key not in truth:
                continue
            sq = sum((float(row[a]) - float(truth[key][a])) ** 2 for a in ['x', 'y', 'z'])
            errors['all'].append(sq)
            matched.append(([float(row[a]) for a in ['x', 'y', 'z']],
                            [float(truth[key][a]) for a in ['x', 'y', 'z']],
                            'static' if kinds.get(key, 0) else 'dynamic'))
            errors['static' if kinds.get(key, 0) else 'dynamic'].append(sq)
        report['matched_truth_observations'] = len(errors['all'])
        for group, values in errors.items():
            if values:
                report[f'{group}_trajectory_rmse_m'] = math.sqrt(sum(values) / len(values))
        report['trajectory_metric'] = 'Raw world coordinates, matched physical observations; no post-hoc rigid/similarity alignment.'
        if align_similarity and len(matched) >= 3:
            import numpy as np
            predicted = np.array([r[0] for r in matched])
            target = np.array([r[1] for r in matched])
            pm, tm = predicted.mean(axis=0), target.mean(axis=0)
            p, t = predicted - pm, target - tm
            variance = np.sum(p * p)
            if variance <= 1e-15 or np.linalg.matrix_rank(p) < 2:
                raise ValueError('Similarity alignment needs a non-degenerate set of 3D points')
            u, singular, vt = np.linalg.svd(t.T @ p)
            correction = np.eye(3)
            correction[-1, -1] = np.linalg.det(u @ vt)
            rotation = u @ correction @ vt
            scale = float(np.sum(singular * np.diag(correction)) / variance)
            aligned = scale * p @ rotation.T + tm
            squared = np.sum((aligned - target) ** 2, axis=1)
            report['similarity_alignment'] = 'One fitted Sim(3) over all matched points, for spatial gauge only; raw errors are also reported.'
            report['similarity_scale'] = scale
            for group in ['all', 'dynamic', 'static']:
                chosen = [float(e) for e, r in zip(squared, matched) if group == 'all' or r[2] == group]
                if chosen:
                    report[f'{group}_similarity_aligned_rmse_m'] = math.sqrt(sum(chosen) / len(chosen))
    else:
        report['trajectory_ground_truth'] = 'unavailable'
    dct_path = result / 'dct.csv'
    if truth_path.exists() and dct_path.exists():
        curves = {}
        for row in rows(dct_path):
            curves.setdefault(int(row['track_id']), []).append((float(row['time']), [float(row[a]) for a in ['x','y','z']]))
        for curve in curves.values():
            curve.sort(key=lambda item: item[0])
        dct_errors = {'all': [], 'dynamic': [], 'static': []}
        for row in trajectory:
            key = point_key(row)
            curve = curves.get(key[0], [])
            if key not in truth or not curve:
                continue
            time = float(row['time'])
            if len(curve) == 1:
                xyz = curve[0][1]
            else:
                if time < curve[0][0] - 1e-9 or time > curve[-1][0] + 1e-9:
                    continue
                index = min(len(curve)-1, max(1, bisect_right([item[0] for item in curve], time)))
                a, b = curve[index-1], curve[index]
                fraction = (time-a[0]) / (b[0]-a[0])
                xyz = [x + fraction*(y-x) for x,y in zip(a[1],b[1])]
            squared = sum((xyz[i]-float(truth[key][a]))**2 for i,a in enumerate(['x','y','z']))
            dct_errors['all'].append(squared)
            dct_errors['static' if kinds.get(key, 0) else 'dynamic'].append(squared)
        report['dct_matched_truth_observations'] = len(dct_errors['all'])
        for group, values in dct_errors.items():
            if values:
                report[f'{group}_dct_interpolated_rmse_m'] = math.sqrt(sum(values)/len(values))
        report['dct_metric'] = 'Raw world coordinates; linearly interpolate exported DCT grid at each reconstructed observation time. Includes grid interpolation error; no extrapolation.'
    camera_truth_path = data / 'true_cameras.csv'
    if camera_truth_path.exists():
        truth = {int(r['camera_id']): r for r in rows(camera_truth_path)}
        estimated = {int(r['camera_id']): r for r in rows(result / 'cameras.csv')}
        common = sorted(truth.keys() & estimated.keys())
        if common:
            reference = common[0]
            estimated_origin = float(estimated[reference]['beta']) / float(estimated[reference]['fps'])
            true_origin = float(truth[reference]['beta']) / float(truth[reference]['fps'])
            errors = []
            for camera in common:
                e, t = estimated[camera], truth[camera]
                sec = (float(e['beta']) / float(e['fps']) - estimated_origin) - (float(t['beta']) / float(t['fps']) - true_origin)
                errors.append({'camera_id': camera, 'error_seconds': sec, 'error_frames': sec * float(t['fps'])})
            report['offset_reference_camera'] = reference
            report['offset_errors'] = errors
            values = [abs(e['error_frames']) for e in errors if e['camera_id'] != reference]
            if values:
                report['offset_mae_frames'] = sum(values) / len(values)
                report['offset_max_error_frames'] = max(values)
    else:
        report['offset_ground_truth'] = 'unavailable'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Input dataset directory; optional truth files')
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--json', type=Path, help='Optional output JSON, refuses an existing file')
    parser.add_argument('--align-similarity', action='store_true', help='Also report a single globally aligned spatial metric; requires NumPy')
    args = parser.parse_args()
    try:
        result = evaluate(args.input, args.result, args.align_similarity)
        text = json.dumps(result, indent=2, allow_nan=False)
        if args.json:
            with args.json.open('x') as f:
                f.write(text + '\n')
        print(text)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
