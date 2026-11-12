#!/usr/bin/env python3
"""Plot ground truth, reconstructed samples, DCT curves and actual 2D inputs.

Requires independent truth.csv and true_cameras.csv in --input. Optional --source
is the imported MoCap CSV used by generate_synthetic.py, with generation.json
providing its time origin and track-id mapping. No spatial alignment is applied.
Optional curve separation affects only the displayed 3D lines, never the metrics.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from generate_synthetic import rodrigues

BLUE, ORANGE, BLACK = '#1675bf', '#e58a19', '#27313c'


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def key(row):
    return tuple(int(row[k]) for k in ('track_id', 'camera_id', 'frame'))


def xyz(row):
    return [float(row[a]) for a in ('x', 'y', 'z')]


def interp(times, points, query):
    return np.column_stack([np.interp(query, times, points[:, d]) for d in range(3)])


def load(args):
    observations = {key(r): r for r in rows(args.input / 'observations.csv')}
    truth = {key(r): r for r in rows(args.input / 'truth.csv')}
    estimated = {key(r): r for r in rows(args.result / 'trajectories.csv')}
    cameras = {int(r['camera_id']): r for r in rows(args.input / 'true_cameras.csv')}
    recovered_cameras = {int(r['camera_id']): r for r in rows(args.result / 'cameras.csv')}
    ref = min(cameras.keys() & recovered_cameras.keys())
    shift = (float(recovered_cameras[ref]['beta']) / float(recovered_cameras[ref]['fps'])
             - float(cameras[ref]['beta']) / float(cameras[ref]['fps']))
    dynamic = sorted({k[0] for k, r in observations.items() if int(r['is_static']) == 0})
    generation = json.loads((args.input / 'generation.json').read_text()) if (args.input / 'generation.json').exists() else {}
    source, labels = {}, {}
    if args.source:
        if 'trajectory_source_start_seconds' not in generation or 'source_track_ids' not in generation:
            raise ValueError('--source requires the MoCap time origin and track mapping in generation.json')
        for r in rows(args.source):
            source.setdefault(int(r['track_id']), []).append([float(r['time']), *xyz(r)])
        metadata = args.source.with_suffix('.metadata.json')
        if metadata.exists():
            labels = json.loads(metadata.read_text()).get('labels', {})
    dct = {}
    if (args.result / 'dct.csv').exists():
        for r in rows(args.result / 'dct.csv'):
            dct.setdefault(int(r['track_id']), []).append([float(r['time']) + shift, *xyz(r)])
    tracks = {}
    for track_id in dynamic:
        keys = [k for k in observations if k[0] == track_id and k in truth and k in estimated]
        if not keys:
            continue
        def time(k):
            c = cameras[k[1]]
            return (k[2] - float(c['beta'])) / float(c['fps'])
        keys.sort(key=time)
        t = np.array([time(k) for k in keys])
        gt = np.array([xyz(truth[k]) for k in keys])
        recovered = np.array([xyz(estimated[k]) for k in keys])
        estimated_t = np.array([float(estimated[k]['time']) + shift for k in keys])
        source_id = generation.get('source_track_ids', [])[track_id] if args.source else track_id
        label = labels.get(str(source_id), f'track {track_id}').split(':')[-1]
        dense_t, dense_gt = t, gt
        if source_id in source:
            samples = np.array(sorted(source[source_id], key=lambda r: r[0]))
            source_t = samples[:, 0] - generation['trajectory_source_start_seconds']
            inside = (source_t > t.min()) & (source_t < t.max())
            dense_t = np.r_[t.min(), source_t[inside], t.max()]
            dense_gt = interp(source_t, samples[:, 1:], dense_t)
        error = np.linalg.norm(recovered - gt, axis=1) * 1000
        curve = np.array(sorted(dct.get(track_id, []), key=lambda r: r[0]))
        dct_error = np.full(len(t), np.nan)
        if curve.size:
            valid = (estimated_t >= curve[0, 0] - 1e-9) & (estimated_t <= curve[-1, 0] + 1e-9)
            dct_error[valid] = np.linalg.norm(interp(curve[:, 0], curve[:, 1:], estimated_t[valid]) - gt[valid], axis=1) * 1000
        tracks[track_id] = dict(id=track_id, label=label, keys=keys, time=t,
                                gt=gt, recovered=recovered, dense_t=dense_t, dense_gt=dense_gt,
                                error=error, dct=curve, dct_error=dct_error,
                                rmse=float(np.sqrt(np.mean(error**2))))
    if not tracks:
        raise ValueError('No dynamic observations match reconstruction and ground truth')
    return tracks, observations, cameras, recovered_cameras


def axes3d(ax, points):
    # Keep small sample markers behind both curves instead of allowing mplot3d
    # to promote the entire scatter collection above the lines by depth sorting.
    ax.computed_zorder = False
    low, high = points.min(axis=0), points.max(axis=0)
    # Equal physical scale along all three axes; never enlarge a spatial error.
    extent = max(float(np.max(high - low)), .05) * 1.12
    center = (low + high) / 2
    for d, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        setter(center[d] - extent / 2, center[d] + extent / 2)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    ax.view_init(elev=23, azim=-60)


def curve_labels(separation_mm):
    if separation_mm:
        half = separation_mm / 2
        return f'GT (display Z -{half:g} mm)', f'DCT (display Z +{half:g} mm)'
    return 'MoCap ground truth', 'DCT output curve'


def display_curve(points, separation_mm, direction):
    # Addition returns a new array; source coordinates remain available for metrics.
    return points + np.array([0., 0., direction * separation_mm / 2000.])


def track3d(ax, track, separation_mm=0):
    gt_label, dct_label = curve_labels(separation_mm)
    gt_display = display_curve(track['dense_gt'], separation_mm, -1)
    ax.plot(*gt_display.T, color=BLACK, linewidth=2.0, zorder=3, label=gt_label)
    visible_points = [gt_display, track['recovered']]
    curve = track['dct']
    if curve.size:
        dct_display = display_curve(curve[:, 1:], separation_mm, 1)
        ax.plot(*dct_display.T, color=ORANGE, linewidth=1.4, linestyle='--', zorder=4, label=dct_label)
        visible_points.append(dct_display)
    ax.scatter(*track['recovered'].T, color=BLUE, s=1.0, alpha=.3,
               edgecolors='none', linewidths=0, depthshade=False, zorder=1,
               label='STBA 3D samples')
    axes3d(ax, np.vstack(visible_points))
    if separation_mm:
        ax.set_zlabel('Z (m; curves shifted for display)')
    ax.set_title(f"{track['id']}: {track['label']} | RMSE {track['rmse']:.2f} mm", fontsize=11)


def legend_handles(separation_mm=0):
    gt_label, dct_label = curve_labels(separation_mm)
    return [Line2D([], [], color=BLACK, label=gt_label),
            Line2D([], [], color=BLUE, marker='o', linestyle='', markersize=2,
                   markeredgewidth=0, alpha=.5, label='STBA 3D samples'),
            Line2D([], [], color=ORANGE, linestyle='--', label=dct_label)]


def project(view, point):
    rotation = rodrigues(np.array([float(view[k]) for k in ('rx', 'ry', 'rz')]))
    p = rotation @ point + np.array([float(view[k]) for k in ('tx', 'ty', 'tz')])
    return np.array([float(view['fx']) * p[0] / p[2] + float(view['cx']),
                     float(view['fy']) * p[1] / p[2] + float(view['cy'])])


def detail(args, track, observations):
    fig = plt.figure(figsize=(13, 9), layout='constrained')
    ax = fig.add_subplot(221, projection='3d')
    track3d(ax, track, args.curve_separation_mm)
    ax.legend(fontsize=8, loc='upper left')
    worst = int(np.argmax(track['error']))
    delta = track['recovered'][worst] - track['gt'][worst]
    dims = sorted(np.argsort(np.abs(delta))[-2:])
    center = track['gt'][worst]
    zoom = fig.add_subplot(222)
    recent = np.abs(track['dense_t'] - track['time'][worst]) < .15
    truth_xy = (track['dense_gt'][recent][:, dims] - center[dims]) * 1000
    zoom.plot(*truth_xy.T, color=BLACK, linewidth=1.6)
    near = np.abs(track['time'] - track['time'][worst]) < .15
    xy = (track['recovered'][near][:, dims] - center[dims]) * 1000
    zoom.scatter(*xy.T, s=16, color=BLUE, alpha=.7)
    curve = track['dct']
    if curve.size:
        near_curve = np.abs(curve[:, 0] - track['time'][worst]) < .15
        zoom.plot(*((curve[near_curve, 1:][:, dims] - center[dims]) * 1000).T,
                  color=ORANGE, linestyle='--')
    endpoint = delta[dims] * 1000
    zoom.scatter(0, 0, s=55, facecolors='white', edgecolors=BLACK, zorder=5)
    zoom.scatter(*endpoint, s=42, color=BLUE, zorder=5)
    zoom.plot([0, endpoint[0]], [0, endpoint[1]], color='#ce4343', linewidth=2, zorder=4)
    radius = max(12, 1.8 * track['error'][worst])
    zoom.set(xlim=(-radius, radius), ylim=(-radius, radius),
             xlabel=f"{'XYZ'[dims[0]]} relative to the truth point (mm)",
             ylabel=f"{'XYZ'[dims[1]]} relative to the truth point (mm)",
             title=f"Worst sample: {track['error'][worst]:.2f} mm in 3D\nCamera {track['keys'][worst][1]}, frame {track['keys'][worst][2]} | spatial error NOT amplified")
    zoom.set_aspect('equal', adjustable='box'); zoom.grid(alpha=.2)
    # The input is 2D. Compare points by physical observation identity, not row order.
    camera = min(k[1] for k in track['keys'])
    indices = sorted([i for i, k in enumerate(track['keys']) if k[1] == camera], key=lambda i: track['keys'][i][2])
    source_views = {(int(v['camera_id']), int(v['frame'])): v for v in rows(args.input / 'views.csv')}
    result_views = {(int(v['camera_id']), int(v['frame'])): v for v in rows(args.result / 'views.csv')}
    measured, actual, predicted = [], [], []
    for i in indices:
        k = track['keys'][i]; observation = observations[k]
        view_key = (camera, k[2])
        sv = source_views.get(view_key, source_views.get((camera, -1)))
        rv = result_views.get(view_key, result_views.get((camera, -1)))
        measured.append([float(observation['u']), float(observation['v'])])
        actual.append(project(sv, track['gt'][i]))
        predicted.append(project(rv, track['recovered'][i]))
    image_ax = fig.add_subplot(223)
    image_ax.plot(*np.array(actual).T, color=BLACK, linewidth=1.4, label='Projected truth (input calibration)')
    image_ax.plot(*np.array(predicted).T, color=ORANGE, linestyle='--', label='Reprojected STBA')
    image_ax.scatter(*np.array(measured).T, color=BLUE, s=24, zorder=5, label='Actual 2D input samples')
    image_ax.invert_yaxis(); image_ax.set_aspect('equal', adjustable='datalim')
    image_ax.set(xlabel='u (pixels)', ylabel='v (pixels)', title=f'Camera {camera}: measured 2D samples at original frames')
    image_ax.grid(alpha=.2); image_ax.legend(fontsize=8)
    err = fig.add_subplot(224)
    err.plot(track['time'], track['error'], color=BLUE, linewidth=.8, marker='.', markersize=3, label='STBA')
    if np.any(np.isfinite(track['dct_error'])):
        err.plot(track['time'], track['dct_error'], color=ORANGE, linewidth=.9, alpha=.8, label='DCT grid interpolation')
    err.set(xlabel='True exposure time relative to reference camera (s)', ylabel='3D position error (mm)',
            title='Errors matched by track / camera / frame')
    err.set_ylim(bottom=0); err.grid(alpha=.2); err.legend(fontsize=8)
    fig.suptitle(f"Track {track['id']} / {track['label']}: truth, sampling and reconstruction", fontsize=15)
    fig.savefig(args.out / f"track_{track['id']:02d}_{track['label']}.png", dpi=180)
    plt.close(fig)


def overview(args, tracks, cameras, recovered_cameras):
    fig = plt.figure(figsize=(14, 9), layout='constrained')
    ax = fig.add_subplot(221, projection='3d')
    all_points = []
    for track, color in zip(tracks.values(), plt.cm.turbo(np.linspace(0, 1, len(tracks)))):
        gt_display = display_curve(track['dense_gt'], args.curve_separation_mm, -1)
        ax.plot(*gt_display.T, color=BLACK, linewidth=1.2, alpha=.85, zorder=3)
        all_points.append(gt_display)
        curve = track['dct']
        if curve.size:
            dct_display = display_curve(curve[:, 1:], args.curve_separation_mm, 1)
            ax.plot(*dct_display.T, color=ORANGE, linewidth=.9, linestyle='--',
                    alpha=.9, zorder=4)
            all_points.append(dct_display)
        ax.scatter(*track['recovered'].T, s=.6, color=color, alpha=.25,
                   edgecolors='none', linewidths=0, depthshade=False, zorder=1)
        all_points.append(track['recovered'])
    axes3d(ax, np.vstack(all_points))
    if args.curve_separation_mm:
        half = args.curve_separation_mm / 2
        ax.set_title(f'Display only: GT Z -{half:g} mm; DCT Z +{half:g} mm\nSmall colored dots: original STBA samples')
        ax.set_zlabel('Z (m; curves shifted for display)')
    else:
        ax.set_title('All tracks: GT (dark) and DCT (orange)\nSmall colored dots: STBA samples')
    ids = list(tracks)
    bars = fig.add_subplot(222)
    bars.bar(np.arange(len(ids)) - .2, [tracks[i]['rmse'] for i in ids], width=.4, color=BLUE, label='STBA')
    bars.bar(np.arange(len(ids)) + .2, [np.sqrt(np.nanmean(tracks[i]['dct_error']**2)) for i in ids],
             width=.4, color=ORANGE, label='DCT grid interpolation')
    bars.set_xticks(np.arange(len(ids)), ids, fontsize=7, rotation=90)
    bars.set(xlabel='Track ID (body labels in per_track_metrics.csv)', ylabel='3D RMSE (mm)', title='Every dynamic track, including the worst cases')
    bars.grid(axis='y', alpha=.2); bars.legend(fontsize=9)
    frame_ax = fig.add_subplot(223)
    frames = sorted({k[2] for t in tracks.values() for k in t['keys']})
    for field, label, color in [('error', 'STBA', BLUE), ('dct_error', 'DCT grid interpolation', ORANGE)]:
        values = [np.sqrt(np.nanmean([t[field][i]**2 for t in tracks.values() for i, k in enumerate(t['keys']) if k[2] == frame])) for frame in frames]
        frame_ax.plot(frames, values, marker='.', color=color, label=label)
    frame_ax.set(xlabel='Local frame index (all tracks and cameras)', ylabel='3D RMSE (mm)', title='Frame-wise error: sequence endpoints vs interior')
    frame_ax.set_ylim(bottom=0); frame_ax.grid(alpha=.2); frame_ax.legend(fontsize=9)
    time_ax = fig.add_subplot(224)
    common = sorted(cameras.keys() & recovered_cameras.keys()); ref = common[0]
    origin = float(recovered_cameras[ref]['beta']) / float(recovered_cameras[ref]['fps']) - float(cameras[ref]['beta']) / float(cameras[ref]['fps'])
    errors = [1000 * (float(recovered_cameras[c]['beta']) / float(recovered_cameras[c]['fps']) - float(cameras[c]['beta']) / float(cameras[c]['fps']) - origin) for c in common]
    time_ax.bar(common, errors, color=BLUE); time_ax.axhline(0, color=BLACK, linewidth=.8)
    time_ax.set(xlabel='Camera ID', ylabel='Relative clock error (ms)', title=f'Time offset error relative to camera {ref}', xticks=common)
    time_ax.grid(axis='y', alpha=.2)
    all_errors = np.concatenate([t['error'] for t in tracks.values()])
    fig.suptitle(f"{len(tracks)} dynamic tracks / {len(common)} cameras | 3D RMSE {np.sqrt(np.mean(all_errors**2)):.3f} mm", fontsize=16)
    fig.savefig(args.out / 'overview.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--source', type=Path, help='Original imported MoCap CSV, in the same spatial coordinates')
    parser.add_argument('--tracks', help='Comma-separated detailed track IDs; default: worst, median and best RMSE')
    parser.add_argument('--curve-separation-mm', type=float, default=0.,
                        help='Display-only Z gap: GT moves down half, DCT up half; errors use original coordinates')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.curve_separation_mm) or args.curve_separation_mm < 0:
        parser.error('--curve-separation-mm must be finite and nonnegative')
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.facecolor': 'white'})
    tracks, observations, cameras, recovered_cameras = load(args)
    ranked = sorted(tracks, key=lambda i: tracks[i]['rmse'], reverse=True)
    selected = list(dict.fromkeys(map(int, args.tracks.split(',')))) if args.tracks else list(dict.fromkeys([ranked[0], ranked[len(ranked)//2], ranked[-1]]))
    if any(i not in tracks for i in selected):
        parser.error('--tracks must name reconstructed dynamic tracks with ground truth')
    overview(args, tracks, cameras, recovered_cameras)
    for track_id in selected:
        detail(args, tracks[track_id], observations)
    with PdfPages(args.out / 'all_tracks.pdf') as pdf:
        ids = list(tracks)
        for start in range(0, len(ids), 6):
            fig = plt.figure(figsize=(13, 8.5), layout='constrained')
            for index, track_id in enumerate(ids[start:start+6]):
                track3d(fig.add_subplot(2, 3, index+1, projection='3d'), tracks[track_id], args.curve_separation_mm)
            coordinate_note = (f'3D curves separated {args.curve_separation_mm:g} mm for display only'
                               if args.curve_separation_mm else 'raw world coordinates')
            fig.suptitle(f'All dynamic tracks | page {start//6+1}/{(len(ids)+5)//6} | {coordinate_note}', fontsize=14)
            fig.legend(handles=legend_handles(args.curve_separation_mm), loc='outside lower center', ncol=3)
            pdf.savefig(fig); plt.close(fig)
    metrics = [{'track_id': t['id'], 'label': t['label'], 'observations': len(t['keys']),
                'rmse_mm': t['rmse'], 'median_mm': float(np.median(t['error'])),
                'p95_mm': float(np.quantile(t['error'], .95)), 'max_mm': float(np.max(t['error'])),
                'dct_interpolated_rmse_mm': float(np.sqrt(np.nanmean(t['dct_error']**2)))} for t in tracks.values()]
    with (args.out / 'per_track_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, metrics[0]); writer.writeheader(); writer.writerows(metrics)
    errors = np.concatenate([t['error'] for t in tracks.values()])
    manifest = dict(dynamic_observations=len(errors), rmse_mm=float(np.sqrt(np.mean(errors**2))),
                    detailed_tracks=selected, worst_rmse_track=ranked[0], source=str(args.source),
                    display_gt_z_shift_mm=-args.curve_separation_mm / 2,
                    display_dct_z_shift_mm=args.curve_separation_mm / 2,
                    interpretation=['3D blue samples are reconstructed points; actual input observations are 2D pixels.',
                                    'Black ground-truth curves are linearly connected source MoCap samples when --source is supplied.',
                                    'DCT curves connect the exported grid; DCT errors include its linear interpolation error.',
                                    'Errors match physical observations by track/camera/frame using unshifted coordinates.',
                                    'Display shifts affect only 3D GT/DCT curves; zooms, 2D inputs, samples and metrics remain unshifted.'])
    (args.out / 'plot_metrics.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
    print(args.out.resolve())


if __name__ == '__main__':
    main()
