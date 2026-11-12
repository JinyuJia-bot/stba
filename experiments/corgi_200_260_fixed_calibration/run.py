"""Extract SelfCap corgi [200,260), then run the root STBA with fixed calibration.

Use: tmp/walking_stba/venv/bin/python experiments/corgi_200_260_fixed_calibration/run.py prepare
     ... run.py solve all8
     ... run.py solve holdout07
The cached animal detector outputs are measurements, not ground truth.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = Path('/data/datasets/selfcap/corgi-release')
POSE = ROOT / 'tmp/selfcap_corgi/pose'
SOLVER = ROOT / 'build/stba'
IDS = [4, 7, 10, 13, 16, 18, 21, 23]
START, STOP, FPS = 200, 260, 60.
CONF = .5
VIEW_FIELDS = ['view_id', 'camera_id', 'frame', 'fx', 'fy', 'cx', 'cy',
               'rx', 'ry', 'rz', 'tx', 'ty', 'tz']
OBS_FIELDS = ['track_id', 'is_static', 'motion_weight', 'camera_id', 'frame', 'u', 'v', 'weight']


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def write_csv(path, fields, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration():
    fi = cv2.FileStorage(str(DATA / 'optimized/intri.yml'), cv2.FILE_STORAGE_READ)
    fe = cv2.FileStorage(str(DATA / 'optimized/extri.yml'), cv2.FILE_STORAGE_READ)
    assert fi.isOpened() and fe.isOpened()
    out = []
    sync = json.loads((DATA / 'optimized/sync.json').read_text())
    for cam in IDS:
        name = f'{cam:04d}'
        k = fi.getNode('K_' + name).mat()
        d = fi.getNode('D_' + name).mat().ravel()
        stored_r = fe.getNode('Rot_' + name).mat()
        t = fe.getNode('T_' + name).mat().ravel()
        # Use the published angle-axis verbatim: stored Rot has float32 rounding.
        rv = fe.getNode('R_' + name).mat().ravel()
        r = cv2.Rodrigues(rv)[0]
        assert np.allclose(r, stored_r, atol=1e-6, rtol=0)
        assert k[0, 1] == 0 and k[1, 0] == 0 and np.array_equal(k[2], [0, 0, 1])
        out.append(dict(original_id=cam, K=k.tolist(), D=d.tolist(), R=r.tolist(),
                        published_Rot_max_abs_rounding=float(np.max(np.abs(r-stored_r))),
                        rvec=rv.tolist(), T=t.tolist(),
                        official_offset_seconds=sync[name] - sync[f'{IDS[0]:04d}']))
    fi.release()
    fe.release()
    return out


def prepare():
    if (HERE / 'input').exists():
        raise FileExistsError('Input already exists; source and existing results are never overwritten.')
    cams = calibration()
    (HERE / 'calibration_source').mkdir()
    sources = {}
    for name in ['intri.yml', 'extri.yml', 'sync.json']:
        source = DATA / 'optimized' / name
        shutil.copy2(source, HERE / 'calibration_source' / name)
        sources[str(source)] = sha(source)
    sources[str(POSE / 'metadata.json')] = sha(POSE / 'metadata.json')
    sources[str(POSE / 'selected_cameras.json')] = sha(POSE / 'selected_cameras.json')
    frames = np.arange(START, STOP)
    xy, valid, und, conf = [], [], [], []
    all_rows, per_camera, image_rows = [], [], []
    for c, cam in enumerate(cams):
        original = cam['original_id']
        source = POSE / f'cam_{original:02d}.npz'
        sources[str(source)] = sha(source)
        with np.load(source) as z:
            selected = (z['frame'] >= START) & (z['frame'] < STOP)
            assert np.array_equal(z['frame'][selected], frames)
            half = z['xy'][selected].astype(float)
            confidence = z['confidence'][selected].astype(float)
        keep = ((confidence >= CONF) & np.isfinite(half).all(axis=-1)
                & (half[..., 0] >= 0) & (half[..., 0] < 1920)
                & (half[..., 1] >= 0) & (half[..., 1] < 1080))
        # Inverse of INTER_AREA half-size pixel-center mapping. K remains original.
        full = 2 * half + .5
        k, d = np.array(cam['K']), np.array(cam['D'])
        ideal = np.full_like(full, np.nan)
        ideal[keep] = cv2.undistortPointsIter(full[keep].reshape(-1, 1, 2), k, d, None, k,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)).reshape(-1, 2)
        xy.append(full); valid.append(keep); und.append(ideal); conf.append(confidence)
        image_dir = HERE / 'frames' / f'cam{original:02d}'
        image_dir.mkdir(parents=True)
        for i, frame in enumerate(frames):
            image = DATA / 'videos' / f'cam{original:02d}/images_2/{frame:04d}.png'
            if not image.is_file():
                raise FileNotFoundError(image)
            (image_dir / image.name).symlink_to(image)
            image_rows.append([c, original, int(frame), str(image), '1920x1080', 'raw distorted, half size'])
            for joint in np.flatnonzero(keep[i]):
                all_rows.append([int(joint), 0, 1, c, int(frame), *ideal[i, joint], 1.])
        per_camera.append(dict(camera_id=c, original_id=original, frames=len(frames),
                               candidate_observations=int(keep.size), valid_observations=int(keep.sum()),
                               rejected_observations=int((~keep).sum())))
    names = json.loads((POSE / 'metadata.json').read_text())['keypoints']
    np.savez_compressed(HERE / 'observations.npz', frame=frames, camera_ids=IDS,
                        xy_full_distorted=xy, xy_full_undistorted=und,
                        valid=valid, confidence=conf, joint_names=names)
    save(HERE / 'calibration.json', cams)
    write_csv(HERE / 'frame_manifest.csv', ['camera_id', 'original_id', 'frame', 'source', 'resolution', 'format'], image_rows)
    write_csv(HERE / 'track_names.csv', ['track_id', 'name'], enumerate(names))
    for label, retained in [('all8', IDS), ('holdout07', [c for c in IDS if c != 7])]:
        inp = HERE / ('input' if label == 'all8' else 'holdout07/input')
        inp.mkdir(parents=True)
        mapping = {IDS.index(original): local for local, original in enumerate(retained)}
        view_rows, camera_rows = [], []
        for old, new in mapping.items():
            cam = cams[old]; k = np.array(cam['K'])
            view_rows.append([new, new, -1, k[0, 0], k[1, 1], k[0, 2], k[1, 2], *cam['rvec'], *cam['T']])
            camera_rows.append([new, FPS, FPS * cam['official_offset_seconds']])
        obs = []
        for row in all_rows:
            if row[3] in mapping:
                new = row.copy(); new[3] = mapping[row[3]]; obs.append(new)
        for joint in range(len(names)):
            assert len({row[3] for row in obs if row[0] == joint}) >= 2
        write_csv(inp / 'cameras.csv', ['camera_id', 'fps', 'beta'], camera_rows)
        write_csv(inp / 'views.csv', VIEW_FIELDS, view_rows)
        write_csv(inp / 'observations.csv', OBS_FIELDS, obs)
        save(inp / 'metadata.json', dict(camera_original_ids=retained, frames_start=START, frames_stop_exclusive=STOP,
             fps=FPS, observations=len(obs), tracks=len(names), confidence_threshold=CONF,
             pixel_coordinates='Original 3840x2160 calibration coordinates; undistorted with original K,D',
             geometry='Official K,R,T fixed; original distortion used only for undistortion and visualization',
             reference_camera_original=4, time_formula='t=(frame-beta)/60',
             ground_truth=False, length_unit='Unverified SelfCap native world-coordinate unit'))
    (HERE / 'config.yaml').write_text(
        '# Coordinates are 2x previous half-size pixels: lambda 1000->4000, Huber 3->6.\n'
        'motion_weight: 4000\nepsilon_seconds: 0.000001\nmin_dt: 0.0000000001\n'
        'search_radius: 0.6\ncoarse_step: 0.15\nfine_step: 0.025\nmax_iterations: 200\n'
        'reference_camera: 0\noptimize_offsets: true\noptimize_poses: false\n'
        'optimize_intrinsics: false\nhuber_px: 6\ndct_fps: 60\ndct_max_samples: 256\n')
    save(HERE / 'manifest.json', dict(dataset=str(DATA), window='[200,260), zero based, 60 original 60 FPS frames',
        cameras=IDS, tracks=17, observations=len(all_rows), per_camera=per_camera,
        extraction='Slice cached YOLOX-m + ViTPose-s APT36K observations; 480 source-image symlinks',
        inputs='Original frame numbers retained; no outside-window context or temporal resampling',
        observation_selection='confidence >=0.5, finite and within raw image; no geometric residual filtering',
        pixel_conversion='full = 2*half + 0.5; original K unchanged; no camera refinement',
        main_comparison='Same 8-camera observations: official fixed sync vs full STBA initialized at official sync',
        heldout_comparison='Remove camera07 from BOTH time estimation and trajectory estimation; predict its observations with its fixed official reference clock',
        caveats=['17 sparse animal keypoints, not dense point-cloud reconstruction',
                 'No independent 3D or timing ground truth; released PLYs are reconstruction references',
                 'Existing camera selection used frame230 visibility; this is not a new blind benchmark',
                 'Same short temporal segment used for fitting and held-out-view prediction; no temporal generalization claim'],
        source_sha256=sources, solver=str(SOLVER), solver_sha256=sha(SOLVER)))
    print(json.dumps(dict(observations=len(all_rows), cameras=per_camera), indent=2), flush=True)


def solve(label):
    manifest = json.loads((HERE / 'manifest.json').read_text())
    assert sha(SOLVER) == manifest['solver_sha256']
    base = HERE if label == 'all8' else HERE / 'holdout07'
    inp = base / 'input'
    for name, mode in [('baseline', 'fixed'), ('result', 'full')]:
        out = base / name
        if out.exists():
            raise FileExistsError(out)
        cmd = [str(SOLVER), '--input', str(inp), '--output', str(out),
               '--config', str(HERE / 'config.yaml'), '--mode', mode]
        started = time.time()
        print('START', label, name, flush=True)
        with (base / f'{name}.log').open('x') as stream:
            status = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        record = dict(command=cmd, seconds=time.time()-started, returncode=status.returncode,
                      solver_sha256=sha(SOLVER), config_sha256=sha(HERE / 'config.yaml'),
                      input_sha256={p.name:sha(p) for p in inp.glob('*.csv')})
        save(base / f'{name}.command.json', record)
        if status.returncode:
            raise RuntimeError(f'{label}/{name}: see log; result saved if only DCT failed')
        before, after = read_csv(inp / 'views.csv'), read_csv(out / 'views.csv')
        assert len(before) == len(after)
        difference = max(abs(float(a[k])-float(b[k])) for a,b in zip(before, after) for k in VIEW_FIELDS)
        assert difference == 0, 'Calibration changed!'
        if mode == 'fixed':
            assert np.array_equal([float(r['beta']) for r in read_csv(inp/'cameras.csv')],
                                  [float(r['beta']) for r in read_csv(out/'cameras.csv')])
        save(base / f'{name}.calibration_check.json', dict(exact_numeric_match=True, max_abs_difference=difference))
        print('DONE', label, name, round(record['seconds'], 2), 's; calibration unchanged', flush=True)
        print((out / 'solver_report.txt').read_text(), flush=True)


def continue_global(label):
    base = HERE if label == 'all8' else HERE / 'holdout07'
    source, output = base / 'result', base / 'result_refined'
    if output.exists():
        raise FileExistsError(output)
    executable = HERE / 'refine/build/refine_global'
    # Apply the same predeclared iteration cap to both unfinished global solves.
    # Objective, observations, ordering rules, geometry and weights stay the same.
    cmd = [str(executable), str(source), str(output), str(HERE / 'config.yaml'), '1000']
    started = time.time()
    print('CONTINUE', label, flush=True)
    with (base / 'result_refined.log').open('x') as stream:
        status = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'})
    save(base / 'result_refined.command.json', dict(command=cmd, seconds=time.time()-started,
         returncode=status.returncode, solver_sha256=sha(executable),
         linked_root_core_sha256=sha(ROOT / 'build/libstba_core.a'),
         trigger='Initial global solve reached 200 iterations without convergence; continue both solves with identical 1000-iteration cap',
         initial_result_preserved=str(source)))
    if status.returncode:
        raise RuntimeError('Continuation failed; inspect log')
    original, refined = read_csv(base/'input/views.csv'), read_csv(output/'views.csv')
    difference = max(abs(float(a[k])-float(b[k])) for a,b in zip(original,refined) for k in VIEW_FIELDS)
    assert len(original) == len(refined) and difference == 0
    save(base / 'result_refined.calibration_check.json',dict(exact_numeric_match=True,max_abs_difference=difference))
    save(base / 'selected_results.json', dict(baseline='baseline',result='result_refined'))
    print('DONE', label, round(time.time()-started,2), 's', flush=True)
    print((output/'solver_report.txt').read_text(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'solve', 'continue'])
    parser.add_argument('label', nargs='?', choices=['all8', 'holdout07'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        if args.label is None:
            parser.error('solve requires all8 or holdout07')
        (solve if args.action == 'solve' else continue_global)(args.label)
