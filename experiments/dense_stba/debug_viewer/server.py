"""Read-only HTTP frontend for archived Dense STBA experiments.

Only the explicit ROI save endpoint writes, under the viewer workspace. No
training, matching, optical flow or optimizer is imported or invoked here.
"""
from __future__ import annotations

import argparse
import base64
from functools import wraps
import io
import json
from pathlib import Path
import threading

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file

ROOT = Path(__file__).resolve().parent
DENSE_ROOT = ROOT.parent


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def png_response(rgb):
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError('Could not encode image')
    return Response(encoded.tobytes(), mimetype='image/png', headers={'Cache-Control': 'private, max-age=3600'})


def coverage_overlay(payload, size):
    """Send a tiny raster overlay, not the whole query/observation array."""
    points = payload.pop('points', {})
    w, h = map(int, size)
    rgba = np.zeros((h, w, 4), np.uint8)
    legend = payload.get('legend', {})
    for uv, category in zip(points.get('seed_uv', []), points.get('category', [])):
        if not np.isfinite(uv).all():
            continue
        x, y = np.rint(uv).astype(int)
        if 0 <= x < w and 0 <= y < h:
            color = legend.get(str(category), {}).get('color', [160, 160, 160, 190])
            rgba[y, x] = color if len(color) == 4 else [*color, 190]
    ok, image = cv2.imencode('.png', cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise ValueError('Could not encode coverage overlay')
    payload['overlay_png'] = 'data:image/png;base64,' + base64.b64encode(image).decode()
    payload['overlay_size'] = [w, h]
    payload['overlay_semantics'] = 'Exact queried seed pixels only; transparent pixels are not queried/selected.'
    return payload


def source_patch(run, camera_index, frame, xy, radius=10, source=True):
    """Crop the actual source image using calibrated working→distorted mapping."""
    cam = run.cameras[camera_index]
    image = run.image(camera_index, frame)
    center = np.asarray(xy, np.float64)
    scale = 1.
    kind = 'working undistorted image'
    source_path = None
    if source and run.manifest.get('dataset') == 'SelfCap corgi-release':
        from experiments.roma_lsm.coordinates import resize_K
        raw_id = cam['raw_frame_ids'][frame]
        source_path = Path('/data/datasets/selfcap/corgi-release') / f"videos/cam{int(cam['camera_id']):02d}/images_2/{raw_id:04d}.png"
        if not source_path.exists():
            raise FileNotFoundError(f'Original distorted image is unavailable: {source_path}')
        original = cv2.imread(str(source_path))
        if original is None:
            raise ValueError('Unreadable source image')
        image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        ksource = resize_K(np.asarray(cam['raw_K']), cam['raw_size'], cam['source_size'])
        ray = np.linalg.solve(np.asarray(cam['K']), np.r_[center, 1.])
        center = cv2.projectPoints(ray[None], np.zeros(3), np.zeros(3), ksource,
                                  np.asarray(cam['raw_D']))[0].reshape(2)
        scale = image.shape[1] / cam['size'][0]
        kind = 'original distorted source image'
    if not np.isfinite(center).all():
        raise ValueError('No finite patch center')
    h, w = image.shape[:2]
    cx, cy = center
    if not (0 <= cx <= w - 1 and 0 <= cy <= h - 1):
        raise ValueError('Patch center lies outside the image')
    rr = int(np.clip(round(radius * scale), 2, 256))
    # Padding is explicit black, not reflected image content.
    x0, y0 = int(np.floor(cx)) - rr, int(np.floor(cy)) - rr
    patch = np.zeros((2 * rr + 1, 2 * rr + 1, 3), np.uint8)
    ax, ay, bx, by = max(0, x0), max(0, y0), min(w, x0 + 2 * rr + 1), min(h, y0 + 2 * rr + 1)
    patch[ay-y0:by-y0, ax-x0:bx-x0] = image[ay:by, ax:bx]
    patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_NEAREST)
    result = png_response(patch)
    result.headers['X-Patch-Space'] = kind
    result.headers['X-Patch-Center'] = json.dumps(center.tolist())
    return result


def create_app(root=DENSE_ROOT, workspace=None):
    from .adapter import DatasetStore
    app = Flask(__name__, template_folder=str(ROOT / 'templates'), static_folder=str(ROOT / 'static'))
    app.config['MAX_CONTENT_LENGTH'] = 256 * 1024
    app.json.sort_keys = False
    store = DatasetStore(Path(root), max_runs=2)
    lock = threading.RLock()
    workspace = Path(workspace or ROOT / 'workspace')
    protected = [Path(root).resolve() / name for name in
                 ['data', 'cache', 'cache_density1', 'results', 'results_density1', 'stability', 'ablations']]
    if any(workspace.resolve().is_relative_to(path) for path in protected):
        raise ValueError('Viewer workspace must not be an experiment input or result directory')
    app.extensions['dense_store'] = store

    def json_api(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with lock:
                return jsonify(clean(fn(*args, **kwargs)))
        return wrapped

    def run_arg(prefix=''):
        run_id = request.args.get(prefix + 'run', 'results_density1/corgi_raw')
        method = request.args.get(prefix + 'method', 'B3')
        return store.load(run_id, method)

    def roi_arg():
        value = request.args.get('roi', 'full')
        return json.loads(value) if value.startswith('{') else value

    def integer(name, default=0, minimum=0, maximum=10000000):
        value = int(request.args.get(name, default))
        if not minimum <= value <= maximum:
            raise ValueError(f'{name} outside allowed range')
        return value

    @app.errorhandler(Exception)
    def error(exc):
        from werkzeug.exceptions import HTTPException
        status = exc.code if isinstance(exc, HTTPException) else (404 if isinstance(exc, FileNotFoundError) else 400)
        if not isinstance(exc, (ValueError, KeyError, IndexError, FileNotFoundError, HTTPException)):
            app.logger.exception('Debug viewer request failed')
            status = 500
        return jsonify({'error': str(exc), 'type': type(exc).__name__}), status

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/health')
    def health():
        return jsonify({'status': 'ok', 'name': 'Dense STBA Debug Studio', 'read_only_experiments': True})

    @app.get('/vendor/plotly.min.js')
    def plotly_js():
        import plotly
        return send_file(Path(plotly.__file__).parent / 'package_data/plotly.min.js', conditional=True, max_age=86400)

    @app.get('/api/runs')
    @json_api
    def runs():
        return {'runs': store.catalog(), 'root': str(Path(root).resolve())}

    @app.get('/api/run')
    @json_api
    def summary():
        return run_arg().summary()

    @app.get('/api/image')
    def image():
        with lock:
            run = run_arg()
            camera, frame = integer('camera', 0, maximum=len(run.cameras)-1), integer('frame')
            if request.args.get('patch') == '1':
                xy = [float(request.args['x']), float(request.args['y'])]
                return source_patch(run, camera, frame, xy, integer('radius', 10, 2, 64), request.args.get('source', '1') == '1')
            return png_response(run.image(camera, frame))

    @app.get('/api/pixel')
    @json_api
    def pixel():
        from .inspector import query_pixel
        return query_pixel(run_arg(), float(request.args['x']), float(request.args['y']), frame=integer('frame'))

    @app.get('/api/point')
    @json_api
    def point():
        from .inspector import inspect_point
        return inspect_point(run_arg(), integer('point'), integer('frame'), include_gt=request.args.get('gt') == '1')

    @app.get('/api/coverage')
    @json_api
    def coverage():
        from .diagnostics import coverage as analyze
        run = run_arg()
        payload = analyze(run, roi_arg(), integer('camera', 1, maximum=len(run.cameras)-1), request.args.get('stage', 'first_failure'))
        return coverage_overlay(payload, run.cameras[0]['size'])

    @app.get('/api/points')
    @json_api
    def points():
        from .diagnostics import point_list
        run = run_arg()
        return point_list(run, roi=roi_arg(), camera_index=integer('camera', 1, maximum=len(run.cameras)-1),
                          reason=request.args.get('reason') or None, sort=request.args.get('sort', 'point_id'),
                          descending=request.args.get('descending') == '1', offset=integer('offset'),
                          limit=integer('limit', 40, 1, 200), stage=request.args.get('stage', 'first_failure'))

    @app.get('/api/flow')
    @json_api
    def flow():
        from .flow_debug import flow_diagnostics
        return flow_diagnostics(run_arg(), integer('point'), camera_index=integer('camera', 1), frame=integer('frame'))

    @app.get('/api/alignment')
    @json_api
    def alignment():
        from .alignment import alignment_panel
        return alignment_panel(run_arg(), roi=roi_arg())

    @app.get('/api/scene')
    @json_api
    def scene():
        from .scene import scene_data
        return scene_data(run_arg(), integer('frame'), point_id=integer('point') if request.args.get('point') else None,
                          color=request.args.get('color', 'speed'), max_points=integer('max_points', 1500, 10, 5000),
                          include_gt=request.args.get('gt') == '1', roi=roi_arg())

    @app.get('/api/compare')
    @json_api
    def compare():
        from .comparison import compare_runs
        left, right = run_arg(), run_arg('right_')
        return compare_runs(left, right,
                            point_id=integer('point') if request.args.get('scope') == 'point' else None,
                            roi=roi_arg(), frame=None if request.args.get('time_scope') == 'all' else integer('frame'))

    @app.get('/api/rois')
    @json_api
    def rois():
        directory = workspace / 'rois'
        entries = []
        if directory.exists():
            for p in sorted(directory.glob('*.json')):
                try:
                    entries.append(json.loads(p.read_text()))
                except (OSError, ValueError):
                    continue
        return {'rois': entries, 'scope': 'Reference seed-image coordinates; reuse only with the same reference image.'}

    @app.post('/api/rois')
    @json_api
    def save_roi():
        # The sole write action is explicit and never enters an experiment dir.
        import re
        from .rois import resolve_roi
        body = request.get_json(force=True)
        name = str(body.get('name', '')).strip()
        if not re.fullmatch(r'[\w\- ]{1,60}', name, re.UNICODE):
            raise ValueError('ROI name must contain 1–60 letters, digits, spaces, underscores or hyphens')
        run = store.load(body['run'], body.get('method', 'B3'))
        roi = body['roi']
        # Validate before writing; custom polygons/rectangles only.
        if not isinstance(roi, dict) or roi.get('type') not in ['rectangle', 'polygon']:
            raise ValueError('Save a drawn rectangle or polygon')
        resolve_roi(run, roi)
        seed = int(run.obs['seed_frame'])
        seed_identity = run.summary().get('reference_seed', {})
        record = {'name': name, 'roi': roi, 'run_id': run.run_id,
                  'reference_camera_id': run.cameras[0]['camera_id'],
                  'seed_image_path': run.cameras[0]['image_paths'][seed],
                  'seed_image_sha256': seed_identity.get('image_sha256'),
                  'size': run.cameras[0]['size'], 'coordinate_space': 'working undistorted reference seed image'}
        directory = workspace / 'rois'
        if directory.is_symlink() or directory.resolve().parent != workspace.resolve():
            raise ValueError('ROI directory must remain inside the viewer workspace')
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (name + '.json')
        if path.is_symlink():
            raise ValueError('Refusing a symlink ROI destination')
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        return record

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DENSE_ROOT)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8917)
    parser.add_argument('--workspace', type=Path, default=ROOT / 'workspace')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    app = create_app(args.root, args.workspace)
    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
