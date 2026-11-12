"""Read-only reference-SEED ROI selection, with explicitly requested local saves."""
from pathlib import Path
import json
import re

import numpy as np


WORKSPACE = Path(__file__).resolve().parent / 'workspace'
BUILTIN_BOXES = {'dog': [207, 145, 241, 213], 'person': [231, 50, 303, 236]}
COORDINATES = 'reference_seed_image_integer_pixel_centers_xy'
CAUTION = ('Manual broad seed-frame regions, not semantic or identity ground truth. '
           'Boxes may overlap and include background; selection is never applied at current-frame UV.')


def _box(points, bounds):
    x0, y0, x1, y1 = bounds
    return ((points[:, 0] >= x0) & (points[:, 0] < x1) &
            (points[:, 1] >= y0) & (points[:, 1] < y1))


def normalize_roi(roi=None):
    if roi is None:
        roi = 'full'
    if isinstance(roi, str):
        if roi not in ['full', 'dog', 'person', 'background']:
            raise ValueError('Unknown ROI; use full/dog/person/background or an explicit ROI object')
        return {'type': roi}
    if not isinstance(roi, dict):
        raise ValueError('ROI must be a name or object')
    kind = roi.get('type', roi.get('kind'))
    if kind in ['full', 'dog', 'person', 'background']:
        return {'type': kind}
    if kind in ['rectangle', 'rect']:
        values = np.asarray(roi.get('bounds', roi.get('xyxy')), dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all() or not (values[0] < values[2] and values[1] < values[3]):
            raise ValueError('Rectangle requires finite increasing [x0,y0,x1,y1] bounds')
        return {'type': 'rectangle', 'bounds': values.tolist(), 'boundary': 'left/top inclusive, right/bottom exclusive'}
    if kind == 'polygon':
        values = np.asarray(roi.get('vertices'), dtype=float)
        if values.ndim != 2 or values.shape[1:] != (2,) or len(values) < 3 or len(values) > 1000 or not np.isfinite(values).all():
            raise ValueError('Polygon requires 3..1000 finite xy vertices')
        area2 = np.sum(values[:, 0] * np.roll(values[:, 1], -1) - values[:, 1] * np.roll(values[:, 0], -1))
        if abs(area2) < 1e-9:
            raise ValueError('Polygon has zero signed area')
        return {'type': kind, 'vertices': values.tolist(), 'boundary': 'edge included; even-odd interior'}
    if kind == 'point_ids':
        values = roi.get('point_ids')
        if not isinstance(values, list) or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in values):
            raise ValueError('point_ids must be an integer list')
        return {'type': kind, 'point_ids': sorted(set(int(v) for v in values))}
    raise ValueError('Unsupported ROI type')


def _polygon(points, vertices):
    vertices = np.asarray(vertices, dtype=float)
    x, y = points.T
    inside = np.zeros(len(points), bool)
    edge = np.zeros(len(points), bool)
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        dx, dy = b - a
        cross = (x - a[0]) * dy - (y - a[1]) * dx
        edge |= (np.abs(cross) <= 1e-8 * max(1., np.hypot(dx, dy))) & (x >= min(a[0], b[0]) - 1e-9) & (x <= max(a[0], b[0]) + 1e-9) & (y >= min(a[1], b[1]) - 1e-9) & (y <= max(a[1], b[1]) + 1e-9)
        if dy:
            inside ^= ((a[1] > y) != (b[1] > y)) & (x < a[0] + (y - a[1]) * dx / dy)
    return inside | edge


def resolve_roi(run, roi=None):
    """Return query-index mask and JSON metadata; all geometry uses seed UV."""
    spec = normalize_roi(roi)
    points = np.asarray(run.obs['seed_uv'])
    kind = spec['type']
    if kind == 'full':
        mask = np.ones(len(points), bool)
    elif kind in BUILTIN_BOXES:
        mask = _box(points, BUILTIN_BOXES[kind])
        spec['bounds'] = list(BUILTIN_BOXES[kind])
    elif kind == 'background':
        mask = ~(_box(points, BUILTIN_BOXES['dog']) | _box(points, BUILTIN_BOXES['person']))
        spec['definition'] = 'Original queried seed ROI minus the union of manual dog/person boxes'
    elif kind == 'rectangle':
        mask = _box(points, spec['bounds'])
    elif kind == 'polygon':
        mask = _polygon(points, spec['vertices'])
    else:
        mask = np.isin(run.point_ids, spec['point_ids'])
    # Archive queries are already the intersection with the original query ROI.
    mask &= np.isfinite(points).all(axis=-1)
    spec.update(coordinate_system=COORDINATES, seed_frame=int(np.asarray(run.obs['seed_frame'])),
                reference_camera_id=run.cameras[0]['camera_id'], image_size=list(run.cameras[0]['size']),
                selected_query_count=int(mask.sum()), total_query_count=len(points), caution=CAUTION,
                builtin_boxes_source='data_frontend_diagnostics.py reference camera07 raw frame230; descriptive only')
    if kind in ['dog', 'person', 'background'] and run.cameras[0]['camera_id'] != 7:
        spec['warning'] = 'Built-in corgi boxes have no established semantic meaning for this dataset'
    return mask, spec


def _saved_path(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name):
        raise ValueError('ROI name must be 1..80 alphanumeric, underscore or hyphen characters')
    if WORKSPACE.is_symlink():
        raise ValueError('Viewer workspace must not redirect ROI writes through a symlink')
    workspace = WORKSPACE.resolve()
    folder = WORKSPACE / 'rois'
    if folder.exists() and folder.resolve().parent != workspace:
        raise ValueError('ROI directory must remain inside the viewer workspace')
    path = folder / (name + '.json')
    if path.is_symlink():
        raise ValueError('Refusing a symlink ROI destination')
    return path


def save_roi(name, roi, *, run=None, explicit=False):
    """Called only for an explicit UI save action; never from resolve/coverage."""
    if not explicit:
        raise ValueError('Saving a custom ROI requires an explicit save action')
    spec = normalize_roi(roi)
    context = resolve_roi(run, spec)[1] if run is not None else {'coordinate_system': COORDINATES}
    record = {'schema_version': 1, 'name': name, 'roi': spec, 'context': context,
              'scope': 'Viewer-only selection; original experiment input/output unchanged'}
    path = _saved_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    return {'path': str(path), **record}


def load_roi(name):
    record = json.loads(_saved_path(name).read_text())
    record['roi'] = normalize_roi(record['roi'])
    return record


def list_rois():
    folder = WORKSPACE / 'rois'
    return [load_roi(p.stem) for p in sorted(folder.glob('*.json')) if not p.is_symlink()] if folder.exists() else []
