"""Explicit, caller-supplied initialization snapshots in the viewer workspace.

This module never reconstructs an initialization from final controls, invokes an
optimizer, or edits experiment artifacts. A future caller must pass the actual
arrays at initialization time and describe their source in ``provenance``.
Missing historical arrays cannot be recovered by this helper.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

from .adapter import sha256, to_jsonable

WORKSPACE_ROOT = Path(__file__).resolve().parent / 'workspace'
_REQUIRED = {'controls', 'knots', 'beta', 'point_ids'}
_OPTIONAL = {'triangulated_xyz', 'triangulated_times', 'triangulated_valid'}


def _workspace_path(path):
    """Resolve symlinks before enforcing the single permitted write/read root."""
    if WORKSPACE_ROOT.is_symlink():
        raise ValueError('The viewer workspace root must not be a symlink to experiment/source directories')
    root = WORKSPACE_ROOT.resolve()
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Initialization sidecars must stay under {root}')
    return path


def _sources(run):
    sources = {}
    for label, path in [('observations', run.observations_path), ('manifest', run.manifest_path)]:
        expected = run.method_meta.get(label + '_sha256', run.meta.get(label + '_sha256'))
        if not expected:
            raise ValueError(f'Cannot bind initialization: frozen {label} SHA256 was not recorded')
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f'Cannot bind initialization: frozen {label} SHA256 mismatch')
        sources[label + '_sha256'] = actual
        sources[label + '_path'] = str(Path(path).resolve())
    return sources


def _validated_arrays(run, controls, knots, beta, point_ids, *,
                      triangulated_xyz=None, triangulated_times=None, triangulated_valid=None):
    ids = np.asarray(point_ids)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids):
        raise ValueError('point_ids must be a nonempty one-dimensional integer array')
    if len(np.unique(ids)) != len(ids) or any(int(p) not in run.id_to_index for p in ids):
        raise ValueError('point_ids must be unique identities already queried in this run')
    controls = np.asarray(controls, np.float64)
    knots = np.asarray(knots, np.float64)
    beta = np.asarray(beta, np.float64)
    if controls.ndim != 3 or controls.shape[0] != len(ids) or controls.shape[2] != 3 or controls.shape[1] < 4:
        raise ValueError('controls must have shape [len(point_ids), K>=4, 3]')
    if knots.shape != (controls.shape[1] + 4,) or not np.isfinite(knots).all() or np.any(np.diff(knots) < 0) or knots[3] >= knots[-4]:
        raise ValueError('knots must define a nonempty cubic B-spline domain with K+4 nondecreasing values')
    if not np.isfinite(controls).all():
        raise ValueError('controls must be finite; omit uninitialized point IDs instead of inventing their controls')
    if beta.shape != (len(run.cameras),) or not np.isfinite(beta).all():
        raise ValueError('beta must have one finite offset in seconds per fixed camera')
    reference = int(run.meta.get('reference_camera_index', 0))
    if abs(float(beta[reference])) > 1e-12:
        raise ValueError('The reference camera beta must be zero (fixed temporal gauge)')
    arrays = dict(controls=controls, knots=knots, beta=beta, point_ids=ids.astype(np.int64))
    if triangulated_xyz is None:
        if triangulated_times is not None or triangulated_valid is not None:
            raise ValueError('triangulated_times/valid require triangulated_xyz')
    else:
        xyz = np.asarray(triangulated_xyz, np.float64)
        times = np.asarray(triangulated_times, np.float64)
        if xyz.ndim != 3 or xyz.shape[0] != len(ids) or xyz.shape[2] != 3:
            raise ValueError('triangulated_xyz must have shape [len(point_ids), T, 3]')
        if times.shape != (xyz.shape[1],) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('triangulated_times must contain T strictly increasing physical times in seconds')
        valid = np.isfinite(xyz).all(-1) if triangulated_valid is None else np.asarray(triangulated_valid)
        if valid.dtype.kind != 'b' or valid.shape != xyz.shape[:2]:
            raise ValueError('triangulated_valid must be a boolean [len(point_ids), T] array')
        if np.any(valid & ~np.isfinite(xyz).all(-1)):
            raise ValueError('Valid triangulated samples must have finite XYZ')
        arrays.update(triangulated_xyz=xyz, triangulated_times=times, triangulated_valid=valid)
    return {k: np.array(v, copy=True, order='C') for k, v in arrays.items()}


def _ids_sha256(ids):
    return hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes(order='C')).hexdigest()


def capture_initialization(run, controls, knots, beta, point_ids, *, provenance,
                           workspace=None, triangulated_xyz=None,
                           triangulated_times=None, triangulated_valid=None):
    """Save genuine caller-provided initial arrays as an independent sidecar.

    ``point_ids`` may be a unique subset of the run. Coordinates are world
    meters; beta and optional triangulation times are seconds. ``workspace``
    may select a subdirectory but must remain inside debug_viewer/workspace.
    Every call creates new files; existing snapshots/results are never replaced.
    """
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError('provenance must explicitly describe the source of these initial arrays')
    directory = _workspace_path(workspace if workspace is not None else WORKSPACE_ROOT / 'initialization')
    arrays = _validated_arrays(run, controls, knots, beta, point_ids,
        triangulated_xyz=triangulated_xyz, triangulated_times=triangulated_times,
        triangulated_valid=triangulated_valid)
    sources = _sources(run)
    token = 'initialization_' + uuid.uuid4().hex
    npz_path = _workspace_path(directory / (token + '.npz'))
    json_path = _workspace_path(directory / (token + '.json'))
    directory.mkdir(parents=True, exist_ok=True)
    # Recheck after mkdir: an existing symlink must not redirect the whitelist.
    _workspace_path(directory)
    metadata = dict(schema_version=1, artifact_kind='caller_supplied_initialization',
        created_at_utc=datetime.now(timezone.utc).isoformat(), run_id=run.run_id,
        method=run.method, provenance=provenance.strip(), sources=sources,
        camera_ids=[c['camera_id'] for c in run.cameras],
        point_ids_sha256=_ids_sha256(arrays['point_ids']), beta_seconds=arrays['beta'].tolist(),
        reference_camera_index=int(run.meta.get('reference_camera_index', 0)),
        time_convention='physical_time = nominal_timestamp + beta_seconds',
        units=dict(controls='meters', knots='seconds', beta='seconds', triangulated_xyz='meters', triangulated_times='seconds'),
        array_shapes={k: list(v.shape) for k, v in arrays.items()}, npz_name=npz_path.name,
        reconstruction_run=False,
        note='Caller supplied these arrays; no inference from final results. This sidecar is not part of the archived main result.')
    created = []
    try:
        with npz_path.open('xb') as handle:
            created.append(npz_path)
            np.savez_compressed(handle, **arrays)
        metadata['npz_sha256'] = sha256(npz_path)
        with json_path.open('x', encoding='utf-8') as handle:
            created.append(json_path)
            json.dump(to_jsonable(metadata), handle, indent=2, allow_nan=False)
            handle.write('\n')
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return dict(npz_path=str(npz_path), json_path=str(json_path), point_count=len(arrays['point_ids']),
                run_id=run.run_id, method=run.method)


def load_initialization(run, json_path):
    """Validate an explicitly selected sidecar; never auto-select a snapshot.

    Returns ``{available, meta, arrays}``; all arrays are read-only. This loader
    rejects different runs/methods, changed frozen sources and corrupted NPZs.
    It does not assert that a caller's provenance statement is independently
    certified, and it never changes the archived estimation result.
    """
    json_path = _workspace_path(json_path)
    with json_path.open(encoding='utf-8') as handle:
        metadata = json.load(handle)
    if metadata.get('schema_version') != 1 or metadata.get('artifact_kind') != 'caller_supplied_initialization':
        raise ValueError('Unrecognized initialization sidecar schema')
    if metadata.get('run_id') != run.run_id or metadata.get('method') != run.method:
        raise ValueError('Initialization sidecar belongs to another run or method')
    if not isinstance(metadata.get('provenance'), str) or not metadata['provenance'].strip():
        raise ValueError('Initialization sidecar has no source provenance')
    current = _sources(run)
    if any(metadata.get('sources', {}).get(k) != current[k] for k in ['observations_sha256', 'manifest_sha256']):
        raise ValueError('Initialization sidecar frozen-source identity does not match this run')
    if metadata.get('camera_ids') != [c['camera_id'] for c in run.cameras]:
        raise ValueError('Initialization sidecar camera order differs from fixed calibration')
    npz_path = _workspace_path(json_path.with_suffix('.npz'))
    if metadata.get('npz_name') != npz_path.name or sha256(npz_path) != metadata.get('npz_sha256'):
        raise ValueError('Initialization NPZ name or SHA256 mismatch')
    with np.load(npz_path, allow_pickle=False) as archive:
        if not _REQUIRED.issubset(archive.files) or set(archive.files) - _REQUIRED - _OPTIONAL:
            raise ValueError('Initialization NPZ has missing or unrecognized arrays')
        supplied = {k: archive[k] for k in archive.files}
    arrays = _validated_arrays(run, **supplied)
    if metadata.get('point_ids_sha256') != _ids_sha256(arrays['point_ids']):
        raise ValueError('Initialization point-ID checksum mismatch')
    if metadata.get('array_shapes') != {k: list(v.shape) for k, v in arrays.items()}:
        raise ValueError('Initialization array shapes disagree with metadata')
    if not np.array_equal(np.asarray(metadata.get('beta_seconds')), arrays['beta']):
        raise ValueError('Initialization beta disagrees with metadata')
    for value in arrays.values():
        value.setflags(write=False)
    return dict(available=True, meta=metadata, arrays=arrays)
