"""Export existing debug diagnostics into viewer workspace without estimation.

Example (run from repository root):
  python -m experiments.dense_stba.debug_viewer.export_diagnostics \
    --run results_density1/corgi_raw --method B3 --point 86142 \
    --camera 1 --frame 15 --roi dog \
    --out experiments/dense_stba/debug_viewer/workspace/exports/failure_86142

--camera is the zero-based camera index, not camera ID. --roi accepts a built-in
name, a JSON rectangle/polygon/point_ids object, or saved:NAME. All ROI geometry
belongs to the reference SEED image. Outputs are diagnostics only: no optimizer,
optical-flow estimator or matching network runs. Missing initial controls cannot
be recovered from final controls. Replayed cycle values are labelled separately
from original recordings. No hook is inserted into the existing pipeline.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .adapter import DatasetStore, frame_identity, sha256, to_jsonable
from .alignment import alignment_panel
from .diagnostics import coverage
from .flow_debug import flow_diagnostics
from .inspector import inspect_point
from .rois import load_roi, normalize_roi


WORKSPACE = Path(__file__).resolve().parent / 'workspace'


def _output_directory(out):
    """Check destination before loading/processing inputs or creating files."""
    if WORKSPACE.is_symlink():
        raise ValueError('Viewer workspace must not be redirected by a symlink')
    base = WORKSPACE.resolve()
    path = Path(out).expanduser().absolute()
    if not path.resolve().is_relative_to(base) or path.resolve() == base:
        raise ValueError('Diagnostic exports must be in a subdirectory of ' + str(base))
    for ancestor in [path, *path.parents]:
        if ancestor == base:
            break
        if ancestor.is_symlink():
            raise ValueError('Export destination must not contain symlink components')
    if path.exists() and not path.is_dir():
        raise ValueError('Export destination must be a directory')
    return path


def _write_json(directory, filename, value):
    destination = directory / filename
    if destination.is_symlink() or (destination.exists() and destination.stat().st_nlink != 1):
        raise ValueError('Refusing symlink or multiply-linked export file: ' + str(destination))
    destination.write_text(json.dumps(to_jsonable(value), indent=2, allow_nan=False) + '\n')
    return {'path': str(destination), 'sha256': sha256(destination), 'size_bytes': destination.stat().st_size}


def _optional(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except (KeyError, FileNotFoundError, ValueError) as error:
        return {'available': False, 'missing': [f'{type(error).__name__}: {error}'],
                'note': 'Unavailable diagnostic was not fabricated or regenerated'}


def _existing_paths(run, inspection=None, flow=None, alignment=None):
    paths = [run.directory / 'meta.json', run.directory / f'{run.method}.json',
             run.directory / f'{run.method}.npz', run.observations_path, run.manifest_path]
    for optional in [run.directory / 'time_initialization.json',
                     run.cache_dir / ('observations_lsm.json' if run.method == 'B4' else 'observations.json')]:
        if optional.is_file():
            paths.append(optional)
    for value in run.manifest.get('calibration_sources', {}):
        if Path(value).is_file():
            paths.append(Path(value))
    for camera in (inspection or {}).get('cameras', []):
        if camera.get('image_path') and Path(camera['image_path']).is_file():
            paths.append(Path(camera['image_path']))
        for key in ['raw_roma', 'seed_roma']:
            record = camera.get(key, {})
            if record.get('available') and record.get('cache_path'):
                path = Path(record['cache_path'])
                paths.append(path)
                for adjacent in [path.with_suffix('.json'), path.parent / 'provenance.json']:
                    if adjacent.is_file():
                        paths.append(adjacent)
    replay = (flow or {}).get('cache_replay', {})
    paths.extend(Path(value) for value in replay.get('verified_cache_paths', []))
    if replay.get('metadata_path'):
        paths.append(Path(replay['metadata_path']))
    # Include actual RGB input files whose grayscale hashes verified flow pairs.
    camera_index = (flow or {}).get('camera_index')
    if camera_index is not None:
        camera = run.cameras[camera_index]
        for record in replay.get('verified_cache_files', []):
            for frame in record['source_frames']:
                if camera.get('image_paths'):
                    paths.append(Path(camera['image_paths'][frame]))
                elif camera.get('frames_path'):
                    paths.append(Path(camera['frames_path']))
    external = (alignment or {}).get('external_sync')
    if external and external.get('status') == 'available':
        paths.append(Path(external['path']))
    for record in (alignment or {}).get('window_stability', {}).get('records', []):
        path = Path(record['source'])
        paths.extend([path, path.with_suffix('.npz'), path.parent / 'meta.json'])
    return sorted({path.resolve() for path in paths if path.is_file()})


def export_diagnostics(run, point_id, *, camera_index=1, frame=None, roi=None, out):
    """Write inspect/coverage/flow/alignment JSON plus hashes; inputs stay read-only."""
    directory = _output_directory(out)
    run.index(point_id)  # Invalid requested identities fail instead of an empty export.
    frame = int(run.meta.get('seed_frame', run.manifest.get('seed_frame', 0))) if frame is None else int(frame)
    frame_identity(run, camera_index, frame)
    roi = normalize_roi(roi)
    before = {str(path): sha256(path) for path in _existing_paths(run)}
    inspection = _optional(inspect_point, run, point_id, frame, include_gt=False)
    region = _optional(coverage, run, roi, camera_index, 'first_failure')
    flow = _optional(flow_diagnostics, run, point_id, camera_index, frame)
    alignment = _optional(alignment_panel, run, roi)
    missing = []
    for name, payload in [('inspect', inspection), ('coverage', region), ('flow', flow), ('alignment', alignment)]:
        for key in ['missing', 'missing_fields']:
            missing.extend(name + ': ' + str(value) for value in payload.get(key, []))
    if 'control_initial' not in run.result:
        missing.append('result.control_initial was not archived; final controls cannot recover the optimization initialization')
    if 'cycle_error' not in run.obs:
        missing.append('obs.cycle_error was not archived; flow.json may contain separately labelled frozen-cache replay only')
    replay = flow.get('cache_replay', {})
    if replay and not replay.get('available'):
        missing.append('flow.cache_replay: ' + replay.get('missing_reason', 'no verified replay available; see flow.json'))
    for name in ['forward', 'backward', 'direct_vs_composed']:
        branch = replay.get(name, {})
        if branch and not branch.get('available'):
            missing.append('flow.cache_replay.' + name + ': ' + branch.get('missing_reason', 'unavailable'))
    for name in ['iteration_history', 'coarse_profiles', 'objective_profile']:
        if alignment.get(name, {}).get('status') in ['missing', 'unsupported']:
            missing.append('alignment.' + name + ': ' + alignment[name].get('reason', alignment[name]['status']))
    paths = _existing_paths(run, inspection, flow, alignment)
    hashes = {str(path): sha256(path) for path in paths}
    changed = [path for path, digest in before.items() if hashes.get(path) != digest]
    if changed:
        raise RuntimeError('Experiment input changed during export; refusing output: ' + ', '.join(changed))
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, payload in [('inspect', inspection), ('coverage', region), ('flow', flow), ('alignment', alignment)]:
        artifacts[name] = _write_json(directory, name + '.json', payload)
    export = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'run_id': run.run_id, 'method': run.method, 'point_id': int(point_id), 'camera_index': camera_index,
        'camera_id': run.cameras[camera_index]['camera_id'], 'frame': frame, 'roi': roi,
        'artifacts': artifacts, 'input_sha256': hashes,
        'input_validation': {'core_sources_unchanged_during_export': True,
            'validated_before_and_after': list(before), 'loaded_run_source_validation': run.source_validation,
            'cached_diagnostics_note': 'Extra cache/image hashes identify files read for optional diagnostic replay; original cycle errors remain missing.'},
        'code_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                       ['export_diagnostics.py', 'adapter.py', 'inspector.py', 'diagnostics.py', 'rois.py', 'flow_debug.py', 'alignment.py']},
        'missing': sorted(set(missing)), 'no_estimation_performed': True, 'gt_evaluation_disabled': True,
        'initial_controls_recovered': False,
        'availability': {'archived_initial_controls': 'control_initial' in run.result,
                         'archived_cycle_errors': 'cycle_error' in run.obs,
                         'separate_frozen_cache_replay': bool(replay.get('available'))},
        'policy': ['Output is a viewer-only sidecar; no original NPZ, cache, manifest, calibration or result is modified.',
                   'No RoMa inference, optical flow estimation, tracking or optimization is launched.',
                   'Unstored initial controls cannot be reconstructed; no fitted controls are relabelled as initial controls.',
                   'Cycle diagnostics, when available, are explicitly recomputed by sampling frozen verified flow fields, not represented as original saved values.',
                   'Future pipelines can archive initial controls at the moment they exist; this exporter does not insert hooks or change the old pipeline.']}
    _write_json(directory, 'export.json', export)
    return export


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True, help='Run ID, e.g. results_density1/corgi_raw')
    parser.add_argument('--method', default='B3', choices=['B0', 'B1', 'B2', 'B3', 'B4'])
    parser.add_argument('--point', required=True, type=int, help='Persistent reference-seed point ID')
    parser.add_argument('--camera', type=int, default=1, help='Zero-based camera index (1 means cam09 for corgi)')
    parser.add_argument('--frame', type=int, help='Local frame index; default is recorded seed frame')
    parser.add_argument('--roi', default='full', help='full/dog/person/background, JSON object, or saved:NAME')
    parser.add_argument('--out', required=True, type=Path, help='Output subdirectory inside debug_viewer/workspace')
    args = parser.parse_args(argv)
    _output_directory(args.out)
    roi = load_roi(args.roi[6:])['roi'] if args.roi.startswith('saved:') else json.loads(args.roi) if args.roi.lstrip().startswith('{') else args.roi
    store = DatasetStore()
    try:
        run = store.load(args.run, args.method)
        result = export_diagnostics(run, args.point, camera_index=args.camera, frame=args.frame, roi=roi, out=args.out)
        print(json.dumps({'export': str(args.out / 'export.json'), 'artifacts': list(result['artifacts']),
                          'input_files_hashed': len(result['input_sha256']), 'missing': result['missing']}, indent=2))
    finally:
        store.close()


if __name__ == '__main__':
    main()
