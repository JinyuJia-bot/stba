"""Read-only CPU audit of existing dense pair outputs and fixed calibration.

Recomputes full source/target patch support for every accepted pixel. Does not
run RoMa/LSM, write matches, load correspondence GT, or use a GPU.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from experiment import patch_diagnostics, read_image, read_mask
from lsm import LSMConfig

ROOT = Path(__file__).resolve().parent
METHODS = ['A0', 'A0g', 'A1_translation', 'A1_affine', 'A2_affine']
PRIMARY = ['A0', 'A1_affine', 'A2_affine']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_pair(directory, manifest):
    meta = json.loads((directory/'meta.json').read_text())
    ref_camera, target_camera = meta['cameras']['ref'], meta['cameras']['target']
    if meta.get('dataset_metadata_path'):
        original = json.loads(Path(meta['dataset_metadata_path']).read_text())
    else:
        original = manifest
    original_cameras = {cam['camera_id']: cam for cam in original['cameras']}
    checks = {}
    for label, cam in [('reference', ref_camera), ('target', target_camera)]:
        source = original_cameras[cam['camera_id']]
        checks[f'{label}_calibration_equals_input'] = all(
            np.array_equal(np.asarray(cam[key]),np.asarray(source[key]))
            for key in ['K','R','t','D','size'])
    ref, target = read_image(ref_camera['image_path']), read_image(target_camera['image_path'])
    ref_mask = read_mask(ref_camera,'valid_mask_path',ref.shape)
    target_mask = read_mask(target_camera,'valid_mask_path',target.shape)
    if meta['mode'] == 'asynchronous_unknown':
        if 'static_mask_path' not in ref_camera or 'static_mask_path' not in target_camera:
            raise ValueError('Unknown asynchrony requires explicit static masks in both views')
        ref_mask &= read_mask(ref_camera,'static_mask_path',ref.shape)
        target_mask &= read_mask(target_camera,'static_mask_path',target.shape)
    elif meta['mode'] not in ['static','synchronized']:
        raise ValueError(f"Unknown mode: {meta['mode']}")
    cfg = LSMConfig(**meta['config']['lsm'])
    with np.load(meta['cache_path']) as archive:
        cached_raw = archive['uv_AB']
    yy,xx = np.mgrid[:ref.shape[0],:ref.shape[1]]
    grid = np.stack([xx,yy],-1)
    records = {};valid_fields = {};shared_roi = None;shared_raw_valid = None
    for method in METHODS:
        file = directory/f'{method}.npz'
        if not file.exists():
            continue
        with np.load(file) as archive:
            data = {key:archive[key] for key in archive.files}
        valid,success,attempt,roi = (data[key].astype(bool) for key in
            ['valid_refined','refined_mask','attempt_mask','reference_roi'])
        if shared_roi is None:
            shared_roi = roi.copy();shared_raw_valid = data['valid_raw'].copy()
        valid_fields[method] = valid
        raw = data['uv_raw'];delta = np.linalg.norm(data['uv_refined']-raw,axis=-1)
        check = dict(
            raw_correspondence_bitwise_unchanged=(raw.shape==cached_raw.shape and
                raw.dtype==cached_raw.dtype and raw.tobytes()==cached_raw.tobytes()),
            reference_roi_identical_across_methods=np.array_equal(roi,shared_roi),
            raw_valid_identical_across_methods=np.array_equal(data['valid_raw'],shared_raw_valid),
            valid_subset_of_roi=not bool((valid & ~roi).any()),
            success_subset_of_valid_and_attempted=not bool((success & ~(valid & attempt)).any()),
            status_success_matches_accepted_attempts=np.array_equal(
                success,attempt & valid & (data['refinement_status']==1)),
            accepted_coordinates_finite=bool(np.isfinite(data['uv_refined'][valid]).all()),
            geometry_only_in_reference_allowed_mask=not bool(
                (data['geometry_applicable_mask'] & ~ref_mask).any()),
            controls_not_counted_as_refinement=(not attempt.any() and not success.any())
                if method in ['A0','A0g'] else True)
        displacement_bound_applies = method.startswith(('A1','A2'))
        max_displacement = float(np.max(delta[attempt])) if attempt.any() else None
        check['center_update_within_bound'] = (
            max_displacement is None or max_displacement <= cfg.max_correction+1e-4
        ) if displacement_bound_applies else True
        bounds = {}
        if valid.any():
            A = data['A'][valid]
            sv = np.linalg.svd(A,compute_uv=False);det = np.linalg.det(A)
            gains = data['gain'][valid];biases=data['bias'][valid]
            check['accepted_affine_in_bounds'] = bool(
                sv.max()<=cfg.affine_sv_max+1e-5 and sv.min()>=cfg.affine_sv_min-1e-5
                and det.min()>cfg.affine_det_min-1e-5)
            check['accepted_photometric_parameters_in_bounds'] = bool(
                gains.min()>=cfg.gain_min-1e-6 and gains.max()<=cfg.gain_max+1e-6
                and np.abs(biases).max()<=cfg.bias_limit+1e-6)
            _,patch_valid = patch_diagnostics(ref,target,grid[valid],data['uv_refined'][valid],
                A,ref_mask,target_mask,cfg,'cpu',gains,biases)
            check['all_accepted_source_and_target_patches_allowed'] = bool(patch_valid.all())
            bounds = dict(min_affine_singular_value=float(sv.min()),
                max_affine_singular_value=float(sv.max()),min_affine_determinant=float(det.min()),
                min_gain=float(gains.min()),max_gain=float(gains.max()),max_abs_bias=float(np.abs(biases).max()))
        else:
            check.update(accepted_affine_in_bounds=True,
                accepted_photometric_parameters_in_bounds=True,
                all_accepted_source_and_target_patches_allowed=True)
        records[method] = dict(passed=all(check.values()),checks=check,
            roi_pixels=int(roi.sum()),raw_valid_pixels=int(data['valid_raw'].sum()),
            attempted_pixels=int(attempt.sum()),accepted_refinement_pixels=int(success.sum()),
            valid_pixels=int(valid.sum()),max_displacement_among_attempted_px=max_displacement,
            max_correction_bound_px=cfg.max_correction if displacement_bound_applies else None,
            parameter_bounds_observed=bounds)
    if not records:
        raise ValueError('No existing A0/A1/A2 result files')
    common = np.logical_and.reduce(list(valid_fields.values()))
    primary = [name for name in PRIMARY if name in valid_fields]
    common_primary = np.logical_and.reduce([valid_fields[name] for name in primary]) if primary else np.zeros_like(common)
    metrics_file = directory/'metrics.json'
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())
        checks['reported_common_all_support_matches'] = metrics['common_all']['pixels']==int(common.sum())
        checks['reported_common_primary_support_matches'] = metrics['common_primary']['pixels']==int(common_primary.sum())
        checks['reported_valid_counts_match'] = all(metrics['methods'][name]['valid_pixels']==record['valid_pixels'] for name,record in records.items())
    return dict(pair=directory.name,mode=meta['mode'],passed=all(checks.values()) and all(
        record['passed'] for record in records.values()),checks=checks,
        common_all_pixels=int(common.sum()),common_primary_pixels=int(common_primary.sum()),
        methods=records)


def run(results, output):
    torch.set_num_threads(2)
    manifest = json.loads((ROOT/'data_manifest.json').read_text())
    calibration = {}
    for path,expected in manifest['calibration_sources'].items():
        current = sha256(path)
        calibration[path] = dict(expected_sha256=expected,current_sha256=current,
                                 unchanged=current==expected)
    pairs = []
    for directory in sorted(Path(results).iterdir()):
        if not directory.is_dir() or not (directory/'meta.json').exists():
            continue
        try:
            pairs.append(audit_pair(directory,manifest))
        except Exception as error:
            pairs.append(dict(pair=directory.name,passed=False,error=f'{type(error).__name__}: {error}'))
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        audit_type='Independent read-only CPU audit of existing pair artifacts',
        command=[sys.executable]+sys.argv,
        passed=bool(pairs) and all(p['passed'] for p in pairs) and all(
            c['unchanged'] for c in calibration.values()),
        pairs_checked=len(pairs),method_outputs_checked=sum(len(p.get('methods',{})) for p in pairs),
        accepted_refinement_pixels_checked=sum(m['accepted_refinement_pixels'] for p in pairs for m in p.get('methods',{}).values()),
        source_calibration=calibration,pairs=pairs,
        code_review={
            'ground_truth_separation':'experiment.py and lsm.py do not load GT; evaluate.py loads GT only after results are saved.',
            'fixed_geometry':'F derives from fixed input K/R/t. Asynchronous inputs require both static masks; patch validity is checked throughout optimization.',
            'evaluation_denominator':'Reference ROI is fixed before matching; synthetic accuracy denominator is reference_roi & true visibility, independent of method acceptance.',
            'fallback_policy':'Raw field is retained exactly. Rejected optimizer proposals may remain in uv_refined but valid_refined/refined_mask are false; no fallback counts as success.',
            'common_regions':'All-method and primary-method supports are intersections of their accepted valid fields.',
            'limitations':'This audit verifies code/artifact invariants, not whether manually drawn static masks are semantically perfect or whether matches have true geometric accuracy.'})
    Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ['passed','pairs_checked','method_outputs_checked','accepted_refinement_pixels_checked']}))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    parser.add_argument('--output',type=Path,default=ROOT/'audit.json')
    args=parser.parse_args()
    result=run(args.results,args.output)
    sys.exit(0 if result['passed'] else 1)
