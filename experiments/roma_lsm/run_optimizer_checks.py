"""Controlled synthetic LSM verification. This is NOT the RoMa A0/A1/A2 test.

The initialization is a known synthetic correspondence plus a declared smooth
0.5/2/6-pixel perturbation. Visibility and accurate GT enter evaluation only.
Every ROI pixel is passed to batched LSM; no sparse-point interpolation is used.
"""
from pathlib import Path
import argparse
from dataclasses import asdict
import csv
import json
import platform
import sys
import time

import numpy as np
import torch

from geometry import fundamental
from lsm import LSMConfig, STATUS, affine_from_dense, bilinear, refine_points


ROOT = Path(__file__).resolve().parent
ERROR_KEYS = {0.5: 'e0p5', 2.: 'e2', 6.: 'e6'}


def error_stats(error):
    values = np.asarray(error, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0, mean=None, median=None, p90=None,
                    fraction_lt_0p5=None, fraction_lt_1=None, fraction_lt_2=None)
    return dict(count=int(len(values)), mean=float(values.mean()),
                median=float(np.median(values)), p90=float(np.percentile(values, 90)),
                fraction_lt_0p5=float(np.mean(values < .5)),
                fraction_lt_1=float(np.mean(values < 1)),
                fraction_lt_2=float(np.mean(values < 2)))


def coverage(error, accepted, evaluable):
    denominator = int(evaluable.sum())
    return {f'correct_coverage_lt_{str(threshold).replace(".", "p")}':
            float(np.sum(accepted & evaluable & (error < threshold))/denominator)
            if denominator else None for threshold in [.5, 1., 2.]}


def sample_field(field, points):
    with torch.no_grad():
        values, valid = bilinear(torch.as_tensor(field, dtype=torch.float64),
                                 torch.as_tensor(points, dtype=torch.float64))
    return values.numpy(), valid.numpy()


def raw_field_valid(uv, size):
    width, height = size
    return (np.isfinite(uv).all(-1) & (uv[..., 0] >= 0) & (uv[..., 0] <= width-1)
            & (uv[..., 1] >= 0) & (uv[..., 1] <= height-1))


def fb_against_fixed_reverse(forward, reverse, reference):
    back, inside = sample_field(reverse, forward)
    error = np.linalg.norm(back-reference, axis=-1)
    error[~inside | ~np.isfinite(back).all(-1)] = np.nan
    return error


def evaluate_run(initial, proposal, refined, raw_valid, truth, evaluable, status,
                 regions, occluded):
    """Evaluation-only function: GT is never returned to the optimizer."""
    raw_error = np.linalg.norm(initial-truth, axis=-1)
    new_error = np.linalg.norm(proposal-truth, axis=-1)
    own = refined & evaluable & np.isfinite(new_error)
    before_own = own & raw_valid
    metrics = dict(
        attempted_pixels=int(len(initial)),
        original_evaluable_pixels=int(evaluable.sum()),
        successful_pixels=int(refined.sum()),
        successful_evaluable_pixels=int(own.sum()),
        success_rate=float(refined.mean()), failure_rate=float((~refined).mean()),
        success_evaluable_coverage=float(own.sum()/evaluable.sum()) if evaluable.any() else None,
        raw_valid_evaluable_coverage=float((raw_valid & evaluable).sum()/evaluable.sum()) if evaluable.any() else None,
        raw_own_valid_epe=error_stats(raw_error[raw_valid & evaluable]),
        refined_own_valid_epe=error_stats(new_error[own]),
        paired_raw_epe=error_stats(raw_error[before_own]),
        paired_refined_epe=error_stats(new_error[before_own]),
        worsening_ratio=float(np.mean(new_error[before_own] > raw_error[before_own]+1e-6)) if before_own.any() else None,
        failure_histogram={name: int(np.sum(status == code)) for code, name in STATUS.items()},
        raw_correct_coverage=coverage(raw_error, raw_valid, evaluable),
        refined_correct_coverage=coverage(new_error, refined, evaluable),
        occluded_attempts=int(occluded.sum()),
        occluded_accepted=int(np.sum(occluded & refined)),
        evaluation_denominator='Fixed ROI pixel centers with true target visibility; never used by acceptance',
    )
    metrics['regions'] = {}
    for region, name in [(0, 'textured_background'), (1, 'weak_texture'), (2, 'foreground')]:
        selected = evaluable & (regions == region)
        accepted = selected & refined
        metrics['regions'][name] = dict(
            evaluable_pixels=int(selected.sum()), successful_pixels=int(accepted.sum()),
            raw_epe=error_stats(raw_error[selected & raw_valid]),
            paired_raw_epe=error_stats(raw_error[accepted & raw_valid]),
            refined_epe=error_stats(new_error[accepted]),
            **coverage(new_error, refined, selected))
    return metrics, raw_error, new_error


def flatten_record(record):
    flat = {k: record[k] for k in ['case', 'init_error_px', 'method', 'attempted_pixels',
            'original_evaluable_pixels', 'successful_pixels', 'success_rate',
            'success_evaluable_coverage', 'worsening_ratio']}
    for group in ['raw_own_valid_epe', 'refined_own_valid_epe', 'paired_raw_epe',
                  'paired_refined_epe', 'common_refined_epe', 'common_raw_epe']:
        for key, value in record.get(group, {}).items():
            flat[f'{group}.{key}'] = value
    flat.update({f'refined.{k}': v for k,v in record['refined_correct_coverage'].items()})
    flat['seconds_lsm'] = record['timing']['seconds']
    flat['peak_cuda_allocated_bytes'] = record['timing']['peak_cuda_allocated_bytes']
    return flat


def run(args):
    torch.set_num_threads(args.cpu_threads)
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text())
    requested_cases = set(args.cases.split(',')) if args.cases else None
    config = dict(experiment='controlled_optimizer_verification_NOT_RoMa',
                  command=[sys.executable]+sys.argv, suite=str(suite_path), seed=suite['seed'],
                  roi=args.roi, stride=args.stride, device=args.device,
                  python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                  input_images='independently rendered float32 images_float; no PNG quantization',
                  coordinate_system='ideal pinhole working-image pixels, x/y, integer pixel centers',
                  initialization='Known synthetic mapping + declared smooth perturbation; not network output',
                  acceptance='LSM diagnostics only; all regular-grid points attempted; no GT visibility gating',
                  affine_initialization='Fit perturbed dense mapping with fixed raw FB<=3 px screening; unreliable fits use identity',
                  confidence='All ones: neutral synthetic score, NOT a RoMa confidence',
                  forward_backward='Diagnostics against the same fixed perturbed reverse field before/after; reverse is not refined',
                  ground_truth_usage='Evaluation only after initialization fields are loaded; no GT accepts/rejects or weights')
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    records = []
    for metadata_path in suite['cases']:
        path = Path(metadata_path)
        if not path.exists():
            path = suite_path.parent/path.parent.name/path.name
        meta = json.loads(path.read_text()); name = meta['case']
        if requested_cases and name not in requested_cases:
            continue
        array_path = Path(meta['array_path'])
        if not array_path.exists():
            array_path = path.parent/'arrays.npz'
        with np.load(array_path) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        images = arrays['images_float']; height, width = images[0].shape
        x0,y0,x1,y1 = args.roi
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f'ROI {args.roi} outside image {(width,height)}')
        yy,xx = np.mgrid[y0:y1:args.stride, x0:x1:args.stride]
        p = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
        ids = (yy.ravel(), xx.ravel())
        dense_reference = arrays['reference_uv']
        F = fundamental(meta['cameras'][0], meta['cameras'][1])
        applicable = arrays['geometry_applicable_mask'][ids]
        # Truth/visibility/region labels are used solely below by evaluate_run.
        truth = arrays['uv_gt_01'][ids]; evaluable = arrays['valid_gt_01'][ids]
        regions = arrays['region_labels'][ids]; occluded = arrays['occluded_gt_01'][ids]
        for magnitude in args.errors:
            key = ERROR_KEYS[magnitude]
            forward = arrays[f'init_uv_01_{key}']; reverse = arrays[f'init_uv_10_{key}']
            initial = forward[ids]
            raw_valid_dense = raw_field_valid(forward, (width,height))
            raw_valid = raw_valid_dense[ids]
            fb_dense = fb_against_fixed_reverse(forward, reverse, dense_reference)
            fitting_valid = raw_valid_dense & np.isfinite(fb_dense) & (fb_dense <= 3.)
            initial_A, initial_A_reliable = affine_from_dense(
                forward, np.ones((height,width), np.float32), fitting_valid, p,
                LSMConfig(batch_size=args.batch_size), device=args.device)
            group = []
            for model in args.models:
                for mode in args.geometry_modes:
                    method = f'{model}_{mode}'
                    cfg = LSMConfig(model=model, geometry_mode=mode, batch_size=args.batch_size,
                                    max_iterations=args.max_iterations, patch_size=args.patch_size,
                                    max_correction=args.max_correction)
                    print(f'{name} init={magnitude:g}px {method}: {len(p)} stride-{args.stride} pixels', flush=True)
                    result = refine_points(images[0], images[1], p, initial, cfg,
                                           A0=initial_A if model == 'affine' else None,
                                           geometry_applicable=applicable, F=F, device=args.device)
                    refined = result['refined_mask']; proposal = result['uv_refined']
                    final_uv = np.where(refined[:,None], proposal, initial)
                    started = time.perf_counter()
                    metrics, raw_error, new_error = evaluate_run(
                        initial, proposal, refined, raw_valid, truth, evaluable,
                        result['refinement_status'], regions, occluded)
                    metrics.update(case=name, init_error_px=magnitude, method=method,
                                   lsm_config=asdict(cfg), timing=result['timing'],
                                   affine_initialization_reliable_fraction=float(initial_A_reliable.mean()),
                                   fb_before=error_stats(fb_dense[ids]),
                                   fb_after_against_fixed_raw_reverse=error_stats(
                                       fb_against_fixed_reverse(proposal,reverse,p)[refined]),
                                   evaluation_seconds=time.perf_counter()-started)
                    payload = {k:v for k,v in result.items() if k != 'timing'}
                    payload.update(reference_uv=p, uv_raw=initial, uv_refined=final_uv,
                                   uv_refined_proposal=proposal, confidence_raw=np.ones(len(p), np.float32),
                                   valid_raw=raw_valid, valid_refined=refined,
                                   geometry_applicable_mask=applicable,
                                   forward_backward_error_before=fb_dense[ids],
                                   forward_backward_error_after=fb_against_fixed_reverse(proposal,reverse,p),
                                   initial_A=initial_A, initial_A_reliable=initial_A_reliable,
                                   evaluation_only_epe_raw=raw_error,
                                   evaluation_only_epe_refined=new_error,
                                   evaluation_only_visible=evaluable,
                                   grid_shape=np.array(xx.shape), image_size=np.array([width,height]),
                                   stride=np.array(args.stride), roi=np.array(args.roi))
                    run_dir = out/name/key/method; run_dir.mkdir(parents=True, exist_ok=True)
                    save_started = time.perf_counter()
                    np.savez_compressed(run_dir/'matches.npz', **payload)
                    metrics['save_seconds'] = time.perf_counter()-save_started
                    (run_dir/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
                    records.append(metrics)
                    group.append((metrics, refined, raw_error, new_error, run_dir))
                    print(f"  success={metrics['success_rate']:.3f}, paired EPE "
                          f"{metrics['paired_raw_epe']['mean']} -> {metrics['paired_refined_epe']['mean']}", flush=True)
            common = evaluable & raw_valid
            for _,refined,_,_,_ in group:
                common &= refined
            for metrics,_,raw_error,new_error,run_dir in group:
                metrics['common_evaluable_pixels'] = int(common.sum())
                metrics['common_raw_epe'] = error_stats(raw_error[common])
                metrics['common_refined_epe'] = error_stats(new_error[common])
                metrics['common_comparison_methods'] = [item[0]['method'] for item in group]
                (run_dir/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    (out/'metrics.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    rows = [flatten_record(record) for record in records]
    if rows:
        with (out/'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f'Wrote {len(records)} controlled verification runs to {out}',flush=True)
    return records


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--suite', default=str(ROOT/'data/synthetic/suite.json'))
    result.add_argument('--out', default=str(ROOT/'results/optimizer_checks'))
    result.add_argument('--device', default='cpu')
    result.add_argument('--cases', help='comma-separated cases; default all')
    result.add_argument('--errors', nargs='+', type=float, choices=list(ERROR_KEYS), default=[.5,2.,6.])
    result.add_argument('--models', nargs='+', choices=['translation','affine'], default=['translation','affine'])
    result.add_argument('--geometry-modes', nargs='+', choices=['off','soft_epipolar'], default=['off','soft_epipolar'])
    result.add_argument('--roi', nargs=4, type=int, default=[64,48,128,96], metavar=('X0','Y0','X1','Y1'))
    result.add_argument('--stride', type=int, default=1)
    result.add_argument('--patch-size', type=int, default=11)
    result.add_argument('--max-iterations', type=int, default=15)
    result.add_argument('--max-correction', type=float, default=4.)
    result.add_argument('--batch-size', type=int, default=512)
    result.add_argument('--cpu-threads', type=int, default=2)
    return result


if __name__ == '__main__':
    args = parser().parse_args()
    if args.stride < 1 or args.batch_size < 1:
        raise ValueError('stride and batch-size must be positive')
    run(args)
