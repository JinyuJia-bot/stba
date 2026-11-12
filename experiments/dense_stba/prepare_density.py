"""Isolated stride-one real frontends, reusing frozen RoMa/temporal field caches.

Run from repository root:
  python -m experiments.dense_stba.prepare_density > experiments/dense_stba/density_frontend.log 2>&1

This does not run RoMa, modify the default stride4 setup, or optimize 3D. The
backend can subsequently use --cache-root experiments/dense_stba/cache_density1
with configs/density1.json and a separate --results-root.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import torch

if __package__ in [None,'']:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from experiments.roma_lsm.prepare_data import sha256
from experiments.dense_stba.cross_view_association import AssociationConfig,build_observations
from experiments.dense_stba.data_frontend_diagnostics import summarize_frontend


ROOT=Path(__file__).resolve().parent


def ensure_directory_link(link,target):
    link,target=Path(link),Path(target).resolve()
    if not target.is_dir():raise FileNotFoundError(target)
    if link.is_symlink():
        if link.resolve()!=target:raise ValueError(f'Existing symlink points elsewhere: {link}')
        return
    if link.exists():raise ValueError(f'Refusing to replace existing non-symlink: {link}')
    link.parent.mkdir(parents=True,exist_ok=True)
    link.symlink_to(os.path.relpath(target,link.parent),target_is_directory=True)


def prepare(source_cache=ROOT/'cache',output_cache=ROOT/'cache_density1'):
    source_cache,output_cache=Path(source_cache),Path(output_cache)
    if source_cache.resolve()==output_cache.resolve():raise ValueError('Density experiment must use an independent directory')
    output_cache.mkdir(parents=True,exist_ok=True)
    cases=[('raw','corgi_raw'),('integer_aligned','corgi_integer_aligned')]
    immutable={}
    for mode,tag in cases:
        manifest_path=ROOT/f'data/real/{mode}/manifest.json'
        manifest=json.loads(manifest_path.read_text())
        for path in [manifest_path,source_cache/tag/'observations.npz',source_cache/tag/'observations.json']:
            immutable[str(path.resolve())]=sha256(path)
        for path in manifest['calibration_sources']:
            immutable[str(Path(path).resolve())]=sha256(path)
        for path in (source_cache/tag/'roma').glob('*.npz'):
            immutable[str(path.resolve())]=sha256(path)
    immutable[str((ROOT/'config.json').resolve())]=sha256(ROOT/'config.json')
    ensure_directory_link(output_cache/'temporal',source_cache/'temporal')
    for _,tag in cases:
        ensure_directory_link(output_cache/tag/'roma',source_cache/tag/'roma')
    torch.set_num_threads(4)
    summary=dict(schema_version=1,level=3,stride=1,default_stride_unchanged=4,
        source_cache=str(source_cache.resolve()),output_cache=str(output_cache.resolve()),
        network_inference_rerun=False,frontend_device='CPU',
        actual_optimized_trajectory_count=0,
        optimization_state='Frontend complete only; candidates require time-aware initialization and a separate actual backend run. No interpolated stride4 trajectories are used.',
        collision_policy='Unchanged0.75px target-space seed competition threshold; at stride1 legitimate neighboring projected samples can compete. Collision rejection is recorded, not tuned for density.',
        config=asdict(AssociationConfig(stride=1)),cases={},frozen_source_sha256=immutable)
    began=time.perf_counter()
    for mode,tag in cases:
        print('BUILD_STRIDE1',tag,flush=True)
        observations,metadata=build_observations(ROOT/f'data/real/{mode}/manifest.json',
                         output_cache/tag,AssociationConfig(stride=1),device='cpu')
        diagnostics=summarize_frontend(output_cache/tag)
        summary['cases'][tag]=dict(metadata=metadata,diagnostics=diagnostics,
            optimization_candidates_before_time_aware_initialization=int(observations['multiview_seed'].sum()),
            all_three_camera_candidates=int(observations['all_three_seed'].sum()),
            observations_sha256=sha256(output_cache/tag/'observations.npz'))
        print(json.dumps({'case':tag,'metadata':metadata,'diagnostics':diagnostics},indent=2),flush=True)
        del observations
    changed=[path for path,digest in immutable.items() if sha256(path)!=digest]
    summary['source_hashes_unchanged']=not changed
    summary['changed_sources']=changed
    summary['total_seconds']=time.perf_counter()-began
    (output_cache/'frontend_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if changed:raise RuntimeError(f'Frozen main inputs changed during density preparation: {changed}')
    print('STRIDE1_FRONTENDS_COMPLETE',json.dumps({tag:{k:v for k,v in value.items() if k not in ['metadata','diagnostics']} for tag,value in summary['cases'].items()}),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-cache',type=Path,default=ROOT/'cache')
    parser.add_argument('--output-cache',type=Path,default=ROOT/'cache_density1')
    args=parser.parse_args();prepare(args.source_cache,args.output_cache)
