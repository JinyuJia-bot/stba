"""Read-only reproducibility/numerical audit of completed Dense STBA artifacts.

This audit never loads synthetic GT or published synchronization offsets. It
checks provenance, actual measurement isolation, fixed geometry, timing bounds,
trajectory projection, and supported-validity bookkeeping. Accuracy relative to
GT is evaluated separately. Run only after all selected result files are closed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
try:
    from .trajectory_spline import evaluate_spline
    from .time_init import training_split,interpolate_tracks,triangulate_samples
except ImportError:
    from trajectory_spline import evaluate_spline
    from time_init import training_split,interpolate_tracks,triangulate_samples
from experiments.roma_lsm.geometry import project

METHODS=['B0','B1','B2','B3','B4']


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def load_npz(path):
    with np.load(path,allow_pickle=False) as archive:return {key:archive[key] for key in archive.files}


def require(condition,message):
    if not bool(condition):raise AssertionError(message)


def same(actual,expected,label):
    require(np.array_equal(actual,expected,equal_nan=True),label+' differs')


def close(actual,expected,label,tolerance=1e-9):
    require(actual.shape==expected.shape,label+' shape differs')
    require(np.array_equal(np.isfinite(actual),np.isfinite(expected)),label+' finite mask differs')
    finite=np.isfinite(actual)
    maximum=float(np.max(np.abs(actual[finite]-expected[finite]))) if finite.any() else 0.
    require(maximum<=tolerance,f'{label}: maximum error {maximum} exceeds {tolerance}')
    return maximum


def support_counts(obs,valid,beta,samples,max_gap_frames=2.1):
    fps=1/np.median(np.diff(obs['timestamps'][0]));masks=[]
    for camera in range(valid.shape[1]):
        _,support=interpolate_tracks(obs['uv'][:,camera],valid[:,camera],obs['timestamps'][camera],
                                     samples-beta[camera],max_gap_frames/fps)
        masks.append(support)
    return np.stack(masks,-1).sum(-1)


def audit_case(directory):
    directory=Path(directory);common=json.loads((directory/'meta.json').read_text())
    declared=common['methods'];methods=[name for name in METHODS if name in declared]
    require(bool(methods),'No declared experimental methods')
    for method in methods:
        require((directory/f'{method}.npz').is_file() and (directory/f'{method}.json').is_file(),
                f'{method}: incomplete method output; wait for the run to finish')
    manifest_path=Path(common['manifest_path']);manifest=json.loads(manifest_path.read_text())
    manifest_hash=sha256(manifest_path)
    require(common.get('manifest_sha256')==manifest_hash,'Case manifest hash changed')
    cameras=manifest['cameras'];C=len(cameras);reference=int(common['reference_camera_index'])
    camera_ids=np.array([cam['camera_id'] for cam in cameras])
    calibration_hashes={}
    for path,expected in manifest.get('calibration_sources',{}).items():
        actual=sha256(path);require(actual==expected,f'Calibration source changed: {path}')
        calibration_hashes[path]=actual
    previous_path=ROOT.parent/'roma_lsm/data_manifest.json'
    reused_calibration=None
    if calibration_hashes and previous_path.exists():
        previous=json.loads(previous_path.read_text());old={cam['camera_id']:cam for cam in previous['cameras']}
        for cam in cameras:
            for key in ['K','R','t','D','raw_K','raw_D','size']:
                if key in cam:same(np.asarray(cam[key]),np.asarray(old[cam['camera_id']][key]),f'Fixed working camera {cam["camera_id"]} {key}')
        reused_calibration=True
    output=dict(directory=str(directory.resolve()),name=common['name'],all_passed=True,methods={},
        manifest_sha256=manifest_hash,calibration_source_sha256=calibration_hashes,
        working_calibration_matches_prior_roma_lsm=reused_calibration,window=common.get('window'),
        no_GT_or_external_sync_accessed=True,
        interpretation='This is a numerical/provenance audit, not an independent accuracy claim. Real beta disagreement with published synchronization is not a failure criterion.')
    arrays={};metadata={};observations={}
    for method in methods:
        data=load_npz(directory/f'{method}.npz');arrays[method]=data
        meta=json.loads((directory/f'{method}.json').read_text());metadata[method]=meta
        require(meta.get('calibration_fixed') is True and common.get('calibration_fixed') is True,'Fixed-calibration declaration missing')
        require(meta.get('manifest_sha256')==manifest_hash,f'{method}: manifest hash changed')
        obs_path=Path(meta['observations_path']);obs_hash=sha256(obs_path)
        require(meta.get('observations_sha256')==obs_hash,f'{method}: frozen observations hash changed')
        require(obs_path.parent.resolve()==directory.resolve(),f'{method}: observations are not frozen beside this result')
        if obs_hash not in observations:observations[obs_hash]=load_npz(obs_path)
        obs=observations[obs_hash];cfg=meta['config']
        for key in ['point_ids','seed_uv','timestamps','query_mask','seed_roi']:
            same(data[key],obs[key],f'{method}: frozen {key}')
        same(data['observed_uv'],obs['uv'],f'{method}: frozen uv')
        same(data['camera_ids'],camera_ids,f'{method}: camera order')
        train,holdout=training_split(obs['visible'],cfg['holdout_every'],cfg['holdout_phase'])
        if common.get('window') is not None:
            keep=np.zeros(train.shape[-1],bool);keep[slice(*common['window'])]=True
            train &= keep[None,None];holdout &= keep[None,None]
        same(data['train_visible'],train,f'{method}: train split/window')
        same(data['holdout_visible'],holdout,f'{method}: holdout split/window')
        require(not np.any(train&holdout),f'{method}: train/holdout overlap')
        beta=np.asarray(data['beta']);initial=np.asarray(data['initial_beta'])
        require(beta.shape==(C,) and np.isfinite(beta).all(),f'{method}: invalid beta shape/value')
        require(beta[reference]==0 and initial[reference]==0,f'{method}: reference time gauge changed')
        fps=1/np.median(np.diff(data['timestamps'][reference]))
        synthetic=manifest['dataset']=='analytic_dynamic_plane'
        bound_frames=cfg['synthetic_beta_bound_frames'] if synthetic else (
            cfg['integer_beta_bound_frames'] if manifest['mode']=='integer_aligned' else cfg['real_raw_beta_bound_frames'])
        bound=bound_frames/fps
        require(np.all(np.abs(beta)<=bound+1e-12),f'{method}: beta hard box violated')
        require(np.all(np.abs(initial)<=bound+1e-12),f'{method}: beta initialization outside box')
        if method in ['B0','B1','B2']:same(beta,initial,f'{method}: fixed beta')
        if method=='B0':same(beta,np.zeros(C),'B0 assumed-synchronous beta')
        require(bool(meta.get('oracle_beta'))==(method=='B2'),f'{method}: oracle-beta label inconsistent')
        valid=np.asarray(data['valid'],bool);traj=np.asarray(data['trajectory_valid'],bool)
        xyz=np.asarray(data['xyz']);samples=np.asarray(data['trajectory_times'])
        require(xyz.shape==valid.shape+(3,),f'{method}: xyz/valid shape mismatch')
        require(np.all(np.isfinite(xyz[valid])),f'{method}: nonfinite valid 3D point')
        require(not np.any(valid&~traj[:,None]),f'{method}: valid sample from rejected trajectory')
        require(np.all(data['view_support'][valid]>=2),f'{method}: valid 3D samples with fewer than two views')
        require(not np.any(valid&data['model_only_mask']),f'{method}: measured and model-only masks overlap')
        for cam in cameras:
            _,z=project(cam,xyz,True)
            require(np.all(z[valid]>0),f'{method}: valid sample behind a camera')
        replay={};inlier_verified=False
        if 'control' in data:
            knots=data['knots'];control=data['control']
            replay['xyz_max_abs_error']=close(xyz,evaluate_spline(control,samples,knots),f'{method}: spline XYZ')
            for key,derivative in [('velocity',1),('acceleration',2)]:
                if key in data:replay[key+'_max_abs_error']=close(data[key],evaluate_spline(control,samples,knots,derivative),f'{method}: spline {key}')
            actual_prediction=np.empty_like(data['predicted_uv']);actual_depth=np.empty(actual_prediction.shape[:-1])
            for j,cam in enumerate(cameras):
                at_obs=evaluate_spline(control,data['timestamps'][j]+beta[j],knots)
                actual_prediction[:,j],actual_depth[:,j]=project(cam,at_obs,True)
            replay['projection_max_abs_error_px']=close(data['predicted_uv'],actual_prediction,f'{method}: fixed-camera projection')
            expected_support=support_counts(obs,data['observation_inlier'],beta,samples)
            same(data['view_support'],expected_support,f'{method}: inlier-based view support')
            before=support_counts(obs,train,beta,samples)
            same(data['view_support_before_residual_screen'],before,f'{method}: pre-screen view support')
            inlier=np.asarray(data['observation_inlier'],bool)
            require(not np.any(inlier&~train),f'{method}: inlier includes holdout/unobserved entry')
            residual=np.linalg.norm(actual_prediction-data['observed_uv'],axis=-1)
            threshold=cfg['optimizer']['inlier_threshold_px']
            require(np.all(residual[inlier]<=threshold+1e-9),f'{method}: inlier residual exceeds threshold')
            require(np.all(actual_depth[inlier]>0),f'{method}: inlier behind camera')
            interval=data['support_interval'];inside=(samples[None]>=interval[:,0,None])&(samples[None]<=interval[:,1,None])
            expected=traj[:,None]&inside&(expected_support>=2)&np.isfinite(xyz).all(-1)
            for cam in cameras:expected &= project(cam,xyz,True)[1]>0
            same(valid,expected,f'{method}: valid 3D support mask')
            model_only=traj[:,None]&inside&(expected_support<2)&np.isfinite(xyz).all(-1)
            same(data['model_only_mask'],model_only,f'{method}: model-only support mask')
            inlier_verified=True
            require(np.all(knots[3]<=data['timestamps']+beta[:,None]+1e-12) and np.all(knots[-4]>=data['timestamps']+beta[:,None]-1e-12),f'{method}: evaluation outside knot domain')
        elif method=='B0':
            # B0 deliberately assumes synchronous local indices. The current
            # datasets have identical local nominal time arrays in all cameras.
            require(np.allclose(data['timestamps'],samples[None],atol=1e-12,rtol=0),'B0 baseline assumes matching nominal time grids')
            tri=triangulate_samples(obs['uv'],train,obs['weight'],obs['timestamps'],cameras,beta,samples)
            enough=tri['valid'].sum(-1)>=cfg['minimum_initialization_samples']
            same(valid,tri['valid']&enough[:,None],'B0 triangulated support')
            replay['xyz_max_abs_error']=close(xyz,tri['xyz'],'B0 triangulation')
            actual=np.stack([project(cam,xyz) for cam in cameras],1)
            replay['projection_max_abs_error_px']=close(data['predicted_uv'],actual,'B0 fixed-camera projection')
        else:
            require(not valid.any() and not traj.any(),f'{method}: missing controls with valid output')
        if 'near_static' in common['name'] and method in ['B3','B4']:
            status=meta['observability']['status'];reliable=meta['observability']['reliable']
            require(all(status[j]=='insufficient_motion' and not reliable[j] for j in range(C) if j!=reference),
                    'Near-static time estimates must be explicitly unobservable')
            same(beta,initial,'Near-static beta remains unestimated')
        output['methods'][method]=dict(all_passed=True,observations_sha256=obs_hash,
            frozen_uv_ids_timestamps_identical=True,train_holdout_window_exact=True,beta_gauge_and_boxes_valid=True,
            beta_seconds=beta.tolist(),beta_status=meta['observability']['status'],
            query_tracks=len(traj),valid_tracks=int(traj.sum()),valid_samples=int(valid.sum()),
            train_observations=int(train.sum()),holdout_observations=int(holdout.sum()),
            inlier_support_replayed=inlier_verified,numerical_replay=replay,
            source_calibration_unchanged=True,oracle_beta=(method=='B2'))
    first=arrays[methods[0]]
    for method in methods[1:]:
        for key in ['point_ids','seed_uv','trajectory_times']:
            same(arrays[method][key],first[key],f'Across methods {key}')
    if 'B1' in arrays and 'B3' in arrays:
        for key in ['initial_beta','seed_eligible','observed_uv','train_visible','holdout_visible']:
            same(arrays['B1'][key],arrays['B3'][key],f'B1/B3 shared initialization {key}')
        if 'knots' in arrays['B1'] or 'knots' in arrays['B3']:
            require('knots' in arrays['B1'] and 'knots' in arrays['B3'],'B1/B3 spline availability differs')
            same(arrays['B1']['knots'],arrays['B3']['knots'],'B1/B3 shared initialization knots')
        require(metadata['B1']['observations_sha256']==metadata['B3']['observations_sha256'],'B1/B3 frozen observations differ')
        before=[metadata[m]['objective_initial'] for m in ['B1','B3']]
        if all(v is not None for v in before):require(np.isclose(*before,atol=1e-8,rtol=1e-10),'B1/B3 initial objective differs')
        if all('control_initial' in arrays[m] for m in ['B1','B3']):
            same(arrays['B1']['control_initial'],arrays['B3']['control_initial'],'B1/B3 initial controls')
            control_evidence='Persisted initial controls are identical'
        else:
            control_evidence=('Initial controls were not persisted, so byte identity cannot be proved from artifacts. '
                'run_experiment.py caches the same (tri,fit,selected) tuple for B1/B3; frozen observations, '
                'initial beta, seed eligibility, knots, split and initial objective agree.')
        output['B1_B3_initialization_evidence']=control_evidence
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots',type=Path,nargs='+',default=[ROOT/'results'])
    parser.add_argument('--cases',nargs='+',help='Only audit these leaf directory names')
    parser.add_argument('--output',type=Path,default=ROOT/'results_audit.json')
    args=parser.parse_args();torch.set_num_threads(4)
    directories=[]
    for root in args.roots:
        candidates=[root] if (root/'meta.json').exists() else [p.parent for p in root.rglob('meta.json')]
        for directory in candidates:
            if not any((directory/f'{m}.npz').exists() for m in METHODS):continue
            if args.cases and directory.name not in args.cases:continue
            if directory not in directories:directories.append(directory)
    require(bool(directories),'No selected completed result directories found')
    records=[];failed=[]
    for directory in sorted(directories):
        try:
            result=audit_case(directory);records.append(result)
            print(json.dumps(dict(directory=str(directory),all_passed=True,methods=list(result['methods']))),flush=True)
        except Exception as error:
            failure=dict(directory=str(directory),all_passed=False,error=f'{type(error).__name__}: {error}')
            failed.append(failure);print(json.dumps(failure),flush=True)
    output=dict(all_passed=not failed,checked_case_count=len(records),failed_case_count=len(failed),
        audited_methods=sum(len(r['methods']) for r in records),cases=records,failures=failed,
        auditor_sha256=sha256(__file__),purpose='Reproducibility, numerical consistency and support isolation; no GT accuracy or real-sync agreement assertions')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
