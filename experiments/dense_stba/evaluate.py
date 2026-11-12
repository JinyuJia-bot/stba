"""Independent Dense STBA evaluation. Never called by selection or optimization.

Synthetic truth is read only here. Real published synchronization is an external
reference, not independent exposure-time GT. All methods retain the query grid.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np

try:
    from .data_loader import timestamps
    from .synthetic_dynamic import load_truth
    from .trajectory_spline import evaluate_spline
except ImportError:
    from data_loader import timestamps
    from synthetic_dynamic import load_truth
    from trajectory_spline import evaluate_spline

METHODS=['B0','B1','B2','B3','B4']
PRIMARY=['B0','B1','B3']


def summary(values,mask=None):
    values=np.asarray(values,dtype=float)
    selected=values.ravel() if mask is None else values[np.asarray(mask,bool)]
    selected=selected[np.isfinite(selected)]
    if not len(selected):return dict(count=0,mean=None,rmse=None,p50=None,p90=None)
    return dict(count=int(len(selected)),mean=float(selected.mean()),
        rmse=float(np.sqrt(np.mean(selected**2))),p50=float(np.median(selected)),
        p90=float(np.percentile(selected,90)))


def ratio(n,d):
    return float(n/d) if d else None


def resolve(path,directory):
    path=Path(path)
    if path.is_absolute() or path.exists():return path
    for prefix in [Path(directory),Path(__file__).resolve().parents[2]]:
        if (prefix/path).exists():return prefix/path
    raise FileNotFoundError(path)


def load_results(directory):
    directory=Path(directory);common={}
    if (directory/'meta.json').exists():common=json.loads((directory/'meta.json').read_text())
    arrays={};metas={}
    for method in METHODS:
        file=directory/f'{method}.npz'
        if not file.exists():continue
        with np.load(file,allow_pickle=False) as data:arrays[method]={key:data[key] for key in data.files}
        metadata=dict(common)
        if (directory/f'{method}.json').exists():metadata.update(json.loads((directory/f'{method}.json').read_text()))
        metas[method]=metadata
    if not arrays:raise ValueError(f'No B0/B1/B2/B3/B4 results in {directory}')
    first=next(iter(arrays));meta=metas[first]
    manifest_path=resolve(meta['manifest_path'],directory)
    digest=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for method,metadata in metas.items():
        candidate=resolve(metadata['manifest_path'],directory)
        if candidate.resolve()!=manifest_path.resolve():raise ValueError('Methods reference different input manifests')
        if metadata.get('observations_path') and not metadata.get('manifest_sha256'):
            raise ValueError(f'{method}: missing manifest hash for external observations; refusing unfrozen evaluation')
        if metadata.get('manifest_sha256') and metadata['manifest_sha256']!=digest:
            raise ValueError(f'{method}: manifest SHA256 mismatch; refusing evaluation against changed inputs')
    return arrays,metas,json.loads(manifest_path.read_text()),manifest_path


def load_verified_observations(data,metadata,directory):
    """Refuse silently mixing saved estimates with a later frontend cache."""
    if not metadata.get('observations_path'):
        return data,dict(source='embedded observed_uv in result',verified=True)
    path=resolve(metadata['observations_path'],directory)
    expected=metadata.get('observations_sha256')
    if not expected:raise ValueError('Missing observations SHA256; refusing to load an unfrozen frontend cache')
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected:raise ValueError(f'Observations SHA256 mismatch: {path}')
    with np.load(path,allow_pickle=False) as archive:observations={key:archive[key] for key in archive.files}
    if not np.array_equal(data['observed_uv'],observations['uv'],equal_nan=True):
        raise ValueError('Saved observed_uv differs from frozen observations uv; refusing mixed-cache metrics')
    for field in ['point_ids','seed_uv']:
        if not np.array_equal(data[field],observations[field],equal_nan=True):
            raise ValueError(f'Saved {field} differs from frozen observations')
    return observations,dict(source=str(path),expected_sha256=expected,current_sha256=actual,
        saved_observed_uv_matches_frozen=True,verified=True)


def beta_comparison(beta,manifest,truth=None,external=None):
    """Beta is PLUS sign, in seconds; integer raw frame shifts are subtracted."""
    cams=manifest['cameras'];reference=next(i for i,c in enumerate(cams) if c['camera_id']==manifest['reference_camera_id'])
    estimate=np.asarray(beta,dtype=float);estimate=estimate-estimate[reference]
    rows=[]
    for i,cam in enumerate(cams):
        fps=float(cam['fps']);row=dict(camera_id=cam['camera_id'],estimated_beta_seconds=float(estimate[i]),
            estimated_beta_frames=float(estimate[i]*fps))
        if truth is not None:
            actual=float(truth.beta_seconds[i]-truth.beta_seconds[reference]);error=estimate[i]-actual
            row.update(gt_beta_seconds=actual,gt_beta_frames=actual*fps,error_ms=float(error*1000),error_frames=float(error*fps))
        if external is not None:
            entry=next(item for item in external['cameras'] if item['camera_id']==cam['camera_id'])
            shift=float(cam.get('integer_raw_frame_shift',entry.get('raw_frame_shift',0)))
            raw=estimate[i]-shift/fps
            reference_raw=float(entry['raw_beta_seconds'])
            row.update(integer_raw_frame_shift=shift,recovered_total_beta_seconds=float(raw),
                published_total_beta_seconds=reference_raw,
                difference_from_published_ms=float((raw-reference_raw)*1000),
                difference_from_published_frames=float((raw-reference_raw)*fps))
        rows.append(row)
    return dict(reference_camera_id=manifest['reference_camera_id'],cameras=rows,
        comparison_kind='independent_synthetic_GT' if truth is not None else
            ('published_reference_NOT_independent_GT' if external is not None else 'no_external_reference'),
        formula='physical_time=nominal_timestamp+beta; total_raw_beta=estimated_residual_beta-integer_raw_frame_shift/fps')


def frontend_metrics(observations,seed_uv,manifest,truth):
    """GT-based association proxy and drift; this function never filters data."""
    uv=observations.get('uv',observations.get('observed_uv'))
    visible=observations.get('visible')
    if visible is None:
        visible=observations['train_visible']|observations['holdout_visible']
    visible=np.asarray(visible,bool);n,c,t=visible.shape;seed=int(manifest['seed_frame'])
    edge=observations.get('edge_status');out=dict(query_pixels=n,cameras=[])
    for j,cam in enumerate(manifest['cameras']):
        gt=truth.project_material(seed_uv,cam['camera_id'],truth.physical_timestamps[j])
        actual=gt['uv'].transpose(1,0,2);valid_gt=gt['valid'].T
        error=np.linalg.norm(uv[:,j]-actual,axis=-1)
        accepted=(edge[:,j]==1) if edge is not None else visible[:,j,seed]
        evaluable=valid_gt[:,seed]&np.isfinite(uv[:,j,seed]).all(-1)
        accepted_eval=accepted&evaluable
        association_error=error[:,seed]
        drift=np.linalg.norm((uv[:,j]-uv[:,j,seed,None])-(actual-actual[:,seed,None]),axis=-1)
        drift_support=visible[:,j]&valid_gt&visible[:,j,seed,None]&valid_gt[:,seed,None]
        out['cameras'].append(dict(camera_id=cam['camera_id'],
            accepted_seed_associations=int(accepted.sum()),gt_visible_seed_queries=int(valid_gt[:,seed].sum()),
            accepted_association_epe_px=summary(association_error,accepted_eval),
            correct_association_proxy_fraction_lt_1px=ratio(int((accepted_eval&(association_error<1.)).sum()),int(accepted_eval.sum())),
            correct_association_proxy_coverage_lt_1px=ratio(int((accepted_eval&(association_error<1.)).sum()),int(valid_gt[:,seed].sum())),
            accepted_occluded_seed_count=int((accepted&gt['occluded'][seed]).sum()),
            temporal_epe_px=summary(error,visible[:,j]&valid_gt),
            temporal_drift_after_removing_seed_error_px=summary(drift,drift_support),
            temporal_epe_per_frame=[summary(error[:,k],visible[:,j,k]&valid_gt[:,k]) for k in range(t)]))
    out['interpretation']='Association correctness is a declared <1 working-pixel GT endpoint proxy at each camera actual exposure time, not an exact physical-ID certificate. Drift subtracts seed endpoint error; it does not excuse initial identity errors.'
    return out


def support_metrics(data):
    valid=np.asarray(data['valid'],bool)&np.isfinite(data['xyz']).all(-1)&np.asarray(data['trajectory_valid'],bool)[:,None]
    times=data['trajectory_times'];support=data['view_support']
    # Sum supported adjacent intervals; disjoint runs do not acquire fake duration.
    duration=((valid[:,1:]&valid[:,:-1])*np.diff(times)[None]).sum(-1)
    intervals=(valid[:,1:]&~valid[:,:-1]).sum(-1)+valid[:,0]
    obs_visible=np.asarray(data['train_visible'],bool)|np.asarray(data['holdout_visible'],bool)
    n=len(valid)
    return dict(query_grid_count=n,
        successful_reference_temporal_tracks=int(np.sum(obs_visible[:,0].sum(-1)>=2)),
        tracks_with_two_or_more_observed_cameras=int(np.sum(obs_visible.any(-1).sum(-1)>=2)),
        declared_recovered_trajectories=int(np.sum(data['trajectory_valid'])),
        recovered_with_valid_samples=int(np.sum(valid.any(-1))),
        recovered_fraction_of_query_grid=ratio(int(valid.any(-1).sum()),n),
        valid_trajectory_samples=int(valid.sum()),
        valid_samples_with_two_or_more_views=int((valid&(support>=2)).sum()),
        valid_samples_with_fewer_than_two_views=int((valid&(support<2)).sum()),
        supported_duration_seconds=summary(duration,valid.any(-1)),
        fragmented_trajectories=int((intervals>1).sum()),
        duration_definition='Sum of adjacent valid sample intervals; gaps are excluded'),valid


def reprojection_metrics(data):
    error=np.linalg.norm(data['predicted_uv']-data['observed_uv'],axis=-1)
    train=np.asarray(data['train_visible'],bool);holdout=np.asarray(data['holdout_visible'],bool)
    if (train&holdout).any():raise ValueError('Train and holdout observations overlap')
    finite=np.isfinite(error)
    return dict(train=summary(error,train),holdout=summary(error,holdout),
        train_observations=int(train.sum()),holdout_observations=int(holdout.sum()),
        train_finite_prediction_coverage=ratio(int((train&finite).sum()),int(train.sum())),
        holdout_finite_prediction_coverage=ratio(int((holdout&finite).sum()),int(holdout.sum())),
        per_camera=[dict(train=summary(error[:,j],train[:,j]),holdout=summary(error[:,j],holdout[:,j])) for j in range(error.shape[1])],
        holdout_per_frame=[summary(error[:,:,k],holdout[:,:,k]) for k in range(error.shape[2])])


def shared_holdout_metrics(arrays,metas):
    """Shared observation masks eliminate prediction-coverage selection effects.

    B4 may observe a different image-derived correspondence after LSM. The second
    comparison therefore fixes the B0 raw measurements and is explicitly not GT.
    """
    if 'B0' not in arrays:return dict(available=False,reason='B0 raw measurement reference absent')
    raw=arrays['B0'];observed=raw['observed_uv'];base=np.asarray(raw['holdout_visible'],bool)&np.isfinite(observed).all(-1)
    frame_count=base.shape[-1]
    def in_window(method):
        allowed=np.ones(frame_count,bool);window=metas[method].get('window')
        if window is not None:
            allowed[:]=False;allowed[slice(*window)]=True
        return allowed[None,None]
    primary=[name for name in PRIMARY if name in arrays]
    primary_mask=base.copy()
    for name in primary:
        data=arrays[name]
        if not np.array_equal(data['observed_uv'],observed,equal_nan=True):
            raise ValueError(f'{name}: primary methods do not share identical raw observed_uv')
        primary_mask &= np.asarray(data['holdout_visible'],bool)&np.isfinite(data['predicted_uv']).all(-1)&in_window(name)
    all_mask=base.copy()
    for name,data in arrays.items():
        all_mask &= np.isfinite(data['predicted_uv']).all(-1)&in_window(name)
    result=dict(available=True,reference_method='B0',
        primary=dict(methods=primary,observations=int(primary_mask.sum()),metrics={}),
        all_methods_fixed_raw=dict(methods=list(arrays),observations=int(all_mask.sum()),metrics={}),
        interpretation='Primary methods share exactly the same original held-out 2D measurements. All-method comparison fixes B0 observed_uv and holdout mask, intersects finite predictions and every method time window. B4 may use different tracked coordinates internally; this fixed raw measurement reference is not independent GT.')
    for name,data in arrays.items():
        error=np.linalg.norm(data['predicted_uv']-observed,axis=-1)
        if name in primary:
            result['primary']['metrics'][name]=summary(error,primary_mask)
        result['all_methods_fixed_raw']['metrics'][name]=summary(error,all_mask)
        result['all_methods_fixed_raw']['metrics'][name]['per_frame']=[summary(error[:,:,k],all_mask[:,:,k]) for k in range(frame_count)]
    return result


def derivative_metrics(data,truth,seed_uv,scope):
    times=np.asarray(data['trajectory_times']);xyz=data['xyz']
    if len(times)<5:return dict(available=False,reason='Fewer than five trajectory samples')
    if 'control' in data and 'knots' in data:
        velocity=evaluate_spline(data['control'],times,data['knots'],derivative=1)
        acceleration=evaluate_spline(data['control'],times,data['knots'],derivative=2)
        estimator='Analytic cubic-spline derivatives'
    else:
        velocity=np.gradient(xyz,times,axis=1,edge_order=2)
        acceleration=np.gradient(velocity,times,axis=1,edge_order=2)
        estimator='Finite differences of exported positions; includes sampling truncation error'
    eps=1e-4
    center=truth.evaluate_material_trajectory(seed_uv,times).transpose(1,0,2)
    plus=truth.evaluate_material_trajectory(seed_uv,times+eps).transpose(1,0,2)
    minus=truth.evaluate_material_trajectory(seed_uv,times-eps).transpose(1,0,2)
    true_velocity=(plus-minus)/(2*eps);true_acceleration=(plus-2*center+minus)/eps**2
    interior=scope.copy();interior[:,:2]=False;interior[:,-2:]=False
    for shift in [-2,-1,1,2]:interior &= np.roll(scope,shift,axis=1)
    return dict(available=True,predicted_derivative=estimator,
        ground_truth_derivative='Continuous analytic material motion, symmetric difference at 0.1ms',
        support='Only interior samples with five consecutive valid/evaluable samples',
        velocity_error_m_per_s=summary(np.linalg.norm(velocity-true_velocity,axis=-1),interior),
        acceleration_error_m_per_s2=summary(np.linalg.norm(acceleration-true_acceleration,axis=-1),interior))


def evaluate_case(directory):
    directory=Path(directory);arrays,metas,manifest,manifest_path=load_results(directory)
    first=next(iter(arrays));reference=arrays[first];ids=reference['point_ids'];seed_uv=reference['seed_uv'];times=reference['trajectory_times']
    for method,data in arrays.items():
        if not np.array_equal(data['point_ids'],ids) or not np.array_equal(data['seed_uv'],seed_uv):
            raise ValueError('All methods must retain the identical full reference query grid')
        if not np.array_equal(data['trajectory_times'],times):raise ValueError('Method trajectory time grids differ')
        if data['xyz'].shape!=(len(ids),len(times),3):raise ValueError('xyz must have shape [N,S,3]')
    truth=load_truth(manifest_path.parent) if (manifest_path.parent/'gt.npz').exists() else None
    external=None
    if manifest.get('external_sync_reference_path'):
        external=json.loads(resolve(manifest['external_sync_reference_path'],manifest_path.parent).read_text())
    support={};valid={}
    for method,data in arrays.items():support[method],valid[method]=support_metrics(data)
    common=np.logical_and.reduce(list(valid.values()))
    primary=[m for m in PRIMARY if m in valid];common_primary=np.logical_and.reduce([valid[m] for m in primary]) if primary else np.zeros_like(common)
    gt_xyz=gt_evaluable=gt_surface=None
    if truth is not None:
        gt_xyz=truth.evaluate_material_trajectory(seed_uv,times).transpose(1,0,2)
        visibility=[]
        for j,cam in enumerate(manifest['cameras']):
            present=(times>=truth.physical_timestamps[j,0])&(times<=truth.physical_timestamps[j,-1])
            visibility.append(truth.project_material(seed_uv,cam['camera_id'],times)['valid'].T&present[None])
        gt_evaluable=np.sum(visibility,axis=0)>=2
        gt_surface=truth.material_from_reference(seed_uv)['surface_id']
    denominator=int(gt_evaluable.sum()) if truth is not None else None
    output=dict(schema_version=1,manifest_path=str(manifest_path),methods_present=list(arrays),
        ground_truth_available=truth is not None,query_grid_count=len(ids),trajectory_sample_count=len(times),
        original_gt_evaluable_samples=denominator,
        original_gt_evaluable_trajectories=int(gt_evaluable.any(-1).sum()) if truth is not None else None,
        denominator_definition='All original query material points at common exported physical times with GT visibility in at least two cameras whose captured time intervals contain the query time; independent of estimated validity, confidence and method selection',
        common_all=dict(methods=list(arrays),valid_samples=int(common.sum())),
        common_primary=dict(methods=primary,valid_samples=int(common_primary.sum())),
        trajectory_times=times.tolist(),methods={},
        interpretation='Synthetic world errors are raw metres without rigid/similarity alignment. Reprojection reduction alone is not independent 3D accuracy. Real published sync values are reference comparisons, not certified GT.')
    rows=[]
    for method,data in arrays.items():
        meta=metas[method];obs,provenance=load_verified_observations(data,meta,directory)
        if 'temporal_success' in obs:
            support[method]['successful_reference_temporal_tracks']=int(obs['temporal_success'].sum())
        if 'multiview_seed' in obs:
            support[method]['tracks_with_two_or_more_observed_cameras']=int(obs['multiview_seed'].sum())
        if 'all_three_seed' in obs:
            support[method]['tracks_with_all_three_associated_cameras']=int(obs['all_three_seed'].sum())
        support[method]['motion_informative_tracks']=int(np.asarray(data['motion_mask'],bool).sum())
        record=dict(method=method,oracle_beta=bool(meta.get('oracle_beta',False)),
            support=support[method],beta=beta_comparison(data['beta'],manifest,truth,external),
            initial_beta_seconds=np.asarray(data['initial_beta']).tolist(),
            reprojection=reprojection_metrics(data),timing=meta.get('timing',{}),
            observability=meta.get('observability'),config=meta.get('config',{}),scopes={},
            valid_samples_per_time=valid[method].sum(0).tolist(),observations_provenance=provenance)
        if truth is not None:
            error=np.linalg.norm(data['xyz']-gt_xyz,axis=-1)
            record['gt_evaluable_samples_per_time']=gt_evaluable.sum(0).tolist()
            record['gt_evaluable_valid_samples_per_time']=(valid[method]&gt_evaluable).sum(0).tolist()
            for scope,mask in [('own',valid[method]),('common_all',common),('common_primary',common_primary)]:
                if scope=='common_primary' and method not in primary:continue
                selected=mask&gt_evaluable
                stats=summary(error,selected)
                stats['valid_coverage_of_original_evaluable_samples']=ratio(int(selected.sum()),denominator)
                for limit in [.005,.01,.02,.05]:
                    stats[f'correct_coverage_lt_{int(limit*1000)}mm']=ratio(int((selected&(error<limit)).sum()),denominator)
                record['scopes'][scope]=dict(position_error_m=stats,
                    derivatives=derivative_metrics(data,truth,seed_uv,selected))
                rows.append(dict(method=method,scope=scope,position_rmse_m=stats['rmse'],position_mae_m=stats['mean'],
                    position_p50_m=stats['p50'],position_p90_m=stats['p90'],valid_samples=stats['count'],
                    coverage=stats['valid_coverage_of_original_evaluable_samples'],oracle_beta=record['oracle_beta']))
            record['position_error_per_time']=[summary(error[:,i],valid[method][:,i]&gt_evaluable[:,i]) for i in range(len(times))]
            record['surface_regions']={}
            for surface in np.unique(gt_surface):
                region=gt_evaluable&(gt_surface==surface)[:,None];chosen=region&valid[method]
                record['surface_regions'][str(int(surface))]=dict(
                    meaning='dynamic_material_plane' if surface==0 else 'static_occluder',
                    original_evaluable_samples=int(region.sum()),position_error_m=summary(error,chosen),
                    coverage=ratio(int(chosen.sum()),int(region.sum())))
            record['frontend_accuracy']=frontend_metrics(obs,seed_uv,manifest,truth)
        else:
            for scope,mask in [('own',valid[method]),('common_all',common),('common_primary',common_primary)]:
                if scope=='common_primary' and method not in primary:continue
                record['scopes'][scope]=dict(valid_samples=int(mask.sum()),query_time_grid_coverage=ratio(int(mask.sum()),mask.size))
            rows.append(dict(method=method,scope='own',valid_samples=int(valid[method].sum()),
                coverage=ratio(int(valid[method].sum()),valid[method].size),oracle_beta=record['oracle_beta'],
                train_reprojection_rmse_px=record['reprojection']['train']['rmse'],
                holdout_reprojection_rmse_px=record['reprojection']['holdout']['rmse']))
        output['methods'][method]=record
    output['shared_holdout_comparison']=shared_holdout_metrics(arrays,metas)
    shared=output['shared_holdout_comparison']
    if shared['available']:
        for method,record in output['methods'].items():
            record['reprojection']['fixed_raw_holdout_common_all']=shared['all_methods_fixed_raw']['metrics'][method]
            if method in shared['primary']['metrics']:
                record['reprojection']['same_observations_holdout_common_primary']=shared['primary']['metrics'][method]
        for row in rows:
            method=row['method'];row['fixed_raw_common_holdout_rmse_px']=shared['all_methods_fixed_raw']['metrics'][method]['rmse']
            if method in shared['primary']['metrics']:row['primary_common_holdout_rmse_px']=shared['primary']['metrics'][method]['rmse']
    (directory/'metrics.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (directory/'metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    args=parser.parse_args();result=evaluate_case(args.directory)
    print(json.dumps(dict(directory=str(args.directory),methods=result['methods_present'],
        ground_truth_available=result['ground_truth_available'],query_grid_count=result['query_grid_count']),indent=2))
