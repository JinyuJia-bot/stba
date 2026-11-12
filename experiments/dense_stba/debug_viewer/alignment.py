"""Read-only timing diagnostics from archived optimizer and frontend artifacts."""
from pathlib import Path
import json

import numpy as np

from .adapter import to_jsonable
from .rois import resolve_roi


def _stats(values):
    x=np.asarray(values,float).ravel();x=x[np.isfinite(x)]
    if not len(x):return dict(count=0,mean=None,rmse=None,p50=None,p90=None)
    return dict(count=len(x),mean=float(x.mean()),rmse=float(np.sqrt(np.mean(x*x))),
                p50=float(np.median(x)),p90=float(np.percentile(x,90)))


def _read_json(path):
    path=Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def _manifest_path(run):
    if getattr(run,'manifest_path',None) is not None:return Path(run.manifest_path)
    value=run.method_meta.get('manifest_path',run.meta.get('manifest_path'))
    return Path(value) if value else None


def _timestamps(run,camera_index):
    if 'timestamps' in run.result:return np.asarray(run.result['timestamps'][camera_index],float)
    if 'timestamps' in run.obs:return np.asarray(run.obs['timestamps'][camera_index],float)
    cam=run.cameras[camera_index]
    for key in ['timestamps_s','nominal_timestamps','timestamps']:
        if key in cam:return np.asarray(cam[key],float)
    raise ValueError('Archived nominal timestamps are missing; frame/FPS times are not invented')


def _reference_index(run):
    return next(i for i,c in enumerate(run.cameras) if c['camera_id']==run.manifest['reference_camera_id'])


def _window_mask(run,count):
    mask=np.ones(count,bool);window=run.method_meta.get('window',run.meta.get('window'))
    if window is not None:
        mask[:]=False;mask[slice(*window)]=True
    return mask


def _bounds(run):
    if 'beta_bounds_seconds' in run.result:
        return np.asarray(run.result['beta_bounds_seconds'],float),'archived result.beta_bounds_seconds'
    config=run.method_meta.get('config',run.meta.get('config',{}))
    key=('synthetic_beta_bound_frames' if run.manifest.get('dataset')=='analytic_dynamic_plane' else
         'integer_beta_bound_frames' if run.manifest.get('mode')=='integer_aligned' else 'real_raw_beta_bound_frames')
    if key not in config:return None,'missing archived bound configuration'
    ref=_reference_index(run);fps=float(run.cameras[ref]['fps']);bound=float(config[key])/fps
    bounds=np.tile([-bound,bound],(len(run.cameras),1));bounds[ref]=0
    return bounds,f'archived config.{key} / reference FPS; no optimization performed'


def _external_reference(run):
    path=run.manifest.get('external_sync_reference_path')
    if not path:return None
    content=_read_json(path)
    return dict(status='available' if content is not None else 'missing',path=path,
        independent_ground_truth=False,content=content,
        interpretation='Published synchronization reference, not independently verified exposure-time GT')


def _window_stability(run):
    """Only inspect stored siblings; never start fitting another window."""
    root=Path(__file__).resolve().parents[1]/'stability';records=[];skipped=[]
    selected_manifest=run.method_meta.get('manifest_sha256',run.meta.get('manifest_sha256'))
    if not selected_manifest:return dict(status='missing',records=[],span=[],skipped=[],reason='Selected run has no verified manifest hash for matching archived windows')
    selected_stride=run.method_meta.get('config',{}).get('association',{}).get('stride')
    for path in sorted(root.glob(f'*/{Path(run.directory).name}/{run.method}.json')):
        method=_read_json(path);common=_read_json(path.parent/'meta.json') or {}
        other_hash=method.get('manifest_sha256',common.get('manifest_sha256'))
        if selected_manifest and other_hash!=selected_manifest:
            skipped.append(dict(path=path,reason='different or unverified manifest identity'));continue
        npz=path.with_suffix('.npz')
        if not npz.is_file():continue
        with np.load(npz,allow_pickle=False) as data:
            if 'beta' not in data:continue
            beta=np.asarray(data['beta'],float)
        if beta.shape!=(len(run.cameras),):continue
        total=beta-np.array([c.get('integer_raw_frame_shift',0)/c['fps'] for c in run.cameras])
        stride=method.get('config',{}).get('association',{}).get('stride')
        records.append(dict(source=path,window=method.get('window',common.get('window')),
            beta_seconds=beta,total_beta_seconds=total,
            total_beta_frames=total*np.array([c['fps'] for c in run.cameras]),
            observability=method.get('observability'),stride=stride,
            matches_selected_density=stride==selected_stride))
    span=[]
    for i,cam in enumerate(run.cameras):
        values=[r['total_beta_seconds'][i] for r in records]
        span.append(dict(camera_id=cam['camera_id'],windows=len(values),
            span_ms=float(np.ptp(values)*1000) if len(values)>1 else None,
            span_frames=float(np.ptp(values)*cam['fps']) if len(values)>1 else None))
    return dict(status='available' if records else 'missing',records=records,span=span,skipped=skipped,
        interpretation='Independent archived fits, not a shared global clock estimate. Repeated boundary solutions or different grid densities do not establish stability.')


def alignment_panel(run,roi=None):
    selected,spec=resolve_roi(run,roi);indices=np.flatnonzero(selected);ref=_reference_index(run)
    beta=np.asarray(run.result['beta'],float)
    initial=np.asarray(run.result['initial_beta'],float) if 'initial_beta' in run.result else None
    bounds,bound_source=_bounds(run);external=_external_reference(run)
    gt=dict(available=False,independent=False,source=None,beta_seconds=None,beta_frames=None)
    manifest_path=_manifest_path(run)
    gt_path=None if manifest_path is None else manifest_path.parent/'gt.npz'
    if run.manifest.get('dataset')=='analytic_dynamic_plane' and gt_path is not None and gt_path.is_file():
        with np.load(gt_path,allow_pickle=False) as archive:
            true_beta=np.asarray(archive['beta_seconds'],float) if 'beta_seconds' in archive else None
        if true_beta is not None and true_beta.shape==beta.shape and np.isfinite(true_beta).all():
            gt=dict(available=True,independent=True,source=str(gt_path),beta_seconds=true_beta,
                beta_frames=true_beta*np.array([cam['fps'] for cam in run.cameras]),
                interpretation='Independent synthetic generation truth; viewer-only overlay, never an optimizer or ROI-selection input.')
    published={c['camera_id']:c for c in (external or {}).get('content',{}).get('cameras',[])} if (external or {}).get('content') else {}
    observable=run.method_meta.get('observability') or {};states=observable.get('status',[])
    rows=[]
    for i,cam in enumerate(run.cameras):
        fps=float(cam['fps']);shift=float(cam.get('integer_raw_frame_shift',0));total=beta[i]-shift/fps
        row=dict(camera_id=cam['camera_id'],camera_index=i,reference_gauge=i==ref,
            final_beta_seconds=beta[i],final_beta_frames=beta[i]*fps,
            initial_beta_seconds=None if initial is None else initial[i],
            initial_beta_frames=None if initial is None else initial[i]*fps,
            integer_raw_frame_shift=shift,total_raw_beta_seconds=total,total_raw_beta_frames=total*fps,
            observability_status=states[i] if i<len(states) else 'missing',
            bounds_seconds=None if bounds is None else bounds[i],
            bounds_frames=None if bounds is None else bounds[i]*fps)
        if bounds is not None:
            distance=min(beta[i]-bounds[i,0],bounds[i,1]-beta[i])
            row.update(at_hard_boundary=i!=ref and distance<=1e-5/fps,
                       distance_to_boundary_frames=distance*fps)
        if cam['camera_id'] in published:
            reference=published[cam['camera_id']].get('raw_beta_seconds')
            if reference is not None:row.update(published_total_beta_seconds=reference,
                published_total_beta_frames=reference*fps,difference_from_published_ms=(total-reference)*1000,
                difference_from_published_frames=(total-reference)*fps)
        if gt['available']:
            row.update(gt_beta_seconds=gt['beta_seconds'][i],gt_beta_frames=gt['beta_frames'][i],
                error_to_gt_ms=(beta[i]-gt['beta_seconds'][i])*1000,
                error_to_gt_frames=(beta[i]-gt['beta_seconds'][i])*fps)
        rows.append(row)
    history=[];missing=[];log=run.method_meta.get('log',[])
    for index,entry in enumerate(log):
        values=entry.get('beta_seconds')
        if values is None or np.asarray(values).shape!=(len(beta),):
            missing.append(index);continue
        values=np.asarray(values,dtype=float)
        if not np.isfinite(values).all():missing.append(index);continue
        history.append(dict(log_index=index,iteration=entry.get('iteration'),beta_seconds=values,
            beta_frames=np.asarray(values)*np.array([c['fps'] for c in run.cameras]),
            objective=entry.get('objective'),image_cost=entry.get('image_cost'),motion_cost=entry.get('motion_cost'),
            accepted=entry.get('accepted'),reprojection_rmse_px=entry.get('reprojection_rmse_px')))
    profiles=run.method_meta.get('coarse_profiles');profile_source=f'{run.method}.json:coarse_profiles'
    if profiles is None:
        archived=_read_json(Path(run.directory)/'time_initialization.json')
        profiles=None if archived is None else archived.get('profiles');profile_source='time_initialization.json:profiles'
    observed=np.asarray(run.result['observed_uv'])[indices]
    predicted=np.asarray(run.result['predicted_uv'])[indices]
    error=np.linalg.norm(predicted-observed,axis=-1)
    residuals={};window=_window_mask(run,observed.shape[2])
    for kind,field in [('train','train_visible'),('holdout','holdout_visible')]:
        mask=np.asarray(run.result[field],bool)[indices]&window[None,None]
        finite_observation=mask&np.isfinite(observed).all(-1)
        finite=finite_observation&np.isfinite(predicted).all(-1)
        residuals[kind]=dict(**_stats(error[finite]),original_observations=int(finite_observation.sum()),
            finite_prediction_coverage=float(finite.sum()/finite_observation.sum()) if finite_observation.any() else None,
            per_camera=[dict(camera_id=cam['camera_id'],**_stats(error[:,j][finite[:,j]])) for j,cam in enumerate(run.cameras)])
    moving=np.asarray(run.result.get('motion_mask',np.zeros(len(selected),bool)),bool)
    config=run.method_meta.get('config',{}).get('time_init',{});minimum=int(config.get('min_dynamic_tracks',8))
    train=np.asarray(run.result['train_visible'],bool)[indices]&window[None,None]
    dynamic_by_camera=[int(np.sum(moving[indices]&(train[:,j].sum(-1)>=config.get('min_samples_per_track',6)))) for j in range(len(beta))]
    initialized=run.result.get('seed_eligible')
    final_valid=run.result.get('trajectory_valid')
    initialized_motion=None if initialized is None else [int(np.sum(moving[indices]&np.asarray(initialized,bool)[indices]&train[:,j].any(-1))) for j in range(len(beta))]
    final_motion=None if final_valid is None else [int(np.sum(moving[indices]&np.asarray(final_valid,bool)[indices]&train[:,j].any(-1))) for j in range(len(beta))]
    payload=dict(run_id=run.run_id,method=run.method,roi=spec,beta_rows=rows,
        beta_convention='physical_time=local_timestamp+residual_beta; total_raw_beta=residual_beta-integer_raw_frame_shift/fps',
        bounds_source=bound_source,bounds_optimized_in_this_arm=run.method in ['B3','B4'],
        iteration_history=dict(status='available' if history else 'missing',records=history,
            total_archived_log_rows=len(log),rows_without_beta=missing,
            policy='Only actual beta_seconds entries are shown; no interpolated or invented iterations.'),
        coarse_profiles=dict(status='available' if profiles is not None else 'missing',source=profile_source,
            profiles=profiles,is_reoptimized_stba_objective=False,
            interpretation='Archived coarse time-initialization score on tracked 2D observations; not a profile with spline controls reoptimized at every beta. ROI changes do not recompute it.'),
        objective_profile=dict(status='unsupported',reason='No archived reoptimized objective profile is consumed by this viewer.',
            online_optimization=False,offline_plan=dict(action='export_plan_only',roi=spec,
                required_work='For each beta grid value, separately refit trajectory controls with all other experiment settings fixed; archive costs, support and input hashes.')),
        observability=observable,external_sync=external,synthetic_gt=gt,
        oracle_beta=bool(run.method_meta.get('oracle_beta',False)),
        window_stability=_window_stability(run),residuals=residuals,
        roi_constraints=dict(query_count=len(indices),motion_mask_count=int(moving[indices].sum()),
            candidate_dynamic_training_tracks_per_camera=dynamic_by_camera,
            dynamic_training_tracks_per_camera=dynamic_by_camera,
            dynamic_training_tracks_interpretation='Candidate moving queries with enough training frames; includes queries that were not optimized.',
            initialized_motion_candidates_per_camera=initialized_motion,
            final_valid_motion_candidates_per_camera=final_motion,
            initialized_motion_candidates=initialized_motion,final_valid_motion_candidates=final_motion,
            archived_global_optimizer_dynamic_tracks_per_camera=observable.get('dynamic_tracks_per_camera'),
            archived_global_optimizer_count_source='method_meta.observability.dynamic_tracks_per_camera, unchanged global values; not recomputed for ROI',
            min_dynamic_tracks=minimum,
            meets_track_count_only=[v>=minimum for v in dynamic_by_camera],
            min_motion_span_px=config.get('min_motion_span_px'),min_motion_speed_px_frame=config.get('min_motion_speed_px_frame'),
            interpretation='ROI statistics use archived image-derived motion flags. The minimum-track test is candidate eligibility only, not an observability certificate or exact beta contribution count. Initialized/final counts additionally require saved selection and any training observation. Actual optimizer counts are shown separately as archived global values. Displayed beta remains the saved global estimate; no ROI refit occurs.'),
        read_only=True)
    return to_jsonable(payload)
