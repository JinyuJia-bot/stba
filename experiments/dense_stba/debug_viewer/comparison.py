"""Fair read-only comparisons: original exposure keys and physical time differ."""
from pathlib import Path
from functools import lru_cache
import hashlib

import numpy as np

from .adapter import to_jsonable
from .alignment import _stats,_timestamps,_reference_index,_window_mask,_manifest_path
from .rois import resolve_roi
from .scene import trajectory_at


@lru_cache(maxsize=128)
def _hash_cached(path,mtime,size):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def _hash(path):
    path=Path(path).resolve();stat=path.stat()
    return _hash_cached(str(path),stat.st_mtime_ns,stat.st_size)


def _raw_ids(camera):
    if 'raw_frame_ids' in camera:return np.asarray(camera['raw_frame_ids'],int),'original_raw_frame_id'
    count=int(camera.get('n_frames',len(camera.get('image_paths',[]))))
    return np.arange(count),'local_frame_identity_with_verified_same_dataset_and_GT'


def _seed_identity(run):
    j=_reference_index(run);cam=run.cameras[j]
    seed=int(run.obs.get('seed_frame',run.manifest.get('seed_frame',0)))
    ids,kind=_raw_ids(cam)
    if not 0<=seed<len(ids):raise ValueError('Invalid archived reference seed frame')
    paths=cam.get('image_paths',[])
    if paths and Path(paths[seed]).is_file():digest=_hash(paths[seed]);source=str(paths[seed])
    else:
        image=np.asarray(run.image(j,seed))
        digest=hashlib.sha256(str((image.shape,str(image.dtype))).encode()+image.tobytes()).hexdigest()
        source='decoded reference seed image'
    return dict(camera_id=cam['camera_id'],frame_id=int(ids[seed]),frame_identity_kind=kind,
                image_sha256=digest,image_source=source)


def _gt_identity(run):
    path=_manifest_path(run)
    if path is None or not (path.parent/'gt.npz').is_file():return None
    files=[path.parent/'gt.npz',path.parent/'gt_metadata.json']
    if not all(p.is_file() for p in files):raise ValueError('Incomplete independent GT identity')
    return {p.name:_hash(p) for p in files}


def _clock(run):
    j=_reference_index(run);cam=run.cameras[j];local=_timestamps(run,j)
    absolute=cam.get('raw_presentation_timestamps_s')
    if absolute is not None:
        offset=np.asarray(absolute,float)-local
        if len(offset)!=len(local) or not np.allclose(offset,offset[0],atol=1e-9,rtol=0):
            return dict(status='inconsistent',origin=None,kind='raw_presentation_time')
        return dict(status='known',origin=float(offset[0]),kind='raw_presentation_time')
    raw,kind=_raw_ids(cam)
    return dict(status='mapped',kind=kind,origin=None,frame_to_local={int(k):float(v) for k,v in zip(raw,local)})


def _compatible(left,right):
    if left.manifest.get('dataset')!=right.manifest.get('dataset'):
        raise ValueError('Cannot compare different datasets')
    a,b=_seed_identity(left),_seed_identity(right)
    for key in ['camera_id','frame_id','frame_identity_kind','image_sha256']:
        if a[key]!=b[key]:raise ValueError(f'Cannot pair point identities from different reference seed images/exposures ({key})')
    gta,gtb=_gt_identity(left),_gt_identity(right)
    if gta!=gtb:raise ValueError('Cannot compare runs with different independent GT identities')
    left_cameras={c['camera_id']:c for c in left.cameras};right_cameras={c['camera_id']:c for c in right.cameras}
    common=[c['camera_id'] for c in left.cameras if c['camera_id'] in right_cameras]
    if not common:raise ValueError('Runs have no shared cameras')
    for cid in common:
        for field in ['K','R','t','D','size']:
            if not np.array_equal(np.asarray(left_cameras[cid].get(field)),np.asarray(right_cameras[cid].get(field))):
                raise ValueError(f'Camera {cid} {field} changed; pixel/world coordinate systems cannot be mixed')
    ua=left.manifest.get('translation_unit',left.manifest.get('units'))
    ub=right.manifest.get('translation_unit',right.manifest.get('units'))
    if ua!=ub:raise ValueError('World translation units differ')
    ca,cb=_clock(left),_clock(right);shared_clock=False
    if ca['status']=='known' and cb['status']=='known':
        shared_clock=ca['kind']==cb['kind'] and np.isclose(ca['origin'],cb['origin'],atol=1e-9,rtol=0)
    elif ca['status']=='mapped' and cb['status']=='mapped' and ca['kind']==cb['kind']:
        keys=set(ca['frame_to_local'])&set(cb['frame_to_local'])
        shared_clock=bool(keys) and all(np.isclose(ca['frame_to_local'][k],cb['frame_to_local'][k],atol=1e-9,rtol=0) for k in keys)
    for run in [left,right]:
        if abs(float(run.result['beta'][_reference_index(run)]))>1e-10:shared_clock=False
    return dict(dataset=left.manifest.get('dataset'),same_reference_seed=True,reference_seed=a,
        same_gt_identity=True,ground_truth_present=gta is not None,shared_camera_ids=common,
        fixed_calibration_equal=True,world_units=ua,shared_reference_clock=bool(shared_clock),
        left_clock=ca,right_clock=cb)


def _exact_observation_equal(a,b):
    if a.dtype!=b.dtype:return np.zeros(a.shape[:-1],bool)
    shape=a.shape[:-1]+(a.shape[-1]*a.dtype.itemsize,)
    left=np.ascontiguousarray(a).view(np.uint8).reshape(shape)
    right=np.ascontiguousarray(b).view(np.uint8).reshape(shape)
    return np.all(left==right,axis=-1)


def _bounded_ids(values,limit=2000):
    values=np.asarray(values)
    return values if len(values)<=limit else values[np.linspace(0,len(values)-1,limit,dtype=int)]


def _own_coverage(run,mask):
    indices=np.flatnonzero(mask);observed=np.asarray(run.result['observed_uv'])[indices]
    predicted=np.asarray(run.result['predicted_uv'])[indices]
    held=np.asarray(run.result['holdout_visible'],bool)[indices]&_window_mask(run,observed.shape[2])[None,None]
    original=held&np.isfinite(observed).all(-1);finite=original&np.isfinite(predicted).all(-1)
    valid=np.asarray(run.result['valid'],bool)[indices]&np.asarray(run.result['trajectory_valid'],bool)[indices,None]
    return dict(query_count=len(indices),heldout_denominator=int(original.sum()),finite_predictions=int(finite.sum()),
        finite_prediction_coverage=float(finite.sum()/original.sum()) if original.any() else None,
        heldout_error_px=_stats(np.linalg.norm(predicted-observed,axis=-1)[finite]),
        recovered_trajectories=int(valid.any(-1).sum()),supported_trajectory_samples=int(valid.sum()),
        query_time_grid_coverage=float(valid.mean()) if valid.size else None)


def compare_runs(left,right,point_id=None,roi=None,frame=None):
    compatibility=_compatible(left,right)
    left_mask,roi_left=resolve_roi(left,roi);right_mask,roi_right=resolve_roi(right,roi)
    shared=np.intersect1d(np.asarray(left.point_ids)[left_mask],np.asarray(right.point_ids)[right_mask])
    if point_id is not None:
        if int(point_id) not in shared:raise ValueError('Requested point ID is not in both selected reference grids')
        shared=np.array([int(point_id)],dtype=np.int64)
        left_mask=np.asarray(left.point_ids)==int(point_id);right_mask=np.asarray(right.point_ids)==int(point_id)
    left_grid_only=np.setdiff1d(np.asarray(left.point_ids)[left_mask],shared)
    right_grid_only=np.setdiff1d(np.asarray(right.point_ids)[right_mask],shared)
    il=np.asarray([left.index(int(p)) for p in shared],int);ir=np.asarray([right.index(int(p)) for p in shared],int)
    if not np.array_equal(np.asarray(left.result['seed_uv'])[il],np.asarray(right.result['seed_uv'])[ir]):
        raise ValueError('Same point IDs have different reference seed pixels')
    rows=[];examples=[];left_errors=[];right_errors=[];delta_measurements=[]
    total_original=total_same=total_changed=total_common=0
    lcamera={c['camera_id']:i for i,c in enumerate(left.cameras)};rcamera={c['camera_id']:i for i,c in enumerate(right.cameras)}
    # Decompress each requested member once; selected copies avoid holding two
    # complete million-observation archives just to display a small ROI.
    lobs=np.asarray(left.result['observed_uv'])[il];robs=np.asarray(right.result['observed_uv'])[ir]
    lpred=np.asarray(left.result['predicted_uv'])[il];rpred=np.asarray(right.result['predicted_uv'])[ir]
    lh=np.asarray(left.result['holdout_visible'],bool)[il];rh=np.asarray(right.result['holdout_visible'],bool)[ir]
    ltrain=np.asarray(left.result['train_visible'],bool)[il];rtrain=np.asarray(right.result['train_visible'],bool)[ir]
    lbeta=np.asarray(left.result['beta']);rbeta=np.asarray(right.result['beta'])
    for cid in compatibility['shared_camera_ids']:
        lj,rj=lcamera[cid],rcamera[cid];lf,lk=_raw_ids(left.cameras[lj]);rf,rk=_raw_ids(right.cameras[rj])
        if lk!=rk:raise ValueError('Raw exposure frame identity conventions differ')
        frames=np.intersect1d(lf,rf);lm={int(v):i for i,v in enumerate(lf)};rm={int(v):i for i,v in enumerate(rf)}
        li=np.array([lm[int(v)] for v in frames],int);ri=np.array([rm[int(v)] for v in frames],int)
        a,b=lobs[:,lj][:,li],robs[:,rj][:,ri];pa,pb=lpred[:,lj][:,li],rpred[:,rj][:,ri]
        mutual=lh[:,lj][:,li]&rh[:,rj][:,ri]
        mutual &= _window_mask(left,len(lf))[li][None]&_window_mask(right,len(rf))[ri][None]
        mutual &= np.isfinite(a).all(-1)&np.isfinite(b).all(-1)
        equal=_exact_observation_equal(a,b)
        same=mutual&equal;changed=mutual&~same
        good=same&np.isfinite(pa).all(-1)&np.isfinite(pb).all(-1)
        ea=np.linalg.norm(pa-a,axis=-1);eb=np.linalg.norm(pb-b,axis=-1)
        left_errors.extend(ea[good]);right_errors.extend(eb[good])
        delta_measurements.extend(np.linalg.norm(a-b,axis=-1)[changed])
        total_original+=int(mutual.sum());total_same+=int(same.sum());total_changed+=int(changed.sum());total_common+=int(good.sum())
        row=dict(camera_id=cid,frame_identity_kind=lk,shared_raw_frames=frames,
            mutually_heldout_original_exposures=int(mutual.sum()),identical_measured_uv=int(same.sum()),
            changed_measured_uv=int(changed.sum()),common_finite_predictions=int(good.sum()),
            left_error_px=_stats(ea[good]),right_error_px=_stats(eb[good]))
        rows.append(row)
        candidates=np.argwhere(np.isfinite(a).all(-1)&np.isfinite(b).all(-1))
        if frame is not None and len(candidates):
            candidates=candidates[li[candidates[:,1]]==int(frame)]
        if len(candidates)>60:candidates=candidates[np.linspace(0,len(candidates)-1,60,dtype=int)]
        lt,rt=_timestamps(left,lj),_timestamps(right,rj)
        for point,k in candidates:
            tl=float(lt[li[k]]+lbeta[lj]);tr=float(rt[ri[k]]+rbeta[rj])
            examples.append(dict(point_id=int(shared[point]),camera_id=cid,raw_frame_id=int(frames[k]),
                left_local_frame=int(li[k]),right_local_frame=int(ri[k]),left_observed_uv=a[point,k],right_observed_uv=b[point,k],
                same_measurement=bool(equal[point,k]),mutually_heldout=bool(mutual[point,k]),
                left_train=bool(ltrain[point,lj,li[k]]),right_train=bool(rtrain[point,rj,ri[k]]),
                left_holdout=bool(lh[point,lj,li[k]]),right_holdout=bool(rh[point,rj,ri[k]]),
                left_in_window=bool(_window_mask(left,len(lf))[li[k]]),right_in_window=bool(_window_mask(right,len(rf))[ri[k]]),
                both_predictions_finite=bool(np.isfinite(pa[point,k]).all()&np.isfinite(pb[point,k]).all()),
                left_estimated_physical_time=tl,right_estimated_physical_time=tr,
                estimated_time_difference_seconds=tr-tl if compatibility['shared_reference_clock'] else None,
                interpretation='Same archived raw exposure; independently fitted clock estimates can assign different physical times.'))
    le,re=np.asarray(left_errors),np.asarray(right_errors)
    comparison=dict(left_run_id=left.run_id,right_run_id=right.run_id,compatibility=compatibility,
        roi=dict(left=roi_left,right=roi_right),shared_point_ids=_bounded_ids(shared),shared_point_count=len(shared),
        query_grid_pairing=dict(shared_count=len(shared),left_only_count=len(left_grid_only),right_only_count=len(right_grid_only),
            shared_point_ids=_bounded_ids(shared),left_only_point_ids=_bounded_ids(left_grid_only),right_only_point_ids=_bounded_ids(right_grid_only),
            id_lists_are_bounded_display_samples=True,id_list_limit=2000,
            interpretation='Only shared seed identities enter paired statistics. One-grid-only queries are coverage differences, not missing observations of a shared trajectory.'),
        own_coverage=dict(left=_own_coverage(left,left_mask),right=_own_coverage(right,right_mask)),
        exposure_pairing=dict(key=['camera_id','original_raw_frame_id','reference_seed_point_id'],
            per_camera=rows,examples=examples,examples_may_be_display_subsampled=True,
            examples_frame_filter='Finite original UV exposure pairs at the requested left local frame, including training/non-heldout examples; summary statistics remain strictly mutually held-out within both saved windows.',
            physical_time_formula='Each run: archived local timestamp + fitted residual beta; matching exposure keys does not imply equal estimated physical times.'),
        same_measurement_holdout=dict(mutually_heldout_exposures=total_original,identical_target_observations=total_same,
            common_finite_predictions=total_common,common_prediction_coverage=float(total_common/total_same) if total_same else None,
            left=_stats(le),right=_stats(re),paired_error_change_right_minus_left_px=_stats(re-le),
            right_lower_fraction=float(np.mean(re<le)) if len(le) else None,
            interpretation='Only exact same archived UV targets, held out in both runs, with finite predictions in both. No residual threshold or GT selects this set.'),
        changed_measurements=dict(count=total_changed,endpoint_change_px=_stats(delta_measurements),excluded_from_same_measurement_comparison=True,
            interpretation='B4/LSM or changed frontend tracks can alter observed UV for the same physical exposure. Those observations are not mixed into an identical-measurement accuracy comparison.'),
        trajectory_pairing=dict(status='unavailable'),warnings=[],read_only=True)
    if total_changed:comparison['warnings'].append(f'{total_changed} mutually held-out exposures have changed measured UV; excluded from same-measurement statistics.')
    if compatibility['shared_reference_clock'] and len(shared):
        left_times=np.asarray(left.result['trajectory_times'],float);right_times=np.asarray(right.result['trajectory_times'],float)
        if frame is not None:
            if not 0<=int(frame)<len(left_times):raise IndexError('Reference frame outside left trajectory grid')
            queries=left_times[[int(frame)]]
        else:queries=left_times[(left_times>=right_times[0])&(left_times<=right_times[-1])]
        ax,av,_,ak=trajectory_at(left,il,queries);bx,bv,_,bk=trajectory_at(right,ir,queries)
        common=av&bv;error=np.linalg.norm(bx-ax,axis=-1)
        trajectory=dict(status='available',physical_times=queries,common_supported_samples=int(common.sum()),
            left_only_supported_samples=int((av&~bv).sum()),right_only_supported_samples=int((bv&~av).sum()),
            neither_supported_samples=int((~av&~bv).sum()),shared_grid_time_samples=int(common.size),
            common_supported_trajectories=int(common.any(-1).sum()),difference_world_units=_stats(error[common]),
            left_interpolation=ak,right_interpolation=bk,
            interpretation='Paired same reference material-seed identity at the same shared reference-clock physical times, with declared support in both runs. This is trajectory disagreement, not independent accuracy. No extrapolation or model-only continuation enters statistics.')
        if point_id is not None:trajectory.update(point_id=int(point_id),left_xyz=ax[0],right_xyz=bx[0],common_valid=common[0])
        elif frame is not None:
            shown=np.flatnonzero(common[:,0]);shown=shown[np.linspace(0,len(shown)-1,min(2000,len(shown)),dtype=int)] if len(shown) else shown
            trajectory.update(point_ids=shared[shown],left_xyz=ax[shown,0],right_xyz=bx[shown,0],display_subsampled=len(shown)<int(common[:,0].sum()))
        if frame is not None:
            trajectory['support_state_point_ids']=dict(common=_bounded_ids(shared[common[:,0]]),
                left_only_supported=_bounded_ids(shared[av[:,0]&~bv[:,0]]),right_only_supported=_bounded_ids(shared[bv[:,0]&~av[:,0]]),
                neither_supported=_bounded_ids(shared[~av[:,0]&~bv[:,0]]),id_list_limit=2000)
        comparison['trajectory_pairing']=trajectory
    else:
        comparison['trajectory_pairing']['reason']='No verified common reference clock or no shared seed IDs; raw exposure pairing is a separate operation.'
    return to_jsonable(comparison)
