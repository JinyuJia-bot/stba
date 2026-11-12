"""One Plotly-compatible 3D scene, entirely from fixed cameras and saved results."""
from pathlib import Path

import numpy as np

from .adapter import to_jsonable,frame_identity
from .alignment import _timestamps,_reference_index,_manifest_path
from .rois import resolve_roi


def _sample_support(times,flags,queries,max_gap):
    """Exact samples or short valid brackets only; no extrapolation across gaps."""
    times=np.asarray(times,float);flags=np.asarray(flags,bool);q=np.asarray(queries,float)
    n=len(times);out=np.zeros((len(flags),len(q)),bool)
    right=np.searchsorted(times,q,side='left')
    for j,(value,r) in enumerate(zip(q,right)):
        if r<n and np.isclose(times[r],value,atol=1e-10,rtol=0):out[:,j]=flags[:,r]
        elif r>0 and r<n and times[r]-times[r-1]<=max_gap+1e-10:
            out[:,j]=flags[:,r-1]&flags[:,r]
    return out


def trajectory_at(run,indices,times):
    """Return world samples and declared support at a common reference clock.

    This read-only helper is also used by experiment comparison. Spline positions
    can be finite outside observation support, but those never become supported.
    """
    indices=np.asarray(indices,int);query=np.atleast_1d(np.asarray(times,float))
    exported=np.asarray(run.result['trajectory_times'],float)
    if len(exported)<1 or np.any(np.diff(exported)<=0):raise ValueError('Trajectory sample times must be strictly increasing')
    flags=np.asarray(run.result['valid'],bool)[indices]
    traj_ok=np.asarray(run.result['trajectory_valid'],bool)[indices]
    flags &= traj_ok[:,None]
    model=np.asarray(run.result.get('model_only_mask',np.zeros_like(run.result['valid'])),bool)[indices]&traj_ok[:,None]
    ref=_reference_index(run);fps=float(run.cameras[ref]['fps'])
    gap=float(run.method_meta.get('config',{}).get('time_init',{}).get('max_gap_frames',2.1))/fps
    supported=_sample_support(exported,flags,query,gap)
    modeled=_sample_support(exported,flags|model,query,gap)&~supported
    if 'control' in run.result and 'knots' in run.result:
        from ..trajectory_spline import evaluate_spline
        xyz=evaluate_spline(np.asarray(run.result['control'])[indices],query.copy(),np.array(run.result['knots'],copy=True))
        kind='archived_cubic_spline'
    else:
        source=np.asarray(run.result['xyz'])[indices];xyz=np.full((len(indices),len(query),3),np.nan)
        for j,value in enumerate(query):
            r=int(np.searchsorted(exported,value))
            if r<len(exported) and np.isclose(exported[r],value,atol=1e-10,rtol=0):xyz[:,j]=source[:,r]
            elif 0<r<len(exported) and exported[r]-exported[r-1]<=gap+1e-10:
                a=(value-exported[r-1])/(exported[r]-exported[r-1]);xyz[:,j]=(1-a)*source[:,r-1]+a*source[:,r]
        kind='linear_interpolation_of_exported_positions_not_a_fitted_continuous_model'
    finite=np.isfinite(xyz).all(-1);supported &= finite;modeled &= finite
    return xyz,supported,modeled,kind


def _center(camera):
    return -np.asarray(camera['R'],float).T@np.asarray(camera['t'],float)


def _rays(camera,uv):
    xy=np.atleast_2d(np.asarray(uv,float));homogeneous=np.column_stack([xy,np.ones(len(xy))])
    ray=homogeneous@np.linalg.inv(np.asarray(camera['K'],float)).T@np.asarray(camera['R'],float)
    return ray/np.linalg.norm(ray,axis=-1,keepdims=True)


def _camera_shape(camera,scale):
    w,h=camera['size'];corners=np.array([[-.5,-.5],[w-.5,-.5],[w-.5,h-.5],[-.5,h-.5]])
    center=_center(camera);world=center+scale*_rays(camera,corners)
    lines=[]
    for corner in world:lines.extend([center.tolist(),corner.tolist(),None])
    for i in range(4):lines.extend([world[i].tolist(),world[(i+1)%4].tolist(),None])
    return dict(camera_id=camera['camera_id'],center=center,frustum_corners=world,
        line_xyz=lines,frustum_ray_length=scale,frustum_length_is_display_only=True,
        calibration_fixed=True)


def _color_values(run,indices,frame,color,xyz):
    if color in ['support','view_support']:
        return np.asarray(run.result['view_support'])[indices,frame],None,'number of supporting views'
    if color in ['confidence','quality']:
        value=run.result.get('confidence')
        return (np.asarray(value)[indices] if value is not None else np.full(len(indices),np.nan)),None,'heuristic confidence, not calibrated probability'
    if color in ['point_id','identity']:
        return np.asarray(run.point_ids)[indices],None,'reference seed pixel ID'
    if color in ['residual','reprojection']:
        observed=np.asarray(run.result['observed_uv'])[indices,:,frame]
        predicted=np.asarray(run.result['predicted_uv'])[indices,:,frame]
        visible=(np.asarray(run.result['train_visible'])[indices,:,frame]|
                 np.asarray(run.result['holdout_visible'])[indices,:,frame])
        error=np.linalg.norm(predicted-observed,axis=-1);good=visible&np.isfinite(error)
        total=np.where(good,error,0).sum(-1);count=good.sum(-1)
        value=np.divide(total,count,out=np.full(len(indices),np.nan),where=count>0)
        return value,None,'mean residual at local frame indices; camera exposure times may differ'
    if color=='rgb':
        ref=_reference_index(run);seed=int(run.obs.get('seed_frame',run.manifest.get('seed_frame',0)))
        image=np.asarray(run.image(ref,seed));pixel=np.round(np.asarray(run.result['seed_uv'])[indices]).astype(int)
        pixel[:,0]=pixel[:,0].clip(0,image.shape[1]-1);pixel[:,1]=pixel[:,1].clip(0,image.shape[0]-1)
        return None,image[pixel[:,1],pixel[:,0],:3],'RGB from reference seed image; visualization only'
    if color!='speed':raise ValueError(f'Unsupported 3D color: {color}')
    if 'velocity' in run.result:
        velocity=np.asarray(run.result['velocity'])[indices,frame]
        return np.linalg.norm(velocity,axis=-1),None,'archived spline speed in world units/second'
    if 'control' in run.result and 'knots' in run.result:
        from ..trajectory_spline import evaluate_spline
        t=np.asarray(run.result['trajectory_times'])[frame]
        v=evaluate_spline(np.asarray(run.result['control'])[indices],[t],np.array(run.result['knots'],copy=True),1)[:,0]
        return np.linalg.norm(v,axis=-1),None,'derivative of archived spline, world units/second'
    ts=np.asarray(run.result['trajectory_times']);lo=max(frame-1,0);hi=min(frame+1,len(ts)-1)
    source=np.asarray(run.result['xyz'])[indices];valid=np.asarray(run.result['valid'])[indices]
    value=np.full(len(indices),np.nan)
    if hi>lo:
        okay=valid[:,lo]&valid[:,hi]&np.isfinite(source[:,lo]).all(-1)&np.isfinite(source[:,hi]).all(-1)
        value[okay]=np.linalg.norm(source[okay,hi]-source[okay,lo],axis=-1)/(ts[hi]-ts[lo])
    return value,None,'finite-difference speed where both neighboring samples are supported'


def scene_data(run,frame,point_id=None,color='speed',max_points=2000,include_gt=False,roi=None):
    frame=int(frame);max_points=int(max_points)
    if not 1<=max_points<=50000:raise ValueError('max_points must lie in [1,50000]')
    times=np.asarray(run.result['trajectory_times'],float)
    if not 0<=frame<len(times):raise IndexError('Reference trajectory frame outside exported time grid')
    roi_mask,roi_spec=resolve_roi(run,roi)
    time=float(times[frame]);xyz=np.asarray(run.result['xyz'])[:,frame]
    trajectory_valid=np.asarray(run.result['trajectory_valid'],bool)
    supported=np.asarray(run.result['valid'],bool)[:,frame]&trajectory_valid&np.isfinite(xyz).all(-1)&roi_mask
    model=np.asarray(run.result.get('model_only_mask',np.zeros_like(run.result['valid'])),bool)[:,frame]&trajectory_valid&np.isfinite(xyz).all(-1)&~supported&roi_mask
    candidates=np.flatnonzero(supported|model);indices=candidates
    if len(indices)>max_points:indices=indices[np.linspace(0,len(indices)-1,max_points,dtype=int)]
    selected=None if point_id is None else run.index(int(point_id))
    if selected is not None and (supported[selected] or model[selected]) and selected not in indices:
        indices=np.r_[indices[:-1],selected] if len(indices)>=max_points else np.r_[indices,selected]
    values,rgb,color_description=_color_values(run,indices,frame,color,xyz)
    points=xyz[indices]
    scale=max(float(np.linalg.norm(np.ptp(points,axis=0)))*.08,.1) if len(points) else .5
    cameras=[_camera_shape(cam,scale) for cam in run.cameras]
    units=run.manifest.get('translation_unit',run.manifest.get('units','dataset-native units, scale unknown'))
    payload=dict(run_id=run.run_id,method=run.method,frame=frame,reference_physical_time=time,
        roi=roi_spec,selected_outside_roi=bool(selected is not None and not roi_mask[selected]),
        time_definition='Archived trajectory_times, in the fixed reference-camera clock; not target-camera same-frame exposures.',
        units=units,cloud=dict(point_ids=np.asarray(run.point_ids)[indices],xyz=points,
            valid=supported[indices],model_only=model[indices],color_values=values,rgb=rgb,
            color_mode=color,color_description=color_description),cameras=cameras,
        sampling=dict(query_trajectories=len(run.point_ids),total_query_count=len(run.point_ids),
            roi_query_count=int(roi_mask.sum()),supported_count=int(supported.sum()),
            model_only_count=int(model.sum()),displayed_count=len(indices),max_points=max_points,
            method='deterministic display subset; selected supported point retained; no change to optimization or statistics'),
        selected_trajectory=None,observation_rays=[],gt=dict(status='disabled'),read_only=True)
    if selected is not None:
        dense_times=np.linspace(times[0],times[-1],max(120,len(times)))
        line,good,modeled,kind=trajectory_at(run,[selected],dense_times)
        current,good_now,model_now,_=trajectory_at(run,[selected],[time])
        payload['selected_trajectory']=dict(point_id=int(point_id),times=dense_times,xyz=line[0],
            valid=good[0],model_only=modeled[0],kind=kind,
            current_xyz=current[0,0],current_valid=bool(good_now[0,0]),current_model_only=bool(model_now[0,0]),
            display_policy='Draw supported and model-only portions separately; unsupported finite spline continuation must not be presented as measured 3D.')
        beta=np.asarray(run.result['beta'],float)
        observed_for_point=np.asarray(run.result['observed_uv'])[selected].copy()
        predicted_for_point=np.asarray(run.result['predicted_uv'])[selected].copy()
        train_for_point=np.asarray(run.result['train_visible'])[selected].copy()
        holdout_for_point=np.asarray(run.result['holdout_visible'])[selected].copy()
        for j,cam in enumerate(run.cameras):
            nominal=_timestamps(run,j);physical=nominal+beta[j]
            local=int(np.argmin(np.abs(physical-time)))
            uv=observed_for_point[j,local]
            identity=frame_identity(run,j,local)
            train=bool(train_for_point[j,local]);holdout=bool(holdout_for_point[j,local])
            observation=dict(**identity,nominal_timestamp=float(nominal[local]),
                estimated_physical_time=float(physical[local]),difference_from_cloud_time=float(physical[local]-time),
                observed_uv=uv,train=train,holdout=holdout,draw_recommended=train or holdout)
            if np.isfinite(uv).all():
                center=_center(cam);direction=_rays(cam,uv)[0]
                position,support,model_support,_=trajectory_at(run,[selected],[physical[local]])
                point=position[0,0];length=float(np.dot(point-center,direction)) if np.isfinite(point).all() else scale*5
                if not np.isfinite(length) or length<=0:length=scale*5
                observation.update(origin=center,direction=direction,endpoint=center+length*direction,
                    trajectory_at_exposure_time=point,trajectory_supported_at_exposure=bool(support[0,0]),
                    trajectory_model_only_at_exposure=bool(model_support[0,0]),
                    predicted_uv=predicted_for_point[j,local])
            else:observation.update(origin=None,direction=None,endpoint=None,reason='nonfinite archived observation')
            payload['observation_rays'].append(observation)
        payload['rays_interpretation']='Each ray is from its own archived observation and has its own estimated physical timestamp; they are not asserted simultaneous with each other or with the cloud.'
    if include_gt:
        manifest=_manifest_path(run)
        if manifest is None or not (manifest.parent/'gt.npz').is_file():
            payload['gt']=dict(status='unavailable',reason='No independent synthetic GT; published real sync is not 3D GT.')
        else:
            from ..synthetic_dynamic import load_truth
            truth=load_truth(manifest.parent)
            gi=np.flatnonzero(roi_mask)
            if len(gi)>max_points:gi=gi[np.linspace(0,len(gi)-1,max_points,dtype=int)]
            seeds=np.asarray(run.result['seed_uv'])[gi]
            world=truth.evaluate_material_trajectory(seeds,[time])[0]
            visible=np.zeros(len(gi),int)
            for j,cam in enumerate(run.cameras):
                if truth.physical_timestamps[j,0]<=time<=truth.physical_timestamps[j,-1]:
                    visible+=truth.project_material(seeds,cam['camera_id'],[time])['valid'][0]
            gt=dict(status='available',point_ids=np.asarray(run.point_ids)[gi],xyz=world,
                at_least_two_visible_cameras=visible>=2,visibility_count=visible,
                layer='separate_GT_layer_only; does not select estimated cloud or influence estimation')
            if selected is not None:
                gt['selected_trajectory']=dict(times=dense_times,
                    xyz=truth.evaluate_material_trajectory(np.asarray(run.result['seed_uv'])[[selected]],dense_times)[:,0])
            payload['gt']=gt
    return to_jsonable(payload)
