"""Single-candidate inspection; no inference, tracking, re-optimization or edits."""
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np
from scipy.interpolate import BSpline
try:
    from .adapter import LazyNPZ,to_jsonable,frame_identity
except ImportError:
    from adapter import LazyNPZ,to_jsonable,frame_identity

TEMPORAL_STATUS={0:'unvisited',1:'tracked',2:'outside_image',3:'fb_inconsistent_possible_occlusion',
    4:'low_seed_texture',5:'multi_frame_drift',6:'lost_after_failure',7:'invalid_seed',8:'invalid_flow'}
EDGE_STATUS={0:'not_attempted',1:'accepted_candidate_identity',2:'RoMa_score_FB_or_bounds',
    3:'competing_identity',4:'temporal_or_multi_anchor_inconsistent'}


def _sample(field,xy):
    field=np.asarray(field);xy=np.asarray(xy,dtype=float)
    h,w=field.shape[:2]
    inside=xy.shape==(2,) and np.isfinite(xy).all() and 0<=xy[0]<=w-1 and 0<=xy[1]<=h-1
    if not inside:return None,False
    x,y=xy;i,j=int(np.floor(x)),int(np.floor(y));i1,j1=min(i+1,w-1),min(j+1,h-1)
    dx,dy=x-i,y-j
    value=(1-dy)*((1-dx)*field[j,i]+dx*field[j,i1])+dy*((1-dx)*field[j1,i]+dx*field[j1,i1])
    return value,bool(np.isfinite(value).all())


def working_to_original(camera,uv):
    """Ideal working pixels -> distorted calibration-native and source pixels.

    Pixel-center resize uses (u+.5)*new_size/old_size-.5. Lens distortion acts
    in normalized camera coordinates, never on already scaled pixel positions.
    """
    value=np.asarray(uv,dtype=float) if uv is not None else np.full(2,np.nan)
    raw_size=camera.get('raw_size',camera['size']);source_size=camera.get('source_size',raw_size)
    result=dict(available=False,uv_work=value,raw_size=raw_size,source_size=source_size,
        convention='x/y integer pixel centers; ideal working ray -> source lens distortion -> native K; half-pixel resize to extracted source')
    if value.shape!=(2,) or not np.isfinite(value).all():result['reason']='nonfinite_work_coordinate';return to_jsonable(result)
    K=np.asarray(camera['K'],float);raw_K=np.asarray(camera.get('raw_K',K),float)
    distortion=np.asarray(camera.get('raw_D',camera.get('D',[])),float).reshape(-1)
    ray=np.linalg.inv(K)@np.r_[value,1.];ray=ray/ray[2]
    model=str(camera.get('raw_distortion_model',camera.get('distortion_model','opencv_radtan'))).lower()
    if distortion.size==0 or np.all(distortion==0):distorted=ray[:2]
    elif 'fisheye' in model:
        if len(distortion)!=4:raise ValueError('Fisheye distortion requires four coefficients')
        distorted=cv2.fisheye.distortPoints(ray[:2].reshape(1,1,2),np.eye(3),distortion)[0,0]
    else:
        if len(distortion) not in [4,5,8,12,14]:raise ValueError('Unsupported OpenCV distortion coefficient count')
        distorted=cv2.projectPoints(ray.reshape(1,3),np.zeros(3),np.zeros(3),np.eye(3),distortion)[0][0,0]
    projected=raw_K@np.r_[distorted,1.];raw=projected[:2]/projected[2]
    source=(raw+.5)*(np.asarray(source_size,float)/np.asarray(raw_size,float))-.5
    inside=lambda q,size:bool(np.isfinite(q).all() and np.all(q>=0) and np.all(q<=np.asarray(size)-1))
    result.update(available=True,uv_raw_distorted=raw,uv_source_distorted=source,
                  inside_raw=inside(raw,raw_size),inside_source=inside(source,source_size),
                  distortion_model=model,distortion_applied=bool(distortion.size and np.any(distortion!=0)))
    return to_jsonable(result)


def _roma_archive(run,path):
    with run._roma_lock:
        if not hasattr(run,'_debug_roma_archives'):run._debug_roma_archives=OrderedDict()
        key=str(path);cache=run._debug_roma_archives
        if key in cache:
            value=cache.pop(key);cache[key]=value;return value
        value=LazyNPZ(path,max_cache_bytes=16*1024**2);cache[key]=value
        while len(cache)>2:
            _,old=cache.popitem(last=False);old.close()
        return value


def _roma_direction(archive,point,source_size,target_size,direction):
    if 'uv_'+direction in archive:
        q,inside=_sample(archive['uv_'+direction],point)
        score,score_ok=_sample(archive['confidence_'+direction],point) if 'confidence_'+direction in archive else (None,False)
        valid,mask_ok=_sample(archive['valid_'+direction],point) if 'valid_'+direction in archive else (None,False)
        return q,score,bool(inside and score_ok and mask_ok and float(valid)>=.999),dict(
            decoding='Stored audited working-image field, bilinear subpixel query',field='uv_'+direction)
    if 'native_warp' not in archive or 'native_certainty' not in archive:
        return None,None,False,dict(reason='No decoded field or native symmetric warp stored')
    warp=archive['native_warp'];certainty=archive['native_certainty']
    if warp.ndim==4:warp=warp[0]
    if certainty.ndim==3:certainty=certainty[0]
    h,double_w=warp.shape[:2];w=double_w//2
    if warp.shape[-1]!=4 or double_w%2:raise ValueError('Unexpected native RoMa v1 symmetric layout')
    xy=(np.asarray(point)+.5)*(np.array([w,h])/np.asarray(source_size))-.5
    field=warp[:,:w,2:] if direction=='AB' else warp[:,w:,:2]
    source_field=warp[:,:w,:2] if direction=='AB' else warp[:,w:,2:]
    source_value,source_ok=_sample(source_field,xy)
    expected_source=2*(xy+.5)/np.array([w,h])-1
    if source_ok and not np.allclose(source_value,expected_source,atol=2e-6,rtol=0):
        raise ValueError('Unexpected native RoMa source lattice/layout; refusing ambiguous decode')
    scores=certainty[:,:w] if direction=='AB' else certainty[:,w:]
    native_good=np.isfinite(field).all(-1)&np.isfinite(scores)&(scores>0)
    normalized,inside=_sample(field,xy);score,score_ok=_sample(scores,xy);mask,mask_ok=_sample(native_good.astype(np.float32),xy)
    q=(np.asarray(normalized)+1)*np.asarray(target_size)/2-.5 if normalized is not None else None
    good=inside and score_ok and mask_ok and float(mask)>=1-1e-6
    if q is not None:good &= bool(np.all(q>=0)&np.all(q<=np.asarray(target_size)-1))
    return q,score,bool(good),dict(decoding='Native v1 symmetric warp -> target pixels; align_corners=False',
        native_source_uv=xy,native_size=[w,h],target_size=target_size)


def raw_roma_at(run,reference_uv,camera_index,frame):
    """Query a cached anchor only. Never substitute a nearby temporal frame."""
    camera_index=int(camera_index);frame=int(frame);reference=int(run.meta.get('reference_camera_index',0))
    anchors=run.obs.get('anchor_frames',np.array([run.meta.get('seed_frame',0)]))
    point=np.asarray(reference_uv,float) if reference_uv is not None else np.full(2,np.nan)
    if camera_index==reference:
        return to_jsonable(dict(available=bool(np.isfinite(point).all()),kind='reference_coordinate',
            reference_uv=point,target_uv=point,frame=frame,confidence=None,forward_backward_error=0.,
            note='Reference coordinates; no self-pair model inference'))
    ref_id=run.cameras[reference]['camera_id'];target_id=run.cameras[camera_index]['camera_id']
    path=run.roma_cache_dir/f'pair_{ref_id}f{frame:04d}_{target_id}f{frame:04d}.npz'
    base=dict(available=False,frame=frame,reference_uv=point,available_anchor_frames=np.asarray(anchors).tolist(),
              cache_path=str(path),source='Pre-existing raw RoMa cache; no model inference',
              frozen_with_result=False)
    if not path.exists():return to_jsonable(dict(base,reason='No RoMa inference was cached for this exact frame'))
    if point.shape!=(2,) or not np.isfinite(point).all():return to_jsonable(dict(base,reason='Reference candidate coordinate is unavailable'))
    field=_roma_archive(run,path);sizes=[run.cameras[reference]['size'],run.cameras[camera_index]['size']]
    q,score,good,decoded=_roma_direction(field,point,sizes[0],sizes[1],'AB')
    back,backscore,back_good,_=_roma_direction(field,q,sizes[1],sizes[0],'BA') if q is not None else (None,None,False,{})
    fb=float(np.linalg.norm(np.asarray(back)-point)) if back is not None else None
    cfg=run.method_meta.get('config',run.meta.get('config',{})).get('association',{})
    accepted=good and back_good and score is not None and float(score)>=cfg.get('confidence_min',.15)
    accepted &= fb is not None and fb<=cfg.get('roma_fb_max',2.)
    return to_jsonable(dict(base,available=True,target_uv=q,confidence=score,backward_uv=back,
        forward_backward_error=fb,raw_model_valid=good,backward_valid=back_good,
        passes_raw_candidate_checks=bool(accepted),decoding=decoded,
        score_meaning='Uncalibrated model score; not a probability or localization variance',
        note='Availability is not accepted physical identity; temporal/anchor/association gates remain separate'))


def _continuous(run,index,row,missing):
    if 'control' not in run.result or 'knots' not in run.result:
        return dict(available=False,reason='No spline controls/knots saved for this method')
    control=run.field_slice('result','control',index);knots=run.result['knots']
    if not np.isfinite(control).all():return dict(available=False,reason='Candidate has no finite reconstructed spline')
    interval=row.get('support_interval');domain=[float(knots[3]),float(knots[-4])]
    observed=interval is not None and np.isfinite(interval).all() and interval[1]>=interval[0]
    low,high=(max(domain[0],float(interval[0])),min(domain[1],float(interval[1]))) if observed else domain
    if high<low:return dict(available=False,reason='No supported spline interval')
    times=np.linspace(low,high,200);spline=BSpline(knots,control,3,extrapolate=False,axis=0)
    return dict(available=True,times=times,xyz=spline(times),velocity=spline(times,nu=1),acceleration=spline(times,nu=2),
        within_observation_support=np.full(len(times),bool(observed)),
        sampling='200 samples of saved spline only; not new measurements or re-optimization',
        knot_domain=domain,observation_support_known=bool(observed))


def inspect_point(run,point_id,frame,include_gt=True):
    index=run.index(point_id);frame=int(frame)
    reference=int(run.meta.get('reference_camera_index',0));frame_identity(run,reference,frame)
    seed=int(run.meta.get('seed_frame',run.manifest.get('seed_frame',0)));missing=[]
    def row(source,key):
        mapping=run.obs if source=='obs' else run.result
        if key not in mapping:missing.append(f'{source}.{key}');return None
        return run.field_slice(source,key,index)
    o={key:row('obs',key) for key in ['uv','visible','weight','status','temporal_fb_error','edge_status',
        'roma_confidence','roma_fb_error','anchor_error','anchor_support','lsm_seed_refined','lsm_seed_status']}
    r={key:row('result',key) for key in ['predicted_uv','train_visible','holdout_visible','observation_inlier',
        'xyz','velocity','acceleration','valid','model_only_mask','view_support','support_interval','trajectory_valid']}
    if 'control_initial' not in run.result:missing.append('result.control_initial (not saved; no initial trajectory will be fabricated)')
    if not run.method_meta.get('log'):missing.append('method_meta.log (not saved or optimizer was not run)')
    seed_uv=np.asarray(run.result['seed_uv'][index]);timestamps=np.asarray(run.result.get('timestamps',run.obs.get('timestamps')))
    beta=np.asarray(run.result.get('beta',np.zeros(len(run.cameras))))
    anchors=np.asarray(run.obs.get('anchor_frames',[]));cameras=[]
    def take(value,*indices):return None if value is None else value[indices]
    for c,cam in enumerate(run.cameras):
        identity=frame_identity(run,c,frame);candidate=take(o['uv'],c,frame);status=take(o['status'],c,frame)
        pred=take(r['predicted_uv'],c,frame);visible=take(o['visible'],c,frame)
        observed=candidate;residual=None
        if pred is not None and observed is not None and np.isfinite(pred).all() and np.isfinite(observed).all():residual=pred-observed
        ref_point=take(o['uv'],reference,frame)
        raw=raw_roma_at(run,ref_point,c,frame);seed_raw=raw_roma_at(run,seed_uv,c,seed)
        anchor_rows=[]
        for k,anchor in enumerate(anchors):
            anchor_rows.append(dict(frame=int(anchor),error_px=take(o['anchor_error'],c,k),
                supported=take(o['anchor_support'],c,k),independent_of_seed=int(anchor)!=seed))
        inside_patch=lambda uv:bool(uv is not None and np.isfinite(np.asarray(uv,float)).all() and np.all(np.asarray(uv,float)>=5) and np.all(np.asarray(uv,float)<=np.asarray(cam['size'])-6))
        centers=dict(candidate_work=candidate,predicted_work=pred,raw_roma_work=raw.get('target_uv'))
        mapping={name:working_to_original(cam,value) for name,value in centers.items()}
        edge=take(o['edge_status'],c)
        cameras.append(dict(**identity,timestamp=float(timestamps[c,frame]),beta_seconds=float(beta[c]),
            actual_time=float(timestamps[c,frame]+beta[c]),image_size=cam['size'],
            candidate=dict(uv=candidate,status=status,status_label=TEMPORAL_STATUS.get(int(status),'unknown') if status is not None else 'missing',
                temporal_visible=None if status is None else int(status)==1,visible_after_association=visible,
                visibility_note='Temporal visibility inferred from saved status==tracked; final visible additionally requires association'),
            observation=dict(uv=observed,visible=visible,weight=take(o['weight'],c,frame),
                             train=take(r['train_visible'],c,frame),holdout=take(r['holdout_visible'],c,frame)),
            prediction=dict(uv=pred,residual_vector=residual,residual_px=None if residual is None else float(np.linalg.norm(residual)),
                            inlier=take(r['observation_inlier'],c,frame),evaluated_at='camera nominal timestamp + its estimated global beta'),
            association=dict(edge_status=edge,edge_label=EDGE_STATUS.get(int(edge),'unknown') if edge is not None else 'missing',
                roma_confidence=take(o['roma_confidence'],c),roma_fb_error=take(o['roma_fb_error'],c),
                temporal_fb_error=take(o['temporal_fb_error'],c,frame),anchors=anchor_rows,
                lsm_seed_refined=take(o['lsm_seed_refined'],c),lsm_seed_status=take(o['lsm_seed_status'],c)),
            raw_roma=raw,seed_roma=seed_raw,coordinate_mapping=mapping,
            patch=dict(default_size=11,half_size=5,centers=centers,
                fully_inside_working_image={name:inside_patch(value) for name,value in centers.items()},
                note='Patch centers are saved candidates/predictions; no patch was refined by this inspector'),
            timeline=dict(uv=take(o['uv'],c),predicted_uv=take(r['predicted_uv'],c),visible=take(o['visible'],c),
                status=take(o['status'],c),temporal_fb_error=take(o['temporal_fb_error'],c),
                train=take(r['train_visible'],c),holdout=take(r['holdout_visible'],c),inlier=take(r['observation_inlier'],c),
                timestamps=timestamps[c],actual_times=timestamps[c]+beta[c])))
    times=run.result.get('trajectory_times',timestamps[reference])
    def current_value(key):
        value=r[key]
        return take(value,frame) if value is not None and frame<len(value) else None
    current_time=times[frame] if frame<len(times) else None
    interval=r['support_interval']
    current_support=None if interval is None or not np.isfinite(interval).all() or current_time is None else bool(interval[0]<=current_time<=interval[1])
    trajectory=dict(times=times,xyz=r['xyz'],velocity=r['velocity'],acceleration=r['acceleration'],
        valid=r['valid'],model_only=r['model_only_mask'],view_support=r['view_support'],support_interval=r['support_interval'],
        continuous=_continuous(run,index,r,missing),
        current=dict(sample_index=frame,time=current_time,reference_camera_id=run.cameras[reference]['camera_id'],
            time_kind='saved common reference trajectory time at the UI frame index',
            xyz=current_value('xyz'),velocity=current_value('velocity'),acceleration=current_value('acceleration'),
            valid=current_value('valid'),model_only=current_value('model_only_mask'),
            view_support=current_value('view_support'),within_observation_support=current_support,
            units=dict(time='seconds',xyz='meters',velocity='meters/second',acceleration='meters/second^2')))
    continuous=trajectory['continuous']
    if continuous['available']:
        if r['observation_inlier'] is None:
            continuous.update(view_support=None,valid=None,model_only=None,
                support_note='Final observation-inlier masks were not saved; continuous support cannot be certified')
        else:
            queries=continuous['times'];counts=np.zeros(len(queries),np.int16)
            max_gap=2.1*np.median(np.diff(timestamps[reference]))
            for c in range(len(run.cameras)):
                available=r['observation_inlier'][c]
                tt=timestamps[c,available]+beta[c]
                if not len(tt):continue
                right=np.searchsorted(tt,queries,side='right');left=np.clip(right-1,0,len(tt)-1);right=np.clip(right,0,len(tt)-1)
                exact=np.isclose(queries,tt[left],atol=1e-10,rtol=0);right=np.where(exact,left,right)
                supported=(queries>=tt[left]-1e-10)&(queries<=tt[right]+1e-10)&(tt[right]-tt[left]<=max_gap+1e-10)
                counts+=supported
            valid=np.full(len(queries),bool(r['trajectory_valid']))&continuous['within_observation_support']&(counts>=2)
            for cam in run.cameras:
                depth=(continuous['xyz']@np.asarray(cam['R']).T+np.asarray(cam['t']).reshape(3))[:,2]
                valid &= np.isfinite(depth)&(depth>0)
            continuous.update(view_support=counts,valid=valid,
                model_only=np.full(len(queries),bool(r['trajectory_valid']))&continuous['within_observation_support']&(counts<2),
                support_note='Saved final inlier endpoints within 2.1 reference frames; no extrapolation or invented measurements')
    gt=dict(available=False,independent=True,reason='No synthetic truth is stored for this run')
    if not include_gt:gt['reason']='GT explicitly disabled for this inspection'
    elif (run.manifest_path.parent/'gt.npz').exists():
        try:
            from experiments.dense_stba.synthetic_dynamic import load_truth
            truth=load_truth(run.manifest_path.parent);gt_xyz=truth.evaluate_material_trajectory(seed_uv[None],times)[:,0]
            gt=dict(available=True,independent=True,source=str(run.manifest_path.parent/'gt.npz'),
                note='Independent synthetic GT displayed after estimation; never used to accept/reject this candidate',
                beta_seconds=truth.beta_seconds,trajectory_times=times,xyz=gt_xyz,
                position_error=None if r['xyz'] is None else np.linalg.norm(r['xyz']-gt_xyz,axis=-1),
                current=dict(sample_index=frame,time=current_time,xyz=gt_xyz[frame],
                    time_kind='same saved common reference trajectory time as trajectory.current',
                    position_error=None if current_value('xyz') is None else np.linalg.norm(current_value('xyz')-gt_xyz[frame])),cameras=[])
            if trajectory['continuous']['available']:
                gt['continuous_xyz']=truth.evaluate_material_trajectory(seed_uv[None],trajectory['continuous']['times'])[:,0]
            for c,cam in enumerate(run.cameras):
                projection=truth.project_material(seed_uv[None],cam['camera_id'],truth.physical_timestamps[c]);uv_gt=projection['uv'][:,0]
                epe=None if o['uv'] is None else np.linalg.norm(o['uv'][c]-uv_gt,axis=-1)
                gt['cameras'].append(dict(camera_index=c,camera_id=cam['camera_id'],physical_times=truth.physical_timestamps[c],
                    uv=uv_gt,valid=projection['valid'][:,0],occluded=projection['occluded'][:,0],
                    candidate_epe=epe,
                    current=dict(frame=frame,physical_time=truth.physical_timestamps[c,frame],
                        time_kind='independent GT acquisition physical time at the UI frame index; never estimated beta',
                        uv=uv_gt[frame],valid=projection['valid'][frame,0],occluded=projection['occluded'][frame,0],
                        candidate_epe=None if epe is None else epe[frame])))
        except (OSError,ValueError,KeyError,ImportError) as error:gt=dict(available=False,independent=True,reason=f'GT unavailable: {error}')
    return to_jsonable(dict(schema_version=1,run_id=run.run_id,method=run.method,point_id=int(point_id),point_index=index,
        frame=frame,seed_frame=seed,seed_uv=seed_uv,reconstructed=bool(r['trajectory_valid']) if r['trajectory_valid'] is not None else False,
        cameras=cameras,trajectory=trajectory,gt=gt,source_validation=run.source_validation,
        missing_fields=sorted(set(missing)),warnings=list(run.warnings),
        identity='point_id is the reference SEED pixel ID; later image coordinates move with that candidate physical point'))


def query_pixel(run,x,y,frame=None):
    """Resolve a reference-image click without inventing unqueried identities."""
    seed=int(run.meta.get('seed_frame',run.manifest.get('seed_frame',0)))
    frame=seed if frame is None else int(frame);reference=int(run.meta.get('reference_camera_index',0))
    frame_identity(run,reference,frame);point=np.array([float(x),float(y)])
    w,h=run.cameras[reference]['size'];inside=bool(np.isfinite(point).all() and 0<=point[0]<=w-1 and 0<=point[1]<=h-1)
    result=dict(schema_version=1,run_id=run.run_id,method=run.method,frame=frame,seed_frame=seed,query_uv=point,
                status='outside_image',point_id=None,distance_px=None,candidates=[],ambiguous=False,raw_roma=[])
    if not inside:return to_jsonable(result)
    if frame==seed:
        pixel=np.rint(point).astype(int);encoded=int(pixel[1]*w+pixel[0]);result['pixel_xy']=pixel
        seeds=run.result.get('seed_uv',run.obs.get('seed_uv'));distance=np.linalg.norm(seeds-point,axis=-1)
        nearest=np.argsort(distance,kind='stable')[:5]
        result['candidates']=[dict(point_id=int(run.point_ids[i]),distance_px=float(distance[i]),seed_uv=seeds[i],suggestion_only=True) for i in nearest]
        if encoded in run.id_to_index and np.array_equal(seeds[run.index(encoded)],pixel):
            result.update(status='exact',point_id=encoded,distance_px=float(np.linalg.norm(pixel-point)))
        else:
            result.update(status='not_queried',nearest_point_id=int(run.point_ids[nearest[0]]) if len(nearest) else None,
                nearest_distance_px=float(distance[nearest[0]]) if len(nearest) else None,
                note='This seed pixel was not queried; nearby IDs are suggestions only, no trajectory is fabricated')
    else:
        uv=run.obs['uv'][:,reference,frame]
        if 'status' in run.obs:accepted=run.obs['status'][:,reference,frame]==1
        else:accepted=run.obs['visible'][:,reference,frame].copy()
        accepted &= np.isfinite(uv).all(-1)
        distance=np.linalg.norm(uv-point,axis=-1);indices=np.flatnonzero(accepted&(distance<=3.))
        indices=indices[np.argsort(distance[indices],kind='stable')]
        result['candidates']=[dict(point_id=int(run.point_ids[i]),distance_px=float(distance[i]),current_uv=uv[i]) for i in indices[:16]]
        if len(indices):
            result.update(status='nearest',point_id=int(run.point_ids[indices[0]]),distance_px=float(distance[indices[0]]),
                ambiguous=len(indices)>1,candidates_within_radius=len(indices),search_radius_px=3.,
                note='Nearest tracked temporal candidate at the selected frame; this is not a seed-pixel identity lookup')
        else:result.update(status='no_identity',search_radius_px=3.,note='No tracked reference candidate within 3 pixels; no identity assigned')
    result['raw_roma']=[dict(camera_index=c,camera_id=cam['camera_id'],**raw_roma_at(run,point,c,frame)) for c,cam in enumerate(run.cameras)]
    return to_jsonable(result)
