"""Time alignment and trajectory seeds from observed 2-D tracks only.

Time convention: physical_time = nominal_timestamp + beta_seconds. No synthetic
GT or real sync.json is accepted by these routines. Interpolated image samples
are used for initialization; the optimizer receives original observations only.
"""
from dataclasses import dataclass
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.roma_lsm.geometry import fundamental, epipolar_residual, project, camera_center, world_rays


def interpolate_tracks(uv,valid,times,query,max_gap):
    """Linear interpolation between actually valid endpoints, with no extrapolation.

    uv[P,T,2], valid[P,T], sorted timestamps[T], shared queries[S]. Gaps longer
    than max_gap remain invalid; failed/occluded tracks are not filled as data.
    """
    uv=np.asarray(uv);valid=np.asarray(valid,bool)&np.isfinite(uv).all(-1)
    times=np.asarray(times,float);query=np.asarray(query,float)
    p,t=valid.shape
    if uv.shape!=(p,t,2) or times.shape!=(t,) or query.ndim!=1:raise ValueError('Unexpected track shapes')
    if not np.all(np.diff(times)>0):raise ValueError('Timestamps must increase')
    indices=np.arange(t)[None,:]
    previous=np.maximum.accumulate(np.where(valid,indices,-1),axis=1)
    following=np.minimum.accumulate(np.where(valid,indices,t)[:,::-1],axis=1)[:,::-1]
    right=np.searchsorted(times,query,side='right')
    low=previous[:,np.clip(right-1,0,t-1)];high=following[:,np.clip(right,0,t-1)]
    # An exact valid endpoint is a supported sample even without a later point.
    base=np.clip(right-1,0,t-1)
    exact=np.isclose(query,times[base],rtol=0,atol=1e-10)[None,:]&valid[:,base]
    low=np.where(exact,base[None,:],low);high=np.where(exact,base[None,:],high)
    li=low.clip(0,t-1);hi=high.clip(0,t-1)
    tl=times[li];th=times[hi];gap=th-tl
    support=(low>=0)&(high<t)&(query[None]>=tl-1e-10)&(query[None]<=th+1e-10)&(gap<=max_gap+1e-10)
    alpha=np.divide(query[None]-tl,gap,out=np.zeros_like(gap),where=gap>1e-12)
    row=np.arange(p)[:,None]
    result=(1-alpha[...,None])*uv[row,li]+alpha[...,None]*uv[row,hi]
    result[~support]=np.nan
    return result,support


def training_split(visible,holdout_every=5,holdout_phase=2):
    heldout=np.zeros_like(visible,bool)
    if holdout_every:
        heldout[...,np.arange(visible.shape[-1])%holdout_every==holdout_phase]=True
    heldout &= visible
    return visible&~heldout,heldout


def motion_diagnostics(uv,visible,times,reference=0):
    """Robust measured displacement; a heuristic observability gate, not GT."""
    p=uv.shape[0];span=np.zeros(p);speed=np.zeros(p);counts=visible[:,reference].sum(-1)
    for i in range(p):
        mask=visible[i,reference]&np.isfinite(uv[i,reference]).all(-1)
        samples=uv[i,reference,mask];tt=times[reference,mask]
        if len(samples)<2:continue
        median=np.median(samples,axis=0)
        span[i]=2*np.quantile(np.linalg.norm(samples-median,axis=-1),.9)
        speed[i]=np.median(np.linalg.norm(np.diff(samples,axis=0),axis=-1)/np.diff(tt))
    return dict(span_px=span,speed_px_s=speed,reference_observation_count=counts)


@dataclass
class TimeInitConfig:
    search_bound_frames:float=6.
    step_frames:float=.1
    max_gap_frames:float=2.1
    min_motion_span_px:float=2.
    min_motion_speed_px_frame:float=.02
    min_dynamic_tracks:int=8
    min_samples_per_track:int=6
    huber_px:float=1.
    separated_profile_gap_min:float=1e-4


def estimate_beta(uv,visible,weight,times,cameras,config=None,reference=0):
    """Common-support coarse profile followed by bounded scalar refinement.

    Only measured moving tracks constrain time. Candidate dynamic associations
    are tested geometrically AFTER shifting their time axes, never by requiring
    synchronous epipolar consistency at equal raw frame indices.
    """
    cfg=config or TimeInitConfig();uv=np.asarray(uv);times=np.asarray(times)
    p,c,t,_=uv.shape;fps=1/np.median(np.diff(times[reference]));bound=cfg.search_bound_frames/fps
    beta=np.zeros(c);motion=motion_diagnostics(uv,visible,times,reference)
    moving=(motion['span_px']>=cfg.min_motion_span_px)&(motion['speed_px_s']>=cfg.min_motion_speed_px_frame*fps)
    profiles=[];reliable=np.zeros(c,bool);reliable[reference]=True
    grid=np.arange(-cfg.search_bound_frames,cfg.search_bound_frames+cfg.step_frames*.25,cfg.step_frames)/fps
    refuv=uv[:,reference];refvalid=visible[:,reference];sample_time=times[reference]
    for j in range(c):
        if j==reference:continue
        F=fundamental(cameras[reference],cameras[j]);gap=cfg.max_gap_frames/fps
        common=refvalid&moving[:,None]
        # Use exactly the same original reference samples for every candidate.
        for candidate in grid:
            _,support=interpolate_tracks(uv[:,j],visible[:,j],times[j],sample_time-candidate,gap)
            common &= support
        track_use=common.sum(-1)>=cfg.min_samples_per_track
        common &= track_use[:,None]
        count=int(track_use.sum());sample_count=int(common.sum())
        mean_target_weight=np.where(visible[:,j],weight[:,j],0).sum(-1)/visible[:,j].sum(-1).clip(1)
        fixed_weight=np.clip(weight[:,reference],.01,1)*np.clip(mean_target_weight,.01,1)[:,None]
        fixed_weight=np.nan_to_num(fixed_weight);fixed_weight*=common
        def score(candidate):
            q,support=interpolate_tracks(uv[:,j],visible[:,j],times[j],sample_time-candidate,gap)
            e=np.abs(epipolar_residual(F,refuv,q));a=np.nan_to_num(e,nan=1e6)
            loss=np.where(a<=cfg.huber_px,a*a,2*cfg.huber_px*a-cfg.huber_px**2)
            # Equal trajectory contribution prevents long/easy tracks dominating.
            per=(loss*fixed_weight).sum(-1)/fixed_weight.sum(-1).clip(1e-12)
            return float(np.mean(per[track_use])) if count else float('inf')
        costs=np.array([score(g) for g in grid]);status='insufficient_dynamic_support'
        best_index=int(np.argmin(costs));best=float(grid[best_index]);chosen=0.
        curvature=0.;contrast=0.;at_boundary=False;basins=[];basin_gap=None
        if count>=cfg.min_dynamic_tracks and np.isfinite(costs).all():
            minima=[k for k in range(len(grid))
                    if (k==0 or costs[k]<=costs[k-1]) and (k==len(grid)-1 or costs[k]<=costs[k+1])]
            # Refine EVERY local basin before comparing ambiguity. Comparing a
            # continuous winner to unrefined grid values hides periodic aliases.
            for k in minima:
                left=grid[max(0,k-1)];right=grid[min(len(grid)-1,k+1)]
                result=minimize_scalar(score,bounds=(left,right),method='bounded',options={'xatol':1e-7})
                basins.append({'beta_frames':float(result.x*fps),'cost':float(result.fun),'grid_index':int(k)})
            basins.sort(key=lambda b:b['cost'])
            chosen=basins[0]['beta_frames']/fps;at_boundary=abs(chosen)>=.98*bound
            h=.1/fps
            if -bound+h<chosen<bound-h:
                curvature=(score(chosen+h)+score(chosen-h)-2*score(chosen))/(.1**2)
            separated=np.abs(grid-chosen)>=.5/fps
            contrast=float(np.min(costs[separated])-score(chosen)) if separated.any() else 0.
            alternate=[b['cost'] for b in basins if abs(b['beta_frames']-chosen*fps)>=.5]
            basin_gap=float(min(alternate)-basins[0]['cost']) if alternate else None
            status=('search_boundary' if at_boundary else 'flat_profile' if curvature<=1e-5 else
                    'ambiguous_separated_minima' if (contrast<cfg.separated_profile_gap_min or
                        (basin_gap is not None and basin_gap<cfg.separated_profile_gap_min)) else 'observable_candidate')
            reliable[j]=status=='observable_candidate'
            beta[j]=chosen
        profiles.append(dict(camera_id=cameras[j]['camera_id'],camera_index=j,
            beta_grid_frames=(grid*fps).tolist(),costs=[float(v) if np.isfinite(v) else None for v in costs],
            beta_initial_seconds=float(beta[j]),beta_initial_frames=float(beta[j]*fps),status=status,
            candidate_dynamic_tracks=count,common_original_samples=sample_count,
            curvature_per_frame2=float(curvature),separated_half_frame_cost_gap=contrast,
            refined_local_minima=basins,second_separated_refined_basin_gap=basin_gap,
            at_boundary=at_boundary,score_definition='Mean across moving tracks of weighted Huber squared normalized point-to-line distance; fixed sample support across candidates'))
    return dict(beta=beta,reliable=reliable,profiles=profiles,moving_tracks=moving,motion=motion,
                reference_camera_index=reference,convention='time=nominal_timestamp+beta_seconds')


def triangulate_samples(uv,valid,weight,times,cameras,beta,sample_times,max_gap_frames=2.1,min_angle_deg=.15):
    """Fixed-camera multiview DLT of time-aligned candidates, only for seeding."""
    p,c,t,_=uv.shape;s=len(sample_times);fps=1/np.median(np.diff(times[0]))
    aligned=[];masks=[]
    for j in range(c):
        q,v=interpolate_tracks(uv[:,j],valid[:,j],times[j],np.asarray(sample_times)-beta[j],max_gap_frames/fps)
        aligned.append(q);masks.append(v)
    q=np.stack(aligned,axis=2);mask=np.stack(masks,axis=2)
    views=mask.sum(-1);rows=[]
    for j,cam in enumerate(cameras):
        P=np.asarray(cam['K'])@np.concatenate([np.asarray(cam['R']),np.asarray(cam['t']).reshape(3,1)],axis=1)
        xy=np.nan_to_num(q[:,:,j]);rows.extend([(xy[...,0,None]*P[2]-P[0])*mask[:,:,j,None],
                                               (xy[...,1,None]*P[2]-P[1])*mask[:,:,j,None]])
    matrix=np.stack(rows,axis=-2)
    _,_,vh=np.linalg.svd(matrix,full_matrices=False);homogeneous=vh[...,-1,:]
    xyz=np.divide(homogeneous[...,:3],homogeneous[...,3,None],out=np.full((p,s,3),np.nan),where=np.abs(homogeneous[...,3,None])>1e-10)
    supported=(views>=2)&np.isfinite(xyz).all(-1);angles=[];errors=[]
    centers=np.stack([camera_center(cam) for cam in cameras])
    rays=xyz[:,:,None]-centers[None,None];rays/=np.linalg.norm(rays,axis=-1,keepdims=True).clip(1e-12)
    for j in range(c):
        pred,z=project(cameras[j],xyz,True)
        supported &= (~mask[:,:,j])|(z>0)
        errors.append(np.where(mask[:,:,j],np.linalg.norm(pred-q[:,:,j],axis=-1),np.nan))
        for k in range(j):
            angle=np.degrees(np.arccos(np.clip(np.sum(rays[:,:,j]*rays[:,:,k],axis=-1),-1,1)))
            angle=np.minimum(angle,180-angle)
            angles.append(np.where(mask[:,:,j]&mask[:,:,k],angle,0))
    angle=np.max(angles,axis=0);supported &= angle>=min_angle_deg
    xyz[~supported]=np.nan
    errors=np.stack(errors,-1);finite=np.isfinite(errors)
    reprojection=np.divide(np.where(finite,errors,0).sum(-1),finite.sum(-1),
        out=np.full((p,s),np.nan),where=finite.sum(-1)>0)
    return dict(xyz=xyz,valid=supported,view_support=views,ray_angle_deg=angle,
                reprojection_error_px=reprojection,sample_times=np.asarray(sample_times),
                aligned_uv=q,aligned_visible=mask)
