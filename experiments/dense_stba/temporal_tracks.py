"""Grid-query temporal tracking from cached dense optical flow, on CPU.

Adjacent bidirectional flows move persistent seed identities, never fixed image
pixels. Failed identities terminate in that time direction; no filling or blind
re-identification. Visibility is a conservative acceptance prediction, not GT.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

if __package__ in [None,'']:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


STATUS={0:'unvisited',1:'tracked',2:'outside_image',3:'fb_inconsistent_possible_occlusion',
        4:'low_seed_texture',5:'multi_frame_drift',6:'lost_after_failure',7:'invalid_seed',8:'invalid_flow'}
DEFAULT_TRACKING=dict(fb_max=.75,fb_relative=.03,seed_texture_min=.005,
                      cycle_max=1.5,cycle_fb_max=2.,use_cycle=True)
DEFAULT_FLOW=dict(method='dis',preset='medium',finest_scale=0,
                  gradient_descent_iterations=25,variational_refinement_iterations=5,
                  cv2_threads=4,skip_stride=5)


def bilinear_sample(field,xy):
    """Exact integer-center bilinear sampling with explicit validity, no padding."""
    field=np.asarray(field);xy=np.asarray(xy,np.float32)
    h,w=field.shape[:2]
    valid=np.isfinite(xy).all(-1)&(xy[...,0]>=0)&(xy[...,0]<=w-1)&(xy[...,1]>=0)&(xy[...,1]<=h-1)
    clean=np.nan_to_num(xy,nan=0,posinf=0,neginf=0)
    x=np.clip(clean[...,0],0,w-1);y=np.clip(clean[...,1],0,h-1)
    ix=np.minimum(np.floor(x).astype(np.int64),w-2);iy=np.minimum(np.floor(y).astype(np.int64),h-2)
    dx=x-ix;dy=y-iy
    if field.ndim==3:dx=dx[...,None];dy=dy[...,None]
    value=(1-dy)*((1-dx)*field[iy,ix]+dx*field[iy,ix+1])+dy*((1-dx)*field[iy+1,ix]+dx*field[iy+1,ix+1])
    valid &= np.isfinite(value).all(-1) if field.ndim==3 else np.isfinite(value)
    return value.astype(np.float32),valid


def _gray_sequence(images):
    images=np.asarray(images)
    if images.dtype!=np.uint8:raise ValueError('Optical flow requires explicit uint8 input, no implicit rescaling')
    if images.ndim==3:return images
    if images.ndim!=4 or images.shape[-1]!=3:raise ValueError('Expected RGB[T,H,W,3] or gray[T,H,W]')
    return np.stack([cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) for image in images])


def _flow(a,b,settings):
    if settings['method']=='dis':
        estimator=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        estimator.setFinestScale(settings['finest_scale'])
        estimator.setGradientDescentIterations(settings['gradient_descent_iterations'])
        estimator.setVariationalRefinementIterations(settings['variational_refinement_iterations'])
        estimator.setUseSpatialPropagation(True)
        return estimator.calc(a,b,None)
    if settings['method']=='farneback':
        return cv2.calcOpticalFlowFarneback(a,b,None,.5,5,21,5,7,1.5,0)
    raise ValueError('Supported explicit CPU methods are dis and farneback')


def build_flow_cache(images,cache_dir,frame_ids=None,camera_id=None,method='dis',config=None):
    """Return dense temporal fields; per-raw-frame pair files share shifted windows.

    Output forward[t]=flow(t→t+1), backward[t]=flow(t+1→t), each [T-1,H,W,2].
    skip_flows[(a,b)] stores directly computed a↔b flow for multi-step checks.
    Cache identity hashes actual frame contents, source indices, settings and cv2.
    """
    settings={**DEFAULT_FLOW,**(config or {}),'method':method}
    cv2.setNumThreads(int(settings['cv2_threads']))
    gray=_gray_sequence(images);T,H,W=gray.shape
    if T<2:raise ValueError('At least two video frames are required')
    ids=np.arange(T) if frame_ids is None else np.asarray(frame_ids)
    if len(ids)!=T or len(np.unique(ids))!=T:raise ValueError('frame_ids must be unique and match sequence length')
    root=Path(cache_dir)/(f'cam{camera_id}' if camera_id is not None else 'camera');root.mkdir(parents=True,exist_ok=True)
    hashes=[hashlib.sha256(image.tobytes()).hexdigest() for image in gray]
    identity=dict(settings=settings,opencv=cv2.__version__,size=[W,H],camera_id=camera_id)
    encoded=json.dumps(identity,sort_keys=True)
    hits=0;computed=0;compute_seconds=0.;save_seconds=0.;began=time.perf_counter()
    def get_pair(a,b):
        nonlocal hits,computed,compute_seconds,save_seconds
        key=hashlib.sha256((encoded+hashes[a]+hashes[b]).encode()).hexdigest()[:20]
        path=root/f'raw_{ids[a]}_{ids[b]}_{key}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as cache:
                ab,ba=cache['forward'].copy(),cache['backward'].copy()
            hits+=1
        else:
            start=time.perf_counter();ab=_flow(gray[a],gray[b],settings);ba=_flow(gray[b],gray[a],settings)
            compute_seconds+=time.perf_counter()-start;computed+=1
            start=time.perf_counter();np.savez_compressed(path,forward=ab,backward=ba)
            save_seconds+=time.perf_counter()-start
        if ab.shape!=(H,W,2) or ba.shape!=(H,W,2):raise ValueError('Corrupted flow cache shape')
        return ab,ba
    adjacent=[get_pair(t,t+1) for t in range(T-1)]
    skip={};stride=int(settings['skip_stride'])
    if stride>1:
        for a in range(0,T-stride,stride):
            skip[(a,a+stride)]=get_pair(a,a+stride)
    summary=dict(**identity,n_frames=T,raw_frame_ids=ids.tolist(),pair_cache_hits=hits,
        pairs_computed=computed,flow_compute_seconds=compute_seconds,flow_save_seconds=save_seconds,
        total_seconds=time.perf_counter()-began,device='CPU',peak_gpu_memory_bytes=0,
        flow_semantics='Dense pixel displacement (dx,dy), integer-center source samples, exact bilinear subpixel query',
        skip_semantics='Independent direct flow every5frames by default, compared to composed adjacent tracks only where direct FB is reliable')
    digest=hashlib.sha256(''.join(hashes).encode()).hexdigest()[:16]
    (root/f'sequence_{ids[0]}_{ids[-1]}_{digest}.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'flow_cache':str(root),**summary}),flush=True)
    return dict(forward=np.stack([p[0] for p in adjacent]),backward=np.stack([p[1] for p in adjacent]),
                gray_frames=gray,skip_flows=skip,frame_ids=ids,metadata=summary)


def track_queries(cache,seed_frame,seed_uv,point_ids=None,config=None):
    """Track arbitrary float seed UV in both directions, returning [P,T,...].

    Local flow FB checks and reliable direct-vs-composed multi-frame flow cycles
    reject likely occlusion/drift. Scores are heuristic, not calibrated visibility
    probabilities. No GT, camera geometry, beta or cross-view identity is used.
    """
    cfg={**DEFAULT_TRACKING,**(config or {})}
    fwd,back=cache['forward'],cache['backward'];T=fwd.shape[0]+1
    H,W=fwd.shape[1:3];seed=int(seed_frame);uv=np.asarray(seed_uv,np.float32)
    if not 0<=seed<T or uv.ndim!=2 or uv.shape[1]!=2:raise ValueError('Invalid seed frame/UV dimensions')
    P=len(uv);ids=np.arange(P,dtype=np.int64) if point_ids is None else np.asarray(point_ids,np.int64)
    if ids.shape!=(P,) or len(np.unique(ids))!=P:raise ValueError('point_ids must be unique per seed')
    tracks=np.full((P,T,2),np.nan,np.float32);visible=np.zeros((P,T),bool)
    confidence=np.zeros((P,T),np.float32);status=np.zeros((P,T),np.uint8)
    fb_error=np.full((P,T),np.nan,np.float32);cycle_error=fb_error.copy();photo_error=fb_error.copy()
    reliable_cycle=np.zeros((P,T),bool)
    tracks[:,seed]=uv
    _,good=bilinear_sample(np.zeros((H,W),np.float32),uv)
    status[:,seed]=np.where(good,1,7)
    gray=cache.get('gray_frames')
    texture=np.full(P,.1,np.float32)
    if gray is not None:
        seed_gray=gray[seed].astype(np.float32)/255
        mean=cv2.boxFilter(seed_gray,-1,(7,7),normalize=True)
        variance=np.maximum(0,cv2.boxFilter(seed_gray**2,-1,(7,7),normalize=True)-mean**2)
        texture,_=bilinear_sample(np.sqrt(variance),uv)
        weak=good&(texture<cfg['seed_texture_min']);status[weak,seed]=4;good &= ~weak
    visible[:,seed]=good
    seed_conf=np.clip(texture/(texture+.01),0,1)
    confidence[:,seed]=np.where(good,seed_conf,0);fb_error[good,seed]=0
    started=time.perf_counter()
    for direction in [1,-1]:
        active=good.copy()
        for t in range(seed+direction,T if direction==1 else -1,direction):
            previous=t-direction
            status[~active,t]=6
            indices=np.flatnonzero(active)
            if not len(indices):continue
            qprev=tracks[indices,previous]
            field=fwd[previous] if direction==1 else back[t]
            reverse=back[previous] if direction==1 else fwd[t]
            delta,valid_flow=bilinear_sample(field,qprev);qnext=qprev+delta
            inverse,inside=bilinear_sample(reverse,qnext)
            error=np.linalg.norm(delta+inverse,axis=-1)
            tolerance=cfg['fb_max']+cfg['fb_relative']*np.linalg.norm(delta,axis=-1)
            keep=valid_flow&inside&np.isfinite(error)&(error<=tolerance)
            reason=np.ones(len(indices),np.uint8)
            reason[~valid_flow]=8;reason[valid_flow&~inside]=2
            reason[valid_flow&inside&(~np.isfinite(error)|(error>tolerance))]=3
            tracks[indices,t]=qnext;fb_error[indices,t]=error
            quality=np.exp(-.5*(error/np.maximum(tolerance,1e-6))**2)
            confidence[indices,t]=np.minimum(confidence[indices,previous],quality)
            if gray is not None:
                a,va=bilinear_sample(gray[previous].astype(np.float32)/255,qprev)
                b,vb=bilinear_sample(gray[t].astype(np.float32)/255,qnext)
                photo_error[indices,t]=np.where(va&vb,abs(a-b),np.nan)
            if cfg['use_cycle']:
                for (lo,hi),(ab,ba) in cache.get('skip_flows',{}).items():
                    if (direction==1 and t==hi and lo>=seed) or (direction==-1 and t==lo and hi<=seed):
                        anchor=lo if direction==1 else hi
                        origin=tracks[indices,anchor]
                        direct=ab if direction==1 else ba;direct_reverse=ba if direction==1 else ab
                        dd,ok0=bilinear_sample(direct,origin);qdirect=origin+dd
                        rr,ok1=bilinear_sample(direct_reverse,qdirect)
                        usable=visible[indices,anchor]&ok0&ok1&(np.linalg.norm(dd+rr,axis=-1)<=cfg['cycle_fb_max'])
                        drift=np.linalg.norm(qnext-qdirect,axis=-1)
                        cycle_error[indices,t]=np.where(usable,drift,np.nan)
                        reliable_cycle[indices,t]=usable
                        bad=keep&usable&(drift>cfg['cycle_max']);reason[bad]=5;keep &= ~bad
                        confidence[indices,t]=np.minimum(confidence[indices,t],np.where(usable,np.exp(-.5*(drift/cfg['cycle_max'])**2),1))
            status[indices,t]=reason;visible[indices,t]=keep
            confidence[indices[~keep],t]=0
            active[indices[~keep]]=False
    # Unaccepted finite proposals are preserved for debugging, never observations.
    displacement=np.linalg.norm(tracks-uv[:,None],axis=-1)
    metadata=dict(config=cfg,status_legend=STATUS,tracking_seconds=time.perf_counter()-started,
        seed_frame=seed,query_count=P,accepted_seed_count=int(good.sum()),accepted_observations=int(visible.sum()),
        tracks_with_ge_10_observations=int((visible.sum(1)>=10).sum()),
        tracks_with_ge_30_observations=int((visible.sum(1)>=30).sum()),
        visibility_semantics='Accepted optical-flow observation; actual occlusion vs identity error is not independently known.',
        identity_policy='Persistent seed point_id; follow flow at moving coordinates in each direction. After first failure terminate that branch; no reidentification or holes filled.',
        confidence_semantics='Heuristic minimum of seed texture and flow consistency scores, not calibrated probability/inverse variance',
        cumulative_drift_policy='Direct skip-frame flow vs composed adjacent tracks, gated by reliable direct FB. No guarantee all drift is detected.',
        device='CPU',peak_gpu_memory_bytes=0)
    return dict(tracks=tracks,visible=visible,confidence=confidence,status=status,fb_error=fb_error,
                cycle_error=cycle_error,cycle_check_reliable=reliable_cycle,photo_error=photo_error,
                displacement_from_seed=displacement,point_ids=ids,seed_uv=uv,seed_frame=np.asarray(seed),
                frame_ids=np.asarray(cache.get('frame_ids',np.arange(T))),metadata=metadata)


def save_tracks(path,result):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**{k:v for k,v in result.items() if k!='metadata'})
    path.with_suffix('.json').write_text(json.dumps(result['metadata'],indent=2)+'\n')


if __name__=='__main__':
    from experiments.dense_stba.data_loader import load_case,load_camera_sequence,query_grid
    parser=argparse.ArgumentParser()
    parser.add_argument('manifest',type=Path);parser.add_argument('--camera',type=int,default=7)
    parser.add_argument('--stride',type=int,default=4);parser.add_argument('--cache-dir',type=Path,default=Path('experiments/dense_stba/cache/temporal'))
    parser.add_argument('--out',type=Path,default=Path('experiments/dense_stba/results/temporal_preview.npz'))
    args=parser.parse_args();manifest=load_case(args.manifest)
    cam=next(c for c in manifest['cameras'] if c['camera_id']==args.camera)
    images=load_camera_sequence(cam)
    flow=build_flow_cache(images,args.cache_dir,cam.get('raw_frame_ids'),args.camera)
    mask=cv2.imread(manifest['seed_roi_mask_path'],0)>0 if 'seed_roi_mask_path' in manifest else None
    uv,ids,_=query_grid(cam['size'],args.stride,mask,margin=5)
    result=track_queries(flow,manifest['seed_frame'],uv,ids)
    save_tracks(args.out,result);print(json.dumps(result['metadata'],indent=2))
