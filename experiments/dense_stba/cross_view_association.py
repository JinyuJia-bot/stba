"""RoMa candidates plus temporal propagation, with explicit identity checks.

No same-frame epipolar test is used for asynchronous moving points. Dense RoMa
fields are retained at every anchor, while actual optimized seeds remain a
reported regular sub-grid. Candidate associations can fail; no holes are filled.
"""
from pathlib import Path
import sys
import json
import time
from dataclasses import dataclass,asdict
import numpy as np
import cv2
import torch
from scipy.spatial import cKDTree

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.roma_lsm.roma_adapter import run as run_roma
from experiments.roma_lsm.lsm import bilinear,LSMConfig,refine_points,affine_from_dense
try:
    from .data_loader import load_case,load_camera_sequence,timestamps,query_grid
except ImportError:
    from data_loader import load_case,load_camera_sequence,timestamps,query_grid


@dataclass
class AssociationConfig:
    stride:int=4
    margin:int=8
    confidence_min:float=.15
    roma_fb_max:float=2.
    anchor_error_max:float=1.5
    anchor_min_support:int=2
    collision_distance:float=.75
    min_track_observations:int=8
    use_lsm:bool=False


def sample(field,points):
    value,valid=bilinear(torch.as_tensor(np.asarray(field),dtype=torch.float32),
                         torch.as_tensor(np.asarray(points),dtype=torch.float32))
    return value.numpy(),valid.numpy()


def anchor_frames(manifest):
    n=len(manifest['cameras'][0]['image_paths']);s=manifest.get('seed_frame',n//2)
    return sorted(set([max(0,s-n//4),s,min(n-1,s+n//4)]))


def roma_manifest(manifest,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    reference=next(c for c in manifest['cameras'] if c['camera_id']==manifest['reference_camera_id'])
    cameras=[];pairs=[]
    for frame in anchor_frames(manifest):
        for cam in manifest['cameras']:
            cameras.append(dict(camera_id=f"{cam['camera_id']}f{frame:04d}",image_path=cam['image_paths'][frame],size=cam['size']))
        for cam in manifest['cameras']:
            if cam['camera_id']==reference['camera_id']:continue
            pairs.append(dict(reference_id=f"{reference['camera_id']}f{frame:04d}",target_id=f"{cam['camera_id']}f{frame:04d}"))
    path=output/'roma_manifest.json';path.write_text(json.dumps(dict(cameras=cameras,pairs=pairs),indent=2))
    return path


def prepare_roma(manifest_path,output,device='cuda'):
    manifest=load_case(manifest_path);path=roma_manifest(manifest,output)
    return run_roma(path,Path(output)/'roma',device=device)


def load_roma_field(output,reference_id,target_id,frame):
    path=Path(output)/'roma'/f'pair_{reference_id}f{frame:04d}_{target_id}f{frame:04d}.npz'
    with np.load(path) as data:
        return {k:data[k] for k in ['uv_AB','uv_BA','confidence_AB','confidence_BA','valid_AB','valid_BA']}


def query_correspondence(field,p,cfg):
    q,inside=sample(field['uv_AB'],p);score,_=sample(field['confidence_AB'],p)
    valid,_=sample(field['valid_AB'].astype(np.float32),p)
    back,back_inside=sample(field['uv_BA'],q);back_valid,_=sample(field['valid_BA'].astype(np.float32),q)
    fb=np.linalg.norm(back-p,axis=-1)
    ok=inside&back_inside&(valid>.999)&(back_valid>.999)&(score>=cfg.confidence_min)&np.isfinite(q).all(-1)&(fb<=cfg.roma_fb_max)
    return q,score,fb,ok


def collision_mask(q,valid,score,distance):
    """Competing seeds mapping to the same target location keep only best score."""
    result=np.zeros(len(q),bool);indices=np.flatnonzero(valid)
    if len(indices)<2:return result
    pairs=cKDTree(q[indices]).query_pairs(distance,output_type='ndarray')
    if len(pairs):
        a,b=indices[pairs[:,0]],indices[pairs[:,1]]
        losers=np.where(score[a]>=score[b],b,a);result[losers]=True
    return result


def build_observations(manifest_path,cache_dir,config=None,device='cpu'):
    """Associate regular reference seeds, saving temporal AND cross-view edges."""
    try:
        from .temporal_tracks import build_flow_cache,track_queries,DEFAULT_TRACKING
    except ImportError:
        from temporal_tracks import build_flow_cache,track_queries,DEFAULT_TRACKING
    cfg=config or AssociationConfig();manifest=load_case(manifest_path);cache_dir=Path(cache_dir)
    cache_dir.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    cameras=manifest['cameras'];reference=next(i for i,c in enumerate(cameras) if c['camera_id']==manifest['reference_camera_id'])
    if reference!=0:raise ValueError('Reference camera must be first in this minimal prototype')
    seed=manifest.get('seed_frame',len(cameras[0]['image_paths'])//2);anchors=anchor_frames(manifest)
    size=cameras[0]['size'];roi=np.ones((size[1],size[0]),bool)
    if manifest.get('seed_roi_mask_path'):
        roi=cv2.imread(manifest['seed_roi_mask_path'],cv2.IMREAD_GRAYSCALE)>0
    p,ids,gridmask=query_grid(size,cfg.stride,roi,cfg.margin)
    sequence=[];flows=[]
    for cam in cameras:
        images=load_camera_sequence(cam);sequence.append(images)
        flows.append(build_flow_cache(images,cache_dir.parent/'temporal',
            frame_ids=cam.get('raw_frame_ids'),camera_id=cam['camera_id']))
    tracking_config=DEFAULT_TRACKING.copy()
    ref=track_queries(flows[0],seed,p,point_ids=ids,config=tracking_config)
    n=len(p);c=len(cameras);t=len(sequence[0])
    uv=np.full((n,c,t,2),np.nan,np.float32);visible=np.zeros((n,c,t),bool)
    weight=np.zeros((n,c,t),np.float32);status=np.zeros((n,c,t),np.int16)
    temporal_fb=np.full((n,c,t),np.nan,np.float32)
    uv[:,0]=ref['tracks'];visible[:,0]=ref['visible'];weight[:,0]=ref['confidence'];status[:,0]=ref['status'];temporal_fb[:,0]=ref['fb_error']
    edge_status=np.zeros((n,c),np.int16);edge_status[:,0]=1
    score=np.ones((n,c),np.float32);roma_fb=np.zeros((n,c),np.float32)
    anchor_error=np.full((n,c,len(anchors)),np.nan,np.float32);anchor_support=np.zeros_like(anchor_error,bool)
    lsm_accepted=np.zeros((n,c),bool);lsm_status=np.zeros((n,c),np.int16)
    for j,cam in enumerate(cameras[1:],1):
        raw=load_roma_field(cache_dir,cameras[0]['camera_id'],cam['camera_id'],seed)
        q,qs,fb,initial_valid=query_correspondence(raw,p,cfg);score[:,j]=qs;roma_fb[:,j]=fb
        edge_status[~initial_valid,j]=2
        if cfg.use_lsm:
            selected=initial_valid
            gray=lambda image:cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY).astype(np.float32)/255
            A,good=affine_from_dense(raw['uv_AB'],raw['confidence_AB'],raw['valid_AB'],p[selected])
            if selected.any():
                fit=refine_points(gray(sequence[0][seed]),gray(sequence[j][seed]),p[selected],q[selected],
                    LSMConfig(model='affine',geometry_mode='off'),A0=A,device=device)
                lsm_status[selected,j]=fit['refinement_status'];lsm_accepted[selected,j]=fit['refined_mask']
                chosen=np.flatnonzero(selected)[fit['refined_mask']];q[chosen]=fit['uv_refined'][fit['refined_mask']]
        conflict=collision_mask(q,initial_valid,qs,cfg.collision_distance)
        edge_status[conflict,j]=3;initial_valid &= ~conflict
        target=track_queries(flows[j],seed,q,point_ids=ids,config=tracking_config)
        uv[:,j]=target['tracks'];temporal_fb[:,j]=target['fb_error'];status[:,j]=target['status']
        for k,frame in enumerate(anchors):
            field=load_roma_field(cache_dir,cameras[0]['camera_id'],cam['camera_id'],frame)
            aq,ac,af,agood=query_correspondence(field,ref['tracks'][:,frame],cfg)
            support=agood&ref['visible'][:,frame]&target['visible'][:,frame]
            error=np.linalg.norm(aq-target['tracks'][:,frame],axis=-1)
            anchor_error[:,j,k]=np.where(support,error,np.nan);anchor_support[:,j,k]=support
        # The seed-frame agreement is zero by construction; it is NOT an
        # independent identity check and must not dilute temporal disagreement.
        independent=anchor_support[:,j]&(np.asarray(anchors)!=seed)[None,:]
        count=independent.sum(-1)
        disagreement=np.max(np.where(independent,anchor_error[:,j],-np.inf),axis=-1)
        disagreement=np.where(count>0,disagreement,np.inf)
        temporal_ok=target['visible'].sum(-1)>=cfg.min_track_observations
        stable=(count>=cfg.anchor_min_support)&np.isfinite(disagreement)&(disagreement<=cfg.anchor_error_max)&temporal_ok
        edge_status[initial_valid&~stable,j]=4
        accepted=initial_valid&stable
        edge_status[accepted,j]=1
        visible[:,j]=target['visible']&accepted[:,None]
        reliability=np.exp(-.5*(disagreement/cfg.anchor_error_max)**2)
        weight[:,j]=target['confidence']*np.clip(qs,0,1)[:,None]*reliability[:,None]
        weight[:,j]*=visible[:,j]
        print(json.dumps({'camera':cam['camera_id'],'queried':n,'roma_candidate':int(initial_valid.sum()),
            'associated':int(accepted.sum()),'anchor_support_2':int((count>=2).sum())}),flush=True)
    ref_temporal=visible[:,0].sum(-1)>=cfg.min_track_observations
    associated=(edge_status[:,1:]==1).sum(-1)>=1
    multiview_seed=ref_temporal&associated
    all_three=ref_temporal&(edge_status[:,1:]==1).all(-1)
    # Keep failed/unassociated identities and original temporal measurements in
    # the archive; eligibility, not hole filling, determines optimization input.
    observation=dict(point_ids=ids,seed_uv=p,query_mask=gridmask,seed_roi=roi,uv=uv,visible=visible,
        weight=weight,status=status,temporal_fb_error=temporal_fb,edge_status=edge_status,
        roma_confidence=score,roma_fb_error=roma_fb,anchor_error=anchor_error,anchor_support=anchor_support,
        temporal_success=ref_temporal,multiview_seed=multiview_seed,all_three_seed=all_three,
        lsm_seed_refined=lsm_accepted,lsm_seed_status=lsm_status,
        timestamps=np.stack([timestamps(cam) for cam in cameras]),anchor_frames=np.asarray(anchors),seed_frame=np.asarray(seed))
    tag='observations_lsm' if cfg.use_lsm else 'observations'
    np.savez_compressed(cache_dir/(tag+'.npz'),**observation)
    meta=dict(manifest_path=str(Path(manifest_path).resolve()),config=asdict(cfg),tracking_config=tracking_config,
        camera_ids=[cam['camera_id'] for cam in cameras],seed_frame=seed,anchor_frames=anchors,
        query_pixels=n,temporal_success=int(ref_temporal.sum()),associated_two_or_more=int(multiview_seed.sum()),
        associated_all_three=int(all_three.sum()),valid_observations=int(visible.sum()),seconds=time.perf_counter()-started,
        edge_status_legend={0:'not_attempted',1:'accepted_candidate_identity',2:'RoMa_score_FB_or_bounds',3:'competing_identity',4:'temporal_or_multi_anchor_inconsistent'},
        temporal_edge_definition='Same point_id across adjacent valid observations within each camera, propagated using actual dense optical flow',
        cross_camera_edge_definition='RoMa seed association plus reciprocal checks and TWO independent non-seed anchors, each within threshold; candidate physical identity, not GT-certified',
        geometry_policy='No equal-frame epipolar rejection on asynchronous dynamic candidates; time-aware geometry is applied later',
        density='Full dense RoMa and optical flow front end; actual trajectory seeds are regular stride grid with IDs retained; no interpolated refined tracks')
    (cache_dir/(tag+'.json')).write_text(json.dumps(meta,indent=2))
    return observation,meta


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('manifest');parser.add_argument('--out',required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--roma-only',action='store_true');parser.add_argument('--skip-roma',action='store_true')
    parser.add_argument('--stride',type=int,default=4);parser.add_argument('--lsm',action='store_true')
    args=parser.parse_args()
    if not args.skip_roma:prepare_roma(args.manifest,args.out,args.device)
    if not args.roma_only:build_observations(args.manifest,args.out,AssociationConfig(stride=args.stride,use_lsm=args.lsm),args.device)
