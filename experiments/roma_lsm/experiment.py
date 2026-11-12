"""Stride-one RoMa -> fixed-camera local matching experiment.

Run from repository root. Inputs are cached full bidirectional RoMa fields.
Geometry is gated by explicit static/synchronized/asynchronous_unknown modes.
GT arrays are NEVER loaded here: evaluation is a separate module.
"""
from pathlib import Path
import argparse
import json
import time
from dataclasses import asdict
import cv2
import numpy as np
import torch

from lsm import LSMConfig, STATUS, bilinear, patch_offsets, refine_points, affine_from_dense, synchronize
from geometry import fundamental, epipolar_residual, triangulate_pair, camera_center

ROOT=Path(__file__).resolve().parent


def read_image(path):
    im=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if im is None:raise FileNotFoundError(path)
    return im.astype(np.float32)/255


def read_mask(cam,key,shape):
    if key not in cam:return np.ones(shape,bool)
    mask=cv2.imread(str(cam[key]),cv2.IMREAD_GRAYSCALE)
    if mask is None:raise FileNotFoundError(cam[key])
    if mask.shape!=shape:raise ValueError('Image and mask dimensions differ')
    return mask>0


def erode(mask,radius):
    return cv2.erode(mask.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8),
                     borderType=cv2.BORDER_CONSTANT,borderValue=0)>0


@torch.no_grad()
def sample_numpy(image,xy,device='cpu'):
    value,inside=bilinear(torch.as_tensor(image,device=device,dtype=torch.float32),
                          torch.as_tensor(xy,device=device,dtype=torch.float32))
    return value.cpu().numpy(),inside.cpu().numpy()


def fb_error(cache,p,q,device):
    back,inside=sample_numpy(cache['uv_BA'],q,device)
    valid,_=sample_numpy(cache['valid_BA'].astype(np.float32),q,device)
    error=np.linalg.norm(back-p,axis=-1)
    error[~inside|(valid<.999)|~np.isfinite(back).all(-1)]=np.nan
    return error


@torch.no_grad()
def patch_diagnostics(ref,tgt,p,q,A,ref_mask,tgt_mask,cfg,device,gain=None,bias=None):
    """Batched exact bilinear diagnostic, preserving invalid patches as NaN."""
    tensor=lambda v:torch.as_tensor(v,device=device,dtype=torch.float32)
    ref,tgt,rm,tm=map(tensor,[ref,tgt,ref_mask,tgt_mask])
    offsets=patch_offsets(cfg.patch_size,device,torch.float32)
    outputs=[];validities=[]
    for start in range(0,len(p),cfg.batch_size):
        sl=slice(start,start+cfg.batch_size);pp,qq,aa=map(tensor,[p[sl],q[sl],A[sl]])
        rp=pp[:,None]+offsets;tp=qq[:,None]+torch.einsum('bij,pj->bpi',aa,offsets)
        rv,ri=bilinear(ref,rp);tv,ti=bilinear(tgt,tp)
        rmask,_=bilinear(rm,rp);tmask,_=bilinear(tm,tp)
        good=(ri&ti&(rmask>.999)&(tmask>.999)).all(-1)
        gg=torch.ones(len(pp),device=device) if gain is None else tensor(gain[sl])
        bb=torch.zeros(len(pp),device=device) if bias is None else tensor(bias[sl])
        photo=(tv-gg[:,None]*rv-bb[:,None]).square().mean(-1).sqrt()
        photo[~good]=torch.nan
        outputs.append(photo.cpu().numpy());validities.append(good.cpu().numpy())
    return np.concatenate(outputs),np.concatenate(validities)


def dataset_jobs(selection):
    jobs=[]
    if selection in ['all','real']:
        meta=ROOT/'data_manifest.json';manifest=json.loads(meta.read_text())
        cams={c['camera_id']:c for c in manifest['cameras']}
        for target in [9,10]:
            jobs.append(dict(name=f'real_7_{target}',ref=cams[7],target=cams[target],
                cache=ROOT/f'cache/pair_7_{target}.npz',dataset=None,mode=manifest['mode']))
    for case in ['translation','affine_brightness','occlusion_weak_noise']:
        if selection not in ['all','synthetic',case]:continue
        meta=ROOT/f'data/synthetic/{case}/metadata.json';manifest=json.loads(meta.read_text())
        cams=manifest['cameras']
        for target in ([1,2] if case=='affine_brightness' else [1]):
            jobs.append(dict(name=f'synthetic_{case}_0_{target}',ref=cams[0],target=cams[target],
                cache=ROOT/f'cache/synthetic_{case}/pair_0_{target}.npz',dataset=str(meta),mode='static'))
    return jobs


def run_pair(job,config,device='cuda',output=None,methods=None):
    if job['mode'] not in ['static','synchronized','asynchronous_unknown']:raise ValueError('Unknown temporal mode')
    cfg=LSMConfig(**config['lsm']);stride=config['stride']
    if stride<1:raise ValueError('stride must be positive')
    out=(Path(output) if output else ROOT/'results')/job['name'];out.mkdir(parents=True,exist_ok=True)
    ref,tgt=read_image(job['ref']['image_path']),read_image(job['target']['image_path'])
    with np.load(job['cache']) as loaded:
        cache={k:loaded[k] for k in ['uv_AB','uv_BA','confidence_AB','confidence_BA','valid_AB','valid_BA']}
    h,w=ref.shape;yy,xx=np.mgrid[:h,:w];grid=np.stack([xx,yy],-1).astype(np.float32)
    rm=read_mask(job['ref'],'valid_mask_path',ref.shape)
    tm=read_mask(job['target'],'valid_mask_path',tgt.shape)
    if job['mode']=='asynchronous_unknown':
        if 'static_mask_path' not in job['ref'] or 'static_mask_path' not in job['target']:
            raise ValueError('This real geometry experiment requires reliable static masks in both views')
        rm &= read_mask(job['ref'],'static_mask_path',ref.shape)
        tm &= read_mask(job['target'],'static_mask_path',tgt.shape)
    roi=erode(rm,cfg.patch_size//2)
    roi &= (xx%stride==0)&(yy%stride==0)
    p=grid[roi];q0=cache['uv_AB'][roi];confidence=cache['confidence_AB']
    F=np.asarray(fundamental(job['ref'],job['target']))
    synchronize(device);started=time.perf_counter()
    fb_full=fb_error(cache,grid,cache['uv_AB'],device)
    raw_model_valid=cache['valid_AB']&np.isfinite(cache['uv_AB']).all(-1)
    raw_quality=raw_model_valid&(confidence>=config['confidence_min'])&np.isfinite(fb_full)&(fb_full<=config['fb_max'])
    A0,affine_reliable=affine_from_dense(cache['uv_AB'],confidence,raw_quality,p,cfg,device)
    photo0,patch_valid=patch_diagnostics(ref,tgt,p,q0,A0,rm,tm,cfg,device)
    raw_valid=roi.copy();raw_valid[roi]=raw_quality[roi]&patch_valid
    applicable=np.zeros((h,w),bool);applicable[roi]=patch_valid
    epi0=np.abs(epipolar_residual(F,p,q0))
    status0=np.zeros((h,w),np.int16)
    status0[roi&~raw_model_valid]=2
    status0[roi&raw_model_valid&(confidence<config['confidence_min'])]=11
    status0[roi&raw_model_valid&(confidence>=config['confidence_min'])&~(np.isfinite(fb_full)&(fb_full<=config['fb_max']))]=12
    status0[roi&raw_quality&~raw_valid]=4
    synchronize(device);screen_seconds=time.perf_counter()-started
    common=dict(uv_raw=cache['uv_AB'],confidence_raw=confidence,valid_model=raw_model_valid,
        valid_raw=raw_valid,reference_roi=roi,geometry_applicable_mask=applicable,
        forward_backward_error_before=fb_full,affine_initialization_reliable=np.zeros((h,w),bool))
    common['affine_initialization_reliable'][roi]=affine_reliable
    def scatter(values,fill=np.nan):
        arr=np.full((h,w)+values.shape[1:],fill,dtype=values.dtype)
        arr[roi]=values;return arr
    common['photo_error_before']=scatter(photo0)
    common['epipolar_error_before']=scatter(np.where(patch_valid,epi0,np.nan).astype(np.float32))
    meta=dict(name=job['name'],cameras={'ref':job['ref'],'target':job['target']},
        cache_path=str(job['cache']),dataset_metadata_path=job['dataset'],mode=job['mode'],
        config=config,stride=stride,ref_size=[w,h],target_size=[tgt.shape[1],tgt.shape[0]],
        coordinate_system='Undistorted working-image x/y; zero-based integer pixel centers; align_corners=False',
        affine_initialization='Weighted local fit to raw RoMa field; bad score/FB/discontinuous neighbors excluded; identity fallback',
        fb_reference='Both before/after query the SAME fixed raw reverse RoMa map (not refined reverse); this is consistency, not GT accuracy',
        calibration_fixed=True,F=F.tolist(),baseline=float(np.linalg.norm(camera_center(job['ref'])-camera_center(job['target']))),
        roi_pixels=int(roi.sum()),raw_valid_pixels=int(raw_valid.sum()),screening_seconds=screen_seconds,
        status_legend=STATUS,methods={},software={'numpy':np.__version__,'torch':torch.__version__,'opencv':cv2.__version__})
    method_list=methods or ['A0','A0g','A1_translation','A1_affine','A2_affine']
    for method in method_list:
        synchronize(device);begin=time.perf_counter();data={k:v.copy() for k,v in common.items()}
        status=status0.copy();q=q0.copy();aa=A0.copy();gain=np.ones(len(p),np.float32);bias=np.zeros(len(p),np.float32)
        attempted=np.zeros((h,w),bool);success=np.zeros((h,w),bool)
        timing={};optimization=None
        if method in ['A1_translation','A1_affine','A2_affine']:
            method_cfg=LSMConfig(**{**asdict(cfg),'model':'translation' if method=='A1_translation' else 'affine',
                'geometry_mode':'soft_epipolar' if method=='A2_affine' else 'off'})
            selected=raw_valid[roi];attempted=raw_valid.copy()
            if selected.any():
                optimization=refine_points(ref,tgt,p[selected],q0[selected],method_cfg,A0=A0[selected],
                    geometry_applicable=applicable[roi][selected],F=F,reference_mask=rm,target_mask=tm,device=device)
                timing=optimization.pop('timing');q[selected]=optimization['uv_refined'];aa[selected]=optimization['A']
                gain[selected]=optimization['gain'];bias[selected]=optimization['bias']
                for key,value in optimization.items():
                    if key in ['uv_refined','A','gain','bias','refinement_status','refined_mask']:continue
                    fill=False if value.dtype==bool else (0 if np.issubdtype(value.dtype,np.integer) else np.nan)
                    field=np.full((h,w)+value.shape[1:],fill,dtype=value.dtype);field[raw_valid]=value;data[key]=field
                status[raw_valid]=optimization['refinement_status']
                success[raw_valid]=optimization['refined_mask']
            valid=success.copy()
        elif method=='A0g':
            method_cfg=None
            line=np.concatenate([p,np.ones((len(p),1))],-1)@F.T
            line/=np.maximum(np.linalg.norm(line[:,:2],axis=-1,keepdims=True),1e-12)
            signed=(q*line[:,:2]).sum(-1)+line[:,2]
            allowed=applicable[roi]&raw_valid[roi]
            q[allowed]-=(signed[:,None]*line[:,:2])[allowed]
            valid=raw_valid.copy();status[valid]=1
        elif method=='A0':
            method_cfg=None;valid=raw_valid.copy();status[valid]=1
        else:raise ValueError(method)
        synchronize(device);optimization_wall=time.perf_counter()-begin
        check_start=time.perf_counter()
        photo_after,patch_ok=patch_diagnostics(ref,tgt,p,q,aa,rm,tm,cfg,device,gain,bias)
        fb_after=fb_error(cache,p,q,device)
        failing=valid[roi]&(~np.isfinite(fb_after)|(fb_after>config['fb_max']))
        region_status=status[roi];region_status[failing]=13
        failing_patch=valid[roi]&~patch_ok;region_status[failing_patch]=4
        status[roi]=region_status
        valid[roi] &= patch_ok&np.isfinite(fb_after)&(fb_after<=config['fb_max'])
        success &= valid
        displacement=np.linalg.norm(q-q0,axis=-1)
        data.update(uv_refined=scatter(q),A=scatter(aa),gain=scatter(gain),bias=scatter(bias),
            valid_refined=valid,refined_mask=success,attempt_mask=attempted,refinement_status=status,
            refinement_displacement=scatter(displacement),photo_error_after=scatter(photo_after),
            forward_backward_error_after=scatter(fb_after),
            epipolar_error_after=scatter(np.where(applicable[roi]&patch_ok,np.abs(epipolar_residual(F,p,q)),np.nan).astype(np.float32)))
        # Raw diagnostics are identical across arms, except model-specific optimizer
        # residuals are retained under explicit keys when available.
        if optimization is not None:
            for key in ['photo_error_before','epipolar_error_before']:
                data['optimizer_'+key]=data[key].copy();data[key]=common[key].copy()
        if valid.any():
            geom_valid=valid&applicable
            tri=triangulate_pair(job['ref'],job['target'],grid[geom_valid],data['uv_refined'][geom_valid])
            for key in ['ray_distance','angle_deg','ray_gap','valid']:
                value=tri[key];fill=False if value.dtype==bool else np.nan
                field=np.full((h,w),fill,dtype=value.dtype);field[geom_valid]=value
                data['triangulation_'+key]=field
        synchronize(device);post_seconds=time.perf_counter()-check_start
        saving=time.perf_counter();np.savez_compressed(out/f'{method}.npz',**data)
        save_seconds=time.perf_counter()-saving
        method_meta=dict(method=method,lsm_config=None if method_cfg is None else asdict(method_cfg),
            timing={**timing,'optimization_and_dispatch_seconds':optimization_wall,
                    'postscreen_seconds':post_seconds,'saving_seconds':save_seconds},
            valid_pixels=int(valid.sum()),attempted_pixels=int(attempted.sum()),refined_pixels=int(success.sum()),
            control_only=method in ['A0','A0g'])
        meta['methods'][method]=method_meta
        (out/'meta.json').write_text(json.dumps(meta,indent=2))
        print(json.dumps({'pair':job['name'],**method_meta}),flush=True)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config.json')
    parser.add_argument('--selection',choices=['all','real','synthetic','translation','affine_brightness','occlusion_weak_noise'],default='all')
    parser.add_argument('--pair',help='Optional exact job name')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--methods',nargs='+')
    parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text())
    np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.set_num_threads(4)
    for job in dataset_jobs(args.selection):
        if args.pair and job['name']!=args.pair:continue
        out=run_pair(job,config,args.device,args.output,args.methods)
        if args.evaluate:
            from evaluate import evaluate_pair
            evaluate_pair(out)


if __name__=='__main__':main()
