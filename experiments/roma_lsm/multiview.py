"""Minimal two-target, shared-positive-depth dense local photometric refinement.

Each supplied reference pixel keeps one unit world ray. Its sole geometric
variable is log(ray range); target centers are fixed-camera projections. Local
affines come from stage one and never change. Weights are frozen before LM.
This module does not accept ground truth, estimate cameras/time, construct tracks,
or optimize independent target-center offsets.
"""
from dataclasses import dataclass, asdict
import time
import numpy as np
import torch
try:
    from .geometry import camera_center, world_rays, triangulate_pair, project_ray_depth
    from .lsm import bilinear, patch_offsets, huber, affine_valid, synchronize
except ImportError:
    from geometry import camera_center, world_rays, triangulate_pair, project_ray_depth
    from lsm import bilinear, patch_offsets, huber, affine_valid, synchronize


STATUS={0:'not_attempted',1:'success',2:'nonfinite_input',3:'reference_boundary',
        4:'target_boundary',5:'low_texture',6:'ill_conditioned',7:'affine_invalid',
        8:'no_convergence',9:'correction_limit',10:'photo_high',13:'fb_after',14:'static_mask_boundary',
        15:'geometry_degenerate',16:'depth_limit',17:'ineligible',18:'invalid_quality'}
WEIGHT_MODES=('uniform','quality','distance','quality_distance')


@dataclass
class MultiViewConfig:
    patch_size:int=11
    max_iterations:int=15
    batch_size:int=256
    sigma_photo:float=.03
    huber_photo:float=1.5
    max_correction:float=4.
    gain_min:float=.5
    gain_max:float=2.
    bias_limit:float=.35
    affine_sv_min:float=.35
    affine_sv_max:float=2.8
    affine_det_min:float=.12
    texture_std_min:float=.012
    texture_eigen_min:float=1e-6
    condition_max:float=1e7
    photo_rmse_max:float=.10
    step_tolerance:float=.003
    objective_tolerance:float=1e-5
    lm_initial:float=.001
    require_convergence:bool=True
    min_angle_deg:float=.2
    min_depth:float=1e-4
    max_relative_depth_change:float=.5
    depth_information_min:float=1e-8
    quality_fb_scale:float=1.5
    quality_photo_scale:float=.05
    quality_floor:float=.05
    distance_scale:float=0.  # 0: median of the two fixed-camera baselines.
    distance_min_weight:float=.2


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)


def initialize_shared_depth(cam_ref,cam_targets,p,q_init,min_angle_deg=.2,eligible=None):
    """One shared initialization from BOTH positive, nondegenerate pair depths.

    Returns reusable NumPy arrays. The same dictionary should be passed to all
    four weight arms. Call only on geometry-applicable pixels: image/mode/mask
    verification is enforced again by refine_shared_depth before optimization.
    The initial range is an unweighted mean, independent of the requested arm.
    """
    p=np.asarray(_numpy(p),dtype=np.float64);q=np.asarray(_numpy(q_init),dtype=np.float64)
    if len(cam_targets)!=2 or p.ndim!=2 or p.shape[1]!=2 or q.shape!=(len(p),2,2):
        raise ValueError('Expected two targets, p Nx2 and q_init Nx2x2')
    pairs=[triangulate_pair(cam_ref,cam,p,q[:,j],min_angle_deg) for j,cam in enumerate(cam_targets)]
    pair_valid=np.stack([x['valid'] for x in pairs],axis=1)
    pair_range=np.stack([x['ray_distance'] for x in pairs],axis=1)
    valid=pair_valid.all(axis=1)
    if eligible is not None:valid &= np.asarray(_numpy(eligible),dtype=bool)
    depth=pair_range.mean(axis=1)
    valid &= np.isfinite(depth)&(depth>0)
    depth=np.where(valid,depth,np.nan)
    return dict(depth=depth,valid=valid,pair_ray_distance=pair_range,pair_valid=pair_valid,
                ray_angle_deg=np.stack([x['angle_deg'] for x in pairs],axis=1),
                ray_gap=np.stack([x['ray_gap'] for x in pairs],axis=1),
                rays=_numpy(world_rays(cam_ref,p)),center=_numpy(camera_center(cam_ref)),
                p=p.copy(),q_init=q.copy(),min_angle_deg=float(min_angle_deg))


def frozen_view_weights(cam_ref,cam_targets,quality_components,weight_mode,config=None):
    """Heuristic frozen scores, normalized to sum to two for every reference point.

    confidence is a raw model score, not calibrated uncertainty. No geometric
    accuracy/ground truth enters this function. A positive floor avoids deleting
    a view through a zero weight; invalid views must be rejected by eligibility.
    """
    cfg=config or MultiViewConfig()
    if weight_mode not in WEIGHT_MODES:raise ValueError('Unknown multiview weight mode')
    confidence=np.asarray(_numpy(quality_components['confidence']),dtype=np.float64)
    fb=np.asarray(_numpy(quality_components['fb_error']),dtype=np.float64)
    photo=np.asarray(_numpy(quality_components['photo_error']),dtype=np.float64)
    if confidence.ndim!=2 or confidence.shape[1]!=2 or fb.shape!=confidence.shape or photo.shape!=confidence.shape:
        raise ValueError('Quality confidence/fb_error/photo_error must be Nx2 arrays')
    valid=np.isfinite(confidence).all(1)&np.isfinite(fb).all(1)&np.isfinite(photo).all(1)
    valid &= (fb>=0).all(1)&(photo>=0).all(1)
    score=np.clip(np.nan_to_num(confidence,nan=0.,posinf=0.,neginf=0.),0,1)
    fb_score=np.exp(-.5*(np.nan_to_num(fb,nan=1e6,posinf=1e6)/cfg.quality_fb_scale)**2)
    photo_score=np.exp(-.5*(np.nan_to_num(photo,nan=1e6,posinf=1e6)/cfg.quality_photo_scale)**2)
    quality=np.clip(score*fb_score*photo_score,cfg.quality_floor,1)
    center=_numpy(camera_center(cam_ref))
    baselines=np.array([np.linalg.norm(_numpy(camera_center(cam))-center) for cam in cam_targets])
    scale=float(cfg.distance_scale or np.median(baselines))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Nonzero baseline scale is required')
    distance=np.clip(np.exp(-(baselines/scale)**2),cfg.distance_min_weight,1)
    weight=np.ones_like(quality)
    if weight_mode in ['quality','quality_distance']:weight *= quality
    if weight_mode in ['distance','quality_distance']:weight *= distance[None,:]
    unnormalized=weight.copy();weight=2*weight/weight.sum(1,keepdims=True)
    return dict(weights=weight,weights_unnormalized=unnormalized,quality_valid=valid,
                confidence_raw=confidence,fb_error_initial=fb,photo_error_initial=photo,
                confidence_component=score,fb_component=fb_score,photo_component=photo_score,
                quality_component=quality,distance_component=distance,baselines=baselines,
                distance_scale=scale,mode=weight_mode)


def shared_patch_residual_jacobian(targets,cam_targets,center,rays,reference_patch,offsets,A_fixed,theta):
    """Residual/Jacobian for theta=[log(range), gain0,bias0,gain1,bias1].

    Returns residual Nx2xP, Jacobian Nx2xPx5, inside Nx2xP, centers Nx2x2,
    patch coordinates Nx2xPx2. Target camera depth must also be positive.
    """
    depth=theta[:,0].exp();residuals=[];jacobians=[];validities=[];uv=[];coordinates=[]
    for j,(target,cam) in enumerate(zip(targets,cam_targets)):
        q,derivative=project_ray_depth(cam,center,rays,depth,True)
        xy=q[:,None,:]+torch.einsum('bij,pj->bpi',A_fixed[:,j],offsets)
        value,gradient,inside=bilinear(target,xy,True)
        gain,bias=theta[:,1+2*j],theta[:,2+2*j]
        residual=value-gain[:,None]*reference_patch-bias[:,None]
        J=torch.zeros((*residual.shape,5),device=theta.device,dtype=theta.dtype)
        J[:,:,0]=(gradient*derivative[:,None,:]).sum(-1)*depth[:,None]
        J[:,:,1+2*j]=-reference_patch;J[:,:,2+2*j]=-1
        R=torch.as_tensor(cam['R'],device=theta.device,dtype=theta.dtype)
        t=torch.as_tensor(cam['t'],device=theta.device,dtype=theta.dtype).reshape(3)
        target_z=((center+depth[:,None]*rays)@R.T+t)[:,2]
        inside &= (target_z>1e-8)[:,None]
        residuals.append(residual);jacobians.append(J);validities.append(inside)
        uv.append(q);coordinates.append(xy)
    return (torch.stack(residuals,1),torch.stack(jacobians,1),torch.stack(validities,1),
            torch.stack(uv,1),torch.stack(coordinates,1))


def _batch(reference,targets,cams,p,q_init,A,depth0,initial_valid,rays,center,weights,
           quality_valid,eligible,cfg,reference_mask,target_masks):
    n=len(p);device=p.device;dtype=p.dtype;offsets=patch_offsets(cfg.patch_size,device,dtype);area=len(offsets)
    status=torch.zeros(n,device=device,dtype=torch.int16)
    def reject(condition,code):status[(status==0)&condition]=code
    reject(~eligible,17)
    reject(~torch.isfinite(p).all(-1)|~torch.isfinite(q_init).all((-1,-2)),2)
    reject(~initial_valid,15);reject(~quality_valid,18)
    reject(~affine_valid(torch.nan_to_num(A),cfg).all(-1),7)
    p=torch.nan_to_num(p);q_init=torch.nan_to_num(q_init)
    rays=torch.nan_to_num(rays);A=torch.nan_to_num(A)
    safe_depth=torch.where(initial_valid,depth0,torch.ones_like(depth0)).clamp_min(cfg.min_depth)
    ref,gradient,inside=bilinear(reference,p[:,None]+offsets,True)
    reject(~inside.all(-1),3)
    if reference_mask is not None:
        mask,ok=bilinear(reference_mask,p[:,None]+offsets)
        reject(~((mask>=.999)&ok).all(-1),14)
    texture=ref.std(-1,unbiased=False)
    tensor=torch.einsum('bpi,bpj->bij',gradient,gradient)/area
    texture_eigen=torch.linalg.eigvalsh(tensor)[:,0]
    reject((texture<cfg.texture_std_min)|(texture_eigen<cfg.texture_eigen_min),5)
    theta=torch.zeros(n,5,device=device,dtype=dtype);theta[:,0]=safe_depth.log();theta[:,1]=theta[:,3]=1
    lower=(safe_depth*(1-cfg.max_relative_depth_change)).clamp_min(cfg.min_depth).log()
    upper=(safe_depth*(1+cfg.max_relative_depth_change)).log()
    eye=torch.eye(5,device=device,dtype=dtype)
    def evaluate(t,normal=False):
        r,J,inside,uv,xy=shared_patch_residual_jacobian(targets,cams,center,rays,ref,offsets,A,t)
        finite=torch.isfinite(r).all((-1,-2))&torch.isfinite(J).all((-1,-2,-3))
        valid=inside.all((-1,-2))&finite
        mask_valid=torch.ones(n,device=device,dtype=torch.bool)
        if target_masks is not None:
            for j,mask_image in enumerate(target_masks):
                sampled,ok=bilinear(mask_image,xy[:,j])
                mask_valid &= ((sampled>=.999)&ok).all(-1)
        valid &= mask_valid
        displacement=torch.linalg.vector_norm(uv-q_init,dim=-1)
        valid &= (displacement<=cfg.max_correction).all(-1)
        # Invalid patches keep their failure status; sanitize their algebra only
        # so one rejected point cannot poison a batched eigensolve or linear solve.
        z=torch.nan_to_num(r/cfg.sigma_photo,nan=0.,posinf=0.,neginf=0.)
        rho,robust_weight=huber(z,cfg.huber_photo)
        objective=(weights*rho.mean(-1)).sum(-1)
        photo=r.square().mean(-1).sqrt()
        if not normal:return objective,valid,photo,uv,mask_valid,displacement
        J=torch.nan_to_num(J/cfg.sigma_photo,nan=0.,posinf=0.,neginf=0.)
        w=weights[:,:,None]*robust_weight/area
        H=torch.einsum('bjpk,bjp,bjpl->bkl',J,w,J)
        g=torch.einsum('bjpk,bjp,bjp->bk',J,w,z)
        scale=torch.diagonal(H,dim1=-2,dim2=-1).clamp_min(1e-12).sqrt()
        corr=H/(scale[:,:,None]*scale[:,None,:])
        eig=torch.linalg.eigvalsh(corr)
        condition=eig[:,-1].clamp_min(1e-12)/eig[:,0].clamp_min(1e-12)
        # Schur information for log range after eliminating view gain/bias.
        nuisance=H[:,1:,1:]
        # Pseudoinverse also handles already rejected constant patches. The
        # full conditioned Hessian and texture checks independently gate use.
        coupling=(torch.linalg.pinv(nuisance,hermitian=True)@H[:,1:,0,None]).squeeze(-1)
        log_information=(H[:,0,0]-(H[:,0,1:]*coupling).sum(-1)).clamp_min(0)
        depth_information=log_information/t[:,0].exp().square()
        return objective,valid,photo,uv,mask_valid,displacement,H,g,condition,depth_information,log_information
    values=evaluate(theta,True)
    energy_before,valid,photo_before,uv_before,mask_ok,initial_displacement,H,g,condition_before,info_before,_=values
    reject(~mask_ok,14);reject((initial_displacement>cfg.max_correction).any(-1),9);reject(~valid,4)
    reject(~torch.isfinite(condition_before)|(condition_before>cfg.condition_max)|(info_before<cfg.depth_information_min),6)
    converged=torch.zeros(n,device=device,dtype=torch.bool)
    iterations=torch.zeros(n,device=device,dtype=torch.int16);accepted_steps=torch.zeros_like(iterations)
    damping=torch.full((n,),cfg.lm_initial,device=device,dtype=dtype)
    for iteration in range(cfg.max_iterations):
        active=(status==0)&~converged
        if not bool(active.any()):break
        energy,valid,photo,uv,mask_ok,displacement,H,g,condition,info,log_info=evaluate(theta,True)
        diagonal=torch.diagonal(H,dim1=-2,dim2=-1).clamp_min(1e-6)
        system=H+torch.diag_embed(damping[:,None]*diagonal)+1e-8*eye
        system=torch.where(active[:,None,None],system,eye)
        rhs=torch.where(active[:,None],-g,torch.zeros_like(g))
        step=torch.linalg.solve(system,rhs[:,:,None]).squeeze(-1)
        accepted=torch.zeros(n,device=device,dtype=torch.bool);selected=theta.clone();chosen_energy=energy.clone()
        pixel_change=torch.zeros(n,device=device,dtype=dtype);photo_change=torch.zeros_like(pixel_change)
        for factor in [1.,.5,.25,.125]:
            trial=theta+factor*step;trial[:,0]=torch.maximum(lower,torch.minimum(upper,trial[:,0]))
            trial[:,[1,3]]=trial[:,[1,3]].clamp(cfg.gain_min,cfg.gain_max)
            trial[:,[2,4]]=trial[:,[2,4]].clamp(-cfg.bias_limit,cfg.bias_limit)
            candidate,ok,_,new_uv,_,_=evaluate(trial)
            good=active&~accepted&ok&torch.isfinite(candidate)&(candidate<energy)
            selected[good]=trial[good];chosen_energy[good]=candidate[good]
            pixel_change[good]=torch.linalg.vector_norm(new_uv[good]-uv[good],dim=-1).max(-1).values
            photo_change[good]=(trial[good,1:]-theta[good,1:]).abs().max(-1).values
            accepted |= good
        small_step=(pixel_change<cfg.step_tolerance)&(photo_change<cfg.step_tolerance)
        small_energy=(energy-chosen_energy).abs()<=cfg.objective_tolerance*(1+energy.abs())
        tiny_gradient=torch.linalg.vector_norm(g/diagonal.sqrt(),dim=-1)<1e-4
        converged |= active&((accepted&small_step&small_energy)|tiny_gradient)
        theta=selected;iterations[active]=iteration+1;accepted_steps[accepted]+=1
        damping=torch.where(accepted,damping*.5,damping*10).clamp(1e-7,1e8)
    energy_after,valid,photo_after,uv_after,mask_ok,displacement,_,_,condition,info,log_info=evaluate(theta,True)
    xyz=center+theta[:,0].exp()[:,None]*rays
    final_angles=[]
    for cam in cams:
        target_ray=xyz-camera_center(cam)
        target_ray=target_ray/torch.linalg.vector_norm(target_ray,dim=-1,keepdim=True).clamp_min(1e-12)
        cosine=(rays*target_ray).sum(-1).clamp(-1,1)
        sine=torch.linalg.vector_norm(torch.linalg.cross(rays,target_ray),dim=-1)
        final_angles.append(torch.rad2deg(torch.atan2(sine,cosine)))
    final_angles=torch.stack(final_angles,-1)
    reject(~mask_ok,14);reject(~valid,4)
    reject((displacement>=cfg.max_correction*.995).any(-1),9)
    reject((theta[:,0]<=lower+1e-6)|(theta[:,0]>=upper-1e-6),16)
    reject(~torch.isfinite(condition)|(condition>cfg.condition_max)|(info<cfg.depth_information_min),6)
    reject(((final_angles<cfg.min_angle_deg)|(final_angles>180-cfg.min_angle_deg)).any(-1),15)
    reject((photo_after>cfg.photo_rmse_max).any(-1),10)
    if cfg.require_convergence:reject(~converged,8)
    status[status==0]=1
    return dict(uv_refined=uv_after,uv_projected_initial=uv_before,depth=theta[:,0].exp(),depth_initial=depth0,
        refined_mask=status==1,valid_refined=status==1,refinement_status=status,converged=converged,
        iterations=iterations,accepted_steps=accepted_steps,gain=theta[:,[1,3]],bias=theta[:,[2,4]],
        photo_error_before=photo_before,photo_error_after=photo_after,objective_before=energy_before,objective_after=energy_after,
        refinement_displacement=displacement,initial_projection_displacement=initial_displacement,
        hessian_condition_before=condition_before,hessian_condition_after=condition,
        depth_information_before=info_before,depth_information_after=info,log_depth_information_after=log_info,
        texture_std=texture,texture_eigenvalue=texture_eigen,ray_angle_deg_after=final_angles)


@torch.no_grad()
def refine_shared_depth(reference,targets,cam_ref,cam_targets,p,q_init,A_fixed,quality_components,
                        config=None,weight_mode='uniform',mode='asynchronous_unknown',reference_mask=None,
                        target_masks=None,eligible=None,device='cpu',dtype=torch.float32,initialization=None):
    """Run all supplied regular-grid candidates in bounded-size torch LM batches.

    For asynchronous_unknown, static masks are mandatory in ALL three images and
    every sampled patch must remain inside them. static/synchronized are explicit
    caller declarations, not inferences from equal frame numbers. No GT argument.
    """
    cfg=config if isinstance(config,MultiViewConfig) else MultiViewConfig(**(config or {}))
    if mode not in ['static','synchronized','asynchronous_unknown']:raise ValueError('Unknown temporal/scene mode')
    if len(targets)!=2 or len(cam_targets)!=2:raise ValueError('This minimal experiment requires exactly two targets')
    if mode=='asynchronous_unknown' and (reference_mask is None or target_masks is None or any(m is None for m in target_masks)):
        raise ValueError('Unknown synchronization requires reliable static masks for all three images')
    if min(cfg.sigma_photo,cfg.lm_initial,cfg.min_depth,cfg.max_correction,cfg.quality_fb_scale,cfg.quality_photo_scale)<=0:
        raise ValueError('All objective/trust-region scales must be positive')
    if not 0<cfg.max_relative_depth_change<1 or not 0<cfg.distance_min_weight<=1 or not 0<cfg.quality_floor<=1:
        raise ValueError('Invalid depth trust region or positive weight floor')
    if cfg.max_iterations<1 or cfg.batch_size<1:raise ValueError('Iterations and batch size must be positive')
    p_np=np.asarray(_numpy(p));q_np=np.asarray(_numpy(q_init));n=len(p_np)
    if not n or p_np.shape!=(n,2) or q_np.shape!=(n,2,2):raise ValueError('Expected nonempty p Nx2, q_init Nx2x2')
    if tuple(A_fixed.shape)!=(n,2,2,2):raise ValueError('A_fixed must have shape Nx2x2x2')
    eligible_np=np.ones(n,dtype=bool) if eligible is None else np.asarray(_numpy(eligible),dtype=bool)
    if eligible_np.shape!=(n,):raise ValueError('eligible must have shape N')
    started_init=time.perf_counter()
    if initialization is None:
        initialization=initialize_shared_depth(cam_ref,cam_targets,p_np,q_np,cfg.min_angle_deg,eligible_np)
    else:
        if not np.array_equal(np.asarray(initialization['p']),p_np,equal_nan=True) or not np.array_equal(np.asarray(initialization['q_init']),q_np,equal_nan=True):
            raise ValueError('Cached initialization belongs to different points/correspondences')
        if float(initialization['min_angle_deg'])<cfg.min_angle_deg:
            raise ValueError('Cached initialization used a weaker angle threshold')
    initial_seconds=time.perf_counter()-started_init
    started_weights=time.perf_counter();scores=frozen_view_weights(cam_ref,cam_targets,quality_components,weight_mode,cfg)
    weight_seconds=time.perf_counter()-started_weights
    if scores['weights'].shape!=(n,2):raise ValueError('Quality scores do not match reference point count')
    def tensor(v):return torch.as_tensor(v,device=device,dtype=dtype)
    reference=tensor(reference);targets=[tensor(t) for t in targets]
    if any(im.ndim!=2 or not torch.isfinite(im).all() for im in [reference,*targets]):
        raise ValueError('All three images must be finite float grayscale')
    if reference_mask is not None:reference_mask=tensor(reference_mask)
    if target_masks is not None:target_masks=[tensor(mask) for mask in target_masks]
    cams=[dict(cam,K=tensor(cam['K']),R=tensor(cam['R']),t=tensor(cam['t'])) for cam in cam_targets]
    p,q,A=tensor(p_np),tensor(q_np),tensor(A_fixed)
    initial_valid=torch.as_tensor(initialization['valid'],device=device,dtype=torch.bool)
    quality_valid=torch.as_tensor(scores['quality_valid'],device=device,dtype=torch.bool)
    eligible_tensor=torch.as_tensor(eligible_np,device=device,dtype=torch.bool)
    depth0=tensor(initialization['depth']);rays=tensor(initialization['rays']);center=tensor(initialization['center'])
    weights=tensor(scores['weights']);output={}
    synchronize(device);started=time.perf_counter()
    if torch.device(device).type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for start in range(0,n,cfg.batch_size):
        sl=slice(start,start+cfg.batch_size)
        values=_batch(reference,targets,cams,p[sl],q[sl],A[sl],depth0[sl],initial_valid[sl],rays[sl],center,
                      weights[sl],quality_valid[sl],eligible_tensor[sl],cfg,reference_mask,target_masks)
        for key,value in values.items():output.setdefault(key,[]).append(value.cpu().numpy())
    synchronize(device);seconds=time.perf_counter()-started
    arrays={key:np.concatenate(value) for key,value in output.items()}
    arrays.update(uv_raw=q_np.copy(),A_fixed=_numpy(A_fixed).copy(),geometry_applicable_mask=eligible_np,
        initial_valid=np.asarray(initialization['valid']),ray_angle_deg=np.asarray(initialization['ray_angle_deg']),
        pair_ray_distance=np.asarray(initialization['pair_ray_distance']),ray_gap=np.asarray(initialization['ray_gap']),
        weights_frozen=scores['weights'],weights_unnormalized=scores['weights_unnormalized'],
        confidence_raw=scores['confidence_raw'],fb_error_initial=scores['fb_error_initial'],photo_quality_initial=scores['photo_error_initial'],
        confidence_component=scores['confidence_component'],fb_component=scores['fb_component'],
        photo_component=scores['photo_component'],quality_component=scores['quality_component'],
        distance_component=scores['distance_component'],baselines=scores['baselines'])
    arrays['timing']=dict(seconds=seconds,initialization_seconds=initial_seconds,weight_seconds=weight_seconds,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else 0,
        points=n,config=asdict(cfg),device=str(device),weight_mode=weight_mode,scene_mode=mode,
        distance_scale=scores['distance_scale'],weights_frozen=True,
        depth_definition='Positive distance along unit reference WORLD ray; not camera Z',
        initialization='Unweighted mean of the same two valid positive pair ranges for all arms')
    return arrays
