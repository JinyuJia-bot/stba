"""Chunked batched local least-squares matching; no camera/time optimization.

All inputs are undistorted working-image x/y pixel centers. Invalid/padded
samples are explicitly excluded. Each reference-grid pixel is optimized on its
own patch; no interpolation of a sparse set of optimized displacements is used.
"""
from dataclasses import dataclass, asdict
import time
import numpy as np
import torch


STATUS = {0:'not_attempted',1:'success',2:'nonfinite_input',3:'reference_boundary',
          4:'target_boundary',5:'low_texture',6:'ill_conditioned',7:'affine_invalid',
          8:'no_convergence',9:'correction_limit',10:'photo_high',11:'low_confidence',
          12:'fb_before',13:'fb_after',14:'static_mask_boundary',15:'geometry_degenerate'}


@dataclass
class LSMConfig:
    patch_size: int = 11
    max_iterations: int = 15
    batch_size: int = 512
    model: str = 'affine'
    geometry_mode: str = 'off'
    sigma_photo: float = .03
    huber_photo: float = 1.5
    sigma_epi: float = 1.
    lambda_epi: float = .1
    huber_epi: float = 2.
    affine_regularization: float = .001
    max_correction: float = 4.
    gain_min: float = .5
    gain_max: float = 2.
    bias_limit: float = .35
    affine_sv_min: float = .35
    affine_sv_max: float = 2.8
    affine_det_min: float = .12
    texture_std_min: float = .012
    texture_eigen_min: float = 1e-6
    condition_max: float = 1e7
    photo_rmse_max: float = .10
    step_tolerance: float = .003
    objective_tolerance: float = 1e-5
    lm_initial: float = .001
    require_convergence: bool = True


def synchronize(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)


def bilinear(image, xy, gradients=False):
    """Bilinear values and exact piecewise analytic xy derivatives.

    image HxW or HxWxC; xy arbitrary leading shape ending in2. Returned validity
    must be checked by caller. NaN/out-of-bounds coordinates are sanitized for
    indexing only; they NEVER become valid padded observations.
    """
    h,w=image.shape[:2]
    if min(h,w)<2:raise ValueError('Bilinear sampling needs image dimensions >=2')
    valid=torch.isfinite(xy).all(-1)&(xy[...,0]>=0)&(xy[...,0]<=w-1)&(xy[...,1]>=0)&(xy[...,1]<=h-1)
    clean=torch.nan_to_num(xy,nan=0.,posinf=0.,neginf=0.)
    x=clean[...,0].clamp(0,w-1);y=clean[...,1].clamp(0,h-1)
    ix=x.floor().long().clamp(0,w-2);iy=y.floor().long().clamp(0,h-2)
    dx=x-ix;dy=y-iy
    if image.ndim==3:dx=dx[...,None];dy=dy[...,None]
    v00=image[iy,ix];v10=image[iy,ix+1];v01=image[iy+1,ix];v11=image[iy+1,ix+1]
    value=(1-dy)*((1-dx)*v00+dx*v10)+dy*((1-dx)*v01+dx*v11)
    if not gradients:return value,valid
    gx=(1-dy)*(v10-v00)+dy*(v11-v01)
    gy=(1-dx)*(v01-v00)+dx*(v11-v10)
    return value,torch.stack([gx,gy],-1),valid


def patch_offsets(size,device,dtype):
    if size<3 or size%2!=1:raise ValueError('patch_size must be odd and >=3')
    r=size//2;y,x=torch.meshgrid(torch.arange(-r,r+1,device=device,dtype=dtype),
                               torch.arange(-r,r+1,device=device,dtype=dtype),indexing='ij')
    return torch.stack([x.reshape(-1),y.reshape(-1)],-1)


def huber(residual,delta):
    absolute=residual.abs()
    rho=torch.where(absolute<=delta,residual.square(),2*delta*absolute-delta*delta)
    weight=torch.where(absolute<=delta,torch.ones_like(residual),delta/absolute.clamp_min(1e-12))
    return rho,weight


def affine_valid(A,cfg):
    finite=torch.isfinite(A).all((-1,-2))
    safe=torch.where(finite[...,None,None],A,torch.eye(2,device=A.device,dtype=A.dtype))
    sv=torch.linalg.svdvals(safe)
    return finite&(sv[...,0]<=cfg.affine_sv_max)&(sv[...,-1]>=cfg.affine_sv_min)&(torch.linalg.det(safe)>cfg.affine_det_min)


def unpack(theta,model):
    if model=='translation':
        A=torch.eye(2,device=theta.device,dtype=theta.dtype).expand(len(theta),2,2)
    else:A=theta[:,2:6].reshape(-1,2,2)
    return theta[:,:2],A,theta[:,-2],theta[:,-1]


def patch_residual_jacobian(target,reference_patch,q0,offsets,theta,model):
    delta,A,gain,bias=unpack(theta,model)
    xy=q0[:,None,:]+delta[:,None,:]+torch.einsum('bij,pj->bpi',A,offsets)
    value,gradient,inside=bilinear(target,xy,True)
    residual=value-gain[:,None]*reference_patch-bias[:,None]
    gx,gy=gradient.unbind(-1);sx,sy=offsets.unbind(-1)
    columns=[gx,gy]
    if model=='affine':columns.extend([gx*sx,gx*sy,gy*sx,gy*sy])
    columns.extend([-reference_patch,-torch.ones_like(value)])
    J=torch.stack(columns,-1)
    return residual,J,inside,xy


def _batch(reference,target,p,q0,A0,applicable,F,cfg,reference_mask=None,target_mask=None):
    n=len(p);dtype=p.dtype;device=p.device;dim=4 if cfg.model=='translation' else 8
    offsets=patch_offsets(cfg.patch_size,device,dtype);area=len(offsets)
    status=torch.zeros(n,device=device,dtype=torch.int16)
    def reject(condition,reason):
        status[(status==0)&condition]=reason
    reject(~torch.isfinite(p).all(-1)|~torch.isfinite(q0).all(-1),2)
    p=torch.nan_to_num(p);q0=torch.nan_to_num(q0)
    ref,ref_gradient,ref_inside=bilinear(reference,p[:,None]+offsets,True)
    reject(~ref_inside.all(-1),3)
    if reference_mask is not None:
        mask,valid=bilinear(reference_mask,p[:,None]+offsets)
        reject(~((mask>=.999)&valid).all(-1),14)
    texture_std=ref.std(-1,unbiased=False)
    texture_tensor=torch.einsum('bpi,bpj->bij',ref_gradient,ref_gradient)/area
    texture_eigen=torch.linalg.eigvalsh(texture_tensor)[:,0]
    reject((texture_std<cfg.texture_std_min)|(texture_eigen<cfg.texture_eigen_min),5)
    if A0 is None:A0=torch.eye(2,device=device,dtype=dtype).expand(n,2,2).clone()
    if cfg.model=='translation':A0=torch.eye(2,device=device,dtype=dtype).expand(n,2,2).clone()
    reject(~affine_valid(A0,cfg),7)
    A0=torch.where(torch.isfinite(A0).all((-1,-2))[:,None,None],A0,torch.eye(2,device=device,dtype=dtype))
    theta=torch.zeros(n,dim,device=device,dtype=dtype);theta[:,-2]=1
    if cfg.model=='affine':theta[:,2:6]=A0.reshape(n,4)
    use_epi=applicable & (cfg.geometry_mode=='soft_epipolar')
    if F is None:
        if use_epi.any():raise ValueError('soft_epipolar needs fixed-calibration F')
        line=torch.zeros(n,3,device=device,dtype=dtype);line[:,1]=1
    else:
        hp=torch.cat([p,torch.ones(n,1,device=device,dtype=dtype)],-1)
        line=hp@F.T;norm=torch.linalg.vector_norm(line[:,:2],dim=-1)
        reject(use_epi & (norm<1e-12),15)
        line=line/norm.clamp_min(1e-12)[:,None]
    def evaluate(t,with_normal=False):
        r,J,inside,xy=patch_residual_jacobian(target,ref,q0,offsets,t,cfg.model)
        valid=inside.all(-1)
        if target_mask is not None:
            mask,mask_inside=bilinear(target_mask,xy)
            valid &= ((mask>=.999)&mask_inside).all(-1)
        delta,A,_,_=unpack(t,cfg.model);valid &= affine_valid(A,cfg)
        z=r/cfg.sigma_photo;rho,w=huber(z,cfg.huber_photo)
        objective=rho.mean(-1)
        epi=((q0+delta)*line[:,:2]).sum(-1)+line[:,2]
        epi_z=epi/cfg.sigma_epi;epi_rho,epi_w=huber(epi_z,cfg.huber_epi)
        objective=objective+cfg.lambda_epi*use_epi*epi_rho
        if cfg.model=='affine':objective += cfg.affine_regularization*(A-A0).square().sum((-1,-2))
        if not with_normal:return objective,valid,r.square().mean(-1).sqrt(),epi
        J=J/cfg.sigma_photo
        H=torch.einsum('bpi,bp,bpj->bij',J,w,J)/area
        g=torch.einsum('bpi,bp,bp->bi',J,w,z)/area
        epi_J=torch.zeros(n,dim,device=device,dtype=dtype);epi_J[:,:2]=line[:,:2]/cfg.sigma_epi
        ew=cfg.lambda_epi*use_epi*epi_w
        H += ew[:,None,None]*epi_J[:,:,None]*epi_J[:,None,:]
        g += (ew*epi_z)[:,None]*epi_J
        if cfg.model=='affine':
            H[:,2:6,2:6] += cfg.affine_regularization*torch.eye(4,device=device,dtype=dtype)
            g[:,2:6] += cfg.affine_regularization*(A-A0).reshape(n,4)
        # Condition measure normalizes columns so arbitrary parameter units do
        # not dominate. Observability also has a separate image-gradient test.
        scale=torch.diagonal(H,dim1=-2,dim2=-1).clamp_min(1e-12).sqrt()
        correlation=H/(scale[:,:,None]*scale[:,None,:])
        eig=torch.linalg.eigvalsh(correlation)
        condition=eig[:,-1].clamp_min(1e-12)/eig[:,0].clamp_min(1e-12)
        return objective,valid,r.square().mean(-1).sqrt(),epi,H,g,condition
    objective,valid,photo_before,epi_before,H,g,cond_before=evaluate(theta,True)
    reject(~valid,4)
    reject(~torch.isfinite(cond_before)|(cond_before>cfg.condition_max),6)
    energy_before=objective.clone();converged=torch.zeros(n,device=device,dtype=torch.bool)
    iterations=torch.zeros(n,device=device,dtype=torch.int16)
    accepted_steps=torch.zeros(n,device=device,dtype=torch.int16)
    damping=torch.full((n,),cfg.lm_initial,device=device,dtype=dtype)
    eye=torch.eye(dim,device=device,dtype=dtype)
    for iteration in range(cfg.max_iterations):
        active=(status==0)&~converged
        if not bool(active.any()):break
        objective,valid,photo,epi,H,g,condition=evaluate(theta,True)
        diagonal=torch.diagonal(H,dim1=-2,dim2=-1).clamp_min(1e-6)
        system=H+torch.diag_embed(damping[:,None]*diagonal)+1e-8*eye
        system=torch.where(active[:,None,None],system,eye)
        rhs=torch.where(active[:,None],-g,torch.zeros_like(g))
        step=torch.linalg.solve(system,rhs.unsqueeze(-1)).squeeze(-1)
        accepted=torch.zeros(n,device=device,dtype=torch.bool);selected=theta.clone()
        selected_energy=objective.clone();selected_change=torch.zeros(n,device=device,dtype=dtype)
        for factor in [1.,.5,.25,.125]:
            trial=theta+factor*step
            delta=trial[:,:2];length=torch.linalg.vector_norm(delta,dim=-1)
            trial[:,:2]=delta*torch.clamp(cfg.max_correction/length.clamp_min(1e-12),max=1)[:,None]
            trial[:,-2]=trial[:,-2].clamp(cfg.gain_min,cfg.gain_max)
            trial[:,-1]=trial[:,-1].clamp(-cfg.bias_limit,cfg.bias_limit)
            energy,inside,_,_=evaluate(trial)
            good=active&~accepted&inside&torch.isfinite(energy)&(energy<objective)
            selected[good]=trial[good];selected_energy[good]=energy[good]
            selected_change[good]=torch.linalg.vector_norm((trial-theta)[good],dim=-1)
            accepted |= good
        tiny_gradient=torch.linalg.vector_norm(g/diagonal.sqrt(),dim=-1)<1e-4
        small_step=selected_change<cfg.step_tolerance
        small_energy=(objective-selected_energy).abs()<=cfg.objective_tolerance*(1+objective.abs())
        converged |= active&((accepted&small_step&small_energy)|tiny_gradient)
        theta=selected
        iterations[active]=iteration+1;accepted_steps[accepted]+=1
        damping=torch.where(accepted,damping*.5,damping*10).clamp(1e-7,1e8)
    energy_after,valid,photo_after,epi_after,_,_,condition=evaluate(theta,True)
    delta,A,gain,bias=unpack(theta,cfg.model)
    displacement=torch.linalg.vector_norm(delta,dim=-1)
    reject(~valid,4)
    reject(displacement>=cfg.max_correction*.995,9)
    reject(~affine_valid(A,cfg),7)
    reject(condition>cfg.condition_max,6)
    reject(photo_after>cfg.photo_rmse_max,10)
    if cfg.require_convergence:reject(~converged,8)
    status[status==0]=1
    return dict(uv_refined=q0+delta,A=A,gain=gain,bias=bias,refinement_status=status,
        refined_mask=status==1,converged=converged,iterations=iterations,accepted_steps=accepted_steps,
        refinement_displacement=displacement,photo_error_before=photo_before,photo_error_after=photo_after,
        epipolar_error_before=epi_before.abs(),epipolar_error_after=epi_after.abs(),
        objective_before=energy_before,objective_after=energy_after,texture_std=texture_std,
        texture_eigenvalue=texture_eigen,hessian_condition_before=cond_before,hessian_condition_after=condition)


@torch.no_grad()
def refine_points(reference,target,p,q0,config=None,A0=None,geometry_applicable=None,F=None,
                  reference_mask=None,target_mask=None,device='cpu',dtype=torch.float32):
    """Optimize ALL supplied regular-grid pixels in bounded-size GPU batches.

    No GT is accepted by this API. Model/confidence/FB screening is performed by
    the caller and preserved separately from optimization status.
    """
    cfg=config if isinstance(config,LSMConfig) else LSMConfig(**(config or {}))
    if cfg.model not in ['translation','affine'] or cfg.geometry_mode not in ['off','soft_epipolar']:
        raise ValueError('Unknown LSM model/geometry mode')
    if min(cfg.sigma_photo,cfg.sigma_epi,cfg.max_correction,cfg.lm_initial)<=0 or cfg.max_iterations<1:
        raise ValueError('LSM scales and iterations must be positive')
    def tensor(x):return torch.as_tensor(x,device=device,dtype=dtype)
    reference,target=tensor(reference),tensor(target)
    if reference.ndim!=2 or target.ndim!=2 or not torch.isfinite(reference).all() or not torch.isfinite(target).all():
        raise ValueError('Reference and target must be finite float grayscale images')
    p,q0=tensor(p),tensor(q0)
    if p.shape!=q0.shape or p.ndim!=2 or p.shape[1]!=2:raise ValueError('p and q0 must be Nx2')
    if A0 is not None:A0=tensor(A0)
    applicable=torch.zeros(len(p),device=device,dtype=torch.bool) if geometry_applicable is None else torch.as_tensor(geometry_applicable,device=device,dtype=torch.bool)
    if F is not None:F=tensor(F)
    if reference_mask is not None:reference_mask=tensor(reference_mask)
    if target_mask is not None:target_mask=tensor(target_mask)
    synchronize(device);started=time.perf_counter()
    if torch.device(device).type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    output={}
    for start in range(0,len(p),cfg.batch_size):
        sl=slice(start,start+cfg.batch_size)
        values=_batch(reference,target,p[sl],q0[sl],None if A0 is None else A0[sl],
                      applicable[sl],F,cfg,reference_mask,target_mask)
        for key,value in values.items():output.setdefault(key,[]).append(value.cpu().numpy())
    synchronize(device);seconds=time.perf_counter()-started
    arrays={k:np.concatenate(v) for k,v in output.items()}
    arrays['timing']=dict(seconds=seconds,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else 0,
                          points=len(p),config=asdict(cfg),device=str(device))
    return arrays


@torch.no_grad()
def affine_from_dense(uv,confidence,valid,p,config=None,device='cpu',fit_radius=2,fit_error_max=1.):
    """Fit local Jacobian from a dense field; exclude unreliable/discontinuous neighbors.

    `valid` should already include raw bidirectional consistency. A fitted matrix
    is accepted only with >=6 neighbors, small fit residual, positive determinant
    and bounded singular values. Otherwise return identity with reliable=False.
    """
    cfg=config or LSMConfig();p=torch.as_tensor(p,device=device,dtype=torch.float32)
    field=torch.as_tensor(uv,device=device,dtype=torch.float32)
    confidence=torch.as_tensor(confidence,device=device,dtype=torch.float32)
    validity=torch.as_tensor(valid,device=device,dtype=torch.float32)
    offsets=torch.tensor([[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]],device=device,dtype=torch.float32)*fit_radius
    result=[];flags=[]
    for start in range(0,len(p),cfg.batch_size*4):
        points=p[start:start+cfg.batch_size*4];center,center_valid=bilinear(field,points)
        q,inside=bilinear(field,points[:,None]+offsets)
        score,_=bilinear(confidence,points[:,None]+offsets);ok,_=bilinear(validity,points[:,None]+offsets)
        change=q-center[:,None]
        mask=inside&(ok>.999)&torch.isfinite(q).all(-1)&(score>.1)
        mask &= torch.linalg.vector_norm(change,dim=-1)<cfg.affine_sv_max*torch.linalg.vector_norm(offsets,dim=-1)+1
        weight=mask*score.clamp_min(0);change=torch.nan_to_num(change)
        H=torch.einsum('pi,bp,pj->bij',offsets,weight,offsets)+1e-6*torch.eye(2,device=device)
        rhs=torch.einsum('pi,bp,bpj->bij',offsets,weight,change)
        A=torch.linalg.solve(H,rhs).transpose(-1,-2)
        residual=change-torch.einsum('bij,pj->bpi',A,offsets)
        error=(residual.square().sum(-1)*weight).sum(-1)/weight.sum(-1).clamp_min(1e-12)
        reliable=(mask.sum(-1)>=6)&center_valid&(error.sqrt()<=fit_error_max)&affine_valid(A,cfg)
        A=torch.where(reliable[:,None,None],A,torch.eye(2,device=device))
        result.append(A.cpu().numpy());flags.append(reliable.cpu().numpy())
    return np.concatenate(result),np.concatenate(flags)
