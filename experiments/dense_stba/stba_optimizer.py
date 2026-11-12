"""Fixed-calibration Dense STBA with one GLOBAL camera-offset vector.

Each trajectory has cubic-spline control points. Batched small trajectory normal
blocks are eliminated into ONE camera-offset Schur system, accumulated over all
chunks. This never constructs a full scene Hessian or independently fits chunk
offsets. Raw observations enter once, at timestamp + beta, in physical seconds.

Objective: sum_o w_o Huber(||projection(X_p(t_o+beta_c))-uv_o||/sigma_px)
         + lambda_motion sum_p integral ||X_p''(t)||^2 dt.
The motion integral uses world-unit^2/seconds^3; the objective is NOT secretly
averaged by track count or observation count. Model scores are heuristic weights.
"""
from dataclasses import dataclass,asdict
import time
import numpy as np
import torch
try:
    from .trajectory_spline import basis_matrix,acceleration_gram
except ImportError:
    from trajectory_spline import basis_matrix,acceleration_gram


@dataclass
class STBAConfig:
    max_iterations:int=30
    batch_size:int=128
    sigma_px:float=1.
    huber_delta:float=2.
    lambda_motion:float=.01
    lm_initial:float=.001
    lm_min:float=1e-8
    lm_max:float=1e9
    beta_step_limit_seconds:float=.05
    beta_step_tolerance_seconds:float=1e-7
    control_step_tolerance:float=1e-7
    objective_tolerance:float=1e-8
    min_observations:int=6
    min_cameras:int=2
    min_motion_span_px:float=.5
    min_dynamic_tracks:int=3
    beta_information_min:float=1e-4
    beta_condition_max:float=1e10
    beta_bound_tolerance_seconds:float=1e-5
    inlier_threshold_px:float=4.
    minimum_camera_depth:float=1e-6
    integration_interval:object=None
    verbose:bool=False


def _camera_tensors(cams,device,dtype):
    if any(cam.get('coordinate_system') in ['raw_distorted','distorted'] for cam in cams):
        raise ValueError('Dense STBA expects once-undistorted pinhole working-image coordinates')
    K=torch.stack([torch.as_tensor(c['K'],device=device,dtype=dtype) for c in cams])
    R=torch.stack([torch.as_tensor(c['R'],device=device,dtype=dtype) for c in cams])
    t=torch.stack([torch.as_tensor(c['t'],device=device,dtype=dtype).reshape(3) for c in cams])
    if K.shape[-2:]!=(3,3) or R.shape[-2:]!=(3,3):raise ValueError('Camera matrices must be 3x3')
    return K,R,t


def _projection(K,R,t,xyz):
    xc=torch.einsum('cij,pctj->pcti',R,xyz)+t[None,:,None]
    h=torch.einsum('cij,pctj->pcti',K,xc)
    denominator=h[...,2:3]
    safe=torch.where(denominator.abs()>1e-12,denominator,torch.full_like(denominator,float('nan')))
    uv=h[...,:2]/safe
    M=K@R
    J=(M[None,:,None,:2,:]*safe[...,None]-h[...,:2,None]*M[None,:,None,2:3,:])/safe[...,None].square()
    return uv,J,xc[...,2]


def observation_model(control,beta,times,knots,cams,jacobian=False):
    """Differentiable torch projection; control[P,K,3], beta[C], times[C,T].

    Optional J_control[P,C,T,2,K*3] and J_time[P,C,T,2] are exact derivatives.
    J_time is per-observation d_uv/d_its_camera_beta, in pixels/second.
    """
    if not isinstance(control,torch.Tensor):control=torch.as_tensor(control,dtype=torch.float64)
    device,dtype=control.device,control.dtype
    beta=torch.as_tensor(beta,device=device,dtype=dtype);times=torch.as_tensor(times,device=device,dtype=dtype)
    knots=torch.as_tensor(knots,device=device,dtype=dtype)
    K,R,t=_camera_tensors(cams,device,dtype)
    actual=times+beta[:,None]
    B=basis_matrix(actual,knots);D=basis_matrix(actual,knots,1)
    xyz=torch.einsum('ctk,pkd->pctd',B,control)
    velocity=torch.einsum('ctk,pkd->pctd',D,control)
    uv,J,depth=_projection(K,R,t,xyz)
    result=dict(xyz=xyz,velocity=velocity,uv=uv,camera_depth=depth,times=actual,basis=B)
    if jacobian:
        result['J_control']=(J[...,None,:]*B[None,:,:,None,:,None]).flatten(-2)
        result['J_time']=torch.einsum('pctij,pctj->pcti',J,velocity)
    return result


def _huber(norm,delta):
    return (torch.where(norm<=delta,norm.square(),2*delta*norm-delta**2),
            torch.where(norm<=delta,torch.ones_like(norm),delta/norm.clamp_min(1e-12)))


def _motion_score(uv,visible):
    values=torch.where(visible[...,None],uv,torch.full_like(uv,float('nan')))
    # Robust displacement span avoids calling frame-to-frame observation noise
    # genuine motion. This is only a motion-information screen, never GT.
    lo=torch.nanquantile(values,.1,dim=2);hi=torch.nanquantile(values,.9,dim=2)
    scores=torch.linalg.vector_norm(hi-lo,dim=-1)
    scores=torch.where(visible.sum(-1)>=4,scores,torch.zeros_like(scores))
    return torch.nan_to_num(scores).max(-1).values


def _synchronize(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)


@torch.no_grad()
def optimize_stba(uv,visible,weight,times,cams,knots,control_init,beta_init,config=None,
                  mode='joint',reference_camera=0,beta_bounds_seconds=None,beta_track_mask=None,
                  device='cpu',dtype=torch.float64):
    """Optimize spline controls and globally shared beta, with fixed K/R/t/D.

    uv[P,C,T,2], visible/weight[P,C,T], timestamps[C,T] in seconds. Bounds must
    be [C,2] seconds (or a shared [2] interval), and knots must cover all allowed
    observation times. mode='fixedbeta' uses exactly the supplied beta; 'joint'
    also updates all non-reference camera offsets through one Schur solve.

    beta_track_mask optionally gates ONLY beta Jacobian/normal accumulation;
    all eligible trajectories still fit X using their actual times+beta. Full
    objective backtracking guards these explicitly motion-gated joint steps.
    No GT, interpolated pseudo-observation, or holdout observation is accepted
    as a special input. The caller supplies training visibility only.
    """
    cfg=config if isinstance(config,STBAConfig) else STBAConfig(**(config or {}))
    if mode not in ['joint','fixedbeta']:raise ValueError('mode must be joint or fixedbeta')
    if min(cfg.max_iterations,cfg.batch_size,cfg.sigma_px,cfg.huber_delta,cfg.lm_initial)<=0 or cfg.lambda_motion<0:
        raise ValueError('Invalid optimizer scales or iteration limits')
    tensor=lambda v:torch.as_tensor(v,device=device,dtype=dtype)
    u=tensor(uv);vis=torch.as_tensor(visible,device=device,dtype=torch.bool).clone()
    weights=tensor(weight);timestamps=tensor(times);knot=tensor(knots)
    control=tensor(control_init).clone();initial_control=control.clone();beta=tensor(beta_init).clone()
    if u.ndim!=4 or u.shape[-1]!=2:raise ValueError('uv must be P,C,T,2')
    P,C,T,_=u.shape;K=len(knot)-4;D=3*K
    if vis.shape!=(P,C,T) or weights.shape!=vis.shape or timestamps.shape!=(C,T):raise ValueError('Observation array dimensions disagree')
    if control.shape!=(P,K,3) or beta.shape!=(C,) or len(cams)!=C:raise ValueError('Control/camera/beta dimensions disagree')
    if not 0<=reference_camera<C or abs(float(beta[reference_camera]))>1e-12:
        raise ValueError('Reference beta must already equal zero to fix the time-origin gauge')
    nonref=[c for c in range(C) if c!=reference_camera];nb=len(nonref)
    if beta_bounds_seconds is None:
        if mode=='joint':raise ValueError('Explicit beta_bounds_seconds are required for joint estimation')
        bounds=torch.stack([beta,beta],-1)
    else:
        bounds=tensor(beta_bounds_seconds)
        if bounds.shape==(2,):bounds=bounds[None].expand(C,2).clone()
        if bounds.shape!=(C,2):raise ValueError('Bounds must be [C,2] or [lower,upper] seconds')
    bounds=bounds.clone();bounds[reference_camera]=0
    if bool((bounds[:,0]>bounds[:,1]).any()) or bool(((beta<bounds[:,0])|(beta>bounds[:,1])).any()):
        raise ValueError('Initial beta must lie inside its ordered hard box bounds')
    finite_time=torch.isfinite(timestamps)
    vis &= torch.isfinite(u).all(-1)&torch.isfinite(weights)&(weights>0)&finite_time[None]
    used_times=vis.any(0)
    earliest=timestamps+bounds[:,0,None];latest=timestamps+bounds[:,1,None]
    if bool(((earliest<knot[3]-1e-10)|(latest>knot[-4]+1e-10))[used_times].any()):
        raise ValueError('Fixed knot domain must cover all training timestamps plus allowed beta bounds')
    timestamps=torch.nan_to_num(timestamps);u=torch.nan_to_num(u)
    weights=torch.where(vis,weights,torch.zeros_like(weights))
    count=vis.sum((-1,-2));camera_count=vis.any(-1).sum(-1)
    initialized=torch.isfinite(control).all((-1,-2))
    eligible=initialized&(count>=cfg.min_observations)&(camera_count>=cfg.min_cameras)
    control=torch.nan_to_num(control)
    Kcam,Rcam,tcam=_camera_tensors(cams,device,dtype)
    Q=acceleration_gram(knot,cfg.integration_interval)
    Q3=torch.kron(Q,torch.eye(3,device=device,dtype=dtype))
    eye=torch.eye(D,device=device,dtype=dtype)
    hot=torch.zeros(C,nb,device=device,dtype=dtype)
    for j,c in enumerate(nonref):hot[c,j]=1
    motion_score=_motion_score(u,vis)
    dynamic=(motion_score>=cfg.min_motion_span_px) if beta_track_mask is None else torch.as_tensor(beta_track_mask,device=device,dtype=torch.bool).clone()
    if dynamic.shape!=(P,):raise ValueError('beta_track_mask must have shape [P]')
    dynamic &= eligible
    # Reject bad initialization rather than allowing invalid camera projection
    # to contaminate all small batched systems.
    B0=basis_matrix(timestamps+beta[:,None],knot)
    for start in range(0,P,cfg.batch_size):
        sl=slice(start,start+cfg.batch_size)
        xyz=torch.einsum('ctk,pkd->pctd',B0,control[sl])
        prediction,_,z=_projection(Kcam,Rcam,tcam,xyz)
        bad=vis[sl]&((z<=cfg.minimum_camera_depth)|~torch.isfinite(prediction).all(-1))
        eligible[sl] &= ~bad.any((-1,-2))
    vis &= eligible[:,None,None];weights=torch.where(vis,weights,torch.zeros_like(weights));dynamic &= eligible
    dynamic_counts=(vis.any(-1)&dynamic[:,None]).sum(0)
    camera_observable=dynamic_counts>=cfg.min_dynamic_tracks
    beta_enabled=mode=='joint' and bool(camera_observable[nonref].any())
    active_beta=camera_observable[nonref]
    hot=hot*active_beta[None]
    initial_beta=beta.clone()

    def evaluate(c,b,normal=False,damping=0.,diagnostic=False,collect=False):
        actual=timestamps+b[:,None];B=basis_matrix(actual,knot);Bt=basis_matrix(actual,knot,1)
        total_image=torch.zeros((),device=device,dtype=dtype);total_motion=total_image.clone()
        weighted_sq=total_image.clone();weight_sum=total_image.clone();invalid=0
        S=torch.zeros(nb,nb,device=device,dtype=dtype);gb=torch.zeros(nb,device=device,dtype=dtype)
        Hbb_sum=torch.zeros_like(S);response=[];solved_gradient=[];control_grad_max=0.
        collected={key:[] for key in ['uv_pred','camera_depth','residual_px','inlier_mask']} if collect else None
        for start in range(0,P,cfg.batch_size):
            sl=slice(start,start+cfg.batch_size);cp=c[sl];good=vis[sl];w0=weights[sl]
            xyz=torch.einsum('ctk,pkd->pctd',B,cp)
            pred,Jproj,z=_projection(Kcam,Rcam,tcam,xyz)
            finite=torch.isfinite(pred).all(-1)&(z>cfg.minimum_camera_depth)
            invalid+=int((good&~finite).sum())
            residual=torch.nan_to_num(pred-u[sl]);r=residual/cfg.sigma_px
            norm=torch.linalg.vector_norm(r,dim=-1);rho,robust=_huber(norm,cfg.huber_delta)
            image=(w0*rho).sum()
            # Q is PSD; clipping only removes roundoff cancellation for constant
            # trajectories whose physical acceleration integral is exactly zero.
            motion=(cfg.lambda_motion*torch.einsum('pkd,kl,pld->',cp*eligible[sl,None,None],Q,cp)).clamp_min(0)
            total_image+=image;total_motion+=motion
            weighted_sq+=(w0*residual.square().sum(-1)).sum();weight_sum+=w0.sum()
            if collect:
                collected['uv_pred'].append(pred.cpu());collected['camera_depth'].append(z.cpu())
                collected['residual_px'].append(torch.linalg.vector_norm(residual,dim=-1).cpu())
                collected['inlier_mask'].append((good&finite&(torch.linalg.vector_norm(residual,dim=-1)<=cfg.inlier_threshold_px)).cpu())
            if not normal:continue
            m=len(cp)
            A=(Jproj[...,None,:]*B[None,:,:,None,:,None]).reshape(m,C*T*2,D)/cfg.sigma_px
            A=torch.nan_to_num(A)
            velocity=torch.einsum('ctk,pkd->pctd',Bt,cp)
            time_jac=torch.einsum('pctij,pctj->pcti',Jproj,velocity)/cfg.sigma_px
            G=(time_jac[...,None]*hot[None,:,None,None,:]).reshape(m,C*T*2,nb)
            G=torch.nan_to_num(G)*dynamic[sl,None,None]
            w=(w0*robust)[...,None].expand(-1,-1,-1,2).reshape(m,-1)
            residual_flat=r.reshape(m,-1)
            Hpp=torch.einsum('pni,pn,pnj->pij',A,w,A)
            gp=torch.einsum('pni,pn,pn->pi',A,w,residual_flat)
            Hpp+=cfg.lambda_motion*Q3[None]*eligible[sl,None,None]
            gp+=cfg.lambda_motion*torch.einsum('kl,pld->pkd',Q,cp).reshape(m,D)*eligible[sl,None]
            Hpb=torch.einsum('pni,pn,pnj->pij',A,w,G)
            Hb=torch.einsum('pni,pn,pnj->pij',G,w,G)
            gradient_beta=torch.einsum('pni,pn,pn->pi',G,w,residual_flat)
            if diagnostic:
                inverse=torch.linalg.pinv(Hpp,hermitian=True,rtol=1e-10)
                solved_g=torch.einsum('pij,pj->pi',inverse,gp)
                solved_B=inverse@Hpb
            else:
                diagonal=torch.diagonal(Hpp,dim1=-2,dim2=-1).abs().clamp_min(1e-8)
                system=Hpp+torch.diag_embed(damping*diagonal)+1e-10*eye
                rhs=torch.cat([gp[:,:,None],Hpb],-1)
                answer=torch.linalg.solve(system,rhs)
                solved_g=answer[:,:,0];solved_B=answer[:,:,1:]
            S+=(Hb-Hpb.transpose(-1,-2)@solved_B).sum(0)
            gb+=(gradient_beta-torch.einsum('pij,pi->pj',Hpb,solved_g)).sum(0)
            Hbb_sum+=Hb.sum(0)
            response.append(solved_B);solved_gradient.append(solved_g)
            control_grad_max=max(control_grad_max,float(gp.abs().max()))
        objective=total_image+total_motion
        if invalid:objective=torch.full_like(objective,float('inf'))
        result=dict(objective=objective,image_cost=total_image,motion_cost=total_motion,
            reprojection_rmse=(weighted_sq/weight_sum.clamp_min(1e-12)).sqrt(),invalid_projection_observations=invalid)
        if normal:
            S=(S+S.T)/2
            result.update(S=S,Hbb=Hbb_sum,gb=gb,response=torch.cat(response),
                          solved_gradient=torch.cat(solved_gradient),control_grad_max=control_grad_max)
        if collect:result.update({key:torch.cat(value).numpy() for key,value in collected.items()})
        return result

    _synchronize(device);started=time.perf_counter()
    if torch.device(device).type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    damping=cfg.lm_initial;logs=[];converged=False
    initial=evaluate(control,beta)
    if not bool(eligible.any()):
        raise ValueError('No trajectory has finite initialization and sufficient positive-depth multiview observations')
    for iteration in range(cfg.max_iterations):
        state=evaluate(control,beta,True,damping)
        if beta_enabled:
            S=state['S']+damping*torch.diag(state['Hbb'].diagonal().abs().clamp_min(1e-8))
            enabled_outer=active_beta[:,None]&active_beta[None,:]
            S=torch.where(enabled_outer,S,torch.zeros_like(S))+torch.diag((~active_beta).to(dtype))
            gb=torch.where(active_beta,state['gb'],torch.zeros_like(state['gb']))
            delta_beta=torch.linalg.solve(S+1e-10*torch.eye(nb,device=device,dtype=dtype),-gb)
            delta_beta=delta_beta.clamp(-cfg.beta_step_limit_seconds,cfg.beta_step_limit_seconds)
            projected=(beta[nonref]+delta_beta).maximum(bounds[nonref,0]).minimum(bounds[nonref,1])
            delta_beta=projected-beta[nonref]
        else:delta_beta=torch.zeros(nb,device=device,dtype=dtype)
        delta_control=-state['solved_gradient']-torch.einsum('pij,j->pi',state['response'],delta_beta)
        delta_control=delta_control.reshape(P,K,3)*eligible[:,None,None]
        accepted=False;new_state=state;accepted_factor=0.
        for factor in [1.,.5,.25,.125,.0625]:
            trial_control=control+factor*delta_control
            trial_beta=beta.clone();trial_beta[nonref]+=factor*delta_beta;trial_beta[reference_camera]=0
            trial=evaluate(trial_control,trial_beta)
            if torch.isfinite(trial['objective']) and float(trial['objective'])<float(state['objective']):
                control,beta,new_state=trial_control,trial_beta,trial;accepted=True;accepted_factor=factor
                break
        cp_step=float(torch.linalg.vector_norm(delta_control,dim=-1).max())*accepted_factor
        b_step=float(delta_beta.abs().max())*accepted_factor if nb else 0.
        improvement=float(state['objective']-new_state['objective'])
        rel=improvement/(1+abs(float(state['objective'])))
        stationary=state['control_grad_max']<1e-8 and (not beta_enabled or float(state['gb'].abs().max())<1e-8)
        converged=bool(stationary or (accepted and cp_step<cfg.control_step_tolerance and b_step<cfg.beta_step_tolerance_seconds)
                       or (accepted and rel<cfg.objective_tolerance and b_step<cfg.beta_step_tolerance_seconds))
        logs.append(dict(iteration=iteration,objective=float(new_state['objective']),image_cost=float(new_state['image_cost']),
            motion_cost=float(new_state['motion_cost']),reprojection_rmse_px=float(new_state['reprojection_rmse']),
            beta_seconds=beta.cpu().tolist(),accepted=accepted,line_search_factor=accepted_factor,lm_damping=damping,
            maximum_control_step=cp_step,maximum_beta_step_seconds=b_step,relative_objective_reduction=rel))
        if cfg.verbose:print(logs[-1],flush=True)
        damping=min(cfg.lm_max,max(cfg.lm_min,damping*(.5 if accepted else 10)))
        if converged or (not accepted and damping>=cfg.lm_max):break
    final=evaluate(control,beta,collect=True)
    information=evaluate(control,beta,True,0.,diagnostic=True)
    _synchronize(device);seconds=time.perf_counter()-started
    inliers=final.pop('inlier_mask');camera_inliers=inliers.any(-1).sum(-1)
    trajectory_valid=(inliers.sum((-1,-2))>=cfg.min_observations)&(camera_inliers>=cfg.min_cameras)&eligible.cpu().numpy()
    physical_times=timestamps+beta[:,None]
    lower=torch.where(vis,physical_times[None],torch.full_like(weights,float('inf'))).amin((-1,-2))
    upper=torch.where(vis,physical_times[None],torch.full_like(weights,-float('inf'))).amax((-1,-2))
    support=torch.stack([lower,upper],-1);support[~eligible]=float('nan')
    schur=information['S'].cpu().numpy();hbb=information['Hbb'].cpu().numpy()
    active_np=active_beta.cpu().numpy();active_schur=schur[active_np][:,active_np]
    if len(active_schur):
        scale=np.sqrt(np.maximum(np.diag(active_schur),1e-12))
        eigenvalues=np.linalg.eigvalsh(active_schur/scale[:,None]/scale[None,:])
        beta_condition=float(max(eigenvalues[-1],1e-12)/max(eigenvalues[0],1e-12))
    else:eigenvalues=np.empty(0);beta_condition=float('inf')
    reliable=np.zeros(C,bool);reasons=['reference_gauge' if c==reference_camera else 'not_estimated' for c in range(C)]
    beta_np=beta.cpu().numpy();bounds_np=bounds.cpu().numpy()
    for j,c in enumerate(nonref):
        if mode=='fixedbeta':reasons[c]='fixed_beta_not_estimated'
        elif int(dynamic_counts[c])<cfg.min_dynamic_tracks:reasons[c]='insufficient_motion'
        elif schur[j,j]<cfg.beta_information_min:reasons[c]='weak_marginal_time_information'
        elif beta_condition>cfg.beta_condition_max:reasons[c]='degenerate_joint_time_information'
        elif min(beta_np[c]-bounds_np[c,0],bounds_np[c,1]-beta_np[c])<=cfg.beta_bound_tolerance_seconds:reasons[c]='at_beta_bound'
        elif not converged:reasons[c]='iteration_limit_or_no_convergence'
        else:reasons[c]='locally_observable';reliable[c]=True
    return dict(control=control.cpu().numpy(),control_initial=initial_control.cpu().numpy(),knots=knot.cpu().numpy(),
        beta=beta_np,beta_initial=initial_beta.cpu().numpy(),beta_bounds_seconds=bounds_np,reference_camera=reference_camera,
        trajectory_valid=trajectory_valid,trajectory_eligible=eligible.cpu().numpy(),observation_used=vis.cpu().numpy(),
        observation_inlier=inliers,uv_pred=final['uv_pred'],residual_px=final['residual_px'],camera_depth=final['camera_depth'],
        support_interval=support.cpu().numpy(),motion_span_px=motion_score.cpu().numpy(),beta_track_mask=dynamic.cpu().numpy(),
        beta_reliable=reliable,beta_status=reasons,beta_schur_information=schur,beta_conditional_information=hbb,
        beta_normalized_schur_eigenvalues=eigenvalues,beta_information_condition=beta_condition,
        dynamic_tracks_per_camera=dynamic_counts.cpu().numpy(),converged=converged,iterations=len(logs),log=logs,
        objective_initial=float(initial['objective']),objective_final=float(final['objective']),
        image_cost_initial=float(initial['image_cost']),image_cost_final=float(final['image_cost']),
        motion_cost_initial=float(initial['motion_cost']),motion_cost_final=float(final['motion_cost']),
        reprojection_rmse_initial=float(initial['reprojection_rmse']),reprojection_rmse_final=float(final['reprojection_rmse']),
        timing=dict(seconds=seconds,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else 0,
                    device=str(device),dtype=str(dtype),point_count=P,eligible_point_count=int(eligible.sum()),raw_training_observations=int(vis.sum()),
                    controls_per_trajectory=K,global_beta_variables=int(active_beta.sum()) if mode=='joint' else 0),
        config=asdict(cfg),mode=mode,calibration_fixed=True,
        objective_definition='SUM of weighted radial Huber normalized pixel residuals plus lambda * continuous squared-acceleration integral',
        observation_policy='Each valid raw training (point,camera,frame) enters once; interpolation is initialization only',
        beta_gradient_policy='All X use times+global beta; beta normal terms use only explicit motion-qualified tracks',
        observability_interpretation='Local motion/Schur/box/convergence diagnostics; no calibrated uncertainty or uniqueness guarantee')
