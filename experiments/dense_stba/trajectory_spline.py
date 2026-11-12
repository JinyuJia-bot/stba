"""Clamped cubic B-splines in physical seconds, with differentiable torch bases.

Control points are world positions. No trajectory identity, timing truth, or
camera parameter is inferred here. The basis is zero outside the knot domain;
trajectory queries outside that domain explicitly return NaN (no extrapolation).
"""
import numpy as np
import torch


def make_knots(t_min,t_max,knot_interval,degree=3):
    if degree!=3:raise ValueError('This prototype uses cubic B-splines only')
    if not np.isfinite([t_min,t_max,knot_interval]).all() or t_max<=t_min or knot_interval<=0:
        raise ValueError('Require finite t_min < t_max and positive knot interval')
    interior=np.arange(t_min+knot_interval,t_max,knot_interval,dtype=np.float64)
    interior=interior[interior<t_max-1e-12*max(1,abs(t_max))]
    return np.r_[np.repeat(float(t_min),degree+1),interior,np.repeat(float(t_max),degree+1)]


def _like(*values):
    for value in values:
        if isinstance(value,torch.Tensor):
            return value.device,value.dtype if value.is_floating_point() else torch.float64
    return torch.device('cpu'),torch.float64


def _basis_tensor(times,knots,degree=3,derivative=0):
    if derivative not in [0,1,2]:raise ValueError('Only position/velocity/acceleration bases are supported')
    if knots.ndim!=1 or len(knots)<2*(degree+1):raise ValueError('Invalid clamped knot vector')
    if bool((knots[1:]<knots[:-1]).any()):raise ValueError('Knots must be nondecreasing')
    if not bool((knots[:degree+1]==knots[0]).all()&(knots[-degree-1:]==knots[-1]).all()):
        raise ValueError('Endpoint knots must have degree+1 multiplicity')
    if not bool(knots[-1]>knots[0]):raise ValueError('Knot domain must have positive duration')
    x=times[...,None]
    # At the right endpoint select the last nonempty degree-zero span; recursion
    # then produces B_last(t_max)=1 and the correct one-sided derivatives.
    ordinary=(x>=knots[:-1])&(x<knots[1:])
    right=(x==knots[-1])&(knots[:-1]<knots[-1])&(knots[1:]==knots[-1])
    values=[(ordinary|right).to(times.dtype)]
    for p in range(1,degree+1):
        count=len(knots)-p-1
        dl=knots[p:p+count]-knots[:count]
        dr=knots[p+1:p+1+count]-knots[1:1+count]
        left=(x-knots[:count])/torch.where(dl>0,dl,torch.ones_like(dl))*(dl>0)
        right=(knots[p+1:p+1+count]-x)/torch.where(dr>0,dr,torch.ones_like(dr))*(dr>0)
        values.append(left*values[-1][...,:count]+right*values[-1][...,1:count+1])
    if derivative==0:return values[degree]
    first=[torch.zeros_like(values[0])]
    for p in range(1,degree+1):
        count=len(knots)-p-1
        dl=knots[p:p+count]-knots[:count];dr=knots[p+1:p+1+count]-knots[1:1+count]
        a=p*(dl>0)/torch.where(dl>0,dl,torch.ones_like(dl))
        b=p*(dr>0)/torch.where(dr>0,dr,torch.ones_like(dr))
        first.append(a*values[p-1][...,:count]-b*values[p-1][...,1:count+1])
    if derivative==1:return first[degree]
    count=len(knots)-degree-1
    dl=knots[degree:degree+count]-knots[:count]
    dr=knots[degree+1:degree+1+count]-knots[1:1+count]
    a=degree*(dl>0)/torch.where(dl>0,dl,torch.ones_like(dl))
    b=degree*(dr>0)/torch.where(dr>0,dr,torch.ones_like(dr))
    return a*first[degree-1][...,:count]-b*first[degree-1][...,1:count+1]


def basis_matrix(times,knots,derivative=0):
    """Return [...,K] basis at seconds. NumPy in -> NumPy; torch stays torch."""
    device,dtype=_like(times,knots)
    t=torch.as_tensor(times,device=device,dtype=dtype)
    k=torch.as_tensor(knots,device=device,dtype=dtype)
    result=_basis_tensor(t,k,3,derivative)
    return result if isinstance(times,torch.Tensor) or isinstance(knots,torch.Tensor) else result.numpy()


def evaluate_spline(control,times,knots,derivative=0):
    """control[P,K,3], times[...] -> world position/velocity/acceleration[P,...,3]."""
    device,dtype=_like(control,times,knots)
    c=torch.as_tensor(control,device=device,dtype=dtype)
    t=torch.as_tensor(times,device=device,dtype=dtype);k=torch.as_tensor(knots,device=device,dtype=dtype)
    if c.ndim!=3 or c.shape[-1]!=3 or c.shape[1]!=len(k)-4:raise ValueError('control must have shape [P,K,3] matching knots')
    B=_basis_tensor(t,k,3,derivative)
    xyz=torch.einsum('...k,pkd->p...d',B,c)
    valid=torch.isfinite(t)&(t>=k[3])&(t<=k[-4])
    xyz=torch.where(valid[None,...,None],xyz,torch.full_like(xyz,float('nan')))
    return xyz if any(isinstance(v,torch.Tensor) for v in [control,times,knots]) else xyz.numpy()


def acceleration_gram(knots,integration_interval=None):
    """Q[k,l]=integral B''_k(t) B''_l(t) dt (units seconds^-3).

    Two-point Gaussian quadrature is exact: cubic-spline acceleration is linear
    within each knot span, and its squared norm is quadratic. A fixed integration
    interval may be supplied; the default is the complete knot domain.
    """
    device,dtype=_like(knots);k=torch.as_tensor(knots,device=device,dtype=dtype)
    lo,hi=(k[3],k[-4]) if integration_interval is None else integration_interval
    lo=torch.as_tensor(lo,device=device,dtype=dtype);hi=torch.as_tensor(hi,device=device,dtype=dtype)
    if not bool((lo>=k[3])&(hi<=k[-4])&(hi>lo)):raise ValueError('Integration interval must lie inside the knot domain')
    left=torch.maximum(k[:-1],lo);right=torch.minimum(k[1:],hi)
    use=right>left;left,right=left[use],right[use]
    mid=(left+right)/2;half=(right-left)/2
    nodes=torch.stack([mid-half/np.sqrt(3),mid+half/np.sqrt(3)],-1).reshape(-1)
    weights=half[:,None].expand(-1,2).reshape(-1)
    D=_basis_tensor(nodes,k,3,2)
    Q=torch.einsum('nk,n,nl->kl',D,weights,D)
    return Q if isinstance(knots,torch.Tensor) else Q.numpy()


def fit_control_points(sample_times,xyz,valid,knots,weights=None,regularization=1e-6,min_samples=4):
    """Fit initialization from triangulated samples; no samples become observations.

    sample_times[S] (shared) or [P,S], xyz[P,S,3], valid[P,S]. Returns control,
    fit_valid, sample_count, fit_rmse, and support_interval[P,2]. A tiny numerical
    diagonal ridge and configurable acceleration penalty resolve unsupported
    control points. They do not establish measurement support outside samples.
    """
    if regularization<0:raise ValueError('Regularization must be nonnegative')
    device,dtype=_like(xyz,sample_times,knots)
    x=torch.as_tensor(xyz,device=device,dtype=dtype)
    times=torch.as_tensor(sample_times,device=device,dtype=dtype)
    k=torch.as_tensor(knots,device=device,dtype=dtype)
    good=torch.as_tensor(valid,device=device,dtype=torch.bool).clone()
    if x.ndim!=3 or x.shape[-1]!=3 or good.shape!=x.shape[:2]:raise ValueError('Expected xyz[P,S,3], valid[P,S]')
    if times.ndim==1:times=times[None].expand(x.shape[:2])
    if times.shape!=good.shape:raise ValueError('Times must have shape S or P,S')
    w=torch.ones_like(times) if weights is None else torch.as_tensor(weights,device=device,dtype=dtype)
    good &= torch.isfinite(x).all(-1)&torch.isfinite(times)&torch.isfinite(w)&(w>0)&(times>=k[3])&(times<=k[-4])
    w=torch.where(good,w,torch.zeros_like(w));safe_x=torch.nan_to_num(x)
    B=_basis_tensor(torch.nan_to_num(times),k)
    H=torch.einsum('psk,ps,psl->pkl',B,w,B)
    rhs=torch.einsum('psk,ps,psd->pkd',B,w,safe_x)
    Q=acceleration_gram(k);H=H+regularization*Q[None]
    ridge=1e-10*torch.diagonal(H,dim1=-2,dim2=-1).abs().mean(-1).clamp_min(1.)
    H=H+ridge[:,None,None]*torch.eye(H.shape[-1],device=device,dtype=dtype)
    c=torch.linalg.solve(H,rhs)
    count=good.sum(-1);fit_valid=(count>=min_samples)&torch.isfinite(c).all((-1,-2))
    residual=torch.einsum('psk,pkd->psd',B,c)-safe_x
    error=(w*residual.square().sum(-1)).sum(-1)/w.sum(-1).clamp_min(1e-12)
    lower=torch.where(good,times,torch.full_like(times,float('inf'))).min(-1).values
    upper=torch.where(good,times,torch.full_like(times,-float('inf'))).max(-1).values
    interval=torch.stack([lower,upper],-1)
    interval=torch.where((count>0)[:,None],interval,torch.full_like(interval,float('nan')))
    c=torch.where(fit_valid[:,None,None],c,torch.full_like(c,float('nan')))
    result=dict(control=c,fit_valid=fit_valid,sample_count=count,fit_rmse=error.sqrt(),support_interval=interval)
    return result if any(isinstance(v,torch.Tensor) for v in [xyz,sample_times,knots]) else {key:value.numpy() for key,value in result.items()}
