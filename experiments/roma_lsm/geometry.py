"""Fixed-calibration geometry for IDEAL PINHOLE working-image pixel coordinates.

Convention: X_camera = R @ X_world + t; C_world = -R.T @ t.
F satisfies p_target.T @ F @ p_reference = 0. Camera dictionary keys: K, R, t,
size=(W,H). Any source distortion D must already have been removed once per camera.
No F/E or camera parameter is fitted from the image matches in this module.
"""
import numpy as np
try:
    from .coordinates import as_float, as_like, is_tensor, torch
except ImportError:
    from coordinates import as_float, as_like, is_tensor, torch


def _like(cam, other=None):
    for item in [cam, other or {}]:
        for key in ['K', 'R', 't']:
            if is_tensor(item.get(key)):
                return as_float(item[key])
    return np.zeros((), dtype=np.float64)


def _camera(cam, like):
    K, R, t = (as_like(cam[key], like) for key in ['K', 'R', 't'])
    if K.shape != (3,3) or R.shape != (3,3) or t.shape not in [(3,), (3,1)]:
        raise ValueError('Camera requires K,R shape (3,3), t shape (3,) or (3,1)')
    # Explicitly marked raw pixels are prohibited, even if their K looks pinhole.
    if cam.get('coordinate_system') in ['raw_distorted', 'distorted']:
        raise ValueError('Geometry requires ideal pinhole/undistorted image coordinates')
    return K, R, t.reshape(3)


def _norm(x, axis=-1, keepdims=False):
    return torch.linalg.vector_norm(x, dim=axis, keepdim=keepdims) if is_tensor(x) else np.linalg.norm(x, axis=axis, keepdims=keepdims)


def _where(condition, a, b):
    return torch.where(condition, a, b) if is_tensor(a) else np.where(condition, a, b)


def _homogeneous(uv):
    ones = torch.ones_like(uv[..., :1]) if is_tensor(uv) else np.ones_like(uv[..., :1])
    return torch.cat((uv, ones), dim=-1) if is_tensor(uv) else np.concatenate((uv, ones), axis=-1)


def _nan_divide(numerator, denominator, eps=1e-12):
    safe = _where(abs(denominator)>eps, denominator, denominator*0+float('nan'))
    return numerator / safe


def camera_center(cam):
    like = _like(cam)
    _, R, t = _camera(cam, like)
    return -(R.T @ t)


def fundamental(cam_ref, cam_target):
    """Known-calibration F, Frobenius normalized; target^T F reference = 0."""
    like = _like(cam_ref, cam_target)
    Kr, Rr, tr = _camera(cam_ref, like)
    Kt, Rt, tt = _camera(cam_target, like)
    relative_R = Rt @ Rr.T
    relative_t = tt - relative_R @ tr
    if float(_norm(relative_t)) <= 1e-12:
        raise ValueError('Coincident camera centers do not define an epipolar baseline')
    x, y, z = relative_t
    if is_tensor(relative_t):
        zero = x*0
        cross = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero)).reshape(3,3)
        invr, invt = torch.linalg.inv(Kr), torch.linalg.inv(Kt)
    else:
        cross = np.array([[0,-z,y],[z,0,-x],[-y,x,0]],dtype=relative_t.dtype)
        invr, invt = np.linalg.inv(Kr), np.linalg.inv(Kt)
    F = invt.T @ cross @ relative_R @ invr
    return F / _norm(F, axis=(-2,-1))


def project(cam, xyz, return_depth=False):
    """Project (...,3) world points. Optional depth is target camera Z, not range.

    Behind-camera points are returned with their signed depth: the caller must
    reject depth<=0. Zero depth returns NaN pixels, never fake padding coordinates.
    """
    xyz = as_float(xyz)
    if xyz.shape[-1] != 3:
        raise ValueError('xyz must end in 3 coordinates')
    K, R, t = _camera(cam, xyz)
    xc = xyz @ R.T + t
    homogeneous = xc @ K.T
    uv = _nan_divide(homogeneous[..., :2], homogeneous[..., 2:3])
    return (uv, xc[...,2]) if return_depth else uv


def world_rays(cam, uv):
    """Unit-length WORLD rays through ideal pinhole pixels (...,2)."""
    uv = as_float(uv)
    if uv.shape[-1] != 2:
        raise ValueError('uv must end in (x,y)')
    K, R, _ = _camera(cam, uv)
    inv = torch.linalg.inv(K) if is_tensor(K) else np.linalg.inv(K)
    rays = (_homogeneous(uv) @ inv.T) @ R
    return rays / _norm(rays, keepdims=True)


def epipolar_residual(F, p_ref, p_target, return_jacobian=False):
    """Signed normalized target point-to-line distance, in target working pixels.

    Jacobian, when requested, is d(distance)/d(target x,y), shape (...,2).
    This is not an algebraic error or an accuracy claim; F must be calibrated.
    """
    like = p_target if is_tensor(p_target) else p_ref
    like = as_float(like)
    p_ref, p_target, F = (as_like(v, like) for v in [p_ref, p_target, F])
    line = _homogeneous(p_ref) @ F.T
    norm = _norm(line[..., :2], keepdims=True)
    jacobian = _nan_divide(line[..., :2], norm)
    distance = _nan_divide((line * _homogeneous(p_target)).sum(-1), norm[...,0])
    return (distance, jacobian) if return_jacobian else distance


def project_ray_depth(cam_target, C_ref, rays, depth, return_jacobian=False):
    """Project C_ref + depth * unit_world_ray; derivative is wrt ray RANGE depth.

    depth has the leading shape of rays, or is scalar. Return (uv, d_uv/d_depth)
    if return_jacobian=True, otherwise uv. No independent target-center offset.
    """
    rays = as_float(rays)
    C_ref, depth = as_like(C_ref, rays), as_like(depth, rays)
    K, R, t = _camera(cam_target, rays)
    xyz = C_ref + depth[...,None] * rays
    xc = xyz @ R.T + t
    h = xc @ K.T
    uv = _nan_divide(h[..., :2], h[..., 2:3])
    if not return_jacobian:
        return uv
    hprime = (rays @ R.T) @ K.T
    derivative = _nan_divide(hprime[..., :2]*h[..., 2:3]-h[..., :2]*hprime[..., 2:3],h[..., 2:3]**2)
    return uv, derivative


def triangulate_pair(cam_ref, cam_target, uv_ref, uv_target, min_angle_deg=.1):
    """Batched DLT diagnostic, returning XYZ and positive-depth/angle validity.

    ray_distance is the signed distance along the UNIT reference ray (not Z).
    angle_deg is the angle between world rays in [0,180]. Both nearly parallel
    and nearly antiparallel rays are rejected. No ground truth enters validity.
    NaN input is sanitized only for SVD then explicitly marked invalid and NaN.
    """
    like = as_float(uv_target if is_tensor(uv_target) else uv_ref)
    a, b = as_like(uv_ref,like), as_like(uv_target,like)
    if a.shape != b.shape or a.shape[-1] != 2:
        raise ValueError('Two correspondence arrays must have the same (...,2) shape')
    finite = torch.isfinite(a).all(-1)&torch.isfinite(b).all(-1) if is_tensor(a) else np.isfinite(a).all(-1)&np.isfinite(b).all(-1)
    a = _where(finite[...,None],a,a*0)
    b = _where(finite[...,None],b,b*0)
    # nan*0 is still nan: use a truly finite neutral point for the SVD only.
    if is_tensor(a):
        a,b = torch.nan_to_num(a),torch.nan_to_num(b)
    else:
        a,b = np.nan_to_num(a),np.nan_to_num(b)
    Kr,Rr,tr = _camera(cam_ref,a); Kt,Rt,tt = _camera(cam_target,a)
    if is_tensor(a):
        invr,invt = torch.linalg.inv(Kr),torch.linalg.inv(Kt)
        Pr = torch.cat((Rr,tr[:,None]),dim=-1);Pt = torch.cat((Rt,tt[:,None]),dim=-1)
    else:
        invr,invt = np.linalg.inv(Kr),np.linalg.inv(Kt)
        Pr = np.concatenate((Rr,tr[:,None]),axis=-1);Pt = np.concatenate((Rt,tt[:,None]),axis=-1)
    an = _homogeneous(a)@invr.T;bn = _homogeneous(b)@invt.T
    equations=[an[...,0:1]*Pr[2]-Pr[0],an[...,1:2]*Pr[2]-Pr[1],
               bn[...,0:1]*Pt[2]-Pt[0],bn[...,1:2]*Pt[2]-Pt[1]]
    matrix = torch.stack(equations,dim=-2) if is_tensor(a) else np.stack(equations,axis=-2)
    _,_,vh = torch.linalg.svd(matrix,full_matrices=False) if is_tensor(a) else np.linalg.svd(matrix,full_matrices=False)
    xyz = _nan_divide(vh[...,-1,:3],vh[...,-1,3:4])
    _,zr = project(cam_ref,xyz,True);_,zt = project(cam_target,xyz,True)
    rays_r,rays_t = world_rays(cam_ref,a),world_rays(cam_target,b)
    dot = (rays_r*rays_t).sum(-1)
    dot = torch.clamp(dot,-1,1) if is_tensor(dot) else np.clip(dot,-1,1)
    angle = torch.rad2deg(torch.acos(dot)) if is_tensor(dot) else np.rad2deg(np.arccos(dot))
    center_r,center_t = -(Rr.T@tr),-(Rt.T@tt)
    ray_distance = ((xyz-center_r)*rays_r).sum(-1)
    cross = torch.linalg.cross(rays_r,rays_t,dim=-1) if is_tensor(rays_r) else np.cross(rays_r,rays_t)
    cross_norm = _norm(cross)
    gap = _nan_divide(abs((cross*(center_t-center_r)).sum(-1)),cross_norm)
    xyz_finite = torch.isfinite(xyz).all(-1) if is_tensor(xyz) else np.isfinite(xyz).all(-1)
    valid = finite & xyz_finite & (zr>1e-8) & (zt>1e-8) & (ray_distance>1e-8)
    valid &= (angle>=min_angle_deg) & (angle<=180-min_angle_deg)
    xyz = _where(valid[...,None],xyz,xyz*0+float('nan'))
    return dict(xyz=xyz,depth_ref=zr,depth_target=zt,ray_distance=ray_distance,
                angle_deg=angle,ray_gap=gap,valid=valid)
