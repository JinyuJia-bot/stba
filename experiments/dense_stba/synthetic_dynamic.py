"""Independently ray-rendered, calibrated asynchronous dynamic RGB videos.

Public manifest/camera records contain nominal frame times only. Exact offsets,
material identities, continuous motion and visibility live in separate GT files.
No source-image resampling is used: every view evaluates an analytic material
texture at its own ray/surface intersection, with 2x2 pixel-area quadrature.
"""
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np


CASES = {
    'fractional': {'beta_frames':[0.,.3,-.5], 'motion_scale':1., 'occlusion':False, 'noise':[0.,0.,0.]},
    'wide': {'beta_frames':[0.,1.2,-.5], 'motion_scale':1., 'occlusion':False, 'noise':[0.,0.,0.]},
    'zero': {'beta_frames':[0.,0.,0.], 'motion_scale':1., 'occlusion':False, 'noise':[0.,0.,0.]},
    'noisy_occlusion': {'beta_frames':[0.,.3,-.5], 'motion_scale':1., 'occlusion':True, 'noise':[.003,.008,.012]},
    'near_static': {'beta_frames':[0.,.3,-.5], 'motion_scale':.001, 'occlusion':False, 'noise':[.002,.002,.002]},
}
OCCLUDER_RECT = [.12,.36,-.65,.55]


def rotation_x(a):
    a=np.asarray(a);r=np.zeros(a.shape+(3,3));c,s=np.cos(a),np.sin(a)
    r[...,0,0]=1;r[...,1,1]=c;r[...,1,2]=-s;r[...,2,1]=s;r[...,2,2]=c
    return r


def rotation_y(a):
    a=np.asarray(a);r=np.zeros(a.shape+(3,3));c,s=np.cos(a),np.sin(a)
    r[...,1,1]=1;r[...,0,0]=c;r[...,0,2]=s;r[...,2,0]=-s;r[...,2,2]=c
    return r


def rotation_z(a):
    a=np.asarray(a);r=np.zeros(a.shape+(3,3));c,s=np.cos(a),np.sin(a)
    r[...,2,2]=1;r[...,0,0]=c;r[...,0,1]=-s;r[...,1,0]=s;r[...,1,1]=c
    return r


def motion_pose(times, params):
    """Continuous material-plane local-to-world rotation and center, metres."""
    t=np.asarray(times,dtype=np.float64);s=float(params['motion_scale']);a=2*np.pi*t
    center=np.stack([s*(.30*np.sin(.8*a)+.055*np.sin(1.7*a+.3)),
                     s*(.24*np.sin(.73*a+.4)+.04*np.sin(1.35*a)),
                     4.3+s*.32*np.sin(.72*a+.2)],-1)
    rotation=(rotation_z(s*.065*np.sin(.55*a-.2)) @
              rotation_y(s*.14*np.sin(.83*a+.7)) @
              rotation_x(s*.105*np.sin(.60*a+.15)))
    return rotation,center


def cameras(size=(192,144)):
    w,h=size;K=np.array([[180*w/192,0,(w-1)/2],[0,178*h/144,(h-1)/2],[0,0,1.]])
    result=[]
    for cid,C in enumerate([[0.,0.,0.],[.32,-.035,0.],[-.38,.075,.03]]):
        C=np.asarray(C);R=rotation_y(np.arctan2(C[0],4.3))
        result.append(dict(camera_id=cid,K=K.tolist(),R=R.tolist(),t=(-R@C).tolist(),
            size=[w,h],D=[0.,0.,0.,0.,0.],distortion_model='none',
            coordinate_system='ideal_pinhole_pixels',units='metres',
            extrinsics_convention='X_camera=R@X_world+t; C_world=-R.T@t'))
    return result


def pixel_grid(size):
    w,h=size;y,x=np.mgrid[:h,:w]
    return np.stack([x,y],-1).astype(np.float64)


def world_rays(camera,uv):
    uv=np.asarray(uv,dtype=np.float64);R=np.asarray(camera['R']);K=np.asarray(camera['K'])
    homogeneous=np.concatenate([uv,np.ones(uv.shape[:-1]+(1,))],-1)
    directions=homogeneous@np.linalg.inv(K).T@R
    directions/=np.linalg.norm(directions,axis=-1,keepdims=True)
    center=-R.T@np.asarray(camera['t'])
    return center,directions


def project(camera,xyz):
    xyz=np.asarray(xyz,dtype=np.float64);R=np.asarray(camera['R']);K=np.asarray(camera['K'])
    xc=xyz@R.T+np.asarray(camera['t']);homogeneous=xc@K.T
    return homogeneous[...,:2]/homogeneous[...,2,None],xc[...,2]


def intersect_scene(camera,uv,time,params):
    """Nearest positive hit on dynamic plane or optional static occluder."""
    C,ray=world_rays(camera,uv);R,center=motion_pose(float(time),params)
    normal=R[:,2];denominator=ray@normal
    distance=((center-C)@normal)/denominator
    xyz=C+distance[...,None]*ray
    material=(xyz-center)@R
    surface=np.zeros(distance.shape,np.int8)
    valid=np.isfinite(distance)&(distance>0)&(np.abs(denominator)>1e-10)
    if params['occlusion']:
        occluder_distance=(3.-C[2])/ray[...,2]
        hit=C+occluder_distance[...,None]*ray
        x0,x1,y0,y1=OCCLUDER_RECT
        closer=((occluder_distance>0)&(occluder_distance<distance)&
                (hit[...,0]>=x0)&(hit[...,0]<=x1)&(hit[...,1]>=y0)&(hit[...,1]<=y1))
        xyz=np.where(closer[...,None],hit,xyz)
        material=np.where(closer[...,None],hit,material)
        distance=np.where(closer,occluder_distance,distance)
        surface=np.where(closer,1,surface).astype(np.int8)
    return dict(xyz=xyz,material_uv=material[...,:2],surface_id=surface,
                distance=distance,valid=valid)


def texture(material_uv,seed,surface_id=None):
    """Continuous, deterministic colored multi-scale material texture."""
    uv=np.asarray(material_uv);u,v=uv[...,0],uv[...,1]
    rng=np.random.default_rng(seed);angles=rng.uniform(-np.pi,np.pi,24)
    magnitudes=rng.uniform(1.4,9.0,24);phases=rng.uniform(-np.pi,np.pi,24)
    weights=rng.uniform(.4,1.,24);weights*=.36/weights.sum()
    value=np.full(u.shape,.5,dtype=np.float64)
    for angle,magnitude,phase,weight in zip(angles,magnitudes,phases,weights):
        value+=weight*np.sin(2*np.pi*magnitude*(np.cos(angle)*u+np.sin(angle)*v)+phase)
    value+=.085*np.sin(2*np.pi*(2.3*u+1.1*v)+.2)
    value+=.065*np.cos(2*np.pi*(1.3*u-3.1*v)-.4)
    rgb=np.stack([value+.035*np.sin(2*np.pi*(.7*u+.4*v)),
                  value+.025*np.cos(2*np.pi*(.5*u-.6*v)),
                  value-.03*np.sin(2*np.pi*(.9*u-.3*v))],-1)
    if surface_id is not None:
        foreground=np.stack([.35+.14*np.sin(27*u+13*v),
            .47+.11*np.sin(19*u-31*v),.40+.1*np.cos(24*u+21*v)],-1)
        rgb=np.where((np.asarray(surface_id)==1)[...,None],foreground,rgb)
    return rgb


def render(camera,time,params,noise_sigma=0.,seed=0,samples_per_axis=2):
    grid=pixel_grid(camera['size']);image=np.zeros(grid.shape[:-1]+(3,))
    offsets=(np.arange(samples_per_axis)+.5)/samples_per_axis-.5
    for oy in offsets:
        for ox in offsets:
            hit=intersect_scene(camera,grid+[ox,oy],time,params)
            if not hit['valid'].all():raise ValueError('Rendering rays lack positive surface intersections')
            image+=texture(hit['material_uv'],params['texture_seed'],hit['surface_id'])
    image/=samples_per_axis**2
    if noise_sigma:
        image+=np.random.default_rng(seed).normal(0,noise_sigma,image.shape)
    clipping=float(np.mean((image<0)|(image>1)))
    return np.clip(image,0,1).astype(np.float32),clipping


def material_world(material_uv,surface_id,times,params):
    material_uv=np.asarray(material_uv).reshape(-1,2);surface_id=np.asarray(surface_id).reshape(-1)
    times=np.atleast_1d(np.asarray(times,dtype=np.float64));R,C=motion_pose(times,params)
    local=np.column_stack([material_uv,np.zeros(len(material_uv))])
    xyz=np.einsum('tij,nj->tni',R,local)+C[:,None,:]
    static=np.column_stack([material_uv,np.full(len(material_uv),3.)])
    return np.where((surface_id==1)[None,:,None],static[None],xyz)


@dataclass
class DynamicTruth:
    directory: Path
    manifest: dict
    params: dict
    beta_seconds: np.ndarray
    physical_timestamps: np.ndarray

    def material_from_reference(self,seed_uv,reference_frame=None):
        frame=self.manifest['seed_frame'] if reference_frame is None else reference_frame
        cam=self.manifest['cameras'][0];time=self.physical_timestamps[0,frame]
        uv=np.asarray(seed_uv,dtype=float).reshape(-1,2)
        hit=intersect_scene(cam,uv,time,self.params)
        w,h=cam['size'];hit['valid']&=(uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1)
        return hit

    def evaluate_material_trajectory(self,seed_uv,times,reference_frame=None):
        material=self.material_from_reference(seed_uv,reference_frame)
        return material_world(material['material_uv'],material['surface_id'],times,self.params)

    def project_material(self,seed_uv,camera_id,times,reference_frame=None):
        times=np.atleast_1d(np.asarray(times,dtype=float))
        seed=self.material_from_reference(seed_uv,reference_frame)
        xyz=material_world(seed['material_uv'],seed['surface_id'],times,self.params)
        cam=next(cam for cam in self.manifest['cameras'] if cam['camera_id']==camera_id)
        uv,z=project(cam,xyz);w,h=cam['size']
        inside=(np.isfinite(uv).all(-1)&(z>0)&(uv[...,0]>=0)&(uv[...,0]<=w-1)&(uv[...,1]>=0)&(uv[...,1]<=h-1))
        visible=np.zeros(inside.shape,bool)
        for i,time in enumerate(times):
            hit=intersect_scene(cam,uv[i],time,self.params)
            visible[i]=hit['valid']&(np.linalg.norm(hit['xyz']-xyz[i],axis=-1)<1e-7)
        valid=inside&visible&seed['valid'][None]
        return dict(uv=uv,xyz=xyz,valid=valid,occluded=inside&~visible&seed['valid'][None],
                    inside=inside,depth_camera_z=z,surface_id=seed['surface_id'])


def load_truth(case_dir):
    directory=Path(case_dir)
    manifest=json.loads((directory/'manifest.json').read_text())
    params=json.loads((directory/'gt_metadata.json').read_text())['motion_and_scene']
    with np.load(directory/'gt.npz') as data:
        beta=data['beta_seconds'];times=data['physical_timestamps']
    return DynamicTruth(directory,manifest,params,beta,times)


def evaluate_material_trajectory(case_dir,seed_uv,times,reference_frame=None):
    return load_truth(case_dir).evaluate_material_trajectory(seed_uv,times,reference_frame)


def generate_case(name,out_dir,frames=32,fps=30.,size=(192,144),seed=20261111):
    if name not in CASES:raise ValueError(f'Unknown case {name}')
    if frames<8 or fps<=0:raise ValueError('Need at least 8 frames and positive fps')
    directory=Path(out_dir).resolve();directory.mkdir(parents=True,exist_ok=True)
    settings=CASES[name];cams=cameras(size);nominal=np.arange(frames,dtype=np.float64)/fps
    beta=np.asarray(settings['beta_frames'])/fps;physical=nominal[None]+beta[:,None]
    params=dict(motion_scale=settings['motion_scale'],occlusion=settings['occlusion'],
                texture_seed=seed,occluder_z=3.,occluder_rectangle=OCCLUDER_RECT)
    clip=[]
    for cid,cam in enumerate(cams):
        target=directory/f'cam{cid:02d}';target.mkdir(exist_ok=True)
        sequence=[];paths=[];fractions=[]
        for frame,time in enumerate(physical[cid]):
            image,clipped=render(cam,time,params,settings['noise'][cid],seed+10000*cid+frame)
            rgb=np.round(image*255).astype(np.uint8);sequence.append(rgb);fractions.append(clipped)
            path=target/f'{frame:04d}.png';cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));paths.append(str(path))
        frame_path=target/'frames.npy';np.save(frame_path,np.stack(sequence))
        cam.update(fps=fps,frames_path=str(frame_path),image_paths=paths,
                   nominal_timestamps=nominal.tolist(),n_frames=frames,color_order='RGB',
                   frames_sha256=hashlib.sha256(frame_path.read_bytes()).hexdigest())
        clip.append(float(np.mean(fractions)))
    manifest=dict(schema_version=1,dataset='analytic_dynamic_plane',case=name,
        reference_camera_id=0,seed_frame=frames//2,fps=fps,n_frames=frames,
        image_size=list(size),cameras=cams,units='metres',
        time_convention='physical_time=nominal_timestamp+unknown_camera_beta_seconds; reference beta fixed to zero',
        color_order='RGB',coordinate_system='ideal pinhole x/y integer pixel centers',
        temporal_mode='asynchronous_unknown',
        renderer='Independent per-camera ray intersections; continuous analytic material RGB texture; 2x2 pixel-area quadrature; no raster warp',
        algorithm_input_policy='Read this manifest, cameras and RGB videos only. Files beginning gt are evaluation/oracle data and prohibited for automatic estimation.')
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (directory/'cameras.json').write_text(json.dumps(cams,indent=2)+'\n')
    seed_grid=pixel_grid(size)
    material=intersect_scene(cams[0],seed_grid,physical[0,frames//2],params)
    np.savez_compressed(directory/'gt.npz',beta_seconds=beta,beta_frames=np.asarray(settings['beta_frames']),
        physical_timestamps=physical,seed_frame=np.array(frames//2),seed_reference_uv=seed_grid.astype(np.float32),
        seed_material_uv=material['material_uv'],seed_surface_id=material['surface_id'],
        seed_valid=material['valid'])
    gt_meta=dict(case=name,seed=seed,motion_and_scene=params,beta_frames=settings['beta_frames'],
        noise_sigma=settings['noise'],clipped_fraction=clip,
        object='Infinite material plane undergoing nonconstant 3D translation, depth variation and rotation about three axes; optional static occluder.',
        motion_model='motion_pose in synthetic_dynamic.py',
        identity='Seed pixel at reference seed_frame maps to material_uv plus surface_id; identity is fixed as the material moves.',
        visibility='Positive target depth, image bounds and nearest-surface ray-intersection agreement.',
        limitations=['Rigid planar dynamic geometry is simpler than real deformable objects.',
            'No rolling shutter, exposure blur, lens distortion or specular appearance.',
            'Pixel-center GT and pixel-area rendering differ at surface boundaries.',
            'Near-static case is intentionally not expected to support reliable beta estimation.'])
    (directory/'gt_metadata.json').write_text(json.dumps(gt_meta,indent=2)+'\n')
    print(f'Generated {name}: {len(cams)} cameras x {frames} RGB frames at {directory}',flush=True)
    return manifest


def generate_suite(out_dir,frames=32,fps=30.,size=(192,144),seed=20261111,cases=None):
    directory=Path(out_dir);directory.mkdir(parents=True,exist_ok=True)
    names=list(CASES) if cases is None else cases
    result=[generate_case(name,directory/name,frames,fps,size,seed) for name in names]
    available=[name for name in CASES if (directory/name/'manifest.json').exists()]
    suite=dict(schema_version=1,case_manifests=[str((directory/name/'manifest.json').resolve()) for name in available])
    (directory/'suite.json').write_text(json.dumps(suite,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'data/synthetic')
    parser.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
    parser.add_argument('--frames',type=int,default=32);parser.add_argument('--fps',type=float,default=30.)
    parser.add_argument('--size',nargs=2,type=int,default=[192,144]);parser.add_argument('--seed',type=int,default=20261111)
    args=parser.parse_args()
    generate_suite(args.out,args.frames,args.fps,args.size,args.seed,args.cases)
