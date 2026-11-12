"""End-to-end dense RoMa/temporal-track continuous-trajectory STBA prototype.

All actual optimizer inputs are image measurements and fixed calibration.
Synthetic truth is accessed ONLY in the explicitly named B2 oracle-beta arm;
other methods never receive beta, XYZ or identities from GT. Real sync is used
by the loader solely for the requested integer-shift input variant.
"""
from pathlib import Path
import sys
import json
import argparse
import time
import hashlib
import shutil
from dataclasses import asdict
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from experiments.roma_lsm.geometry import project
try:
    from .data_loader import load_case
    from .cross_view_association import prepare_roma,build_observations,AssociationConfig
    from .time_init import TimeInitConfig,estimate_beta,training_split,triangulate_samples,interpolate_tracks,motion_diagnostics
    from .trajectory_spline import make_knots,fit_control_points,evaluate_spline
    from .stba_optimizer import optimize_stba,STBAConfig
except ImportError:
    from data_loader import load_case
    from cross_view_association import prepare_roma,build_observations,AssociationConfig
    from time_init import TimeInitConfig,estimate_beta,training_split,triangulate_samples,interpolate_tracks,motion_diagnostics
    from trajectory_spline import make_knots,fit_control_points,evaluate_spline
    from stba_optimizer import optimize_stba,STBAConfig


def json_safe(value):
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,np.ndarray):return json_safe(value.tolist())
    if isinstance(value,np.generic):return json_safe(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def write_json(path,value):
    Path(path).write_text(json.dumps(json_safe(value),indent=2,allow_nan=False)+'\n')


def freeze_observations(source,directory):
    """A result must not silently change when its reusable frontend cache changes."""
    source=Path(source);destination=Path(directory)/source.name
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:
        shutil.copyfile(source,destination)
    return str(destination.resolve()),digest


def select_jobs(selection):
    jobs=[]
    if selection in ['all','synthetic','fractional','wide','zero','noisy_occlusion','near_static']:
        for name in ['fractional','wide','zero','noisy_occlusion','near_static']:
            if selection not in ['all','synthetic',name]:continue
            jobs.append((f'synthetic_{name}',ROOT/f'data/synthetic/{name}/manifest.json'))
    if selection in ['all','real','raw','integer_aligned']:
        for name in ['raw','integer_aligned']:
            if selection in ['all','real',name]:jobs.append((f'corgi_{name}',ROOT/f'data/real/{name}/manifest.json'))
    return jobs


def support_map(obs,train,beta,samples,max_gap_frames=2.1):
    fps=1/np.median(np.diff(obs['timestamps'][0]));views=[]
    for j in range(train.shape[1]):
        _,valid=interpolate_tracks(obs['uv'][:,j],train[:,j],obs['timestamps'][j],samples-beta[j],max_gap_frames/fps)
        views.append(valid)
    return np.stack(views,-1).sum(-1).astype(np.int16)


def make_initialization(obs,train,cameras,beta,knots,config):
    tri=triangulate_samples(obs['uv'],train,obs['weight'],obs['timestamps'],cameras,beta,obs['timestamps'][0])
    samples_ok=tri['valid']&(tri['reprojection_error_px']<=config['initialization_max_reprojection_px'])
    fit=fit_control_points(tri['sample_times'],tri['xyz'],samples_ok,knots,regularization=1e-7,
                           min_samples=config['minimum_initialization_samples'])
    selected=fit['fit_valid']&obs['multiview_seed']
    return tri,fit,selected


def run_case(name,manifest_path,config,device='cuda',methods=None,frontend=True,results_root=None,window=None,cache_root=None):
    manifest=load_case(manifest_path);cameras=manifest['cameras'];synthetic=manifest['dataset']=='analytic_dynamic_plane'
    cache=(Path(cache_root) if cache_root else ROOT/'cache')/name;output=(Path(results_root) if results_root else ROOT/'results')/name
    output.mkdir(parents=True,exist_ok=True)
    methods=list(methods or config['default_methods'])
    if not synthetic:methods=[m for m in methods if m!='B2']
    if frontend:
        prepare_roma(manifest_path,cache,device)
        build_observations(manifest_path,cache,AssociationConfig(**config['association']),device)
    if not (cache/'observations.npz').exists():raise FileNotFoundError(cache/'observations.npz')
    with np.load(cache/'observations.npz') as data:obs={k:data[k] for k in data.files}
    frozen_path,observations_sha=freeze_observations(cache/'observations.npz',output)
    train,holdout=training_split(obs['visible'],config['holdout_every'],config['holdout_phase'])
    if window is not None:
        selected_frames=np.zeros(train.shape[-1],bool);selected_frames[slice(*window)]=True
        train &= selected_frames[None,None];holdout &= selected_frames[None,None]
    fps=1/np.median(np.diff(obs['timestamps'][0]));n,c,t,_=obs['uv'].shape
    bound_frames=(config['synthetic_beta_bound_frames'] if synthetic else
                  config['integer_beta_bound_frames'] if manifest['mode']=='integer_aligned' else config['real_raw_beta_bound_frames'])
    init_start=time.perf_counter()
    coarse=estimate_beta(obs['uv'],train,obs['weight'],obs['timestamps'],cameras,
                         TimeInitConfig(search_bound_frames=bound_frames,**config['time_init']))
    initialization_seconds=time.perf_counter()-init_start
    write_json(output/'time_initialization.json',{**coarse,'seconds':initialization_seconds})
    beta0=coarse['beta'];bound=bound_frames/fps
    knots=make_knots(float(obs['timestamps'].min()-bound),float(obs['timestamps'].max()+bound),config['knot_interval_frames']/fps)
    bounds=np.tile([-bound,bound],(c,1));bounds[0]=0
    samples=obs['timestamps'][0]
    already=json.loads((output/'meta.json').read_text()).get('methods',[]) if (output/'meta.json').exists() else []
    saved_methods=list(dict.fromkeys(already+methods))
    meta=dict(name=name,manifest_path=str(Path(manifest_path).resolve()),observations_path=frozen_path,observations_sha256=observations_sha,
        manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        config=config,camera_ids=[cam['camera_id'] for cam in cameras],fps=fps,methods=saved_methods,
        window=window,seed_frame=int(obs['seed_frame']),seed_grid_size=n,
        calibration_fixed=True,reference_camera_index=0,beta_convention='physical_time=nominal_timestamp+beta_seconds',
        holdout_policy='Every fifth original observation (phase2) withheld from time-profile scoring, 3D initialization, spline and beta optimization. Frontend uses images for tracking/association; this tests prediction of held-out 2D measurements, not independent 3D accuracy.',
        geometry_support_policy='Exported valid 3D samples require >=2 views with observed endpoints within2.1frames, positive camera depth, valid trajectory, and physical support; model-only continuation separately marked. Interpolated2D samples are initialization/support checks, never extra loss observations.',
        software={'torch':torch.__version__,'numpy':np.__version__})
    write_json(output/'meta.json',meta)
    print(json.dumps({'case':name,'coarse_beta_frames':(beta0*fps).tolist(),'reliable':coarse['reliable'].tolist(),'motion_tracks':int(coarse['moving_tracks'].sum())}),flush=True)
    cached_init=None
    for method in methods:
        began=time.perf_counter();method_obs=obs;method_train=train;method_holdout=holdout;method_coarse=coarse
        if method=='B4':
            build_observations(manifest_path,cache,AssociationConfig(**config['association'],use_lsm=True),device)
            with np.load(cache/'observations_lsm.npz') as data:method_obs={k:data[k] for k in data.files}
            method_frozen_path,method_observations_sha=freeze_observations(cache/'observations_lsm.npz',output)
            method_train,method_holdout=training_split(method_obs['visible'],config['holdout_every'],config['holdout_phase'])
            if window is not None:
                method_train &= selected_frames[None,None];method_holdout &= selected_frames[None,None]
            method_coarse=estimate_beta(method_obs['uv'],method_train,method_obs['weight'],method_obs['timestamps'],cameras,
                TimeInitConfig(search_bound_frames=bound_frames,**config['time_init']))
        initial=method_coarse['beta'].copy()
        if method!='B4':method_frozen_path,method_observations_sha=frozen_path,observations_sha
        if method=='B0':initial=np.zeros(c)
        if method=='B2':
            # Isolated oracle arm. This GT access is never performed for B0/B1/B3/B4.
            try:from .synthetic_dynamic import load_truth
            except ImportError:from synthetic_dynamic import load_truth
            initial=np.asarray(load_truth(Path(manifest_path).parent).beta_seconds).copy()
        if method in ['B1','B3'] and cached_init is not None:tri,fit,selected=cached_init
        else:
            tri,fit,selected=make_initialization(method_obs,method_train,cameras,initial,knots,config)
            if method in ['B1','B3']:cached_init=(tri,fit,selected)
        core={};beta=initial.copy()
        fields=dict(point_ids=method_obs['point_ids'],seed_uv=method_obs['seed_uv'],trajectory_times=samples,
            observed_uv=method_obs['uv'],train_visible=method_train,holdout_visible=method_holdout,
            motion_mask=method_coarse['moving_tracks'],initial_beta=initial,
            seed_eligible=selected,query_mask=method_obs['query_mask'],seed_roi=method_obs['seed_roi'],
            camera_ids=np.array([cam['camera_id'] for cam in cameras]),timestamps=method_obs['timestamps'])
        if method=='B0':
            xyz=tri['xyz'];valid=tri['valid'];traj_valid=valid.sum(-1)>=config['minimum_initialization_samples']
            valid &= traj_valid[:,None];support=tri['view_support'];prediction=np.full((n,c,t,2),np.nan)
            for j,cam in enumerate(cameras):prediction[:,j]=project(cam,xyz)
            fields.update(xyz=xyz,valid=valid,trajectory_valid=traj_valid,view_support=support,
                beta=beta,predicted_uv=prediction,model_only_mask=np.zeros_like(valid))
            timing={'seconds':time.perf_counter()-began,'device':'CPU','peak_cuda_allocated_bytes':0}
            observability={'status':['assumed_synchronous']*c,'reliable':[False]*c}
        elif selected.any():
            # B1 and B3 use byte-identical initial spline controls and observations.
            opt_cfg=STBAConfig(**config['optimizer'])
            opt_weight=method_obs['weight'][selected] if config.get('optimizer_use_confidence_weights',True) else np.ones_like(method_obs['weight'][selected])
            core=optimize_stba(method_obs['uv'][selected],method_train[selected],opt_weight,
                method_obs['timestamps'],cameras,knots,fit['control'][selected],initial,opt_cfg,
                mode='joint' if method in ['B3','B4'] else 'fixedbeta',beta_bounds_seconds=bounds,
                beta_track_mask=method_coarse['moving_tracks'][selected],device=device)
            beta=core['beta'];control=np.full((n,len(knots)-4,3),np.nan);control[selected]=core['control']
            xyz=evaluate_spline(control,samples,knots)
            inlier=np.zeros_like(method_train);inlier[selected]=core['observation_inlier']
            support=support_map(method_obs,inlier,beta,samples)
            support_before=support_map(method_obs,method_train,beta,samples)
            traj_valid=np.zeros(n,bool);traj_valid[selected]=core['trajectory_valid']
            interval=np.full((n,2),np.nan);interval[selected]=core['support_interval']
            inside=(samples[None]>=interval[:,0,None])&(samples[None]<=interval[:,1,None])
            valid=traj_valid[:,None]&inside&(support>=2)&np.isfinite(xyz).all(-1)
            # A model estimate unsupported by two views stays available but cannot
            # silently inflate measured 3D trajectory coverage.
            model_only=traj_valid[:,None]&inside&(support<2)&np.isfinite(xyz).all(-1)
            prediction=np.full((n,c,t,2),np.nan);prediction[selected]=core['uv_pred']
            for cam in cameras:
                _,depth=project(cam,xyz,True);valid &= depth>0
            fields.update(xyz=xyz,valid=valid,trajectory_valid=traj_valid,view_support=support,beta=beta,
                predicted_uv=prediction,model_only_mask=model_only,control=control,knots=knots,
                velocity=evaluate_spline(control,samples,knots,1),acceleration=evaluate_spline(control,samples,knots,2),
                support_interval=interval,beta_information=core['beta_schur_information'])
            fields['observation_inlier']=inlier
            fields['view_support_before_residual_screen']=support_before
            final_status=list(core['beta_status']);final_reliable=core['beta_reliable'].copy()
            if method in ['B3','B4']:
                final_reliable &= method_coarse['reliable']
                for j in range(1,c):
                    if not method_coarse['reliable'][j] and core['beta_reliable'][j]:
                        final_status[j]='ambiguous_or_weak_time_profile_despite_local_fit'
            timing=core['timing'];observability={'status':final_status,'reliable':final_reliable,
                'local_optimizer_status':core['beta_status'],
                'dynamic_tracks_per_camera':core['dynamic_tracks_per_camera'],'converged':core['converged'],
                'marginal_schur_information':core['beta_schur_information'],
                'interpretation':core['observability_interpretation']}
        else:
            fields.update(xyz=np.full((n,len(samples),3),np.nan),valid=np.zeros((n,len(samples)),bool),
                trajectory_valid=np.zeros(n,bool),view_support=tri['view_support'],beta=beta,
                predicted_uv=np.full((n,c,t,2),np.nan),model_only_mask=np.zeros((n,len(samples)),bool))
            timing={'seconds':time.perf_counter()-began,'device':device,'peak_cuda_allocated_bytes':0}
            observability={'status':['no_multiview_trajectory_initialization']*c,'reliable':[False]*c}
        observed_weight=np.where(method_obs['visible'],method_obs['weight'],0)
        fields['confidence']=observed_weight.sum((1,2))/method_obs['visible'].sum((1,2)).clip(1)
        saving=time.perf_counter();np.savez_compressed(output/f'{method}.npz',**fields);save_seconds=time.perf_counter()-saving
        method_meta=dict(method=method,oracle_beta=method=='B2',manifest_path=meta['manifest_path'],config=config,
            observations_path=method_frozen_path,observations_sha256=method_observations_sha,manifest_sha256=meta['manifest_sha256'],
            timing={**timing,'saving_seconds':save_seconds,'method_wall_seconds':time.perf_counter()-began},
            observability=observability,beta_frames=beta*fps,initial_beta_frames=initial*fps,
            initialized_trajectories=int(selected.sum()),valid_trajectories=int(fields['trajectory_valid'].sum()),
            valid_3D_samples=int(fields['valid'].sum()),log=core.get('log',[]),
            objective_initial=core.get('objective_initial'),objective_final=core.get('objective_final'),
            calibration_fixed=True,coarse_profiles=method_coarse['profiles'])
        write_json(output/f'{method}.json',method_meta)
        print(json.dumps({'case':name,'method':method,'beta_frames':(beta*fps).tolist(),
            'trajectories':int(fields['trajectory_valid'].sum()),'valid_samples':int(fields['valid'].sum()),
            'status':observability['status'],'seconds':timing['seconds']}),flush=True)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',default='fractional',choices=['all','synthetic','real','raw','integer_aligned','fractional','wide','zero','noisy_occlusion','near_static'])
    parser.add_argument('--config',type=Path,default=ROOT/'config.json');parser.add_argument('--device',default='cuda')
    parser.add_argument('--skip-frontend',action='store_true');parser.add_argument('--methods',nargs='+')
    parser.add_argument('--results-root',type=Path);parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--cache-root',type=Path)
    parser.add_argument('--window',nargs=2,type=int)
    args=parser.parse_args();cfg=json.loads(args.config.read_text());torch.set_num_threads(4)
    np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    for name,manifest in select_jobs(args.selection):
        out=run_case(name,manifest,cfg,args.device,args.methods,not args.skip_frontend,args.results_root,args.window,args.cache_root)
        if args.evaluate:
            try:from .evaluate import evaluate_case
            except ImportError:from evaluate import evaluate_case
            evaluate_case(out)


if __name__=='__main__':main()
