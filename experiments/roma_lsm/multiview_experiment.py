"""Four frozen-weight arms using the same dense A1 affine matches and depth seed.

Run only after the recorded stage-one gate passes. Ground truth is loaded by the
separate evaluation function after all optimization outputs have been saved.
"""
from pathlib import Path
from dataclasses import asdict, fields
import argparse
import csv
import json
import time
import numpy as np
import torch

from multiview import (MultiViewConfig, STATUS, WEIGHT_MODES, initialize_shared_depth,
                       refine_shared_depth)
from experiment import ROOT, dataset_jobs, read_image, read_mask, fb_error, synchronize
from geometry import fundamental, epipolar_residual


def build_config(config):
    """Central LSM defaults plus optional central multiview overrides."""
    keys={f.name for f in fields(MultiViewConfig)}
    selected={k:v for k,v in config.get('lsm',{}).items() if k in keys}
    mv=config.get('multiview',{})
    selected.update({k:v for k,v in mv.items() if k in keys and k!='distance_scale'})
    if 'distance_floor' in mv:selected['distance_min_weight']=mv['distance_floor']
    scale=mv.get('distance_scale',0.)
    selected['distance_scale']=0. if scale=='median_selected_baselines' else float(scale)
    if mv.get('weights_frozen',True) is not True:raise ValueError('This experiment requires frozen weights')
    if tuple(mv.get('weight_modes',WEIGHT_MODES))!=WEIGHT_MODES:
        raise ValueError('All four specified weight arms are required in their declared order')
    return MultiViewConfig(**selected)


def _load_npz(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k] for k in data.files}


def run_group(name,jobs,config,device='cuda',results_root=None,output_root=None):
    started_total=time.perf_counter()
    cfg=build_config(config)
    results_root=Path(results_root or ROOT/'results')
    out=Path(output_root or results_root/'multiview')/name;out.mkdir(parents=True,exist_ok=True)
    first,second=jobs
    if first['ref']['camera_id']!=second['ref']['camera_id'] or first['ref']!=second['ref']:
        raise ValueError('Both pairs must share exactly the same fixed reference calibration and image')
    if first['mode']!=second['mode']:raise ValueError('Pair scene modes disagree')
    mode=first['mode'];cam_ref=first['ref'];cam_targets=[job['target'] for job in jobs]
    source_files=[results_root/job['name']/'A1_affine.npz' for job in jobs]
    data=[_load_npz(path) for path in source_files]
    ref=read_image(cam_ref['image_path']);targets=[read_image(cam['image_path']) for cam in cam_targets]
    height,width=ref.shape;yy,xx=np.mgrid[:height,:width]
    grid=np.stack([xx,yy],axis=-1).astype(np.float32)
    reference_roi=np.logical_and.reduce([d['reference_roi'].astype(bool) for d in data])
    eligible=reference_roi.copy()
    for d in data:
        eligible &= d['valid_refined'].astype(bool)&d['geometry_applicable_mask'].astype(bool)
        eligible &= np.isfinite(d['uv_refined']).all(-1)&np.isfinite(d['A']).all((-1,-2))
    rm=read_mask(cam_ref,'valid_mask_path',ref.shape)
    tm=[read_mask(cam,'valid_mask_path',image.shape) for cam,image in zip(cam_targets,targets)]
    if mode=='asynchronous_unknown':
        if any('static_mask_path' not in cam for cam in [cam_ref,*cam_targets]):
            raise ValueError('Reliable static masks are required in all three asynchronous views')
        rm &= read_mask(cam_ref,'static_mask_path',ref.shape)
        tm=[mask&read_mask(cam,'static_mask_path',image.shape) for mask,cam,image in zip(tm,cam_targets,targets)]
    p=grid[eligible]
    if not len(p):raise ValueError('No jointly valid geometry-applicable A1 affine pixels')
    q=np.stack([d['uv_refined'][eligible] for d in data],axis=1)
    A=np.stack([d['A'][eligible] for d in data],axis=1)
    quality=dict(confidence=np.stack([d['confidence_raw'][eligible] for d in data],1),
        fb_error=np.stack([d['forward_backward_error_before'][eligible] for d in data],1),
        photo_error=np.stack([d['photo_error_before'][eligible] for d in data],1))
    started=time.perf_counter()
    initialization=initialize_shared_depth(cam_ref,cam_targets,p,q,cfg.min_angle_deg)
    initialization_seconds=time.perf_counter()-started
    reverse=[]
    for job in jobs:
        with np.load(job['cache'],allow_pickle=False) as cache:
            reverse.append({key:cache[key] for key in ['uv_BA','valid_BA']})
    Fs=[fundamental(cam_ref,cam) for cam in cam_targets]
    raw=np.stack([d['uv_raw'] for d in data],2)
    stage1=np.stack([d['uv_refined'] for d in data],2)
    raw_valid=reference_roi&np.logical_and.reduce([d['valid_raw'] for d in data])
    def scatter(values):
        values=np.asarray(values)
        fill=False if values.dtype==bool else (0 if np.issubdtype(values.dtype,np.integer) else np.nan)
        full=np.full((height,width)+values.shape[1:],fill,dtype=values.dtype)
        full[eligible]=values
        return full
    meta=dict(name=name,source_method='A1_affine',source_files=[str(p) for p in source_files],
        source_pair_names=[job['name'] for job in jobs],reference_camera=cam_ref,target_cameras=cam_targets,
        mode=mode,calibration_fixed=True,coordinate_system='Undistorted working-image x/y, integer pixel centers; align_corners=False',
        stride=config['stride'],seed=config.get('seed'),config=asdict(cfg),central_config=config,
        reference_roi_pixels=int(reference_roi.sum()),eligible_stage1_pixels=int(eligible.sum()),
        initialization_valid_pixels=int(initialization['valid'].sum()),initialization_seconds=initialization_seconds,
        initialization='Unweighted mean of both valid positive unit-ray ranges; identical for all four arms',
        weight_meaning='Frozen heuristic raw model score x initial FB x initial photometric score; no calibrated probability/uncertainty claim',
        quality_source='Original RoMa confidence, original forward-backward error, original affine-patch photometric error',
        distance_definition='Euclidean separation of fixed camera centers, same calibration translation units',
        depth_definition='Positive distance along reference unit world ray; NOT camera Z',
        covariance_claim=False,geometric_centers='Both centers are projections of the same single 3D point; no independent offsets',
        dataset_metadata_path=first['dataset'],status_legend=STATUS,methods={},
        software=dict(numpy=np.__version__,torch=torch.__version__),
        warning='Real data without GT measures consistency only; fixed-geometry residuals are not independent accuracy evidence')
    for arm in WEIGHT_MODES:
        result=refine_shared_depth(ref,targets,cam_ref,cam_targets,p,q,A,quality,config=cfg,
            weight_mode=arm,mode=mode,reference_mask=rm,target_masks=tm,device=device,initialization=initialization)
        timing=result.pop('timing');synchronize(device);started=time.perf_counter()
        result['forward_backward_error_before']=np.stack([fb_error(cache,p,q[:,j],device) for j,cache in enumerate(reverse)],1)
        result['forward_backward_error_after']=np.stack([fb_error(cache,p,result['uv_refined'][:,j],device) for j,cache in enumerate(reverse)],1)
        result['epipolar_error_before']=np.stack([np.abs(epipolar_residual(F,p,q[:,j])) for j,F in enumerate(Fs)],1)
        result['epipolar_error_after']=np.stack([np.abs(epipolar_residual(F,p,result['uv_refined'][:,j])) for j,F in enumerate(Fs)],1)
        fb_after=result['forward_backward_error_after']
        reject_fb=result['valid_refined']&(~np.isfinite(fb_after).all(-1)|(fb_after>config['fb_max']).any(-1))
        result['refinement_status'][reject_fb]=13
        result['valid_refined'][reject_fb]=False;result['refined_mask'][reject_fb]=False
        synchronize(device);timing['postscreen_seconds']=time.perf_counter()-started
        full={k:(v if k in ['distance_component','baselines'] else scatter(v)) for k,v in result.items()}
        full.update(uv_raw=raw,uv_stage1=stage1,reference_roi=reference_roi,attempt_mask=eligible,
                    valid_stage1=eligible,valid_raw=raw_valid)
        started=time.perf_counter();np.savez_compressed(out/f'{arm}.npz',**full)
        timing['saving_seconds']=time.perf_counter()-started
        meta['methods'][arm]=dict(valid_pixels=int(result['valid_refined'].sum()),timing=timing,
            status_counts={f'{int(k)}:{STATUS.get(int(k),"unknown")}':int((result['refinement_status']==k).sum()) for k in np.unique(result['refinement_status'])},
            distance_scale=timing['distance_scale'],baselines=result['baselines'].tolist(),
            distance_component=result['distance_component'].tolist())
        (out/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(json.dumps(dict(group=name,arm=arm,**meta['methods'][arm])),flush=True)
    meta['total_seconds']=time.perf_counter()-started_total
    (out/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    # Optimization outputs are complete before any truth is loaded.
    evaluate_group(out)
    return out


def _stats(values,mask):
    selected=np.asarray(values)[mask];selected=selected[np.isfinite(selected)]
    if not len(selected):return dict(count=0,mean=None,median=None,p90=None,rmse=None)
    return dict(count=len(selected),mean=float(selected.mean()),median=float(np.median(selected)),
                p90=float(np.quantile(selected,.9)),rmse=float(np.sqrt(np.mean(selected**2))))


def _ratio(n,d):return float(n/d) if d else None


def evaluate_group(directory):
    """Read truth here only; frozen outputs and acceptance are never changed."""
    directory=Path(directory);meta=json.loads((directory/'meta.json').read_text())
    data={arm:_load_npz(directory/f'{arm}.npz') for arm in WEIGHT_MODES}
    initial=data['uniform'];roi=initial['reference_roi'].astype(bool)
    attempted=initial['attempt_mask'].astype(bool);denominator=int(roi.sum())
    valid={arm:d['valid_refined'].astype(bool)&roi for arm,d in data.items()}
    common=np.logical_and.reduce(list(valid.values()))
    truth=None
    if meta['dataset_metadata_path'] is not None:
        dataset=json.loads(Path(meta['dataset_metadata_path']).read_text())
        ref_id=meta['reference_camera']['camera_id']
        keys=[f"{ref_id}{cam['camera_id']}" for cam in meta['target_cameras']]
        with np.load(dataset['array_path'],allow_pickle=False) as arrays:
            truth=dict(uv=np.stack([arrays['uv_gt_'+key] for key in keys],2),
                       valid=np.logical_and.reduce([arrays['valid_gt_'+key] for key in keys]),depth=arrays['depth_gt'])
    output=dict(name=meta['name'],reference_roi_pixels=denominator,attempted_pixels=int(attempted.sum()),
        common_four_arms_pixels=int(common.sum()),ground_truth_available=truth is not None,
        scene_mode=meta['mode'],geometry_fixed=True,calibration_units='Same as source fixed-calibration translations',
        common_scope='Intersection of four final validity masks; original denominator is ROI & both GT-visible masks',
        warning='Real residual reduction does not establish true localization/depth accuracy; weights are heuristics',methods={})
    rows=[]
    gt_region=roi&truth['valid'] if truth is not None else None
    gt_denominator=int(gt_region.sum()) if truth is not None else None
    output['original_gt_evaluable_pixels']=gt_denominator
    for arm,d in data.items():
        method=dict(valid_pixels=int(valid[arm].sum()),success_fraction_of_attempted=_ratio(int(valid[arm].sum()),int(attempted.sum())),
                    status_counts=meta['methods'][arm]['status_counts'],timing=meta['methods'][arm]['timing'],scopes={})
        for scope,support in [('own',valid[arm]),('common',common)]:
            record=dict(pixels=int(support.sum()),coverage_of_reference_roi=_ratio(int(support.sum()),denominator),
                        coverage_of_stage1_eligible=_ratio(int(support.sum()),int(attempted.sum())),views={})
            depth_stats=None
            if truth is not None:
                use=support&gt_region
                depth_stats=_stats(d['depth']-truth['depth'],use)
                record['depth_error_calibration_units']=depth_stats
                record['initial_depth_error_same_pixels']=_stats(d['depth_initial']-truth['depth'],use)
                record['gt_valid_coverage_original_roi']=_ratio(int(use.sum()),gt_denominator)
            for j,cam in enumerate(meta['target_cameras']):
                view={key:_stats(d[key][...,j],support) for key in ['photo_error_before','photo_error_after',
                      'forward_backward_error_before','forward_backward_error_after','epipolar_error_before',
                      'epipolar_error_after','refinement_displacement','ray_angle_deg_after']}
                row=dict(method=arm,scope=scope,target_camera=cam['camera_id'],valid_pixels=int(support.sum()),
                         coverage=record['coverage_of_reference_roi'])
                if truth is not None:
                    after=np.linalg.norm(d['uv_refined'][...,j,:]-truth['uv'][...,j,:],axis=-1)
                    prior=np.linalg.norm(d['uv_stage1'][...,j,:]-truth['uv'][...,j,:],axis=-1)
                    projected=np.linalg.norm(d['uv_projected_initial'][...,j,:]-truth['uv'][...,j,:],axis=-1)
                    raw=np.linalg.norm(d['uv_raw'][...,j,:]-truth['uv'][...,j,:],axis=-1)
                    use=support&gt_region
                    view['endpoint_error_px']=_stats(after,use)
                    view['stage1_endpoint_error_same_pixels_px']=_stats(prior,use)
                    view['projected_initial_endpoint_error_same_pixels_px']=_stats(projected,use)
                    view['raw_endpoint_error_same_pixels_px']=_stats(raw,use)
                    view['worsened_fraction_vs_stage1']=_ratio(int((use&(after>prior+1e-6)).sum()),int(use.sum()))
                    view['worsened_fraction_vs_projected_initial']=_ratio(int((use&(after>projected+1e-6)).sum()),int(use.sum()))
                    for threshold in [.5,1.,2.]:
                        key=str(threshold).replace('.','p')
                        good=int((use&(after<threshold)).sum())
                        view['fraction_lt_'+key+'_px']=_ratio(good,int(use.sum()))
                        view['correct_coverage_lt_'+key+'_px']=_ratio(good,gt_denominator)
                    row.update(epe_mean=view['endpoint_error_px']['mean'],epe_median=view['endpoint_error_px']['median'],
                               epe_p90=view['endpoint_error_px']['p90'],depth_rmse=depth_stats['rmse'],
                               projected_initial_epe_mean=view['projected_initial_endpoint_error_same_pixels_px']['mean'],
                               worsened_fraction_vs_stage1=view['worsened_fraction_vs_stage1'],
                               worsened_fraction_vs_projected_initial=view['worsened_fraction_vs_projected_initial'],
                               correct_coverage_lt_0p5_px=view['correct_coverage_lt_0p5_px'])
                for key in ['photo_error_after','forward_backward_error_after','epipolar_error_after']:
                    row[key]=view[key]['mean']
                record['views'][str(cam['camera_id'])]=view;rows.append(row)
            method['scopes'][scope]=record
        output['methods'][arm]=method
    (directory/'metrics.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (directory/'metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config.json')
    parser.add_argument('--selection',choices=['real','synthetic','all'],default='all')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--results-root',type=Path,default=ROOT/'results')
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    torch.manual_seed(config.get('seed',0));np.random.seed(config.get('seed',0))
    if args.selection in ['synthetic','all']:
        run_group('synthetic_affine_brightness',dataset_jobs('affine_brightness'),config,args.device,args.results_root,args.output_root)
    if args.selection in ['real','all']:
        run_group('real_7_9_10',dataset_jobs('real'),config,args.device,args.results_root,args.output_root)


if __name__=='__main__':main()
