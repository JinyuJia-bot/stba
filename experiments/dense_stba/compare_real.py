"""Compare real B3 runs only on verified identical reference-camera targets.

Default: compare raw/integer cases in results and results_density1, separately
and on their four-way common camera07 held-out support. The video frontend saw
all frames; held-out 2D predictions are diagnostics, not independent 3D/time GT.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/dense_stba_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __package__ in [None,'']:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from experiments.roma_lsm.prepare_data import sha256

ROOT=Path(__file__).resolve().parent
CASES=['corgi_raw','corgi_integer_aligned']


def stats(error,mask):
    values=np.asarray(error)[np.asarray(mask,bool)]
    values=values[np.isfinite(values)]
    if not len(values):return dict(count=0,mean=None,rmse=None,p50=None,p90=None)
    return dict(count=int(len(values)),mean=float(np.mean(values)),
                rmse=float(np.sqrt(np.mean(values**2))),p50=float(np.median(values)),
                p90=float(np.quantile(values,.9)))


def fraction(a,b):return float(a/b) if b else None


def bits_equal(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes(order='C')==b.tobytes(order='C')


def load_run(directory):
    directory=Path(directory)
    required=[directory/name for name in ['B3.npz','B3.json','metrics.json','meta.json']]
    missing=[str(p) for p in required if not p.exists()]
    if missing:raise FileNotFoundError('B3 and evaluation must finish before comparison: '+', '.join(missing))
    method=json.loads((directory/'B3.json').read_text())
    meta=json.loads((directory/'meta.json').read_text())
    metrics=json.loads((directory/'metrics.json').read_text())
    if 'B3' not in metrics.get('methods',{}):raise ValueError(f'B3 metrics not ready: {directory}')
    manifest=json.loads(Path(method['manifest_path']).read_text())
    names=['point_ids','observed_uv','holdout_visible','predicted_uv','beta','camera_ids',
           'timestamps','seed_eligible','trajectory_valid','valid','query_mask']
    with np.load(directory/'B3.npz',allow_pickle=False) as archive:
        arrays={name:archive[name].copy() for name in names}
    cameras={c['camera_id']:c for c in manifest['cameras']}
    ref_id=manifest['reference_camera_id']
    if ref_id!=7:raise ValueError('This real comparison requires reference camera07')
    ref_index=int(np.flatnonzero(arrays['camera_ids']==ref_id)[0])
    frame_ids=np.asarray(cameras[ref_id]['raw_frame_ids'],np.int64)
    if len(np.unique(arrays['point_ids']))!=len(arrays['point_ids']) or len(np.unique(frame_ids))!=len(frame_ids):
        raise ValueError('Point IDs and raw frame IDs must be unique')
    if arrays['observed_uv'].shape!=arrays['predicted_uv'].shape or arrays['observed_uv'].shape[2]!=len(frame_ids):
        raise ValueError('Observed/predicted/frame dimensions disagree')
    fps=float(cameras[ref_id]['fps'])
    beta=np.asarray(arrays['beta'],float)
    shifts=np.array([cameras[int(cid)]['integer_raw_frame_shift'] for cid in arrays['camera_ids']],float)
    camera_fps=np.array([cameras[int(cid)]['fps'] for cid in arrays['camera_ids']],float)
    total_beta=beta-shifts/camera_fps
    external=json.loads(Path(manifest['external_sync_reference_path']).read_text())
    published={c['camera_id']:c['raw_beta_seconds'] for c in external['cameras']}
    cfg=method['config'];bound=float(cfg['integer_beta_bound_frames'] if manifest['mode']=='integer_aligned' else cfg['real_raw_beta_bound_frames'])
    beta_rows=[]
    for j,cid in enumerate(arrays['camera_ids']):
        lower,upper=(0.,0.) if j==ref_index else (-bound,bound)
        residual_frames=float(beta[j]*fps)
        distance=float(min(residual_frames-lower,upper-residual_frames))
        beta_rows.append(dict(camera_id=int(cid),residual_seconds=float(beta[j]),residual_frames=residual_frames,
            integer_raw_frame_shift=int(shifts[j]),total_seconds=float(total_beta[j]),total_frames=float(total_beta[j]*fps),
            published_reference_total_seconds=float(published[int(cid)]),published_reference_total_frames=float(published[int(cid)]*fps),
            difference_from_reference_ms=float((total_beta[j]-published[int(cid)])*1000),
            difference_from_reference_frames=float((total_beta[j]-published[int(cid)])*fps),
            residual_bounds_frames=[lower,upper],total_bounds_frames=[lower-shifts[j]*fps/camera_fps[j],upper-shifts[j]*fps/camera_fps[j]],
            at_boundary=(j!=ref_index and distance<=1e-5),distance_to_boundary_frames=distance,
            observability_status=method.get('observability',{}).get('status',[None]*len(beta))[j]))
    error=np.linalg.norm(arrays['predicted_uv']-arrays['observed_uv'],axis=-1)
    eligible=arrays['holdout_visible']&np.isfinite(arrays['observed_uv']).all(-1)
    finite=eligible&np.isfinite(arrays['predicted_uv']).all(-1)
    own=stats(error,finite)
    own['heldout_observation_denominator']=int(eligible.sum())
    own['finite_prediction_coverage']=fraction(int(finite.sum()),int(eligible.sum()))
    own['per_camera']={str(int(cid)):stats(error[:,j],finite[:,j]) for j,cid in enumerate(arrays['camera_ids'])}
    n,t=len(arrays['point_ids']),len(frame_ids)
    key=f'{directory.parent.name}/{directory.name}'
    info=dict(key=key,directory=str(directory.resolve()),mode=manifest['mode'],
        stride=int(cfg['association']['stride']),camera_ids=arrays['camera_ids'].tolist(),fps=fps,
        frame_range_camera07=[int(frame_ids.min()),int(frame_ids.max())+1],
        own_heldout_all_cameras=own,beta=beta_rows,
        query_trajectories=n,initialized_trajectories=int(arrays['seed_eligible'].sum()),
        valid_trajectories=int(arrays['trajectory_valid'].sum()),valid_trajectory_coverage=fraction(int(arrays['trajectory_valid'].sum()),n),
        supported_3d_samples=int(arrays['valid'].sum()),supported_sample_coverage=fraction(int(arrays['valid'].sum()),n*t),
        timing=method.get('timing',{}),observability=method.get('observability',{}),
        translation_unit=manifest.get('translation_unit','dataset_native_unit'),
        beta_comparison_kind='Published synchronization reference, NOT independent time GT',
        beta_formula='total_beta_seconds = residual_beta_seconds - integer_raw_frame_shift / camera_fps',
        observations_sha256=method.get('observations_sha256'),
        source_files={str(p.resolve()):sha256(p) for p in required},
        heldout_policy=meta.get('holdout_policy'),
        window=meta.get('window'))
    calibration={k:np.asarray(cameras[ref_id][k]) for k in ['K','R','t']}
    return dict(key=key,info=info,arrays=arrays,reference_index=ref_index,
                raw_frame_ids=frame_ids,reference_calibration=calibration)


def common_reference(runs):
    """Align actual seed IDs AND raw frames; require bitwise-identical 2D targets.

    The mask is held out in every run and has finite predictions in every run.
    Equality is tested on ALL mutually held-out measured targets before finite
    prediction filtering, so a missing prediction cannot hide changed targets.
    """
    if len(runs)<2:raise ValueError('A common comparison requires at least two runs')
    point_ids=np.asarray(runs[0]['arrays']['point_ids'])
    frame_ids=np.asarray(runs[0]['raw_frame_ids'])
    for run in runs[1:]:
        point_ids=np.intersect1d(point_ids,run['arrays']['point_ids'])
        frame_ids=np.intersect1d(frame_ids,run['raw_frame_ids'])
        for name in ['K','R','t']:
            if not np.array_equal(run['reference_calibration'][name],runs[0]['reference_calibration'][name]):
                raise ValueError('Reference calibration changed between runs')
    observations=[];predictions=[];heldouts=[];times=[]
    for run in runs:
        a=run['arrays'];j=run['reference_index']
        lookup={int(v):i for i,v in enumerate(a['point_ids'])}
        ids=np.array([lookup[int(v)] for v in point_ids],np.int64)
        lookup_t={int(v):i for i,v in enumerate(run['raw_frame_ids'])}
        frames=np.array([lookup_t[int(v)] for v in frame_ids],np.int64)
        observations.append(a['observed_uv'][ids,j][:,frames])
        predictions.append(a['predicted_uv'][ids,j][:,frames])
        heldouts.append(a['holdout_visible'][ids,j][:,frames])
        times.append(a['timestamps'][j,frames])
    for values in times[1:]:
        if not bits_equal(values,times[0]):raise ValueError('Reference raw-frame timestamps changed')
    original=np.logical_and.reduce(heldouts)
    for obs in observations:
        original &= np.isfinite(obs).all(-1)
    for obs in observations[1:]:
        if not bits_equal(obs[original],observations[0][original]):
            raise ValueError('Original observed_uv is not bitwise equal on common held-out camera07 targets; refusing an unfair comparison')
    common=original.copy()
    for prediction in predictions:common &= np.isfinite(prediction).all(-1)
    indices=np.stack(np.meshgrid(point_ids,frame_ids,indexing='ij'),axis=-1)[common]
    result=dict(reference_camera_id=7,run_keys=[r['key'] for r in runs],shared_seed_ids_count=len(point_ids),
        shared_raw_frames=frame_ids.tolist(),mutual_original_heldout_observations=int(original.sum()),
        common_finite_prediction_heldout_observations=int(common.sum()),
        common_prediction_coverage=fraction(int(common.sum()),int(original.sum())),
        common_seed_ids_with_predictions=int(common.any(1).sum()),
        original_observed_uv_bitwise_equal=True,checked_target_count=int(original.sum()),
        target_identity_sha256=hashlib.sha256(indices.astype(np.int64).tobytes()).hexdigest(),
        common_observed_uv_sha256=hashlib.sha256(observations[0][common].tobytes()).hexdigest(),
        sample_definition='Identical camera07, reference pixel ID and original raw frame ID; held out in every listed run, finite predicted2D in every listed run; no residual-based selection.',
        by_run={})
    errors=[np.linalg.norm(pred-obs,axis=-1) for pred,obs in zip(predictions,observations)]
    for run,error in zip(runs,errors):
        item=stats(error,common)
        item['per_raw_frame']=[dict(raw_frame_id=int(frame),**stats(error[:,j],common[:,j])) for j,frame in enumerate(frame_ids)]
        result['by_run'][run['key']]=item
    if len(runs)==2:
        delta=(errors[1]-errors[0])[common]
        result['paired_change_second_minus_first']=dict(mean_endpoint_error_change_px=float(delta.mean()) if len(delta) else None,
            second_lower_fraction=float((delta<0).mean()) if len(delta) else None,
            note='Paired held-out2D prediction difference, not independently verified physical3D accuracy')
    return result


def window_stability(stability_root):
    root=Path(stability_root)
    result=dict(root=str(root.resolve()),kind='Independent-window diagnostic; these separately optimized offsets are NOT stitched into one reconstruction',
        windows=['window_0_40','window_20_60'],cases={},missing=[])
    for case in CASES:
        records=[]
        for window in result['windows']:
            directory=root/window/case
            if not all((directory/file).exists() for file in ['B3.npz','B3.json','meta.json','metrics.json']):
                result['missing'].append(str(directory));continue
            run=load_run(directory);info=run['info']
            records.append(dict(window=window,frame_window=info['window'],beta=info['beta'],
                                own_heldout_all_cameras=info['own_heldout_all_cameras'],timing=info['timing']))
            del run
        per_camera={}
        if records:
            for j,item in enumerate(records[0]['beta']):
                values=np.array([r['beta'][j]['total_seconds'] for r in records])
                frames=np.array([r['beta'][j]['total_frames'] for r in records])
                per_camera[str(item['camera_id'])]=dict(window_count=len(records),
                    total_beta_seconds=values.tolist(),total_beta_frames=frames.tolist(),
                    span_ms=float(np.ptp(values)*1000) if len(records)>1 else None,
                    span_frames=float(np.ptp(frames)) if len(records)>1 else None,
                    any_at_boundary=any(r['beta'][j]['at_boundary'] for r in records),
                    all_at_boundary=all(r['beta'][j]['at_boundary'] for r in records),
                    span_interpretation='A zero span while constrained at a bound is not evidence of reliable synchronization' if any(r['beta'][j]['at_boundary'] for r in records) else 'Variation across separately fitted windows; no shared global estimate was formed')
        result['cases'][case]=dict(independent_runs=records,per_camera=per_camera)
    return result


def write_csv(path,summary):
    rows=[]
    for run in summary['runs']:
        for beta in run['beta']:
            common=summary['common_reference07_all_runs']['by_run'][run['key']]
            rows.append(dict(row_type='run',run=run['key'],stride=run['stride'],mode=run['mode'],**beta,
                query_trajectories=run['query_trajectories'],valid_trajectories=run['valid_trajectories'],
                trajectory_coverage=run['valid_trajectory_coverage'],supported_3d_samples=run['supported_3d_samples'],
                sample_coverage=run['supported_sample_coverage'],
                own_heldout_count=run['own_heldout_all_cameras']['count'],own_heldout_rmse_px=run['own_heldout_all_cameras']['rmse'],
                common_camera07_count=common['count'],common_camera07_rmse_px=common['rmse'],
                common_camera07_mean_px=common['mean'],common_camera07_p50_px=common['p50'],common_camera07_p90_px=common['p90'],
                solver_seconds=run['timing'].get('seconds'),wall_seconds=run['timing'].get('method_wall_seconds'),
                peak_cuda_allocated_bytes=run['timing'].get('peak_cuda_allocated_bytes')))
    for case,item in summary['window_stability']['cases'].items():
        for cid,values in item['per_camera'].items():
            rows.append(dict(row_type='independent_window_span',run=case,camera_id=int(cid),
                             independent_window_beta_span_frames=values['span_frames'],
                             independent_window_beta_span_ms=values['span_ms'],any_window_at_boundary=values['any_at_boundary']))
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def plot_summary(summary,path):
    runs=summary['runs'];common=summary['common_reference07_all_runs'];n=len(runs)
    labels=[f"stride{r['stride']}\n{'raw' if r['mode']=='raw' else 'integer'}" for r in runs]
    colors=['#377eb8' if r['mode']=='raw' else '#e6862a' for r in runs]
    x=np.arange(n)
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    values=[common['by_run'][r['key']]['rmse'] for r in runs]
    axes[0,0].bar(x,[v if v is not None else 0 for v in values],color=colors)
    axes[0,0].set_title(f"Same camera07 held-out targets (n={common['common_finite_prediction_heldout_observations']})")
    axes[0,0].set_ylabel('2D prediction RMSE [working px]')
    axes[0,0].text(.02,.98,'Original observed UV verified bitwise equal',transform=axes[0,0].transAxes,va='top',fontsize=8)
    own=[r['own_heldout_all_cameras']['rmse'] for r in runs]
    axes[0,1].bar(x,[v if v is not None else 0 for v in own],color=colors)
    axes[0,1].set_title('Own held-out targets, all cameras\nDifferent sets: descriptive, not paired')
    axes[0,1].set_ylabel('2D prediction RMSE [working px]')
    for j,cid in enumerate([9,10]):
        values=[next(b for b in r['beta'] if b['camera_id']==cid)['total_frames'] for r in runs]
        axes[0,2].plot(x,values,'o-',label=f'cam{cid:02d}',linewidth=1.)
        published=next(b for b in runs[0]['beta'] if b['camera_id']==cid)['published_reference_total_frames']
        axes[0,2].axhline(published,color=['#1f77b4','#ff7f0e'][j],linestyle='--',alpha=.6,label=f'cam{cid:02d} published reference')
        for k,r in enumerate(runs):
            b=next(b for b in r['beta'] if b['camera_id']==cid)
            if b['at_boundary']:axes[0,2].plot(k,values[k],'x',color='red',markersize=10,markeredgewidth=2)
    axes[0,2].set_title('Recovered TOTAL beta (reference07=0)\nDashed: published sync, not independent GT; x: bound')
    axes[0,2].set_ylabel('Frames at60 FPS');axes[0,2].legend(fontsize=7)
    axes[1,0].bar(x,[r['valid_trajectory_coverage'] for r in runs],color=colors)
    axes[1,0].set_ylim(0,1);axes[1,0].set_ylabel('Valid trajectories / all query seeds')
    axes[1,0].set_title('Own 3D trajectory support coverage')
    for i,r in enumerate(runs):axes[1,0].text(i,r['valid_trajectory_coverage']+.02,f"{r['valid_trajectories']}/{r['query_trajectories']}",ha='center',fontsize=8)
    axes[1,1].bar(x,[r['timing'].get('method_wall_seconds',r['timing'].get('seconds',0)) for r in runs],color=colors)
    axes[1,1].set_title('B3 fit/optimize/save time; excludes coarse init/frontend')
    axes[1,1].set_ylabel('Seconds')
    for i,r in enumerate(runs):
        memory=r['timing'].get('peak_cuda_allocated_bytes',0)/2**20
        axes[1,1].text(i,0,f'{memory:.0f} MiB',ha='center',va='bottom',rotation=90,fontsize=8,color='black')
    for ax in [axes[0,0],axes[0,1],axes[0,2],axes[1,0],axes[1,1]]:
        ax.set_xticks(x,labels,fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    stable=summary['window_stability']
    for case,item in stable['cases'].items():
        for cid in ['9','10']:
            if cid not in item['per_camera']:continue
            y=item['per_camera'][cid]['total_beta_frames']
            axes[1,2].plot(np.arange(len(y)),y,'o',label=f"{'raw' if case=='corgi_raw' else 'integer'} cam{int(cid):02d}",
                          linestyle='-' if case=='corgi_raw' else '--')
            for i,record in enumerate(item['independent_runs']):
                beta=next(b for b in record['beta'] if b['camera_id']==int(cid))
                if beta['at_boundary']:
                    axes[1,2].plot(i,y[i],'x',color='red',markersize=10,markeredgewidth=2)
    axes[1,2].set_xticks([0,1],['window[0,40)','window[20,60)'],fontsize=8)
    axes[1,2].set_ylabel('Independent total beta [frames]')
    axes[1,2].set_title('Independent windows, NOT a stitched solution\nx: bound; zero span at bound is not reliable stability',fontsize=10)
    axes[1,2].grid(alpha=.2);axes[1,2].legend(fontsize=7)
    fig.suptitle('Fixed-camera real B3 comparisons — held-out tracked2D consistency, not 3D/time accuracy proof',fontsize=13)
    fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)


def write_outputs(directory,summary):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    (directory/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    write_csv(directory/'comparison.csv',summary)
    plot_summary(summary,directory/'comparison.png')


def compare(results_roots,output=ROOT/'real_comparison',stability_root=ROOT/'stability'):
    groups=[];all_runs=[]
    for root in map(Path,results_roots):
        runs=[load_run(root/case) for case in CASES]
        groups.append((root,runs));all_runs.extend(runs)
    if len({r['key'] for r in all_runs})!=len(all_runs):raise ValueError('Results roots need distinct directory names')
    stability=window_stability(stability_root)
    base=dict(schema_version=1,method='B3',ground_truth_available=False,
        interpretation='Held-out tracked2D measurement prediction only. Frontends used full video, including held-out frames. No independent 3D or synchronization GT. Own all-camera errors use different targets and must not be treated as paired raw/integer accuracy comparisons.',
        coordinates='Undistorted working-image integer pixel centers; Euclidean2D errors in working pixels. Translation remains in dataset-native units; no claim of metric3D accuracy.',
        window_stability=stability)
    pairwise={}
    for root,runs in groups:
        common=common_reference(runs);pairwise[root.name]=common
        summary={**base,'runs':[r['info'] for r in runs],'common_reference07_all_runs':common,
                 'common_reference07_by_results_root':{root.name:common}}
        write_outputs(root,summary)
    merged={**base,'runs':[r['info'] for r in all_runs],
            'common_reference07_all_runs':common_reference(all_runs),
            'common_reference07_by_results_root':pairwise}
    write_outputs(output,merged)
    print(json.dumps({'output':str(output),'runs':[r['key'] for r in all_runs],
         'common_reference07_targets':merged['common_reference07_all_runs']['common_finite_prediction_heldout_observations'],
         'bitwise_equal':merged['common_reference07_all_runs']['original_observed_uv_bitwise_equal'],
         'window_stability_missing':stability['missing']},indent=2))
    return merged


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results-root',type=Path,action='append',help='May repeat; default includes results and results_density1')
    parser.add_argument('--output',type=Path,default=ROOT/'real_comparison')
    parser.add_argument('--stability-root',type=Path,default=ROOT/'stability')
    args=parser.parse_args();compare(args.results_root or [ROOT/'results',ROOT/'results_density1'],args.output,args.stability_root)
