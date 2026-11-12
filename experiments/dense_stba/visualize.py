"""Standalone Dense STBA figures and animations; plotting never alters results."""
from pathlib import Path
import argparse
import json
import os

import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/dense_stba_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

try:
    from .evaluate import load_results,evaluate_case,load_verified_observations
    from .data_loader import load_camera_sequence
    from .synthetic_dynamic import load_truth
except ImportError:
    from evaluate import load_results,evaluate_case,load_verified_observations
    from data_loader import load_camera_sequence
    from synthetic_dynamic import load_truth


def save(fig,path,tight=True):
    fig.savefig(path,dpi=160,bbox_inches='tight' if tight else None);plt.close(fig)


def equal_axes(ax,points,unit_label='m'):
    p=np.asarray(points).reshape(-1,3);p=p[np.isfinite(p).all(-1)]
    if not len(p):return
    lo,hi=p.min(0),p.max(0);center=(lo+hi)/2;radius=max(float((hi-lo).max()/2),.01)*1.08
    ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
    ax.set_box_aspect((1,1,1))
    ax.set_xlabel(f'World X ({unit_label})',fontsize=8,labelpad=0)
    ax.set_ylabel(f'World Y ({unit_label})',fontsize=8,labelpad=0)
    ax.set_zlabel(f'World Z ({unit_label})',fontsize=8,labelpad=0)


def track_overlays(directory,manifest,data,observations,label=''):
    uv=observations.get('uv',data['observed_uv'])
    visible=observations.get('visible',data['train_visible']|data['holdout_visible'])
    n,c,t=visible.shape;seed=int(manifest['seed_frame']);frames=sorted(set([0,seed,t-1]))
    colors=plt.cm.turbo(np.linspace(0,1,n))
    chosen=np.linspace(0,n-1,min(100,n),dtype=int)
    fig,axes=plt.subplots(c,len(frames),figsize=(4.1*len(frames),3*c),squeeze=False)
    for j,cam in enumerate(manifest['cameras']):
        images=load_camera_sequence(cam)
        for k,frame in enumerate(frames):
            ax=axes[j,k];ax.imshow(images[frame]);good=chosen[visible[chosen,j,frame]]
            ax.scatter(uv[good,j,frame,0],uv[good,j,frame,1],s=4,c=colors[good],linewidths=0)
            for point in good:
                start=max(0,frame-5);trail=uv[point,j,start:frame+1].copy()
                trail[~visible[point,j,start:frame+1]]=np.nan
                ax.plot(trail[:,0],trail[:,1],color=colors[point],lw=.65)
            ax.set_title(f"cam {cam['camera_id']}, frame {frame}: {visible[:,j,frame].sum()} visible",fontsize=10)
            ax.set_xlim(-.5,cam['size'][0]-.5);ax.set_ylim(cam['size'][1]-.5,-.5);ax.axis('off')
    fig.suptitle(f'{label}: original image-derived 2D tracks; fixed identity colors (display subset only)')
    fig.tight_layout();save(fig,directory/'tracks_2d.png')


def status_map(directory,manifest,data,observations,label='',marker_size=7):
    p=data['seed_uv'];n=len(p);temporal=observations.get('temporal_success',np.ones(n,bool))
    associated=observations.get('multiview_seed',(data['train_visible']|data['holdout_visible']).any(-1).sum(-1)>=2)
    recovered=data['trajectory_valid']&data['valid'].any(-1)
    status=np.zeros(n,int);status[temporal]=1;status[associated]=2;status[recovered]=3
    reference=manifest['cameras'][0];image=load_camera_sequence(reference)[manifest['seed_frame']]
    fig,ax=plt.subplots(figsize=(8,5));ax.imshow(image)
    names=['Temporal failure','Temporal only','Associated, no 3D','Recovered 3D'];colors=['#d73027','#fdae61','#4575b4','#1a9850']
    for code,category_label in enumerate(names):
        use=status==code;ax.scatter(p[use,0],p[use,1],s=marker_size,color=colors[code],label=f'{category_label}: {use.sum()}',linewidths=0)
    ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=9);ax.axis('off')
    ax.set_title(f'{label}: full query grid → supported 3D')
    save(fig,directory/'query_status.png')


def trajectory_figures(directory,manifest,data,truth=None,max_gif_frames=12,label='',
                       max_cloud_points=None,cloud_rgb=False,max_trajectory_lines=18):
    xyz=data['xyz'];valid=data['valid']&data['trajectory_valid'][:,None]&np.isfinite(xyz).all(-1)
    times=data['trajectory_times'];n,s=valid.shape
    gt=truth.evaluate_material_trajectory(data['seed_uv'],times).transpose(1,0,2) if truth is not None else None
    stated_units=manifest.get('translation_unit',manifest.get('units','unknown'))
    unit_label='m' if truth is not None or stated_units in ['m','meters','metres'] else 'native units'
    gt_visible=None
    if truth is not None:
        visible=[]
        for j,cam in enumerate(manifest['cameras']):
            present=(times>=truth.physical_timestamps[j,0])&(times<=truth.physical_timestamps[j,-1])
            visible.append(truth.project_material(data['seed_uv'],cam['camera_id'],times)['valid'].T&present[None])
        gt_visible=np.sum(visible,axis=0)>=2
    colors=plt.cm.turbo(np.linspace(0,1,n));cloud_colors=colors
    if cloud_rgb:
        reference_image=np.asarray(Image.open(manifest['cameras'][0]['image_paths'][manifest['seed_frame']]).convert('RGB'))
        pixel=np.round(data['seed_uv']).astype(int);pixel[:,0]=pixel[:,0].clip(0,reference_image.shape[1]-1);pixel[:,1]=pixel[:,1].clip(0,reference_image.shape[0]-1)
        cloud_colors=reference_image[pixel[:,1],pixel[:,0]]/255.
    cloud_candidates=np.flatnonzero(valid.any(-1));cloud_indices=cloud_candidates
    if max_cloud_points is not None and len(cloud_candidates)>max_cloud_points:
        cloud_indices=cloud_candidates[np.linspace(0,len(cloud_candidates)-1,max_cloud_points,dtype=int)]
    cloud_mask=np.zeros(n,bool);cloud_mask[cloud_indices]=True
    sampling_note=(f'Display {len(cloud_indices):,}/{len(cloud_candidates):,} recovered trajectories; '
                   f'{"reference RGB" if cloud_rgb else "identity"} colors; visualization sampling only') if max_cloud_points is not None else ''
    available=np.flatnonzero(valid.sum(-1)>=3)
    chosen=available[np.linspace(0,len(available)-1,min(max_trajectory_lines,len(available)),dtype=int)] if len(available) else []
    fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(111,projection='3d')
    for index,point in enumerate(chosen):
        trajectory=xyz[point].copy();trajectory[~valid[point]]=np.nan
        ax.plot(*trajectory.T,color=colors[point],lw=.65 if max_cloud_points is not None else 1.3,label='Estimated' if index==0 else None)
        if gt is not None:
            ax.plot(*gt[point].T,color='black',lw=.65,ls='--',alpha=.55,label='GT' if index==0 else None)
    equal_axes(ax,xyz[valid],unit_label);ax.set_title(f'{label}\n{len(chosen)} displayed trajectory lines; raw world coordinates',fontsize=10)
    if len(chosen):ax.legend()
    save(fig,directory/'trajectories_3d.png',tight=False)
    indices=np.linspace(0,s-1,min(max_gif_frames,s),dtype=int)
    fig=plt.figure(figsize=(12,4.8))
    for k,index in enumerate(sorted(set([0,s//2,s-1]))):
        ax=fig.add_subplot(1,3,k+1,projection='3d');good=valid[:,index]&cloud_mask
        if gt is not None:
            ax.scatter(*gt[gt_visible[:,index],index].T,s=.3,color='gray',alpha=.45,label='GT evaluable',linewidths=0)
        ax.scatter(*xyz[good,index].T,s=.30 if max_cloud_points is not None else .8,c=cloud_colors[good],alpha=.8,label='Estimated',linewidths=0,depthshade=not cloud_rgb)
        equal_axes(ax,xyz[valid],unit_label);ax.set_title(f't={times[index]:.3f}s, {good.sum()} displayed / {valid[:,index].sum()} supported',fontsize=8)
        ax.tick_params(labelsize=7)
        if k==0:ax.legend(fontsize=7,loc='upper left')
    fig.suptitle(label+('\n'+sampling_note if sampling_note else ''),fontsize=11)
    fig.subplots_adjust(left=.02,right=.97,bottom=.15,top=.76 if max_cloud_points is not None else .83,wspace=.10)
    save(fig,directory/'pointcloud_snapshots.png',tight=False)
    animation=[]
    for index in indices:
        fig=plt.figure(figsize=(6,5) if max_cloud_points is not None else (5,4),dpi=100)
        ax=fig.add_subplot(111,projection='3d');good=valid[:,index]&cloud_mask
        if gt is not None:ax.scatter(*gt[gt_visible[:,index],index].T,s=.3,color='gray',alpha=.5,linewidths=0)
        ax.scatter(*xyz[good,index].T,s=.45 if max_cloud_points is not None else 1.3,c=cloud_colors[good],alpha=.8,linewidths=0,depthshade=not cloud_rgb)
        equal_axes(ax,xyz[valid],unit_label)
        caption=f'{label}\nt={times[index]:.3f}s | {good.sum()} displayed / {valid[:,index].sum()} supported'
        if sampling_note:caption+='\n'+sampling_note
        ax.set_title(caption,fontsize=8 if max_cloud_points is not None else 10)
        fig.tight_layout();fig.canvas.draw()
        rgb=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy();animation.append(Image.fromarray(rgb));plt.close(fig)
    if animation:
        animation[0].save(directory/'pointcloud.gif',save_all=True,append_images=animation[1:],
            duration=max(80,int(1000*(times[-1]-times[0])/max(1,len(indices)-1))),loop=0)


def metric_figures(directory,metrics,arrays,manifest):
    names=list(metrics['methods']);c=len(manifest['cameras']);positions=np.arange(c)
    fig,ax=plt.subplots(figsize=(8,4));width=.8/max(1,len(names))
    for i,name in enumerate(names):
        rows=metrics['methods'][name]['beta']['cameras']
        # Residual beta is directly comparable to the synthetic GT. For real
        # integer compensation plots report restored raw offsets against reference.
        values=[r.get('recovered_total_beta_seconds',r['estimated_beta_seconds'])*manifest['cameras'][j]['fps'] for j,r in enumerate(rows)]
        label=name+(' (oracle beta)' if metrics['methods'][name].get('oracle_beta') else '')
        reliability=(metrics['methods'][name].get('observability') or {}).get('reliable',[False]*c)
        if name in ['B3','B4'] and not all(reliability[1:]):label+=' (unreliable cameras hatched)'
        bars=ax.bar(positions-.4+width*(i+.5),values,width,label=label)
        if name in ['B3','B4']:
            for j,bar in enumerate(bars):
                if j>0 and not reliability[j]:bar.set_hatch('///');bar.set_edgecolor('black')
    first=metrics['methods'][names[0]]['beta']['cameras'];reference=[]
    for j,row in enumerate(first):
        value=row.get('gt_beta_frames')
        if value is None and 'published_total_beta_seconds' in row:value=row['published_total_beta_seconds']*manifest['cameras'][j]['fps']
        reference.append(np.nan if value is None else value)
    ax.plot(positions,reference,'kx',ms=9,label='GT' if metrics['ground_truth_available'] else 'Published reference (not GT)')
    if 'B3' in arrays:
        initial=np.asarray(arrays['B3']['initial_beta']).copy()
        for j,cam in enumerate(manifest['cameras']):initial[j]-=cam.get('integer_raw_frame_shift',0)/cam['fps']
        ax.plot(positions,[initial[j]*cam['fps'] for j,cam in enumerate(manifest['cameras'])],'o',mfc='none',color='gray',label='B3 initial')
    ax.set_xticks(positions,[str(cam['camera_id']) for cam in manifest['cameras']]);ax.set_xlabel('Camera');ax.set_ylabel('PLUS beta (frames)')
    unreliable='B3' in metrics['methods'] and not any((metrics['methods']['B3'].get('observability') or {}).get('reliable',[False])[1:])
    if not metrics['ground_truth_available']:
        title='Real synchronization unverified: comparison to published reference only'
    else:title='Beta is unobservable: reported values are not reliable synchronization' if unreliable else 'Camera time offsets, fixed reference gauge'
    ax.set_title(title)
    ax.legend(fontsize=8);ax.grid(alpha=.2,axis='y');save(fig,directory/'offsets.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,record in metrics['methods'].items():
        if 'position_error_per_time' in record:
            values=[np.nan if q['rmse'] is None else q['rmse']*1000 for q in record['position_error_per_time']]
            axes[0].plot(metrics['trajectory_times'],values,label=name)
        raw_common=record['reprojection'].get('fixed_raw_holdout_common_all',{})
        values=[np.nan if q['rmse'] is None else q['rmse'] for q in raw_common.get('per_frame',record['reprojection']['holdout_per_frame'])]
        axes[1].plot(np.arange(len(values)),values,label=name)
    axes[0].set_xlabel('Physical reference time (s)');axes[0].set_ylabel('Own-valid position RMSE (mm)')
    axes[0].set_title('Independent synthetic error' if metrics['ground_truth_available'] else 'No independent 3D GT')
    axes[1].set_xlabel('Local frame');axes[1].set_ylabel('Held-out reprojection RMSE (px)')
    axes[1].set_title('Common predictions vs fixed raw B0 holdout (not GT)' if metrics.get('shared_holdout_comparison',{}).get('available') else 'Held-out observations; missing values stay missing')
    for ax in axes:ax.grid(alpha=.25)
    axes[1].legend(fontsize=8);fig.tight_layout();save(fig,directory/'error_curves.png')
    fig,(ax,temporal)=plt.subplots(1,2,figsize=(12,4))
    queried=[];tracked=[];associated=[];reconstructed=[]
    for name in names:
        support=metrics['methods'][name]['support'];queried.append(support['query_grid_count']);tracked.append(support['successful_reference_temporal_tracks']);associated.append(support['tracks_with_two_or_more_observed_cameras']);reconstructed.append(support['recovered_with_valid_samples'])
    x=np.arange(len(names));width=.2
    for i,(label,values) in enumerate([('Query',queried),('Temporal',tracked),('Associated',associated),('3D',reconstructed)]):ax.bar(x+(i-1.5)*width,values,width,label=label)
    ax.set_xticks(x,names);ax.set_ylabel('Reference grid trajectories');ax.set_title('Coverage through the pipeline');ax.legend(fontsize=8)
    for name in names:
        record=metrics['methods'][name]
        data=arrays[name];valid=data['valid']&data['trajectory_valid'][:,None]&np.isfinite(data['xyz']).all(-1)
        denominator=np.asarray(record.get('gt_evaluable_samples_per_time',np.full(valid.shape[1],len(valid))),float)
        numerator=np.asarray(record.get('gt_evaluable_valid_samples_per_time',valid.sum(0)),float)
        fraction=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
        temporal.plot(data['trajectory_times'],fraction,label=name)
    temporal.set_xlabel('Physical reference time (s)')
    temporal.set_ylabel('GT-evaluable query-time coverage' if any('gt_evaluable_samples_per_time' in m for m in metrics['methods'].values()) else 'Fraction of all queried tracks')
    temporal.set_ylim(-.02,1.02)
    temporal.set_title('Temporal coverage; missing support stays missing');temporal.legend(fontsize=8);temporal.grid(alpha=.2)
    fig.tight_layout();save(fig,directory/'coverage.png')


def visualize_case(directory,method='B3',max_gif_frames=12,max_cloud_points=None,
                   cloud_rgb=False,max_trajectory_lines=18):
    directory=Path(directory);arrays,metas,manifest,manifest_path=load_results(directory)
    if method not in arrays:method=next(reversed(arrays))
    data=arrays[method];meta=metas[method];observations,_=load_verified_observations(data,meta,directory)
    metric_path=directory/'metrics.json'
    metrics=json.loads(metric_path.read_text()) if metric_path.exists() else evaluate_case(directory)
    truth=load_truth(manifest_path.parent) if (manifest_path.parent/'gt.npz').exists() else None
    out=directory/'figures';out.mkdir(exist_ok=True)
    label=method+(' (oracle beta)' if meta.get('oracle_beta',False) else '')
    if truth is None:label+=' | real synchronization unverified'
    elif method in ['B3','B4'] and not any((meta.get('observability') or {}).get('reliable',[False])[1:]):label+=' | beta unobservable'
    density_note=None
    if max_cloud_points is not None:
        stride=meta.get('config',{}).get('association',{}).get('stride','unknown')
        roi=manifest.get('dynamic_candidate_roi');size=manifest['cameras'][0]['size']
        region=f'{roi[2]-roi[0]}x{roi[3]-roi[1]} ROI within {size[0]}x{size[1]}' if roi else f'{size[0]}x{size[1]} input'
        density_note=f'Stride {stride}, {region}; {len(data["point_ids"]):,} queried, {meta.get("initialized_trajectories",int(data["trajectory_valid"].sum())):,} optimized'
        label+='\n'+density_note
    track_overlays(out,manifest,data,observations,label)
    status_map(out,manifest,data,observations,label,marker_size=1 if max_cloud_points is not None else 7)
    trajectory_figures(out,manifest,data,truth,max_gif_frames,label,max_cloud_points,cloud_rgb,max_trajectory_lines)
    metric_figures(out,metrics,arrays,manifest)
    files=[str(path) for path in sorted(out.iterdir()) if path.is_file()]
    (out/'visualization.json').write_text(json.dumps(dict(method=method,artifacts=files,
        density_note=density_note,max_displayed_cloud_trajectories=max_cloud_points,
        max_displayed_trajectory_lines=max_trajectory_lines,cloud_color='reference RGB' if cloud_rgb else 'identity color',
        queried_trajectories=len(data['point_ids']),optimized_trajectories=meta.get('initialized_trajectories'),
        interpretation='Plots may subsample tracks for readability; algorithm outputs and metrics retain the full query grid. No geometric offset or post-hoc alignment is applied to GT.'),indent=2)+'\n')
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--method',default='B3');parser.add_argument('--max-gif-frames',type=int,default=12)
    parser.add_argument('--max-cloud-points',type=int);parser.add_argument('--cloud-rgb',action='store_true')
    parser.add_argument('--max-trajectory-lines',type=int,default=18)
    args=parser.parse_args();print(visualize_case(args.directory,args.method,args.max_gif_frames,
        args.max_cloud_points,args.cloud_rgb,args.max_trajectory_lines))
