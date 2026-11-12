"""Static, reproducible matching diagnostics; no GT-based selection or filtering."""
import argparse
import json
import os
from pathlib import Path

import cv2
os.environ.setdefault('MPLCONFIGDIR','/tmp/roma_lsm_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap
import numpy as np
import torch
import torch.nn.functional as functional

try:
    from .evaluate import load_pair,camera_pair,resolve_path,load_truth,STATUS
except ImportError:
    from evaluate import load_pair,camera_pair,resolve_path,load_truth,STATUS


def sample(image,uv):
    """Exact torch bilinear image sampling, invalid pixels explicitly become NaN."""
    image=np.asarray(image,np.float32);uv=np.asarray(uv,np.float32)
    h,w=image.shape
    finite=np.isfinite(uv).all(-1)
    inside=finite&(uv[...,0]>=0)&(uv[...,0]<=w-1)&(uv[...,1]>=0)&(uv[...,1]<=h-1)
    clean=np.nan_to_num(uv,nan=0,posinf=0,neginf=0)
    normalized=2*(clean+.5)/np.asarray([w,h],np.float32)-1
    shape=uv.shape[:-1]
    grid=torch.from_numpy(normalized.reshape(1,-1,1,2))
    with torch.no_grad():
        values=functional.grid_sample(torch.from_numpy(image)[None,None],grid,
                  mode='bilinear',padding_mode='zeros',align_corners=False)[0,0,:,0].numpy().reshape(shape)
    values[~inside]=np.nan
    return values


def masked(value,mask):
    result=np.asarray(value,float).copy()
    result[~mask]=np.nan
    return result


def show(ax,value,title,cmap='viridis',vmin=None,vmax=None):
    artist=ax.imshow(value,cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
    ax.set_title(title,fontsize=9)
    ax.set_xticks([]);ax.set_yticks([])
    return artist


def _read_image(camera,directory):
    path=resolve_path(camera['image_path'],directory)
    image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image.astype(np.float32)/255


def _save(fig,path):
    fig.savefig(path,dpi=160,bbox_inches='tight')
    plt.close(fig)


def visualize_pair(directory):
    torch.set_num_threads(4)
    directory=Path(directory)
    arrays,metas=load_pair(directory)
    first=next(iter(arrays))
    ref_cam,target_cam=camera_pair(metas[first])
    reference,target=(_read_image(c,directory) for c in [ref_cam,target_cam])
    out=directory/'plots';out.mkdir(exist_ok=True)
    names=list(arrays)
    roi=arrays[first]['reference_roi'].astype(bool)
    fig,axes=plt.subplots(1,3,figsize=(13,3.5),constrained_layout=True)
    show(axes[0],reference,'Reference image / evaluation ROI','gray',0,1)
    if roi.any() and not roi.all():
        axes[0].contour(roi.astype(float),levels=[.5],colors=['lime'],linewidths=.6)
    show(axes[1],target,'Target image','gray',0,1)
    a=show(axes[2],masked(arrays[first]['confidence_raw'],roi),'Raw RoMa score inside ROI (not probability)',vmin=0,vmax=1)
    fig.colorbar(a,ax=axes[2],fraction=.046)
    _save(fig,out/'confidence.png')

    n=len(names)
    ys,xs=np.nonzero(roi)
    bounds=(max(0,int(xs.min())-3),min(roi.shape[1]-1,int(xs.max())+3),
            max(0,int(ys.min())-3),min(roi.shape[0]-1,int(ys.max())+3)) if len(xs) else (0,roi.shape[1]-1,0,roi.shape[0]-1)
    aspect=(bounds[3]-bounds[2]+1)/(bounds[1]-bounds[0]+1)
    row_height=min(3,max(1.3,3.4*aspect+.65))
    def crop_axes(ax):
        ax.set_xlim(bounds[0]-.5,bounds[1]+.5)
        ax.set_ylim(bounds[3]+.5,bounds[2]-.5)
    shifts=np.concatenate([data['refinement_displacement'][roi&data['valid_refined'].astype(bool)] for data in arrays.values()])
    shifts=shifts[np.isfinite(shifts)]
    displacement_limit=min(4,max(.2,float(np.quantile(shifts,.99)))) if len(shifts) else 4.
    fig,axes=plt.subplots(n,4,figsize=(14,row_height*n),squeeze=False,constrained_layout=True)
    status_cmap=ListedColormap(plt.get_cmap('tab20').colors[:16])
    statuses_seen=set()
    for row,name in enumerate(names):
        data=arrays[name]
        valid=data['valid_refined'].astype(bool)&roi
        refined=data['refined_mask'].astype(bool)&valid
        state=np.zeros(roi.shape,float);state[valid]=1;state[refined]=2
        show(axes[row,0],masked(state,roi),f'{name}: valid {valid.sum()}, LSM refined {refined.sum()}',
             ListedColormap(['#aaaaaa','#4c78a8','#3ca370']),0,2)
        disp=data['refinement_displacement']
        a=show(axes[row,1],masked(disp,valid),f'Center shift [px], display max {displacement_limit:.2f}',vmin=0,vmax=displacement_limit)
        fig.colorbar(a,ax=axes[row,1],fraction=.035)
        statuses=data['refinement_status']
        statuses_seen.update(int(v) for v in np.unique(statuses[roi]))
        status_image=axes[row,2].imshow(masked(statuses,roi),cmap=status_cmap,
                          norm=BoundaryNorm(np.arange(-.5,16.5),16),interpolation='nearest')
        fig.colorbar(status_image,ax=axes[row,2],fraction=.035,ticks=sorted(int(v) for v in np.unique(statuses[roi])))
        axes[row,2].set_title('Status code (controls: code 1 means valid)',fontsize=9)
        axes[row,2].set_xticks([]);axes[row,2].set_yticks([])
        geo=valid&data['geometry_applicable_mask'].astype(bool)
        a=show(axes[row,3],masked(np.abs(data['epipolar_error_after']),geo),
               'Epipolar distance [px]; not accuracy',vmin=0,vmax=.3)
        fig.colorbar(a,ax=axes[row,3],fraction=.035)
        for ax in axes[row]:
            crop_axes(ax)
    legend=' | '.join(f'{code}: {STATUS.get(code,"unknown")}' for code in sorted(statuses_seen))
    fig.suptitle('Validity: gray=rejected/unattempted, blue=valid control/unrefined, green=accepted LSM\n'+legend,fontsize=9)
    _save(fig,out/'quality_maps.png')

    # Matched support is identical before/after within each displayed method.
    fig,axes=plt.subplots(n,3,figsize=(12,row_height*n),squeeze=False,constrained_layout=True)
    for row,name in enumerate(names):
        data=arrays[name]
        common=roi&data['valid_raw'].astype(bool)&data['valid_refined'].astype(bool)
        before=sample(target,data['uv_raw'])
        after=sample(target,data['uv_refined'])
        gain=data.get('gain',np.ones(roi.shape,np.float32))
        bias=data.get('bias',np.zeros(roi.shape,np.float32))
        values=[abs(before-reference),abs(after-reference),abs(after-gain*reference-bias)]
        titles=[f'{name}: raw |target(q0)-ref|',f'{name}: refined |target(q)-ref|',
                f'{name}: |target(q)-a*ref-b|']
        for col,(value,title) in enumerate(zip(values,titles)):
            a=show(axes[row,col],masked(value,common),title,'magma',0,.15)
            fig.colorbar(a,ax=axes[row,col],fraction=.035)
            crop_axes(axes[row,col])
    fig.suptitle('Center-sample absolute intensity error [0,1]; same raw/refined valid support in each row\nBrightness compensation is shown separately; these are diagnostics, not independent accuracy.',fontsize=10)
    _save(fig,out/'warp_errors.png')

    patch_methods=[name for name in ['A1_affine','A2_affine'] if name in arrays]
    if not patch_methods:
        patch_methods=[names[-1]]
    selected=np.logical_and.reduce([arrays[name]['valid_refined'].astype(bool)&roi for name in patch_methods])
    positions=np.argwhere(selected)
    patch_records=[]
    if len(positions):
        # Deterministic spatial quantiles, NOT best-error examples or GT filtering.
        indices=np.unique(np.rint(np.linspace(.15,.85,3)*(len(positions)-1)).astype(int))
        positions=positions[indices]
        fig,ax=plt.subplots(figsize=(7,4))
        show(ax,reference,'Patch locations: fixed spatial quantiles of common accepted support','gray',0,1)
        for i,(y,x) in enumerate(positions):
            ax.plot(x,y,'o',markerfacecolor='none',markeredgecolor='red',markersize=10)
            ax.text(x+5,y-5,str(i+1),color='red',fontsize=10)
        _save(fig,out/'patch_locations.png')
        rows=len(positions)*len(patch_methods)
        fig,axes=plt.subplots(rows,6,figsize=(15,2.3*rows),squeeze=False,constrained_layout=True)
        raw_data=arrays.get('A0',arrays[first])
        row=0
        for name in patch_methods:
            data=arrays[name]
            cfg=metas[name].get('lsm_config') or metas[name].get('config',{}).get('lsm',{})
            patch_size=int(cfg.get('patch_size',11));r=patch_size//2
            sy,sx=np.mgrid[-r:r+1,-r:r+1]
            offsets=np.stack([sx,sy],-1).astype(np.float32)
            for index,(y,x) in enumerate(positions):
                p=np.array([x,y],np.float32)
                q0=data['uv_raw'][y,x];q=data['uv_refined'][y,x]
                raw_A=raw_data.get('A',np.broadcast_to(np.eye(2),roi.shape+(2,2)))[y,x]
                A=data['A'][y,x]
                ref_patch=sample(reference,p+offsets)
                raw_patch=sample(target,q0+offsets@raw_A.T)
                refined_patch=sample(target,q+offsets@A.T)
                gain=float(data['gain'][y,x]);bias=float(data['bias'][y,x])
                corrected=(refined_patch-bias)/gain if gain>0 else np.full_like(refined_patch,np.nan)
                before=abs(raw_patch-ref_patch);after=abs(refined_patch-gain*ref_patch-bias)
                values=[ref_patch,raw_patch,refined_patch,corrected,before,after]
                titles=[f'{name} / p{index+1}: reference','Raw target, initial local A',
                        'Refined target, fitted local A','Refined (target-b)/a','Raw |target-ref|','Refined |target-a*ref-b|']
                for col,(value,title) in enumerate(zip(values,titles)):
                    show(axes[row,col],value,title,'gray' if col<4 else 'magma',0,1 if col<4 else .15)
                patch_records.append(dict(method=name,point_index=index+1,reference_xy=p.tolist(),
                    raw_target_xy=q0.tolist(),refined_target_xy=q.tolist(),gain=gain,bias=bias,
                    raw_A=raw_A.tolist(),refined_A=A.tolist()))
                row+=1
        fig.suptitle('Deterministic spatial examples; no GT and no best/worst photometric selection. Local patch models may differ.',fontsize=10)
        _save(fig,out/'patch_examples.png')
    else:
        (out/'patch_examples_unavailable.txt').write_text('No common accepted support for affine methods; no fabricated or fallback patches were shown.\n')

    truth=load_truth(metas[first],directory)
    if truth is not None:
        common=np.logical_and.reduce([data['valid_refined'].astype(bool) for data in arrays.values()])&roi&truth['valid']
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        maximum=.2
        for name,data in arrays.items():
            epe=np.linalg.norm(data['uv_refined']-truth['uv'],axis=-1)
            for row,mask in [(0,common),(1,data['valid_refined'].astype(bool)&roi&truth['valid'])]:
                values=np.sort(epe[mask&np.isfinite(epe)])
                if len(values):
                    maximum=max(maximum,float(values[-1]))
                    for col in [0,1]:
                        axes[row,col].plot(values,np.arange(1,len(values)+1)/len(values),label=f'{name} n={len(values)}',linewidth=1.3)
        raw_data=arrays.get('A0',arrays[first])
        unfiltered=roi&truth['valid']&raw_data.get('valid_model',np.isfinite(raw_data['uv_raw']).all(-1))
        raw_error=np.linalg.norm(raw_data['uv_raw']-truth['uv'],axis=-1)
        values=np.sort(raw_error[unfiltered&np.isfinite(raw_error)])
        if len(values):
            maximum=max(maximum,float(values[-1]))
            for col in [0,1]:
                axes[1,col].plot(values,np.arange(1,len(values)+1)/len(values),'--',color='black',
                                 label=f'Raw unfiltered n={len(values)}',linewidth=1.)
        for row,title in [(0,'All methods common valid support'),(1,'Each method own support (selection differs)')]:
            for col,limit in [(0,maximum*1.02),(1,.2)]:
                ax=axes[row,col]
                ax.set_title(title+(' / full range' if col==0 else ' / 0–0.2px zoom'),fontsize=9)
                ax.set_xlabel('Endpoint error [working px]');ax.set_ylabel('Fraction of valid evaluated pixels')
                ax.set_xlim(0,limit);ax.set_ylim(0,1);ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle('Independent synthetic GT; full-range and subpixel detail use identical evaluated samples',fontsize=10)
        _save(fig,out/'synthetic_epe_cdf.png')
        primary=[name for name in ['A0','A1_affine','A2_affine'] if name in arrays]
        support=np.logical_and.reduce([arrays[name]['valid_refined'].astype(bool) for name in primary])&roi&truth['valid']
        fig,axes=plt.subplots(len(primary),2,figsize=(10,3*len(primary)),squeeze=False,constrained_layout=True)
        for row,name in enumerate(primary):
            data=arrays[name]
            error=np.linalg.norm(data['uv_refined']-truth['uv'],axis=-1)
            before=np.linalg.norm(data['uv_raw']-truth['uv'],axis=-1)
            a=show(axes[row,0],masked(error,support),f'{name}: endpoint error [px], display saturates at0.2',vmin=0,vmax=.2)
            fig.colorbar(a,ax=axes[row,0],fraction=.035)
            a=show(axes[row,1],masked(error-before,support),f'{name}: EPE change vs raw; blue=improvement','coolwarm',-.05,.05)
            fig.colorbar(a,ax=axes[row,1],fraction=.035)
        fig.suptitle(f'Identical A0/A1_affine/A2_affine common valid + GT visible support: {support.sum()} pixels\nColor limits aid display; metrics retain all values. GT never used for acceptance.',fontsize=10)
        _save(fig,out/'synthetic_epe_maps.png')
    info=dict(image_source='cv2.IMREAD_GRAYSCALE PNG /255, matching experiment.py',
              pixel_convention='xy integer pixel centers; torch bilinear align_corners=False',
              invalid_samples='Explicitly masked; no padding or interpolation of failed refinement is shown as valid',
              map_display={'crop_xyxy':[bounds[0],bounds[2],bounds[1],bounds[3]],
                           'displacement_color_max_px':displacement_limit,
                           'displacement_color_policy':'Pair-wide99thpercentile of valid corrections, clipped to[0.2,4]; outliers saturate display only, metrics unchanged'},
              patch_selection='Three deterministic spatial quantiles of common accepted affine-method support; no GT or residual-based selection',
              patch_records=patch_records,
              fb_interpretation='Saved forward-backward diagnostics query the same fixed RAW reverse RoMa map; no co-adapted refined reverse field',
              artifacts=[str(p.relative_to(directory)) for p in sorted(out.glob('*.png'))])
    (out/'visualization.json').write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')
    return info


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    print(json.dumps(visualize_pair(args.directory),indent=2))
