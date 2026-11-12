"""Replay stage-two invariant checks and draw a compact common-support summary."""
from pathlib import Path
import os
import json
import hashlib
import numpy as np
import torch
from multiview import MultiViewConfig, WEIGHT_MODES, frozen_view_weights
from multiview_experiment import _load_npz, evaluate_group
from experiment import ROOT, read_mask
from geometry import world_rays, camera_center, project
from lsm import bilinear, patch_offsets


def patch_mask_checks(mask,centers,A,patch_size):
    offset=patch_offsets(patch_size,'cpu',torch.float32)
    mask=torch.as_tensor(mask,dtype=torch.float32)
    failed=0
    for start in range(0,len(centers),512):
        center=torch.as_tensor(centers[start:start+512],dtype=torch.float32)
        affine=torch.as_tensor(A[start:start+512],dtype=torch.float32)
        xy=center[:,None]+torch.einsum('bij,pj->bpi',affine,offset)
        value,inside=bilinear(mask,xy)
        failed+=int((~(inside&(value>=.999)).all(-1)).sum())
    return failed


def audit_group(directory):
    directory=Path(directory);meta=json.loads((directory/'meta.json').read_text())
    cfg=MultiViewConfig(**meta['config'])
    data={arm:_load_npz(directory/f'{arm}.npz') for arm in WEIGHT_MODES}
    source=[_load_npz(path) for path in meta['source_files']]
    source_meta=[json.loads((Path(path).parent/'meta.json').read_text()) for path in meta['source_files']]
    cameras=[meta['reference_camera'],*meta['target_cameras']]
    for j,item in enumerate(source_meta):
        assert cameras[0]==item['cameras']['ref'] and cameras[j+1]==item['cameras']['target']
    calibration=json.dumps([{key:cam.get(key) for key in ['K','R','t','D','size','coordinate_system']} for cam in cameras],sort_keys=True)
    first=data['uniform'];eligible=first['attempt_mask'];height,width=eligible.shape
    yy,xx=np.mgrid[:height,:width];grid=np.stack([xx,yy],-1).astype(np.float32)
    masks=[]
    for cam in cameras:
        size=cam['size'];shape=(size[1],size[0]);mask=read_mask(cam,'valid_mask_path',shape)
        if meta['mode']=='asynchronous_unknown':
            assert 'static_mask_path' in cam
            mask &= read_mask(cam,'static_mask_path',shape)
        masks.append(mask)
    source_eligible=first['reference_roi'].copy()
    for d in source:
        source_eligible &= d['valid_refined']&d['geometry_applicable_mask']
        source_eligible &= np.isfinite(d['uv_refined']).all(-1)&np.isfinite(d['A']).all((-1,-2))
    assert np.array_equal(eligible,source_eligible)
    expected_A=np.stack([d['A'][eligible] for d in source],1)
    expected_q=np.stack([d['uv_refined'][eligible] for d in source],1)
    quality=dict(confidence=first['confidence_raw'][eligible],fb_error=first['fb_error_initial'][eligible],
                 photo_error=first['photo_quality_initial'][eligible])
    common=np.logical_and.reduce([d['valid_refined'] for d in data.values()])
    record=dict(all_passed=True,group=meta['name'],scene_mode=meta['mode'],
        source_calibrations_exactly_equal=True,calibration_sha256=hashlib.sha256(calibration.encode()).hexdigest(),
        eligibility_exactly_joint_A1_affine_and_geometry=True,eligible_pixels=int(eligible.sum()),
        common_four_arms_pixels=int(common.sum()),cpu_patch_replay_threshold=.999,
        depth_initial_bitwise_identical_all_arms=True,weights_frozen_recomputed=True,
        no_ground_truth_used_in_acceptance_or_weights=True,arms={})
    for arm,d in data.items():
        success=d['valid_refined'];s=int(success.sum())
        assert d['depth_initial'].tobytes()==first['depth_initial'].tobytes()
        assert np.array_equal(d['attempt_mask'],eligible)
        assert np.array_equal(d['refined_mask'],success)
        assert np.all(d['refinement_status'][success]==1)
        assert not np.any(success&~eligible)
        assert np.array_equal(d['A_fixed'][eligible],expected_A)
        assert np.array_equal(d['uv_stage1'][eligible],expected_q)
        assert np.all(np.isfinite(d['depth'][success])&(d['depth'][success]>0))
        scores=frozen_view_weights(cameras[0],cameras[1:],quality,arm,cfg)
        np.testing.assert_allclose(d['weights_frozen'][eligible],scores['weights'],atol=0,rtol=0)
        np.testing.assert_allclose(d['weights_frozen'][eligible].sum(-1),2,atol=1e-12,rtol=0)
        assert np.all(d['weights_frozen'][eligible]>0)
        assert meta['methods'][arm]['timing']['weights_frozen'] is True
        p=grid[success];rays=world_rays(cameras[0],p)
        xyz=camera_center(cameras[0])+d['depth'][success][:,None]*rays
        projection_error=[];positive_target=[];patch_fail=[]
        patch_fail.append(patch_mask_checks(masks[0],p,np.tile(np.eye(2),(s,1,1)),cfg.patch_size))
        for j,cam in enumerate(cameras[1:]):
            uv,z=project(cam,xyz,return_depth=True)
            positive_target.append(bool(np.all(z>0)))
            error=np.linalg.norm(uv-d['uv_refined'][success][:,j],axis=-1)
            projection_error.append(float(error.max()))
            assert positive_target[-1] and projection_error[-1]<5e-4
            patch_fail.append(patch_mask_checks(masks[j+1],d['uv_refined'][success][:,j],d['A_fixed'][success][:,j],cfg.patch_size))
        assert sum(patch_fail)==0,patch_fail
        delta=np.abs(d['depth'][common]-first['depth'][common])
        record['arms'][arm]=dict(success_pixels=s,positive_ray_depth=True,positive_target_depth=positive_target,
            patch_mask_failed_counts=patch_fail,maximum_projection_replay_error_px=projection_error,
            affine_exactly_fixed_from_stage1=True,mean_frozen_weights=d['weights_frozen'][eligible].mean(0).tolist(),
            mean_absolute_depth_change_vs_uniform_on_common=float(delta.mean()),
            maximum_absolute_depth_change_vs_uniform_on_common=float(delta.max()),
            optimization_seconds=meta['methods'][arm]['timing']['seconds'],
            peak_allocated_MB=meta['methods'][arm]['timing']['peak_cuda_allocated_bytes']/1e6)
    (directory/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def plot_summary(root):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/stba_roma_lsm_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=json.loads((root/'synthetic_affine_brightness'/'metrics.json').read_text())
    real_metrics=json.loads((root/'real_7_9_10'/'metrics.json').read_text())
    arms=list(WEIGHT_MODES);labels=['Uniform','Quality','Distance','Quality × distance']
    scopes=[metrics['methods'][arm]['scopes']['common'] for arm in arms]
    depths=[s['depth_error_calibration_units']['rmse']*1000 for s in scopes]
    initial=scopes[0]['initial_depth_error_same_pixels']['rmse']*1000
    colors=['#567a9d','#dd9856','#4c9871','#8e6eab']
    fig,axes=plt.subplots(2,2,figsize=(12.5,8),layout='constrained')
    x=np.arange(4)
    ax=axes[0,0];ax.bar(x,depths,color=colors);ax.axhline(initial,color='black',ls='--',label=f'Shared initialization: {initial:.3f} mm')
    for i,v in enumerate(depths):ax.text(i,v+.035,f'{v:.3f}',ha='center',fontsize=10)
    ax.set(xticks=x,xticklabels=labels,ylabel='Ray-range RMSE (mm)',ylim=(0,3.8),title=f"Synthetic: common {metrics['common_four_arms_pixels']:,} pixels")
    ax.legend(fontsize=9,loc='upper right')
    ax=axes[0,1]
    for j,cam in enumerate(['1','2']):
        errors=[s['views'][cam]['endpoint_error_px']['mean'] for s in scopes]
        baseline=scopes[0]['views'][cam]['projected_initial_endpoint_error_same_pixels_px']['mean']
        ax.plot(x,errors,'o-',label=f'Target {cam} refined')
        ax.axhline(baseline,ls='--',color=['C0','C1'][j],label=f'Target {cam} shared initialization')
    ax.set(xticks=x,xticklabels=labels,ylabel='Mean EPE (working pixels)',title='Synthetic: projected seed is the fair baseline')
    ax.legend(fontsize=8)
    ax=axes[1,0]
    for j,cam in enumerate(['9','10']):
        values=[real_metrics['methods'][arm]['scopes']['common']['views'][cam]['forward_backward_error_after']['mean'] for arm in arms]
        before=real_metrics['methods']['uniform']['scopes']['common']['views'][cam]['forward_backward_error_before']['mean']
        ax.plot(x,values,'o-',label=f'Target {cam}: after')
        ax.axhline(before,ls='--',color=['C0','C1'][j],label=f'Target {cam}: A1 before')
    ax.set(xticks=x,xticklabels=labels,ylabel='FB consistency to fixed raw reverse (px)',title=f"Real static ROI: common {real_metrics['common_four_arms_pixels']:,} pixels; no GT")
    ax.legend(fontsize=8)
    ax=axes[1,1]
    for group,label,color in [('synthetic_affine_brightness','Synthetic','#4c9871'),('real_7_9_10','Real static ROI','#567a9d')]:
        m=json.loads((root/group/'metrics.json').read_text())
        coverage=[100*m['methods'][arm]['scopes']['own']['coverage_of_reference_roi'] for arm in arms]
        ax.plot(x,coverage,'o-',color=color,label=label)
    ax.set(xticks=x,xticklabels=labels,ylabel='Own valid / original reference ROI (%)',ylim=(0,100),title='Coverage includes pairwise visibility/quality losses')
    ax.legend(fontsize=9)
    for ax in axes.ravel():
        ax.grid(axis='y',alpha=.2);ax.tick_params(axis='x',labelrotation=12)
    fig.suptitle('Four frozen view weights, one shared positive depth, fixed camera calibration',fontsize=14)
    fig.savefig(root/'multiview_comparison.png',dpi=180)
    plt.close(fig)


def main():
    torch.set_num_threads(2)
    root=ROOT/'results/multiview';records={}
    for name in ['synthetic_affine_brightness','real_7_9_10']:
        evaluate_group(root/name)
        records[name]=audit_group(root/name)
    output=dict(all_passed=True,groups=records,interpretation='No GT is used by this invariant audit; GT is read only by separate metric evaluation and plotting.')
    (ROOT/'multiview_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    plot_summary(root)
    print(json.dumps({name:dict(all_passed=value['all_passed'],eligible=value['eligible_pixels'],common=value['common_four_arms_pixels']) for name,value in records.items()}))


if __name__=='__main__':main()
