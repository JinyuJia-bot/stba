"""Dense pair evaluation; ground truth is read here only, never used for filtering.

The original GT-evaluable denominator is reference_roi & valid_gt, independent of
all five methods. Common support and each method's own support are both retained.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHODS=['A0','A0g','A1_translation','A1_affine','A2_affine']
PRIMARY=['A0','A1_affine','A2_affine']
STATUS={0:'not_attempted',1:'success',2:'nonfinite_input',3:'reference_boundary',
        4:'target_boundary',5:'low_texture',6:'ill_conditioned',7:'affine_invalid',
        8:'no_convergence',9:'correction_limit',10:'photo_high',11:'low_confidence',
        12:'fb_before',13:'fb_after',14:'static_mask_boundary',15:'geometry_degenerate'}


def resolve_path(path, directory):
    path=Path(path)
    if path.is_absolute() or path.exists():
        return path
    for prefix in [Path(directory),Path(__file__).resolve().parents[2]]:
        candidate=prefix/path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def load_pair(directory):
    directory=Path(directory)
    common=json.loads((directory/'meta.json').read_text()) if (directory/'meta.json').exists() else {}
    arrays,metadata={},{}
    for name in METHODS:
        path=directory/f'{name}.npz'
        if not path.exists():
            continue
        with np.load(path,allow_pickle=False) as archive:
            arrays[name]={k:archive[k] for k in archive.files}
        meta={k:v for k,v in common.items() if k!='methods'}
        method_meta=common.get('methods',{}).get(name,{})
        if isinstance(method_meta,dict):
            meta.update(method_meta)
        for candidate in [directory/f'{name}.json',directory/f'{name}_meta.json',directory/name/'meta.json']:
            if candidate.exists():
                meta.update(json.loads(candidate.read_text()))
        if 'metadata_json' in arrays[name]:
            embedded=json.loads(str(arrays[name]['metadata_json'].item()))
            meta={**embedded,**meta}
        metadata[name]=meta
    if not arrays:
        raise ValueError(f'No method NPZ files in {directory}')
    return arrays,metadata


def camera_pair(meta):
    cameras=meta.get('cameras')
    if isinstance(cameras,list) and len(cameras)==2:
        return cameras
    if isinstance(cameras,dict):
        ref=cameras.get('ref',cameras.get('reference'))
        target=cameras.get('target',cameras.get('tgt'))
        if ref is not None and target is not None:
            return ref,target
    if 'reference_camera' in meta and 'target_camera' in meta:
        return meta['reference_camera'],meta['target_camera']
    raise ValueError('Metadata requires cameras={ref:camera,target:camera} or a two-camera list')


def load_truth(meta,directory):
    """GT is explicit opt-in via dataset_metadata_path; real data returns None."""
    metadata_path=meta.get('dataset_metadata_path')
    if metadata_path is None:
        return None
    path=resolve_path(metadata_path,directory)
    dataset=json.loads(path.read_text())
    array_path=resolve_path(dataset.get('array_path','arrays.npz'),path.parent)
    ref,target=camera_pair(meta)
    key=f"{ref['camera_id']}{target['camera_id']}"
    with np.load(array_path,allow_pickle=False) as arrays:
        uv=arrays[f'uv_gt_{key}'].copy()
        valid=arrays[f'valid_gt_{key}'].astype(bool)
        regions=arrays['region_labels'].copy() if 'region_labels' in arrays else None
    if not np.isfinite(uv[valid]).all():
        raise ValueError('GT marks non-finite correspondences valid')
    return dict(uv=uv,valid=valid,regions=regions,region_names=dataset.get('region_labels',{}),
                source=str(path),case=dataset.get('case'))


def summary(values,mask):
    values=np.asarray(values)
    selected=values[np.asarray(mask,dtype=bool)]
    selected=selected[np.isfinite(selected)]
    if not len(selected):
        return dict(count=0,mean=None,median=None,p90=None)
    return dict(count=int(len(selected)),mean=float(selected.mean()),
                median=float(np.median(selected)),p90=float(np.quantile(selected,.9)))


def ratio(count,denominator):
    return float(count/denominator) if denominator else None


def epe_stats(error,mask,denominator):
    mask=mask&np.isfinite(error)
    result=summary(error,mask)
    n=result['count']
    result['valid_coverage_of_original_evaluable_roi']=ratio(n,denominator)
    for threshold in [.5,1.,2.]:
        tag=str(threshold).rstrip('0').rstrip('.').replace('.','p')
        correct=int((mask&(error<threshold)).sum())
        result[f'fraction_lt_{tag}_px']=ratio(correct,n)
        result[f'correct_count_lt_{tag}_px']=correct
        result[f'correct_coverage_lt_{tag}_px']=ratio(correct,denominator)
    return result


def diagnostics(data,mask):
    result={}
    geometry=np.asarray(data.get('geometry_applicable_mask',np.zeros(mask.shape,bool)),bool)
    for name in ['photo_error_before','photo_error_after','forward_backward_error_before',
                 'forward_backward_error_after','epipolar_error_before','epipolar_error_after',
                 'refinement_displacement']:
        if name in data:
            use=mask&geometry if name.startswith('epipolar') else mask
            result[name]=summary(np.abs(data[name]),use)
    return result


def evaluate_pair(directory,truth=None):
    directory=Path(directory)
    arrays,metas=load_pair(directory)
    first=next(iter(arrays))
    roi=np.asarray(arrays[first]['reference_roi'],bool)
    shape=roi.shape
    if truth is None:
        truth=load_truth(metas[first],directory)
    if truth is not None and (truth['valid'].shape!=shape or truth['uv'].shape!=shape+(2,)):
        raise ValueError('GT and working result dimensions disagree')
    valid={}
    for name,data in arrays.items():
        if not np.array_equal(data['reference_roi'],roi):
            raise ValueError('Methods must share reference_roi for a fixed evaluation denominator')
        uv=data['uv_refined']
        if uv.shape!=shape+(2,):
            raise ValueError(f'{name}: unexpected correspondence shape')
        valid[name]=roi&data['valid_refined'].astype(bool)&np.isfinite(uv).all(-1)
    common=np.logical_and.reduce(list(valid.values()))
    primary_names=[name for name in PRIMARY if name in valid]
    common_primary=np.logical_and.reduce([valid[name] for name in primary_names]) if primary_names else np.zeros(shape,bool)
    gt_region=roi&truth['valid'] if truth is not None else None
    denominator=int(gt_region.sum()) if truth is not None else None
    output=dict(schema_version=1,directory=str(directory),methods_present=list(arrays),
                methods_missing=[n for n in METHODS if n not in arrays],
                ground_truth_available=truth is not None,
                interpretation='Synthetic EPE is independent accuracy evidence. Real data without GT reports consistency and diagnostic values only; optimized geometry/photo residuals do not prove true localization accuracy.',
                coordinate_system='Working-image integer pixel centers, x/y; all endpoint errors in working pixels.',
                reference_roi_pixels=int(roi.sum()),
                original_gt_evaluable_roi_pixels=denominator,
                original_gt_evaluable_roi_definition='reference_roi & valid_gt; independent of method output validity and quality filtering',
                common_all={'methods':list(arrays),'pixels':int(common.sum())},
                common_primary={'methods':primary_names,'pixels':int(common_primary.sum())},
                worsening_epsilon_px=1e-6,
                shared_screening_seconds=metas[first].get('screening_seconds'),
                forward_backward_interpretation=metas[first].get('fb_reference'),methods={})
    cache_path=metas[first].get('cache_path')
    if cache_path:
        network_meta=resolve_path(cache_path,directory).with_suffix('.json')
        if network_meta.exists():
            network=json.loads(network_meta.read_text())
            output['cached_network_measurement']={key:network.get(key) for key in [
                'model_load_seconds','inference_seconds','device_to_host_seconds','decode_seconds',
                'save_seconds','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','provenance_path']}
            output['cached_network_measurement']['source']=str(network_meta)
    if truth is not None:
        output['ground_truth_source']=truth.get('source','explicit caller argument')
    rows=[]
    raw_data=arrays.get('A0',arrays[first])
    unscreened=roi&np.asarray(raw_data.get('valid_model',np.isfinite(raw_data['uv_raw']).all(-1)),bool)&np.isfinite(raw_data['uv_raw']).all(-1)
    output['raw_baseline_definitions']={
        'A0':'Raw coordinates after shared confidence/FB/target-patch screening; not all raw model predictions.',
        'unfiltered_raw_model':'Raw coordinates on reference_roi & valid_model, before confidence/FB/patch screening. GT evaluation additionally intersects valid_gt.'}
    output['unfiltered_raw_model']=dict(valid_pixels=int(unscreened.sum()),
        valid_coverage_of_reference_roi=ratio(int(unscreened.sum()),int(roi.sum())))
    raw_row=dict(method='A0_unfiltered_model',scope='own',valid_pixels=int(unscreened.sum()),
                 coverage_of_reference_roi=output['unfiltered_raw_model']['valid_coverage_of_reference_roi'],
                 attempted_pixels=0,successful_refinement_pixels=0)
    if truth is not None:
        raw_error=np.linalg.norm(raw_data['uv_raw']-truth['uv'],axis=-1)
        unfiltered_epe=epe_stats(raw_error,unscreened & gt_region,denominator)
        output['unfiltered_raw_model']['endpoint_error_px']=unfiltered_epe
        raw_row.update({'epe_'+key:value for key,value in unfiltered_epe.items()})
    rows.append(raw_row)
    for name,data in arrays.items():
        method_valid=valid[name]
        attempt=roi&np.asarray(data.get('attempt_mask',np.zeros(shape,bool)),bool)
        refined=roi&np.asarray(data.get('refined_mask',np.zeros(shape,bool)),bool)
        successful=refined&method_valid&attempt
        inconsistent=refined&~method_valid
        if inconsistent.any():
            raise ValueError(f'{name}: refined_mask includes {inconsistent.sum()} invalid outputs')
        status=data.get('refinement_status',np.zeros(shape,np.uint8))
        raw_valid=roi&np.asarray(data['valid_raw'],bool)&np.isfinite(data['uv_raw']).all(-1)
        record=dict(reference_roi_pixels=int(roi.sum()),attempted_pixels=int(attempt.sum()),
                    successful_refinement_pixels=int(successful.sum()),
                    failed_attempted_pixels=int((attempt&~successful).sum()),
                    unattempted_roi_pixels=int((roi&~attempt).sum()),
                    success_fraction_of_attempted=ratio(int(successful.sum()),int(attempt.sum())),
                    failure_fraction_of_attempted=ratio(int((attempt&~successful).sum()),int(attempt.sum())),
                    raw_valid_pixels=int(raw_valid.sum()),valid_pixels=int(method_valid.sum()),
                    valid_coverage_of_reference_roi=ratio(int(method_valid.sum()),int(roi.sum())),
                    attempt_coverage_of_reference_roi=ratio(int(attempt.sum()),int(roi.sum())),
                    valid_but_unrefined_pixels=int((method_valid&~refined).sum()),
                    status_counts_in_reference_roi={f'{int(k)}:{STATUS.get(int(k),"unknown")}':int((roi&(status==k)).sum()) for k in np.unique(status[roi])},
                    status_counts_among_attempted={f'{int(k)}:{STATUS.get(int(k),"unknown")}':int((attempt&(status==k)).sum()) for k in np.unique(status[attempt])},
                    timing=metas[name].get('timing',metas[name].get('timings',{})),scopes={})
        error=np.linalg.norm(data['uv_refined']-truth['uv'],axis=-1) if truth is not None else None
        raw_error=np.linalg.norm(data['uv_raw']-truth['uv'],axis=-1) if truth is not None else None
        for scope,support in [('own',method_valid),('common_all',common),('common_primary',common_primary)]:
            if scope=='common_primary' and name not in primary_names:
                continue
            use=support&method_valid
            stats=dict(valid_pixels=int(use.sum()),
                       coverage_of_reference_roi=ratio(int(use.sum()),int(roi.sum())),
                       diagnostics=diagnostics(data,use))
            row=dict(method=name,scope=scope,valid_pixels=int(use.sum()),
                     coverage_of_reference_roi=stats['coverage_of_reference_roi'],
                     attempted_pixels=record['attempted_pixels'],successful_refinement_pixels=record['successful_refinement_pixels'])
            if truth is not None:
                evaluable=use & gt_region
                stats['endpoint_error_px']=epe_stats(error,evaluable,denominator)
                paired=evaluable&raw_valid&np.isfinite(raw_error)
                stats['raw_endpoint_error_on_same_pixels_px']=epe_stats(raw_error,paired,denominator)
                worsening=paired&(error>raw_error+1e-6)
                improving=paired&(error<raw_error-1e-6)
                stats['paired_comparison_to_raw']=dict(pixels=int(paired.sum()),
                    worsened_pixels=int(worsening.sum()),improved_pixels=int(improving.sum()),
                    unchanged_pixels=int((paired&~worsening&~improving).sum()),
                    worsened_fraction=ratio(int(worsening.sum()),int(paired.sum())),
                    improved_fraction=ratio(int(improving.sum()),int(paired.sum())),
                    mean_epe_change_px=float((error-raw_error)[paired].mean()) if paired.any() else None)
                row.update({'epe_'+key:value for key,value in stats['endpoint_error_px'].items()})
                row['worsened_fraction']=stats['paired_comparison_to_raw']['worsened_fraction']
            for key,value in stats['diagnostics'].items():
                row.update({key+'_'+stat:val for stat,val in value.items()})
            rows.append(row)
            record['scopes'][scope]=stats
        if truth is not None:
            refined_paired=successful & gt_region & raw_valid & np.isfinite(raw_error)
            record['accepted_refinement_worsened_vs_raw']=dict(pixels=int(refined_paired.sum()),
                worsened_pixels=int((refined_paired&(error>raw_error+1e-6)).sum()),
                worsened_fraction=ratio(int((refined_paired&(error>raw_error+1e-6)).sum()),int(refined_paired.sum())))
            if truth.get('regions') is not None:
                record['regions']={}
                for label in np.unique(truth['regions'][roi]):
                    rmask=roi&(truth['regions']==label)
                    rden=int((rmask&truth['valid']).sum())
                    ruse=rmask&method_valid&truth['valid']
                    record['regions'][str(int(label))]=dict(name=truth.get('region_names',{}).get(str(int(label)),str(int(label))),
                        original_evaluable_pixels=rden,endpoint_error_px=epe_stats(error,ruse,rden),
                        raw_on_same_pixels=summary(raw_error,ruse&raw_valid),
                        attempted_pixels=int((attempt&rmask).sum()),successful_pixels=int((successful&rmask).sum()))
        output['methods'][name]=record
    (directory/'metrics.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with (directory/'metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    result=evaluate_pair(args.directory)
    print(json.dumps({'directory':str(args.directory),'methods':result['methods_present'],
                      'ground_truth_available':result['ground_truth_available'],
                      'reference_roi_pixels':result['reference_roi_pixels'],
                      'common_all':result['common_all'],'common_primary':result['common_primary']},indent=2))
