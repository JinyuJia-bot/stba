"""Real frontend coverage only; manual seed boxes are labels for reporting, not GT."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize_frontend(directory):
    directory=Path(directory)
    meta=json.loads((directory/'observations.json').read_text())
    with np.load(directory/'observations.npz') as archive:
        data={key:archive[key] for key in archive.files}
    p=data['seed_uv'];visible=data['visible'];temporal=data['status']==1
    # Boxes refer ONLY to manually viewed reference07 frame230. They are broad
    # diagnostic candidates, can include background and overlap, and never select
    # estimator inputs. No physical identity or segmentation GT is available.
    boxes={'all_candidate_roi':None,'dog_candidate_seed_box':[207,145,241,213],
           'person_candidate_seed_box':[231,50,303,236]}
    displacement=np.linalg.norm(data['uv'][:,0]-p[:,None],axis=-1)
    motion=np.max(np.where(visible[:,0],displacement,-np.inf),axis=1)
    ref_duration=visible[:,0].sum(1)
    result={'ground_truth_available':False,'reference_camera_id':meta['camera_ids'][0],
            'stride':meta['config']['stride'],'query_count':len(p),
            'region_caution':'Manual broad diagnostic boxes at reference frame230; contain possible background, can overlap; never used for filtering/optimization. Counts do not certify correct physical identities.',
            'motion_diagnostic':'Reference image displacement >2px from seed and >=30 accepted observations; not a proof of beta observability.',
            'regions':{},'per_camera':{},'camera_edge_status':{},
            'stage_counts':{'query_grid':len(p),'reference_temporal_success':int(data['temporal_success'].sum()),
                'cross_view_two_or_more_candidates':int(data['multiview_seed'].sum()),
                'cross_view_all_three_candidates':int(data['all_three_seed'].sum()),
                'reference_ge30_and_displacement_gt2':int(((ref_duration>=30)&(motion>2)).sum()),
                'all_three_ge30_and_displacement_gt2':int((data['all_three_seed']&(ref_duration>=30)&(motion>2)).sum())}}
    for name,box in boxes.items():
        region=np.ones(len(p),bool) if box is None else ((p[:,0]>=box[0])&(p[:,0]<box[2])&(p[:,1]>=box[1])&(p[:,1]<box[3]))
        result['regions'][name]={'box_xyxy':box,'queries':int(region.sum()),
            'temporal_success':int((region&data['temporal_success']).sum()),
            'ge30frames':int((region&(ref_duration>=30)).sum()),
            'cross_view_two_or_more':int((region&data['multiview_seed']).sum()),
            'cross_view_all_three':int((region&data['all_three_seed']).sum()),
            'all_three_ge30_and_motion_gt2':int((region&data['all_three_seed']&(ref_duration>=30)&(motion>2)).sum())}
    for j,cid in enumerate(meta['camera_ids']):
        durations=temporal[:,j].sum(1)
        result['per_camera'][str(cid)]={'original_temporal_observations':int(temporal[:,j].sum()),
            'observations_after_cross_view_association':int(visible[:,j].sum()),
            'tracks_ge30_before_association':int((durations>=30).sum()),
            'duration_quantiles_frames':np.quantile(durations,[.1,.5,.9]).tolist(),
            'status_observation_counts':{str(int(k)):int((data['status'][:,j]==k).sum()) for k in np.unique(data['status'][:,j])}}
        if j:
            result['camera_edge_status'][str(cid)]={str(int(k)):int((data['edge_status'][:,j]==k).sum()) for k in np.unique(data['edge_status'][:,j])}
    (directory/'frontend_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    args=parser.parse_args();print(json.dumps(summarize_frontend(args.directory),indent=2))
