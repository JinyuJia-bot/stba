"""Read-only dense overlap evidence for the two baseline-selected real views."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

try:
    from .coordinates import pixel_grid, pixel_to_normalized
    from .geometry import fundamental, epipolar_residual
except ImportError:
    from coordinates import pixel_grid, pixel_to_normalized
    from geometry import fundamental, epipolar_residual


def sample(field,uv,size):
    tensor=torch.as_tensor(field.astype(np.float32))
    if tensor.ndim==2:
        tensor=tensor[...,None]
    tensor=tensor.permute(2,0,1)[None]
    grid=torch.as_tensor(pixel_to_normalized(uv,size).astype(np.float32))[None]
    return functional.grid_sample(tensor,grid,align_corners=False,mode='bilinear',
                                  padding_mode='zeros')[0].permute(1,2,0).numpy()


def run():
    torch.set_num_threads(4)
    base=Path('experiments/roma_lsm')
    manifest=json.loads((base/'data_manifest.json').read_text())
    cams={c['camera_id']:c for c in manifest['cameras']}
    records=[]
    for pair in manifest['pairs']:
        a,b=pair['reference_id'],pair['target_id']
        ca,cb=cams[a],cams[b]
        cache=np.load(base/f'cache/pair_{a}_{b}.npz')
        p=pixel_grid(ca['size']);q=cache['uv_AB']
        masks=[]
        for c in [ca,cb]:
            mask=cv2.imread(c['static_mask_path'],0)>0
            mask=cv2.erode(mask.astype(np.uint8),np.ones((11,11),np.uint8))>0
            masks.append(mask)
        target_support=sample(masks[1],q,cb['size'])[...,0]>=1-1e-6
        reverse_support=sample(cache['valid_BA'],q,cb['size'])[...,0]>=1-1e-6
        support=masks[0]&target_support&reverse_support&cache['valid_AB']
        fb=np.linalg.norm(sample(cache['uv_BA'],q,cb['size'])-p,axis=-1)
        epi=np.abs(epipolar_residual(fundamental(ca,cb),p,q))
        conf=cache['confidence_AB']
        record=dict(reference_id=a,target_id=b,baseline=pair['baseline'],
                    reference_eroded_static_pixels=int(masks[0].sum()),
                    both_eroded_static_and_dense_valid_pixels=int(support.sum()),
                    reference_static_coverage=float(support.sum()/masks[0].sum()),
                    confidence_median=float(np.median(conf[support])),
                    FB_median_px=float(np.median(fb[support])),FB_P90_px=float(np.quantile(fb[support],.9)),
                    epipolar_median_px=float(np.median(epi[support])),
                    epipolar_P90_px=float(np.quantile(epi[support],.9)),
                    confidence_gt_02_and_FB_lt_2_pixels=int((support&(conf>.2)&(fb<2)).sum()))
        records.append(record)
    result={'purpose':'Evidence of actual static co-visibility after baseline/orientation/visual selection. Consistency only: no real GT, no accuracy claim.',
            'mask_support':'Both source and target manually inspected structural-floor masks eroded by5pixels; all bilinear contributors and both dense directions valid.',
            'sampler':'PyTorch bilinear grid_sample, align_corners=False, integer pixel centers.',
            'pairs':records}
    (base/'data_overlap_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    run()
