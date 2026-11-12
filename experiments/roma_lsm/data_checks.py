"""Independent prepared-data/adapter checks, without network inference or GT filtering."""
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from .coordinates import resize_pixel
    from .prepare_data import working_to_raw_distorted
    from .roma_adapter import adapter_self_test
except ImportError:
    from coordinates import resize_pixel
    from prepare_data import working_to_raw_distorted
    from roma_adapter import adapter_self_test


def check(manifest_path='experiments/roma_lsm/data_manifest.json'):
    manifest=json.loads(Path(manifest_path).read_text())
    result={'adapter':adapter_self_test(),'cameras':[]}
    points=np.array([[10,10],[100,100],[239.5,134.5],[470,260],[450,200]],np.float64)
    for cam in manifest['cameras']:
        raw=working_to_raw_distorted(points,cam)
        # Invert original distortion independently with OpenCV, targeting work K.
        inverse=cv2.undistortPoints(raw[:,None,:],np.asarray(cam['raw_K']),
                                    np.asarray(cam['raw_D']),P=np.asarray(cam['K'])).reshape(-1,2)
        roundtrip=float(np.max(np.abs(inverse-points)))
        assert roundtrip<.002, (cam['camera_id'],roundtrip)
        source=resize_pixel(raw,cam['raw_size'],cam['source_size'])
        inverse_half=cv2.undistortPoints(source[:,None,:],np.asarray(cam['source_K']),
                                        np.asarray(cam['raw_D']),P=np.asarray(cam['K'])).reshape(-1,2)
        np.testing.assert_allclose(inverse_half,inverse,atol=1e-9,rtol=0)
        K=np.asarray(cam['K']); raw_K=np.asarray(cam['raw_K'])
        xyz=np.array([[.1,.2,3.],[-1.,.2,5.]])
        raw_ideal=(xyz@raw_K.T);raw_ideal=raw_ideal[:,:2]/raw_ideal[:,2:]
        work=xyz@K.T;work=work[:,:2]/work[:,2:]
        np.testing.assert_allclose(work,resize_pixel(raw_ideal,cam['raw_size'],cam['size']),atol=1e-10)
        mask=cv2.imread(cam['static_mask_path'],0)>0
        valid=cv2.imread(cam['valid_mask_path'],0)>0
        assert np.all(valid[mask])
        R=np.asarray(cam['R']);t=np.asarray(cam['t'])
        np.testing.assert_allclose(R@R.T,np.eye(3),atol=1e-12)
        np.testing.assert_allclose(-R.T@t,cam['C_world'],atol=1e-12)
        result['cameras'].append({'camera_id':cam['camera_id'],
                                  'raw_distortion_roundtrip_max_px':roundtrip,
                                  'static_pixels':int(mask.sum())})
    result['passed']=True
    return result


if __name__=='__main__':
    result=check()
    Path('experiments/roma_lsm/data_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
