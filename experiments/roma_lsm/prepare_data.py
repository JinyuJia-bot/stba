"""Prepare a small fixed-calibration, conservatively static Corgi experiment.

Run from the repository root. No camera, offset, or geometry is estimated.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from .coordinates import resize_K, pixel_grid
except ImportError:
    from coordinates import resize_K, pixel_grid


POLYGONS = {
    7: [[[15, 176], [187, 176], [192, 260], [15, 260]],
        [[330, 174], [465, 174], [465, 260], [330, 260]]],
    9: [[[15, 173], [172, 173], [180, 260], [15, 260]],
        [[312, 173], [465, 173], [465, 260], [312, 260]]],
    10: [[[15, 205], [140, 205], [148, 260], [15, 260]],
         [[303, 175], [465, 175], [465, 260], [303, 260]]],
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def working_to_raw_distorted(uv, camera):
    """Working ideal pixels -> original 3840x2160 distorted integer-center pixels."""
    uv = np.asarray(uv, np.float64)
    homogeneous = np.concatenate([uv, np.ones_like(uv[..., :1])], axis=-1)
    rays = homogeneous @ np.linalg.inv(np.asarray(camera['K'])).T
    result, _ = cv2.projectPoints(rays.reshape(-1, 3), np.zeros(3), np.zeros(3),
                                 np.asarray(camera['raw_K']), np.asarray(camera['raw_D']))
    return result.reshape(uv.shape)


def prepare(root, out):
    out = Path(out)
    data = out / 'data'
    data.mkdir(parents=True, exist_ok=True)
    intri_path, extri_path = root/'optimized/intri.yml', root/'optimized/extri.yml'
    intri = cv2.FileStorage(str(intri_path), cv2.FILE_STORAGE_READ)
    extri = cv2.FileStorage(str(extri_path), cv2.FILE_STORAGE_READ)
    all_cameras = {}
    for i in range(24):
        name = f'{i:04d}'
        K, D = intri.getNode(f'K_{name}').mat(), intri.getNode(f'D_{name}').mat()
        R = cv2.Rodrigues(extri.getNode(f'R_{name}').mat())[0]
        t = extri.getNode(f'T_{name}').mat().reshape(3)
        all_cameras[i] = (K, D, R, t)
    cameras, previews = [], []
    for camera_id in [7, 9, 10]:
        raw_K, D, R, t = all_cameras[camera_id]
        path = root/f'videos/cam{camera_id:02d}/images_2/0230.png'
        raw = cv2.imread(str(path))
        if raw is None or raw.shape[:2] != (1080,1920):
            raise ValueError(f'Unexpected source image: {path}')
        source_size, raw_size, size = [1920,1080], [3840,2160], [480,270]
        source_K = resize_K(raw_K, raw_size, source_size)
        # A single per-camera undistortion, never pairwise epipolar rectification.
        mx, my = cv2.initUndistortRectifyMap(source_K, D, None, source_K,
                                             tuple(source_size), cv2.CV_32FC1)
        undistorted = cv2.remap(raw, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        image = cv2.resize(undistorted, tuple(size), interpolation=cv2.INTER_AREA)
        source_valid = ((mx >= 0)&(mx <= source_size[0]-1)&(my >= 0)&(my <= source_size[1]-1))
        valid = cv2.resize(source_valid.astype(np.float32), tuple(size),
                           interpolation=cv2.INTER_AREA) >= 1-1e-7
        mask = np.zeros((size[1],size[0]), np.uint8)
        cv2.fillPoly(mask, [np.asarray(p,np.int32) for p in POLYGONS[camera_id]], 255)
        mask[~valid] = 0
        prefix = data / f'cam{camera_id:02d}'
        image_path = str(prefix)+'.png'
        static_path = str(prefix)+'_static.png'
        valid_path = str(prefix)+'_valid.png'
        cv2.imwrite(image_path, image)
        cv2.imwrite(static_path, mask)
        cv2.imwrite(valid_path, valid.astype(np.uint8)*255)
        camera = dict(camera_id=camera_id, image_path=image_path, static_mask_path=static_path,
                      valid_mask_path=valid_path, size=size, width=size[0], height=size[1],
                      K=resize_K(source_K,source_size,size).tolist(), R=R.tolist(), t=t.tolist(),
                      C_world=(-R.T@t).tolist(), D=[0.,0.,0.,0.,0.],
                      distortion_model='none_after_undistortion', coordinate_system='undistorted_working',
                      raw_K=raw_K.tolist(),raw_D=D.reshape(-1).tolist(),
                      raw_distortion_model='opencv_radial_tangential_k1_k2_p1_p2_k3',
                      raw_size=raw_size, source_size=source_size, source_K=source_K.tolist(),
                      source_path=str(path), source_image_sha256=sha256(path),
                      frame=230, timestamp=None, static_polygons=POLYGONS[camera_id],
                      static_pixels=int((mask>0).sum()),image_sha256=sha256(image_path))
        mapping = working_to_raw_distorted(pixel_grid(size), camera).astype(np.float32)
        map_path = str(prefix)+'_working_to_raw.npz'
        np.savez_compressed(map_path, uv_raw_distorted=mapping,
                            uv_source_distorted=(mapping+.5)*.5-.5)
        camera['working_to_raw_map_path']=map_path
        cameras.append(camera)
        overlay = image.copy()
        overlay[mask>0] = (overlay[mask>0].astype(float)*.65 + np.array([30,220,30])*.35).astype(np.uint8)
        cv2.polylines(overlay,[np.asarray(p,np.int32) for p in POLYGONS[camera_id]], True,(0,255,0),1)
        cv2.putText(overlay,f'cam{camera_id:02d}: static floor only',(12,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
        previews.append(overlay)
    ref = all_cameras[7]
    Cref = -ref[2].T@ref[3]
    candidates=[]
    for i,(K,D,R,t) in all_cameras.items():
        angle=np.degrees(np.arccos(np.clip(ref[2][2]@R[2],-1,1)))
        candidates.append(dict(camera_id=i,baseline=float(np.linalg.norm(-R.T@t-Cref)),
                               optical_axis_angle_deg=float(angle)))
    candidates.sort(key=lambda row:row['baseline'])
    pairs=[dict(reference_id=7,target_id=j,baseline=float(np.linalg.norm(np.asarray(cameras[0]['C_world'])-
                                                                                       np.asarray(c['C_world']))))
           for j,c in [(9,cameras[1]),(10,cameras[2])]]
    manifest=dict(schema_version=1, seed=20261111, dataset='SelfCap corgi-release',
                  mode='asynchronous_unknown', reference_camera_id=7, cameras=cameras,pairs=pairs,
                  calibration_sources={str(p):sha256(p) for p in [intri_path,extri_path]},
                  translation_unit='dataset_native_common_unit; metric scale not independently verified',
                  extrinsics='X_camera = R_cw @ X_world + t_cw; C_world = -R_cw.T @ t_cw',
                  calibration_policy='All source camera parameters fixed; R obtained by Rodrigues of supplied R_ vector to avoid rounded Rot_ matrix non-orthogonality.',
                  coordinate_convention='xy; integer pixel centers; normalized grid_sample align_corners=False; g=2*(u+0.5)/W-1',
                  processing='Source images_2 are 1920x1080 half-sized from3840x2160. K transformed with half-pixel resize. One undistortion per camera at1920x1080 using same K. INTER_AREA resize to480x270 with exact pixel-center K transform. No crop/padding/rectification.',
                  static_mask_basis='Visually inspected frame230 undistorted images. Conservative manually drawn polygons contain architectural courtyard floor only; all persons,dog,cart,chairs,pots excluded. Floor geometry static despite asynchronous exposure; shadows/illumination can vary. Same frame number is not asserted synchronous. Geometry requires BOTH source and target masks with patch support.',
                  static_mask_preview=str(data/'static_masks.png'),
                  camera_selection={'candidates_by_baseline':candidates,
                                    'reason':'07→09 small baseline and07→10 moderate baseline selected from calibrated centers then visually checked courtyard overlap. Dense network coverage and forward-backward diagnostics determine actual usable support; indices do not encode adjacency.'})
    cv2.imwrite(str(data/'static_masks.png'),np.concatenate(previews,axis=1))
    (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'manifest':str(out/'data_manifest.json'),'pairs':pairs,
                      'static_pixels':{c['camera_id']:c['static_pixels'] for c in cameras}},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/data/datasets/selfcap/corgi-release'))
    parser.add_argument('--out',type=Path,default=Path('experiments/roma_lsm'))
    args=parser.parse_args()
    prepare(args.root,args.out)
