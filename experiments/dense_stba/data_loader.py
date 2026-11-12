"""Fixed-calibration video loading and explicit SelfCap clock bookkeeping.

New Dense STBA convention: physical_time = local_timestamp + beta (PLUS).
SelfCap README: actual_time = raw_frame/fps - sync. Thus relative beta_raw is
sync_reference-sync_camera, opposite in sign to the older STBA executable.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

if __package__ in [None,'']:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from experiments.roma_lsm.coordinates import resize_K
from experiments.roma_lsm.prepare_data import sha256


CAMERA_IDS=[7,9,10]
SOURCE=Path('/data/datasets/selfcap/corgi-release')
ROOT=Path(__file__).resolve().parent


def clock_mapping(sync_c,sync_reference,fps,integer_compensated):
    """Return raw index shift and PLUS-beta in seconds; no trajectory estimator."""
    raw_beta=float(sync_reference-sync_c)
    shift=int(np.rint(-raw_beta*fps)) if integer_compensated else 0
    residual=raw_beta+shift/fps
    return dict(raw_frame_shift=shift,raw_beta_seconds=raw_beta,
                raw_beta_frames=raw_beta*fps,residual_beta_seconds=residual,
                residual_beta_frames=residual*fps)


def probe_video(path):
    """Read integer presentation timestamps without decoding video frames."""
    command=['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_packets',
        '-show_entries','stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,time_base:packet=pts',
        '-of','json',str(path)]
    result=json.loads(subprocess.check_output(command,text=True))
    stream=result['streams'][0]
    pts=np.sort(np.asarray([p['pts'] for p in result['packets'] if 'pts' in p],np.int64))
    if len(np.unique(pts))!=len(pts):
        raise ValueError('Duplicate presentation timestamps; explicit frame mapping required')
    times=pts*float(Fraction(stream['time_base']))
    fps=float(Fraction(stream['avg_frame_rate']))
    return dict(video_path=str(path),stream=stream,fps=fps,pts=pts,times=times,
                timestamp_kind='MP4 encoded presentation time; not independently measured hardware exposure time')


def load_case(path):
    path=Path(path)
    return json.loads(path.read_text())


def timestamps(camera):
    for name in ['timestamps_s','nominal_timestamps','timestamps']:
        if name in camera:
            return np.asarray(camera[name],np.float64)
    return np.arange(camera['n_frames'],dtype=np.float64)/camera['fps']


def load_camera_sequence(camera):
    """Return uint8 RGB [T,H,W,3], supporting real and synthetic public schema."""
    if camera.get('frames_path'):
        images=np.load(camera['frames_path'],mmap_mode='r')
    else:
        loaded=[]
        for path in camera['image_paths']:
            image=cv2.imread(str(path),cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            loaded.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        images=np.stack(loaded)
    if images.dtype!=np.uint8 or images.ndim!=4 or images.shape[-1]!=3:
        raise ValueError('Expected uint8 RGB sequence [T,H,W,3]')
    if tuple(images.shape[1:3][::-1])!=tuple(camera['size']):
        raise ValueError('Image dimensions disagree with fixed working calibration')
    if len(images)!=len(timestamps(camera)):
        raise ValueError('Frame and timestamp counts disagree')
    return images


def query_grid(size,stride=4,mask=None,margin=0):
    """Regular integer-center seeds and full-image pixel IDs; no keypoint detector."""
    w,h=size
    yy,xx=np.mgrid[:h,:w]
    keep=(xx%stride==0)&(yy%stride==0)&(xx>=margin)&(xx<w-margin)&(yy>=margin)&(yy<h-margin)
    if mask is not None:
        keep &= np.asarray(mask,bool)
    return np.stack([xx[keep],yy[keep]],axis=-1).astype(np.float32),(yy[keep]*w+xx[keep]).astype(np.int64),keep


def prepare_real_data(out=ROOT/'data/real',start=200,stop=260):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(4)
    previous=json.loads((ROOT.parent/'roma_lsm/data_manifest.json').read_text())
    fixed={c['camera_id']:c for c in previous['cameras']}
    sync_path=SOURCE/'optimized/sync.json'
    sync=json.loads(sync_path.read_text())
    readme_path=Path('/data/datasets/SelfCap/README.md')
    reference=7;seed=(stop-start)//2
    # Broad, manually selected dynamic candidate region, NOT a foreground GT mask.
    roi=np.zeros((270,480),np.uint8);roi[50:242,190:315]=255
    roi_path=out/'reference_dynamic_candidate_roi.png';cv2.imwrite(str(roi_path),roi)
    camera_outputs={};probes={};mappings={};previews=[]
    began=time.perf_counter()
    for cid in CAMERA_IDS:
        camera=fixed[cid]
        probe=probe_video(SOURCE/f'videos/{cid:04d}.mp4');probes[cid]=probe
        if not np.isclose(probe['fps'],60) or len(probe['times'])!=3500:
            raise ValueError(f'Unexpected source frame rate/count for camera {cid}')
        if not np.allclose(np.diff(probe['times']),1/probe['fps'],atol=1e-12):
            raise ValueError('Source is not the verified CFR stream; revisit integer-shift mapping')
        mapping={mode:clock_mapping(sync[f'{cid:04d}'],sync[f'{reference:04d}'],probe['fps'],mode=='integer_aligned')
                 for mode in ['raw','integer_aligned']}
        mappings[cid]=mapping
        ids=sorted(set(int(n+m['raw_frame_shift']) for m in mapping.values() for n in range(start,stop)))
        camera_dir=out/'working_frames'/f'cam{cid:02d}';camera_dir.mkdir(parents=True,exist_ok=True)
        raw_K=np.asarray(camera['raw_K']);D=np.asarray(camera['raw_D'])
        source_K=resize_K(raw_K,(3840,2160),(1920,1080))
        mx,my=cv2.initUndistortRectifyMap(source_K,D,None,source_K,(1920,1080),cv2.CV_32FC1)
        outputs={}
        for raw_id in ids:
            dest=camera_dir/f'{raw_id:04d}.png'
            source=SOURCE/f'videos/cam{cid:02d}/images_2/{raw_id:04d}.png'
            if not source.exists():
                raise FileNotFoundError(source)
            if not dest.exists():
                image=cv2.imread(str(source),cv2.IMREAD_COLOR)
                if image.shape[:2]!=(1080,1920):raise ValueError('Unexpected raw extraction size')
                ideal=cv2.remap(image,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                work=cv2.resize(ideal,(480,270),interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(dest),work)
            outputs[raw_id]=str(dest.resolve())
        camera_outputs[cid]=outputs
        for raw_id in [start,start+seed,stop-1]:
            preview=cv2.imread(outputs[raw_id]);cv2.putText(preview,f'cam{cid:02d} raw{raw_id}',(10,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
            if cid==7:cv2.rectangle(preview,(190,50),(314,241),(0,255,0),1)
            previews.append(preview)
        print(json.dumps({'camera_id':cid,'prepared_raw_frames':[ids[0],ids[-1]+1],'count':len(ids)}),flush=True)
    cv2.imwrite(str(out/'sequence_preview.png'),np.concatenate([np.concatenate(previews[i:i+3],axis=1) for i in [0,3,6]],axis=0))
    manifests={}
    for mode in ['raw','integer_aligned']:
        mode_dir=out/mode;mode_dir.mkdir(exist_ok=True)
        cameras=[];references=[]
        for cid in CAMERA_IDS:
            old=fixed[cid];probe=probes[cid];mapping=mappings[cid][mode]
            raw_ids=np.arange(start,stop)+mapping['raw_frame_shift']
            raw_times=probe['times'][raw_ids]
            local=raw_times-raw_times[0]
            camera={k:old[k] for k in ['camera_id','K','R','t','D','size','raw_K','raw_D','raw_size','source_size','coordinate_system','valid_mask_path']}
            camera.update(fps=probe['fps'],n_frames=len(raw_ids),image_paths=[camera_outputs[cid][int(i)] for i in raw_ids],
                raw_frame_ids=raw_ids.tolist(),frame_ids=np.arange(start,stop).tolist(),local_frame_ids=np.arange(stop-start).tolist(),
                timestamps_s=local.tolist(),nominal_timestamps=local.tolist(),raw_presentation_timestamps_s=raw_times.tolist(),
                raw_presentation_timestamp_origin_s=float(raw_times[0]),integer_raw_frame_shift=mapping['raw_frame_shift'],
                timestamp_kind=probe['timestamp_kind'],timestamp_time_base=probe['stream']['time_base'],
                local_timestamp_formula='raw_presentation_time(raw_frame_id)-raw_presentation_time(first_selected_raw_frame)')
            cameras.append(camera)
            references.append(dict(camera_id=cid,**mapping,official_sync_seconds=sync[f'{cid:04d}']))
        reference_info=dict(source=str(sync_path),source_sha256=sha256(sync_path),
            independent_ground_truth=False,meaning='Published synchronization reference, not independently verified exposure-time GT',
            plus_beta_definition='t=local_timestamp+beta; beta_reference07=0',
            raw_plus_beta_formula='sync[07]-sync[c]',
            residual_plus_beta_formula='beta_raw+integer_raw_frame_shift/fps',
            recover_total_beta_formula='beta_raw=beta_residual-integer_raw_frame_shift/fps',cameras=references)
        reference_path=mode_dir/'external_sync_reference.json';reference_path.write_text(json.dumps(reference_info,indent=2)+'\n')
        manifest=dict(name=f'corgi_{mode}_{start}_{stop}',dataset='SelfCap corgi-release',mode=mode,
            reference_camera_id=7,cameras=cameras,seed_frame=seed,aligned_frame_window=[start,stop],
            pixel_convention='Working undistorted xy, zero-based integer pixel centers, no pairwise rectification',
            beta_convention='physical relative time = local encoded presentation timestamp + beta_seconds; beta07=0',
            calibration_fixed=True,calibration_sources=previous['calibration_sources'],
            translation_unit=previous['translation_unit'],
            seed_roi_mask_path=str(roi_path.resolve()),dynamic_candidate_roi=[190,50,315,242],
            seed_roi_semantics='Broad manually selected dog/person image box at reference frame230. NOT a segmentation/GT mask. Motion reliability must additionally select informative trajectories; static pixels in box do not constrain beta.',
            external_sync_reference_path=str(reference_path.resolve()),
            preprocessing_uses_external_sync=(mode=='integer_aligned'),
            preprocessing='Same once-per-camera fixed undistortion and pixel-center resizing as verified roma_lsm. Camera calibration not estimated. Frames are indexed/cropped in time only; no temporal image interpolation.',
            tracking_resource_audit='No local CoTracker3 source/checkpoint found in torch hub, listed conda envs and project searches. CPU OpenCV DIS dense optical flow is explicit fallback, with forward/backward and skip-frame cycle checks. No learned visibility or guaranteed physical identity claim.',
            source_readme=str(readme_path),source_readme_sha256=sha256(readme_path),
            preview_path=str((out/'sequence_preview.png').resolve()))
        path=mode_dir/'manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n');manifests[mode]=str(path)
    audit=dict(manifests=manifests,preparation_seconds=time.perf_counter()-began,
        streams={str(cid):{k:probes[cid][k] for k in ['video_path','stream','fps','timestamp_kind']} for cid in CAMERA_IDS},
        sync_sign_evidence='Local official SelfCap README: time computed from frame index - sync.json = actual timestamp.',
        camera_clock_mappings=mappings,
        cotracker_reference='https://github.com/facebookresearch/co-tracker',
        tracker_choice='OpenCV DIS medium preset on CPU; full-resolution dense pairwise flow, arbitrary subpixel grid-query tracking, no sparse feature detector.')
    (out/'resource_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2),flush=True)
    return manifests


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'data/real')
    parser.add_argument('--start',type=int,default=200);parser.add_argument('--stop',type=int,default=260)
    args=parser.parse_args();prepare_real_data(args.out,args.start,args.stop)
