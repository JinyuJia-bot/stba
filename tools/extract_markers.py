#!/usr/bin/env python3
"""Extract identified ArUco corners from calibrated videos; no board rigidity is assumed."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import shutil

OBS = ['track_id', 'is_static', 'motion_weight', 'camera_id', 'frame', 'u', 'v', 'weight']


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def extract(args):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError('Install optional dependency: pip install opencv-python-headless>=4.7') from error
    if not hasattr(cv2, 'aruco') or not hasattr(cv2.aruco, 'ArucoDetector'):
        raise RuntimeError('OpenCV >= 4.7 with the aruco module is required')
    if not args.dictionary.startswith('DICT_') or not hasattr(cv2.aruco, args.dictionary):
        raise ValueError('Unknown ArUco dictionary')
    if args.every < 1 or args.max_frames < 0:
        raise ValueError('--every must be positive and --max-frames must be nonnegative')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Output directory must be empty; existing data is never overwritten')
    cameras = {int(r['camera_id']): r for r in read_csv(args.calibration / 'cameras.csv')}
    views = {(int(r['camera_id']), int(r['frame'])) for r in read_csv(args.calibration / 'views.csv')}
    old_path = args.calibration / 'observations.csv'
    observations = read_csv(old_path) if old_path.exists() else []
    first_id = max((int(r['track_id']) for r in observations), default=-1) + 1
    videos = read_csv(args.videos)
    if len({int(r['camera_id']) for r in videos}) != len(videos):
        raise ValueError('Use one video per camera in videos.csv')
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    detected = defaultdict(list)
    for video in videos:
        camera = int(video['camera_id'])
        if camera not in cameras:
            raise ValueError(f'Unknown logical camera {camera}')
        offset = int(video.get('frame_offset') or 0)
        if offset < 0:
            raise ValueError('frame_offset must be nonnegative')
        path = Path(video['path'])
        if not path.is_absolute():
            path = args.videos.parent / path
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f'Cannot open {path}')
        try:
            metadata_fps = capture.get(cv2.CAP_PROP_FPS)
            expected_fps = float(cameras[camera]['fps'])
            if metadata_fps > 0 and abs(metadata_fps - expected_fps) > 0.01 * expected_fps:
                raise ValueError(f'Camera {camera}: video fps={metadata_fps}, cameras.csv fps={expected_fps}. Correct frame-rate metadata before extracting.')
            index = 0
            while not args.max_frames or index < args.max_frames:
                ok, image = capture.read()
                if not ok:
                    break
                frame = index + offset
                index += 1
                if (index - 1) % args.every:
                    continue
                corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                if ids is None:
                    continue
                if (camera, frame) not in views and (camera, -1) not in views:
                    raise ValueError(f'No calibrated view for camera {camera}, frame {frame}')
                if len(set(ids.flatten().tolist())) != len(ids):
                    raise ValueError('Repeated physical marker IDs in one frame are ambiguous')
                for marker_corners, identifier in zip(corners, ids.flatten()):
                    for corner_id, (u, v) in enumerate(marker_corners.reshape(4, 2)):
                        key = int(identifier) * 4 + corner_id
                        detected[key].append((camera, frame, float(u), float(v)))
        finally:
            capture.release()
    kept = 0
    id_map = []
    for marker_corner, rows in sorted(detected.items()):
        if len({row[0] for row in rows}) < 2:
            continue
        track = first_id + kept
        kept += 1
        id_map.append((track, marker_corner // 4, marker_corner % 4))
        observations.extend(dict(zip(OBS, [track, 0, 1, camera, frame, u, v, 1]))
                            for camera, frame, u, v in rows)
    if not kept:
        raise ValueError('No marker corner is visible in at least two cameras')
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ['cameras.csv', 'views.csv']:
        shutil.copyfile(args.calibration / name, args.out / name)
    with (args.out / 'observations.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, OBS)
        writer.writeheader()
        writer.writerows(observations)
    with (args.out / 'marker_track_ids.csv').open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['track_id', 'marker_id', 'corner_id'])
        writer.writerows(id_map)
    print(f'Added {kept} dynamic corner tracks to {args.out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--videos', type=Path, required=True, help='CSV: camera_id,path,frame_offset (paths relative to this CSV)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dictionary', default='DICT_4X4_50')
    parser.add_argument('--every', type=int, default=1, help='Read every Nth frame; preserve original frame indices/fps')
    parser.add_argument('--max-frames', type=int, default=0, help='0 means all frames')
    args = parser.parse_args()
    try:
        extract(args)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
