#!/usr/bin/env python3
"""Convert a COLMAP text model to the STBA CSV input format (no COLMAP dependency)."""
import argparse
import csv
import math
from pathlib import Path

CAMERAS = ['camera_id', 'fps', 'beta']
VIEWS = ['view_id', 'camera_id', 'frame', 'fx', 'fy', 'cx', 'cy', 'rx', 'ry', 'rz', 'tx', 'ty', 'tz']
OBS = ['track_id', 'is_static', 'motion_weight', 'camera_id', 'frame', 'u', 'v', 'weight']


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def read_cameras(path):
    cameras = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.split()
        identifier, model = int(fields[0]), fields[1]
        parameters = list(map(float, fields[4:]))
        if model == 'PINHOLE' and len(parameters) == 4:
            intrinsics = parameters
        elif model == 'SIMPLE_PINHOLE' and len(parameters) == 3:
            f, cx, cy = parameters
            intrinsics = [f, f, cx, cy]
        else:
            raise ValueError(f'Camera {identifier}: unsupported model {model}. Export an undistorted PINHOLE model and use the matching undistorted images/coordinates.')
        if identifier in cameras or not all(math.isfinite(x) for x in intrinsics) or min(intrinsics[:2]) <= 0:
            raise ValueError(f'Invalid or duplicate COLMAP camera {identifier}')
        cameras[identifier] = intrinsics
    return cameras


def read_images(path):
    images = {}
    # A blank second line is meaningful: an image can have no 2D points.
    with path.open() as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            p = line.split(maxsplit=9)
            if len(p) != 10:
                raise ValueError('Malformed image pose line in images.txt')
            points = next(f, None)
            while points is not None and points.lstrip().startswith('#'):
                points = next(f, None)
            if points is None:
                raise ValueError('Missing points line in images.txt')
            tokens = points.split()
            if len(tokens) % 3:
                raise ValueError('Malformed image points line')
            name = p[9].strip()
            if name in images:
                raise ValueError(f'Duplicate image name: {name}')
            images[name] = (list(map(float, p[1:5])), list(map(float, p[5:8])), int(p[8]),
                            [(float(tokens[i]), float(tokens[i+1]), int(tokens[i+2])) for i in range(0, len(tokens), 3)])
    return images


def angle_axis(quaternion):
    norm = math.sqrt(sum(x*x for x in quaternion))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('Invalid COLMAP rotation quaternion')
    q = [x / norm for x in quaternion]
    if q[0] < 0:
        q = [-x for x in q]
    length = math.sqrt(sum(x*x for x in q[1:]))
    factor = 2.0 if length < 1e-12 else 2 * math.atan2(length, q[0]) / length
    return [factor*x for x in q[1:]]


def convert(args):
    if not math.isfinite(args.scale) or args.scale <= 0:
        raise ValueError('--scale must be positive and finite')
    intrinsics = read_cameras(args.model / 'cameras.txt')
    images = read_images(args.model / 'images.txt')
    mapping = read_csv(args.mapping)
    if not mapping:
        raise ValueError('Empty image mapping')
    logical_cameras, views, selected, seen = {}, [], {}, set()
    for row in mapping:
        name = row['image_name']
        camera, frame = int(row['camera_id']), int(row['frame'])
        fps, beta = float(row['fps']), float(row.get('beta') or 0)
        if camera < 0 or frame < 0 or fps <= 0 or not all(math.isfinite(x) for x in [fps, beta]):
            raise ValueError('Invalid camera_id/frame/fps/beta in mapping')
        if name not in images or name in selected or (camera, frame) in seen:
            raise ValueError(f'Missing image or duplicate mapping: {name}')
        seen.add((camera, frame))
        parameters = {'camera_id': camera, 'fps': fps, 'beta': beta}
        if camera in logical_cameras and logical_cameras[camera] != parameters:
            raise ValueError('fps and beta must be constant within each logical camera')
        logical_cameras[camera] = parameters
        q, t, intrinsic_id, points = images[name]
        values = [len(views), camera, frame, *intrinsics[intrinsic_id], *angle_axis(q), *[x * args.scale for x in t]]
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Non-finite COLMAP view')
        views.append(dict(zip(VIEWS, values)))
        selected[name] = (camera, frame, points)
    if sorted(logical_cameras) != list(range(len(logical_cameras))):
        raise ValueError('Mapping camera_id must be contiguous starting at 0')

    observations = read_csv(args.dynamic_observations) if args.dynamic_observations else []
    keys = set()
    for row in observations:
        if set(row) != set(OBS):
            raise ValueError('Dynamic observations must use the exact observations.csv header')
        camera, frame = int(row['camera_id']), int(row['frame'])
        key = (int(row['track_id']), camera, frame)
        if int(row['is_static']) != 0 or (camera, frame) not in seen or key in keys:
            raise ValueError('Dynamic observations must be unique, non-static, and have mapped views')
        if not all(math.isfinite(float(row[k])) for k in ['u', 'v', 'weight', 'motion_weight']) or float(row['weight']) <= 0 or float(row['motion_weight']) <= 0:
            raise ValueError('Invalid dynamic observation values')
        keys.add(key)
    first_id = max((int(row['track_id']) for row in observations), default=-1) + 1
    tracks = {}
    if not args.no_static_points:
        point_ids = {int(line.split()[0]) for line in (args.model / 'points3D.txt').read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith('#')}
        for camera, frame, points in selected.values():
            for u, v, point_id in points:
                if not math.isfinite(u) or not math.isfinite(v):
                    raise ValueError('Non-finite COLMAP image observation')
                if point_id < 0:
                    continue
                if point_id not in point_ids:
                    raise ValueError(f'Image references absent POINT3D_ID {point_id}')
                tracks.setdefault(point_id, []).append((camera, frame, u, v))
    track_map = []
    for point_id, points in sorted(tracks.items()):
        if len({p[0] for p in points}) < 2:
            continue
        track_id = first_id + len(track_map)
        track_map.append({'track_id': track_id, 'colmap_point3d_id': point_id})
        for camera, frame, u, v in points:
            observations.append(dict(zip(OBS, [track_id, 1, 1, camera, frame, u, v, 1])))
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError('Output directory must be empty; existing data is never overwritten')
    write_csv(args.out / 'cameras.csv', CAMERAS, [logical_cameras[k] for k in sorted(logical_cameras)])
    write_csv(args.out / 'views.csv', VIEWS, views)
    write_csv(args.out / 'observations.csv', OBS, observations)
    write_csv(args.out / 'colmap_track_ids.csv', ['track_id', 'colmap_point3d_id'], track_map)
    print(f'Imported {len(views)} views, {len(track_map)} static tracks, {len(observations)} observations into {args.out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='COLMAP text model directory')
    parser.add_argument('--mapping', type=Path, required=True, help='CSV: image_name,camera_id,frame,fps,beta')
    parser.add_argument('--dynamic-observations', type=Path)
    parser.add_argument('--no-static-points', action='store_true')
    parser.add_argument('--scale', type=float, default=1.0, help='Metres per COLMAP length unit (default: already metric)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        convert(args)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
