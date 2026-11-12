#!/usr/bin/env python3
"""Create a known-ground-truth multi-camera dataset. Requires only NumPy.

Solver input and evaluation truth are written into different files. In particular,
input beta is zero for all cameras; true beta never appears in solver CSV files.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def angle_axis(rotation):
    angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
    if angle < 1e-10:
        return np.zeros(3)
    if np.pi - angle < 1e-5:
        # The eigenvector with eigenvalue 1 is the axis; its sign is immaterial at pi.
        _, vectors = np.linalg.eigh((rotation + rotation.T) / 2)
        axis = vectors[:, -1]
    else:
        axis = np.array([rotation[2, 1] - rotation[1, 2],
                         rotation[0, 2] - rotation[2, 0],
                         rotation[1, 0] - rotation[0, 1]]) / (2 * np.sin(angle))
    return angle * axis


def rodrigues(vector):
    angle = np.linalg.norm(vector)
    if angle < 1e-15:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


def view_at(camera, count, time, moving):
    theta = 2 * np.pi * camera / count
    center = np.array([4 * np.sin(theta), 0.8 + 0.25 * np.sin(2 * theta), -4 * np.cos(theta)])
    if moving:
        center += np.array([0.08 * np.sin(1.3 * time + theta),
                            0.04 * np.cos(1.7 * time + theta),
                            0.06 * np.sin(1.1 * time - theta)])
    forward = -center / np.linalg.norm(center)
    right = np.cross(np.array([0., 1., 0.]), forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack((right, down, forward))
    return rotation, -rotation @ center


def position(track, time):
    phase = 0.63 * track
    return np.array([0.65 * np.sin(2.3 * time + phase) + 0.12 * np.sin(4.1 * time + 0.3 * phase),
                     0.40 * np.cos(2.8 * time + 0.7 * phase) + 0.10 * np.sin(4.7 * time + phase),
                     0.30 * np.sin(3.4 * time + 0.5 * phase) + 0.08 * np.cos(5.2 * time + phase)])


def read_trajectories(path, limit):
    tracks = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["track_id", "time", "x", "y", "z"]:
            raise ValueError("Trajectory CSV header must be track_id,time,x,y,z")
        for row in reader:
            track = int(row["track_id"])
            sample = np.array([float(row[k]) for k in ("time", "x", "y", "z")])
            if track < 0 or not np.all(np.isfinite(sample)):
                raise ValueError("Trajectory ids must be nonnegative and samples finite")
            tracks.setdefault(track, []).append(sample)
    selected = []
    for track, samples in sorted(tracks.items())[:limit]:
        samples = np.array(sorted(samples, key=lambda a: a[0]))
        if len(samples) < 2 or np.any(np.diff(samples[:, 0]) <= 0):
            raise ValueError(f"Trajectory {track} needs at least two strictly increasing times")
        selected.append((track, samples))
    if not selected:
        raise ValueError("Trajectory CSV contains no tracks")
    return selected


def interpolate_trajectory(samples, time, max_gap):
    times = samples[:, 0]
    index = int(np.searchsorted(times, time))
    if index < len(times) and abs(times[index] - time) < 1e-10:
        return samples[index, 1:]
    if index > 0 and abs(times[index - 1] - time) < 1e-10:
        return samples[index - 1, 1:]
    if index == 0 or index == len(times):
        raise ValueError("Requested MoCap sample lies outside the source trajectory")
    gap = times[index] - times[index - 1]
    if gap > max_gap:
        return None  # Retain a source occlusion as a missing image observation.
    alpha = (time - times[index - 1]) / gap
    return (1 - alpha) * samples[index - 1, 1:] + alpha * samples[index, 1:]


def write_csv(path, header, rows):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header.split(","))
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("experiments/synthetic/input"))
    parser.add_argument("--cameras", type=int, default=4)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--tracks", type=int, default=6, help="Number of dynamic tracks")
    parser.add_argument("--static-points", type=int, default=20)
    parser.add_argument("--fps", type=float, default=30.)
    parser.add_argument("--fps-list", help="Optional comma-separated known frame rates, one per camera")
    parser.add_argument("--noise", type=float, default=0.1, help="Per-coordinate image noise std, pixels")
    parser.add_argument("--drop-rate", type=float, default=0.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", help="Comma-separated true beta values in frames, starting with 0")
    parser.add_argument("--pose-noise", type=float, default=0., help="Camera >=2 perturbation std: radians for rotation, metres for translation")
    parser.add_argument("--intrinsics-noise", type=float, default=0., help="Camera >=2 relative focal length perturbation std")
    parser.add_argument("--moving", action="store_true", help="Write a separate view for every camera frame")
    parser.add_argument("--trajectory-csv", type=Path, help="Use imported MoCap trajectories in metres instead of analytic motion")
    parser.add_argument("--start-time", type=float, help="MoCap source time corresponding to camera-0 frame 0; default fits all offset samples in the common window")
    parser.add_argument("--max-gap", type=float, help="Maximum interpolated MoCap gap in seconds; default twice each track's median sample interval")
    args = parser.parse_args()
    values = [args.fps, args.noise, args.drop_rate, args.pose_noise, args.intrinsics_noise]
    if (not np.all(np.isfinite(values)) or args.cameras < 2 or args.frames < 3 or args.tracks < 1 or
            args.static_points < 0 or args.fps <= 0 or args.noise < 0 or args.pose_noise < 0 or
            args.intrinsics_noise < 0 or not 0 <= args.drop_rate < 1):
        parser.error("Invalid dimensions, rates, or noise parameters")
    phases = [0., .37, -.23, .11, -.31, .27, -.13, .19, .07, -.07]
    beta = np.array([phases[c] if c < len(phases) else .6 * ((c * .61803398875) % 1 - .5)
                     for c in range(args.cameras)])
    if args.beta is not None:
        try:
            beta = np.array([float(x) for x in args.beta.split(",")])
        except ValueError:
            parser.error("--beta must contain numeric comma-separated values")
        if len(beta) != args.cameras or not np.all(np.isfinite(beta)) or abs(beta[0]) > 1e-12:
            parser.error("--beta needs one finite value per camera, with beta_0=0")
    fps = np.full(args.cameras, args.fps)
    if args.fps_list is not None:
        try:
            fps = np.array([float(x) for x in args.fps_list.split(",")])
        except ValueError:
            parser.error("--fps-list must contain numeric comma-separated values")
        if len(fps) != args.cameras or not np.all(np.isfinite(fps)) or np.any(fps <= 0):
            parser.error("--fps-list needs one positive finite frame rate per camera")
    trajectories = None
    source_start = None
    max_gaps = None
    if args.trajectory_csv is not None:
        try:
            trajectories = read_trajectories(args.trajectory_csv, args.tracks)
            args.tracks = len(trajectories)
            common_start = max(samples[0, 0] for _, samples in trajectories)
            common_end = min(samples[-1, 0] for _, samples in trajectories)
            local_start = float(np.min(-beta / fps))
            local_end = float(np.max((args.frames - 1 - beta) / fps))
            source_start = common_start - local_start if args.start_time is None else args.start_time
            if (not np.isfinite(source_start) or source_start + local_start < common_start - 1e-10 or
                    source_start + local_end > common_end + 1e-10):
                raise ValueError("Requested frames/offsets exceed the common MoCap time window; reduce --frames or adjust --start-time")
            if args.max_gap is not None and (not np.isfinite(args.max_gap) or args.max_gap <= 0):
                raise ValueError("--max-gap must be positive and finite")
            max_gaps = [args.max_gap if args.max_gap is not None else
                        2.01 * np.median(np.diff(samples[:, 0])) for _, samples in trajectories]
        except (OSError, ValueError) as error:
            parser.error(str(error))
    elif args.start_time is not None or args.max_gap is not None:
        parser.error("--start-time and --max-gap require --trajectory-csv")
    rng = np.random.default_rng(args.seed)
    intrinsics = np.array([1000., 1000., 640., 360.])
    views, true_views, lookup = [], [], {}
    for camera in range(args.cameras):
        frames = range(args.frames) if args.moving else [-1]
        for frame in frames:
            time = (frame - beta[camera]) / fps[camera] if frame >= 0 else 0.
            rotation, translation = view_at(camera, args.cameras, time, args.moving)
            view_id = len(views)
            truth_row = [view_id, camera, frame, *intrinsics, *angle_axis(rotation), *translation]
            true_views.append(truth_row)
            estimated_rotation = rotation.copy()
            estimated_translation = translation.copy()
            estimated_intrinsics = intrinsics.copy()
            # Two accurate views provide a controlled spatial/scale anchor in BA experiments.
            if camera >= 2:
                estimated_rotation = rodrigues(rng.normal(0, args.pose_noise, 3)) @ rotation
                estimated_translation += rng.normal(0, args.pose_noise, 3)
                estimated_intrinsics[:2] *= np.exp(rng.normal(0, args.intrinsics_noise, 2))
            views.append([view_id, camera, frame, *estimated_intrinsics,
                          *angle_axis(estimated_rotation), *estimated_translation])
            lookup[camera, frame] = (rotation, translation)
    static_points = rng.uniform([-1.2, -.7, -.8], [1.2, .8, .8], size=(args.static_points, 3))
    observations, truth = [], []
    for track in range(args.tracks + args.static_points):
        is_static = track >= args.tracks
        for camera in range(args.cameras):
            for frame in range(args.frames):
                if rng.random() < args.drop_rate:
                    continue
                time = (frame - beta[camera]) / fps[camera]
                if is_static:
                    xyz = static_points[track - args.tracks]
                elif trajectories is None:
                    xyz = position(track, time)
                else:
                    xyz = interpolate_trajectory(trajectories[track][1], source_start + time, max_gaps[track])
                    if xyz is None:
                        continue
                rotation, translation = lookup[camera, frame if args.moving else -1]
                p = rotation @ xyz + translation
                if p[2] <= 0:
                    raise RuntimeError("Generated point is behind a camera")
                uv = intrinsics[:2] * p[:2] / p[2] + intrinsics[2:] + rng.normal(0, args.noise, 2)
                observations.append([track, int(is_static), 1., camera, frame, *uv, 1.])
                truth.append([track, camera, frame, *xyz])
    if trajectories is not None:
        for track in range(args.tracks):
            seen = {row[3] for row in observations if row[0] == track}
            if len(seen) < 2:
                parser.error(f"MoCap trajectory {trajectories[track][0]} has fewer than two observed cameras after gaps/dropout")
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "cameras.csv", "camera_id,fps,beta", [(c, fps[c], 0.) for c in range(args.cameras)])
    write_csv(args.out / "views.csv", "view_id,camera_id,frame,fx,fy,cx,cy,rx,ry,rz,tx,ty,tz", views)
    write_csv(args.out / "observations.csv", "track_id,is_static,motion_weight,camera_id,frame,u,v,weight", observations)
    write_csv(args.out / "truth.csv", "track_id,camera_id,frame,x,y,z", truth)
    write_csv(args.out / "true_cameras.csv", "camera_id,fps,beta", [(c, fps[c], beta[c]) for c in range(args.cameras)])
    write_csv(args.out / "true_views.csv", "view_id,camera_id,frame,fx,fy,cx,cy,rx,ry,rz,tx,ty,tz", true_views)
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata["true_beta_frames"] = beta.tolist()
    metadata["fps_per_camera"] = fps.tolist()
    if trajectories is not None:
        metadata["source_track_ids"] = [track for track, _ in trajectories]
        metadata["trajectory_source_start_seconds"] = float(source_start)
        metadata["max_gap_seconds_per_track"] = [float(gap) for gap in max_gaps]
    (args.out / "generation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Created {len(observations)} observations for {args.cameras} cameras at {args.out}")
    print("True beta (frames):", beta.tolist())


if __name__ == "__main__":
    main()
