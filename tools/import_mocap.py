#!/usr/bin/env python3
"""Convert C3D or a long-form XYZ CSV into track_id,time,x,y,z (metres).

CSV accepts track_id,time,x,y,z, or track_id,frame,x,y,z together with --fps.
C3D support is optional: pip install ezc3d. ASF/AMC skeleton angles are not XYZ
marker trajectories and must first be exported with a proper kinematic reader.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def unit_scale(units):
    name = units.strip().lower()
    scales = {"m": 1., "meter": 1., "meters": 1., "metre": 1., "metres": 1.,
              "mm": .001, "millimeter": .001, "millimeters": .001,
              "cm": .01, "centimeter": .01, "centimeters": .01}
    if name not in scales:
        raise ValueError(f"Unknown length unit {units!r}; specify --units m, cm or mm")
    return scales[name]


def from_c3d(path, units):
    try:
        import ezc3d
    except ImportError as error:
        raise RuntimeError("C3D import needs optional package ezc3d: python3 -m pip install ezc3d") from error
    source = ezc3d.c3d(str(path))
    parameters = source["parameters"]["POINT"]
    rate = float(parameters["RATE"]["value"][0])
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("C3D POINT:RATE must be positive")
    if units == "auto":
        values = parameters.get("UNITS", {}).get("value", [])
        if len(values) == 0:
            raise ValueError("C3D has no POINT:UNITS; specify --units explicitly")
        units = str(values[0])
    scale = unit_scale(units)
    points = np.asarray(source["data"]["points"])
    residuals = np.asarray(source["data"]["meta_points"]["residuals"])
    labels = list(parameters["LABELS"]["value"])
    tracks = {}
    for track in range(points.shape[1]):
        samples = []
        for frame in range(points.shape[2]):
            xyz = points[:3, track, frame] * scale
            if residuals[0, track, frame] < 0 or not np.all(np.isfinite(xyz)):
                continue
            samples.append((frame / rate, *xyz))
        if len(samples) >= 2:
            tracks[track] = samples
    metadata = {"format": "c3d", "source_units": units, "source_fps": rate,
                "source_first_frame": int(source["header"]["points"]["first_frame"]),
                "labels": {str(i): str(label) for i, label in enumerate(labels)}}
    return tracks, metadata


def from_csv(path, units, fps):
    scale = unit_scale("m" if units == "auto" else units)
    tracks = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        if not {"track_id", "x", "y", "z"} <= fields:
            raise ValueError("CSV requires track_id,x,y,z columns")
        use_frame = "time" not in fields
        if use_frame and ("frame" not in fields or fps is None or not np.isfinite(fps) or fps <= 0):
            raise ValueError("CSV requires time in seconds, or frame together with positive --fps")
        for row in reader:
            track = int(row["track_id"])
            time = float(row["frame"]) / fps if use_frame else float(row["time"])
            if track < 0 or not np.isfinite(time):
                raise ValueError("Track ids must be nonnegative and times finite")
            try:
                xyz = np.array([float(row[k]) if row[k].strip() else np.nan for k in ("x", "y", "z")]) * scale
            except ValueError as error:
                raise ValueError(f"Invalid XYZ value at CSV line {reader.line_num}") from error
            if not np.all(np.isfinite(xyz)):
                continue
            tracks.setdefault(track, []).append((time, *xyz))
    return tracks, {"format": "csv", "source_units": "m" if units == "auto" else units,
                    "source_fps": fps if use_frame else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--units", default="auto", choices=["auto", "m", "cm", "mm"],
                        help="auto uses C3D metadata; CSV auto means metres")
    parser.add_argument("--fps", type=float, help="Frame rate for a CSV with frame instead of time")
    parser.add_argument("--center", action="store_true", help="Apply one constant translation, centering the full recording's bounding box")
    parser.add_argument("--max-tracks", type=int, help="Keep the first N sorted valid tracks")
    args = parser.parse_args()
    try:
        extension = args.input.suffix.lower()
        if extension == ".c3d":
            tracks, metadata = from_c3d(args.input, args.units)
        elif extension == ".csv":
            tracks, metadata = from_csv(args.input, args.units, args.fps)
        else:
            raise ValueError("Supported inputs are .c3d and long-form .csv; ASF/AMC need an external skeleton/forward-kinematics conversion")
        if args.max_tracks is not None and args.max_tracks < 1:
            raise ValueError("--max-tracks must be positive")
        valid = {}
        for track, samples in sorted(tracks.items()):
            array = np.array(sorted(samples), dtype=float)
            if len(array) < 2:
                continue
            if np.any(np.diff(array[:, 0]) <= 0):
                raise ValueError(f"Duplicate times in track {track}")
            valid[track] = array
        if args.max_tracks is not None:
            valid = dict(list(valid.items())[:args.max_tracks])
        if not valid:
            raise ValueError("No track contains at least two valid samples")
        all_points = np.concatenate([a[:, 1:] for a in valid.values()])
        center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2 if args.center else np.zeros(3)
        origin = min(a[0, 0] for a in valid.values())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["track_id", "time", "x", "y", "z"])
            for track, samples in valid.items():
                for sample in samples:
                    writer.writerow([track, sample[0] - origin, *(sample[1:] - center)])
        metadata.update(source=str(args.input), output_units="m", time_origin_seconds=origin,
                        fixed_center_metres=center.tolist(), tracks=len(valid),
                        valid_samples=sum(len(a) for a in valid.values()))
        args.out.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Wrote {metadata['valid_samples']} samples from {len(valid)} tracks to {args.out}")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
