"""End-to-end regression on saved, independently evaluated synthetic data.

Run from the project root:
  python3 tests/run_integration.py --binary build/stba --work experiments/checks
"""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def run(command, log):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}:\n{result.stdout[-2500:]}")


def metrics(source, output):
    true_cameras = {int(row["camera_id"]): row for row in read_csv(source / "true_cameras.csv")}
    cameras = {int(row["camera_id"]): row for row in read_csv(output / "cameras.csv")}
    if set(cameras) != set(true_cameras):
        raise AssertionError("Reconstruction omitted a camera")
    # Fix the arbitrary global time origin before comparing offsets. Use seconds,
    # then express errors in reference-camera frames, also valid for mixed fps.
    ref = min(true_cameras)
    true_origin = float(true_cameras[ref]["beta"]) / float(true_cameras[ref]["fps"])
    estimated_origin = float(cameras[ref]["beta"]) / float(cameras[ref]["fps"])
    fps = float(true_cameras[ref]["fps"])
    errors = []
    for camera, row in cameras.items():
        truth = true_cameras[camera]
        errors.append(abs((float(row["beta"]) / float(row["fps"]) - estimated_origin) -
                          (float(truth["beta"]) / float(truth["fps"]) - true_origin)) * fps)

    truth = {(int(r["track_id"]), int(r["camera_id"]), int(r["frame"])):
             tuple(float(r[axis]) for axis in ("x", "y", "z")) for r in read_csv(source / "truth.csv")}
    observations = read_csv(source / "observations.csv")
    dynamic_ids = {int(r["track_id"]) for r in observations if r["is_static"] == "0"}
    expected_keys = {(int(r["track_id"]), int(r["camera_id"]), int(r["frame"])) for r in observations}
    reconstructed_keys = set()
    squared_error = 0.0
    count = 0
    for row in read_csv(output / "trajectories.csv"):
        key = (int(row["track_id"]), int(row["camera_id"]), int(row["frame"]))
        if key in reconstructed_keys:
            raise AssertionError("Duplicate reconstructed observation")
        reconstructed_keys.add(key)
        point = tuple(float(row[axis]) for axis in ("x", "y", "z"))
        if not all(math.isfinite(x) for x in point):
            raise AssertionError("Non-finite reconstructed point")
        if key[0] in dynamic_ids:
            squared_error += sum((x - y) ** 2 for x, y in zip(point, truth[key]))
            count += 1
    if reconstructed_keys != expected_keys:
        raise AssertionError("Reconstruction dropped or added observations")
    if count == 0:
        raise AssertionError("No dynamic trajectories were reconstructed")
    for row in read_csv(output / "metrics.csv"):
        if row["usable"] not in ("1", "true", "True") or row["order_valid"] not in ("1", "true", "True"):
            raise AssertionError("Unusable solution or changed temporal ordering")
        if not math.isfinite(float(row["cost"])):
            raise AssertionError("Non-finite objective")
    return {"max_offset_error_frames": max(errors), "dynamic_rmse_m": math.sqrt(squared_error / count),
            "evaluated_observations": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/stba"))
    parser.add_argument("--work", type=Path, default=Path("experiments/checks"))
    parser.add_argument("--extended", action="store_true", help="Also test unequal camera frame rates")
    args = parser.parse_args()
    binary = args.binary.resolve()
    root = Path(__file__).resolve().parents[1]
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    cases = [("signed_offsets", []), ("missing", ["--drop-rate", "0.1"]),
             ("moving_cameras", ["--moving"])]
    if args.extended:
        cases.append(("mixed_fps", ["--fps-list", "30,24,20"]))
    report = {}
    for name, extra in cases:
        directory = work / name
        directory.mkdir(exist_ok=True)
        source, fixed, full = directory / "input", directory / "fixed", directory / "full"
        run([sys.executable, str(root / "tools/generate_synthetic.py"), "--cameras", "3", "--frames", "24",
             "--tracks", "4", "--static-points", "8", "--beta", "0,0.31,-0.27", "--noise", "0.1",
             "--seed", "42", "--out", str(source), *extra], directory / "generate.log")
        # The solver receives only measured data, physically separate from truth.
        solver_input = directory / "solver_input"
        solver_input.mkdir(exist_ok=True)
        for filename in ("cameras.csv", "views.csv", "observations.csv"):
            shutil.copyfile(source / filename, solver_input / filename)
        run([str(binary), "--input", str(solver_input), "--output", str(fixed), "--mode", "fixed"], directory / "fixed.log")
        run([str(binary), "--input", str(solver_input), "--output", str(full)], directory / "full.log")
        baseline, solved = metrics(source, fixed), metrics(source, full)
        if solved["max_offset_error_frames"] > 0.15:
            raise AssertionError(f"{name}: offset regression: {solved}")
        if solved["dynamic_rmse_m"] >= baseline["dynamic_rmse_m"]:
            raise AssertionError(f"{name}: STBA did not improve dynamic 3D error: {solved} vs {baseline}")
        dct_rows = read_csv(full / "dct.csv")
        if not dct_rows or not all(math.isfinite(float(row[key])) for row in dct_rows for key in ("time", "x", "y", "z")):
            raise AssertionError(f"{name}: missing or non-finite DCT output")
        dct_metrics = read_csv(full / "dct_metrics.csv")
        if {int(row["track_id"]) for row in dct_metrics} != {int(row["track_id"]) for row in dct_rows}:
            raise AssertionError(f"{name}: a DCT track was omitted")
        for row in dct_metrics:
            before, after = float(row["cost_before"]), float(row["cost_after"])
            if row["usable"] not in ("1", "true", "True") or not math.isfinite(after) or after > before + 1e-7 * (1 + before):
                raise AssertionError(f"{name}: DCT fit failed or increased its objective")
        report[name] = {"fixed": baseline, "stba": solved}
        (work / "integration_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"PASS {name}: offset {solved['max_offset_error_frames']:.4f} frame; "
              f"dynamic RMSE {baseline['dynamic_rmse_m']*1000:.2f} -> {solved['dynamic_rmse_m']*1000:.2f} mm", flush=True)
    print(work / "integration_report.json")


if __name__ == "__main__":
    main()
