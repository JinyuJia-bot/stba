#!/usr/bin/env python3
"""MoCap import checks; the C3D round-trip is skipped if ezc3d is unavailable."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


class MocapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def run_tool(self, script, *arguments, success=True):
        result = subprocess.run([sys.executable, str(ROOT / "tools" / script), *map(str, arguments)],
                                capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_csv_units_and_offset_sampling(self):
        source = self.folder / "raw.csv"
        with source.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["track_id", "frame", "x", "y", "z"])
            for track in [4, 12]:
                for frame in range(121):
                    time = frame / 120
                    writer.writerow([track, frame, 1000 * (.1 * track + .2 * time),
                                     1000 * (-.4 + .1 * time), 100 * np.sin(2 * time + track)])
        trajectory = self.folder / "trajectory.csv"
        self.run_tool("import_mocap.py", source, "--out", trajectory,
                      "--fps", 120, "--units", "mm", "--center")
        imported = rows(trajectory)
        self.assertEqual(len(imported), 242)
        points = np.array([[float(r[k]) for k in ("x", "y", "z")] for r in imported[:121]])
        np.testing.assert_allclose(points[-1] - points[0], [.2, .1, .1 * (np.sin(6) - np.sin(4))])
        output = self.folder / "input"
        arguments = ["--trajectory-csv", trajectory, "--out", output, "--cameras", 3,
                     "--frames", 18, "--fps-list", "30,24,20", "--beta", "0,0.31,-0.27",
                     "--noise", 0, "--static-points", 3]
        self.run_tool("generate_synthetic.py", *arguments)
        metadata = json.loads((output / "generation.json").read_text())
        self.assertEqual(metadata["source_track_ids"], [4, 12])
        self.assertAlmostEqual(metadata["trajectory_source_start_seconds"], .31 / 24)
        for row in rows(output / "truth.csv"):
            track, camera, frame = [int(row[k]) for k in ("track_id", "camera_id", "frame")]
            if track >= 2:
                continue
            time = metadata["trajectory_source_start_seconds"] + (
                frame - metadata["true_beta_frames"][camera]) / metadata["fps_per_camera"][camera]
            samples = imported[track * 121:(track + 1) * 121]
            times = [float(sample["time"]) for sample in samples]
            for key in ("x", "y", "z"):
                expected = np.interp(time, times, [float(sample[key]) for sample in samples])
                self.assertAlmostEqual(float(row[key]), expected, places=12)
        self.assertTrue(all(float(row["beta"]) == 0 for row in rows(output / "cameras.csv")))
        failed = self.run_tool("generate_synthetic.py", *arguments, "--frames", 100, success=False)
        self.assertIn("common MoCap time window", failed.stderr)

    def test_unsupported_skeleton_format(self):
        failed = self.run_tool("import_mocap.py", self.folder / "motion.amc",
                               "--out", self.folder / "out.csv", success=False)
        self.assertIn("ASF/AMC", failed.stderr)

    def test_missing_interval(self):
        spec = importlib.util.spec_from_file_location("generate_synthetic", ROOT / "tools/generate_synthetic.py")
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        samples = np.array([[0., 0, 0, 0], [.01, 1, 0, 0], [.10, 10, 0, 0]])
        self.assertIsNone(generator.interpolate_trajectory(samples, .05, .03))
        np.testing.assert_array_equal(generator.interpolate_trajectory(samples, .10, .03), [10, 0, 0])
        with self.assertRaises(ValueError):
            generator.interpolate_trajectory(samples, -.01, .03)

    @unittest.skipUnless(importlib.util.find_spec("ezc3d"), "optional ezc3d package is unavailable")
    def test_c3d_units_rate_and_occlusion(self):
        import ezc3d
        source = ezc3d.c3d()
        source["parameters"]["POINT"]["RATE"]["value"] = [120.]
        source["parameters"]["POINT"]["UNITS"]["value"] = ["mm"]
        source["parameters"]["POINT"]["LABELS"]["value"] = ["hip", "hand"]
        points = np.ones((4, 2, 24))
        points[:3, 0, :] = np.array([100, 200, 300])[:, None]
        points[:3, 1, :] = np.array([400, 500, 600])[:, None]
        source["data"]["points"] = points
        source["data"]["meta_points"]["residuals"] = np.zeros((1, 2, 24))
        source["data"]["meta_points"]["residuals"][0, 1, 7] = -1
        recording = self.folder / "markers.c3d"
        source.write(str(recording))
        output = self.folder / "c3d.csv"
        self.run_tool("import_mocap.py", recording, "--out", output)
        imported = rows(output)
        self.assertEqual(len(imported), 47)
        self.assertAlmostEqual(float(imported[0]["x"]), .1)
        self.assertAlmostEqual(float(imported[1]["time"]), 1 / 120)
        self.assertFalse(any(int(row["track_id"]) == 1 and abs(float(row["time"]) - 7 / 120) < 1e-12
                             for row in imported))


if __name__ == "__main__":
    unittest.main()
