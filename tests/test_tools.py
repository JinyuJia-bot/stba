"""Small data-contract tests; optional OpenCV test generates real marker videos."""
import csv
import importlib.util
import math
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools'


def write_csv(path, fields, rows):
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def command(self, tool, *args, success=True):
        completed = subprocess.run([sys.executable, str(TOOLS / tool), *map(str, args)], text=True, capture_output=True)
        if success:
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        else:
            self.assertNotEqual(completed.returncode, 0)
        return completed

    def colmap_fixture(self):
        model = self.root / 'model'
        model.mkdir()
        (model / 'cameras.txt').write_text('3 PINHOLE 640 480 500 501 320 240\n8 SIMPLE_PINHOLE 640 480 550 320 240\n')
        (model / 'images.txt').write_text(
            '# Header\n1 1 0 0 0 0 0 0 3 a.png\n120 130 100 300 200 -1\n'
            '2 0.7071067811865476 0 0.7071067811865475 0 1 0 0 8 b.png\n140 150 100\n'
            '9 1 0 0 0 0 0 0 3 unused.png\n\n')
        (model / 'points3D.txt').write_text('100 0 0 5 255 255 255 0.1 1 0 2 0\n')
        mapping = self.root / 'mapping.csv'
        write_csv(mapping, ['image_name', 'camera_id', 'frame', 'fps', 'beta'],
                  [['a.png', 0, 0, 30, 0], ['b.png', 1, 0, 30, 0.2]])
        return model, mapping

    def test_colmap_pose_tracks_and_preserved_dynamic_ids(self):
        model, mapping = self.colmap_fixture()
        dynamic = self.root / 'dynamic.csv'
        fields = ['track_id','is_static','motion_weight','camera_id','frame','u','v','weight']
        write_csv(dynamic, fields, [[10,0,1,0,0,200,200,1],[10,0,1,1,0,210,200,1]])
        output = self.root / 'converted'
        self.command('import_colmap.py', '--model', model, '--mapping', mapping,
                     '--dynamic-observations', dynamic, '--out', output)
        with (output / 'views.csv').open() as f:
            views = list(csv.DictReader(f))
        self.assertAlmostEqual(float(views[1]['ry']), math.pi / 2, places=12)
        self.assertEqual(float(views[1]['fx']), 550)
        self.assertEqual(float(views[1]['fy']), 550)
        with (output / 'observations.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 4)
        self.assertEqual({int(r['track_id']) for r in rows}, {10, 11})
        self.command('import_colmap.py', '--model', model, '--mapping', mapping, '--out', output, success=False)

    def test_distortion_models_are_rejected(self):
        model, mapping = self.colmap_fixture()
        (model / 'cameras.txt').write_text('3 SIMPLE_RADIAL 640 480 500 320 240 0.1\n')
        result = self.command('import_colmap.py', '--model', model, '--mapping', mapping,
                              '--out', self.root / 'converted', success=False)
        self.assertIn('unsupported model', result.stderr)

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'optional NumPy dependency is absent')
    def test_evaluation_time_gauge_and_optional_spatial_alignment(self):
        data, result = self.root / 'data', self.root / 'result'
        data.mkdir()
        result.mkdir()
        points = [(0,0,0),(1,0,0),(0,2,0),(0,0,3)]
        truth, estimated, observations = [], [], []
        for i, (x,y,z) in enumerate(points):
            camera, frame = i % 2, i // 2
            truth.append([0,camera,frame,x,y,z])
            estimated.append([0,camera,frame,frame/30,-2*y+5,2*x-2,2*z+1])
            observations.append([0,0,1,camera,frame,320,240,1])
        write_csv(data / 'truth.csv', ['track_id','camera_id','frame','x','y','z'], truth)
        write_csv(result / 'trajectories.csv', ['track_id','camera_id','frame','time','x','y','z'], estimated)
        write_csv(data / 'observations.csv', ['track_id','is_static','motion_weight','camera_id','frame','u','v','weight'], observations)
        write_csv(data / 'true_cameras.csv', ['camera_id','fps','beta'], [[0,30,0],[1,30,.37]])
        write_csv(result / 'cameras.csv', ['camera_id','fps','beta'], [[0,30,3],[1,30,3.37]])
        write_csv(result / 'metrics.csv', ['stage','cost','image_cost','motion_cost','reprojection_rmse','usable','order_valid'], [['global',1,.5,.5,.1,1,1]])
        completed = self.command('evaluate_alignment.py', '--input', data, '--result', result, '--align-similarity')
        report = json.loads(completed.stdout)
        self.assertEqual(report['observation_coverage'], 1)
        self.assertGreater(report['all_trajectory_rmse_m'], 1)
        self.assertLess(report['all_similarity_aligned_rmse_m'], 1e-12)
        self.assertLess(report['offset_max_error_frames'], 1e-12)
        no_truth = json.loads(self.command('evaluate_alignment.py', '--result', result).stdout)
        self.assertEqual(no_truth['ground_truth'], 'not provided')
        self.assertNotIn('all_trajectory_rmse_m', no_truth)

    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'optional OpenCV dependency is absent')
    def test_marker_corner_identity_from_rendered_videos(self):
        import cv2
        import numpy as np
        if not hasattr(cv2, 'aruco'):
            self.skipTest('OpenCV has no aruco module')
        calibration = self.root / 'calibration'
        calibration.mkdir()
        write_csv(calibration / 'cameras.csv', ['camera_id','fps','beta'], [[0,30,0],[1,30,0]])
        fields = ['view_id','camera_id','frame','fx','fy','cx','cy','rx','ry','rz','tx','ty','tz']
        write_csv(calibration / 'views.csv', fields,
                  [[0,0,-1,500,500,320,240,0,0,0,0,0,0], [1,1,-1,500,500,320,240,0,0,0,1,0,0]])
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 7, 160)
        for camera in [0,1]:
            path = self.root / f'{camera}.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (640,480))
            self.assertTrue(writer.isOpened())
            for frame in range(6):
                image = np.full((480,640,3), 255, dtype=np.uint8)
                left = 80 + 20*camera + 4*frame
                image[100:260, left:left+160] = marker[:,:,None]
                writer.write(image)
            writer.release()
        videos = self.root / 'videos.csv'
        write_csv(videos, ['camera_id','path','frame_offset'], [[0,'0.avi',0],[1,'1.avi',0]])
        output = self.root / 'tracks'
        self.command('extract_markers.py', '--calibration', calibration, '--videos', videos, '--out', output)
        with (output / 'observations.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 48)
        self.assertEqual(len({r['track_id'] for r in rows}), 4)
        for track in {r['track_id'] for r in rows}:
            self.assertEqual({r['camera_id'] for r in rows if r['track_id']==track}, {'0','1'})
        self.command('extract_markers.py', '--calibration', calibration, '--videos', videos,
                     '--out', output, success=False)


if __name__ == '__main__':
    unittest.main()
