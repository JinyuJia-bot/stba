import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.dense_stba.debug_viewer.server import create_app
try:
    from .test_adapter import make_fixture
except ImportError:
    from test_adapter import make_fixture


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'experiment'
        self.root.mkdir()
        make_fixture(self.root)
        self.workspace = Path(self.temp.name) / 'viewer-only'
        self.app = create_app(self.root, self.workspace)
        self.client = self.app.test_client()
        self.query = 'run=results/toy&method=B3&frame=1&point=26&camera=1&roi=full'

    def tearDown(self):
        self.app.extensions['dense_store'].close()
        self.temp.cleanup()

    def test_image_point_and_unqueried_pixel_http_contract(self):
        self.assertTrue(self.client.get('/health').get_json()['read_only_experiments'])
        point = self.client.get('/api/point?' + self.query)
        self.assertEqual(point.status_code, 200)
        self.assertEqual(point.get_json()['point_id'], 26)
        json.dumps(point.get_json(), allow_nan=False)
        image = self.client.get('/api/image?' + self.query)
        self.assertEqual(image.mimetype, 'image/png')
        self.assertTrue(image.data.startswith(b'\x89PNG'))
        pixel = self.client.get('/api/pixel?' + self.query + '&x=3&y=2').get_json()
        self.assertIsNone(pixel['point_id'])
        self.assertEqual(pixel['status'], 'not_queried')

    def test_coverage_sends_small_raster_and_bounded_point_page(self):
        result = self.client.get('/api/coverage?' + self.query)
        self.assertEqual(result.status_code, 200)
        data = result.get_json()
        self.assertNotIn('points', data)
        raw = base64.b64decode(data['overlay_png'].split(',')[1])
        self.assertTrue(raw.startswith(b'\x89PNG'))
        page = self.client.get('/api/points?' + self.query + '&limit=1').get_json()
        self.assertEqual(len(page['points']), 1)
        self.assertEqual(self.client.get('/api/points?' + self.query + '&limit=1000000').status_code, 400)

    def test_explicit_roi_save_does_not_modify_experiment_files(self):
        files = [p for p in self.root.rglob('*') if p.is_file()]
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        before = {str(p): digest(p) for p in files}
        result = self.client.post('/api/rois', json={'name': 'test_roi', 'run': 'results/toy',
                                 'method': 'B3', 'roi': {'type': 'polygon', 'vertices': [[1, 1], [6, 1], [3, 6]]}})
        self.assertEqual(result.status_code, 200, result.get_json())
        self.assertTrue((self.workspace / 'rois/test_roi.json').is_file())
        self.assertEqual(before, {str(p): digest(p) for p in files})
        self.assertEqual(self.client.get('/api/rois').get_json()['rois'][0]['name'], 'test_roi')

    def test_workspace_and_invalid_path_guards(self):
        with self.assertRaises(ValueError):
            create_app(self.root, self.root / 'results/export')
        result = self.client.post('/api/rois', json={'name': '../../meta', 'run': 'results/toy',
                                 'roi': {'type': 'rectangle', 'bounds': [0, 0, 2, 2]}})
        self.assertEqual(result.status_code, 400)
        self.assertEqual(self.client.get('/api/run?run=../../outside&method=B3').status_code, 400)
        self.assertEqual(self.client.get('/api/point?' + self.query.replace('point=26', 'point=999')).status_code, 400)
        self.assertEqual(self.client.get('/api/image?' + self.query.replace('frame=1', 'frame=-1')).status_code, 400)


if __name__ == '__main__':
    unittest.main()
