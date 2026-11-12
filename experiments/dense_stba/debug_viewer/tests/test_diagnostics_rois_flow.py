import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from experiments.dense_stba.debug_viewer.diagnostics import coverage, point_list
from experiments.dense_stba.debug_viewer import rois
from experiments.dense_stba.debug_viewer.flow_debug import flow_diagnostics


def fixture():
    n, c, t = 6, 3, 11
    points = np.array([[207, 145], [230, 150], [231, 180], [240, 212], [241, 213], [300, 60]], np.float32)
    status = np.ones((n, c, t), np.int16)
    status[3, 1, 2:] = 6
    edge = np.ones((n, c), np.int16)
    edge[:, 1] = [2, 3, 4, 4, 1, 1]
    support = np.ones((n, c, 3), bool)
    support[2, 1, 0] = False
    support[3, 1] = False
    error = np.zeros((n, c, 3), np.float32)
    error[2, 1, 2] = 2.
    multiview = np.array([False, False, False, False, True, True])
    obs = dict(point_ids=np.arange(n), seed_uv=points, seed_frame=np.array(5), status=status,
               uv=np.broadcast_to(points[:, None, None, :], (n, c, t, 2)).copy(),
               edge_status=edge, anchor_support=support, anchor_error=error, anchor_frames=np.array([0, 5, 10]),
               roma_confidence=np.ones((n, c)), roma_fb_error=np.zeros((n, c)),
               temporal_success=np.ones(n, bool), multiview_seed=multiview, all_three_seed=multiview,
               visible=np.zeros((n, c, t), bool), temporal_fb_error=np.zeros((n, c, t)))
    result = dict(seed_eligible=np.array([False, False, False, False, True, True]),
                  trajectory_valid=np.array([False, False, False, False, False, True]))
    cameras = [dict(camera_id=cid, size=[480, 270]) for cid in [7, 9, 10]]
    run = SimpleNamespace(run_id='test', method='B3', obs=obs, result=result, cameras=cameras,
                          point_ids=obs['point_ids'], method_meta={}, meta={}, manifest={}, cache_dir=None)
    run.index = lambda point_id: int(np.flatnonzero(run.point_ids == point_id)[0])
    return run


class ROITests(unittest.TestCase):
    def test_seed_coordinates_half_open_and_background(self):
        run = fixture()
        selected, spec = rois.resolve_roi(run, 'dog')
        np.testing.assert_array_equal(selected, [1, 1, 1, 1, 0, 0])
        run.obs['uv'][:] = 0  # Current trajectories cannot change seed selection.
        np.testing.assert_array_equal(rois.resolve_roi(run, 'dog')[0], selected)
        person = rois.resolve_roi(run, 'person')[0]
        background = rois.resolve_roi(run, 'background')[0]
        np.testing.assert_array_equal(background, ~(selected | person))
        self.assertEqual(spec['seed_frame'], 5)

    def test_polygon_boundary_and_ids(self):
        run = fixture()
        polygon = {'type': 'polygon', 'vertices': [[207, 145], [230, 145], [230, 150], [207, 150]]}
        np.testing.assert_array_equal(rois.resolve_roi(run, polygon)[0], [1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(rois.resolve_roi(run, {'type': 'point_ids', 'point_ids': [3]})[0], [0, 0, 0, 1, 0, 0])

    def test_save_requires_explicit_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(rois, 'WORKSPACE', Path(directory)):
            with self.assertRaises(ValueError):
                rois.save_roi('test', 'full')
            with self.assertRaises(ValueError):
                rois.save_roi('../escape', 'full', explicit=True)
            rois.save_roi('dog_selection', 'dog', run=fixture(), explicit=True)
            self.assertEqual(rois.load_roi('dog_selection')['roi']['type'], 'dog')
            self.assertEqual(len(rois.list_rois()), 1)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ['rois'])

    def test_save_refuses_workspace_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'outside').mkdir()
            (root / 'workspace').symlink_to(root / 'outside', target_is_directory=True)
            with patch.object(rois, 'WORKSPACE', root / 'workspace'), self.assertRaises(ValueError):
                rois.save_roi('test', 'full', explicit=True)
            self.assertEqual(list((root / 'outside').iterdir()), [])


class CoverageTests(unittest.TestCase):
    def test_recorded_order_and_overlapping_edge4_reasons(self):
        run = fixture()
        result = coverage(run)
        self.assertEqual(result['points']['category'], ['roma_initial', 'collision', 'flow_or_anchors', 'flow_or_anchors', 'trajectory', 'success'])
        subdivision = result['edge4_subdivision']['counts']
        self.assertEqual(subdivision['target_observation_count_insufficient'], 1)
        self.assertEqual(subdivision['independent_anchor_support_insufficient'], 2)
        self.assertEqual(subdivision['independent_anchor_error_above_threshold'], 1)
        self.assertEqual(result['stage_counts']['flow']['count'], 5)  # visible is all false, status remains original.
        self.assertEqual(result['stage_counts']['roma_initial_valid']['count'], 5)
        self.assertEqual(result['stage_counts']['collision_free']['count'], 4)
        json.dumps(result, allow_nan=False)

    def test_seed_anchor_never_counts_as_independent(self):
        run = fixture()
        run.obs['anchor_support'][4, 1] = [False, True, False]
        result = point_list(run, reason=None)
        self.assertEqual(result['points'][4]['independent_anchor_support'], 0)
        self.assertIsNone(result['points'][4]['max_anchor_error'])

    def test_missing_and_point_pagination(self):
        run = fixture()
        del run.obs['anchor_support']
        result = coverage(run)
        self.assertIn('obs.anchor_support', result['missing_fields'])
        self.assertIsNone(result['edge4_subdivision']['counts']['independent_anchor_support_insufficient'])
        self.assertEqual(point_list(run, reason='collision')['points'][0]['point_id'], 1)
        page = point_list(run, sort='point_id', descending=True, offset=1, limit=2)
        self.assertEqual([p['point_id'] for p in page['points']], [4, 3])
        self.assertEqual(coverage(run, stage='trajectory_valid')['points']['category'], ['fail'] * 5 + ['pass'])


class FlowCacheTests(unittest.TestCase):
    def make_cache(self, root):
        length, width, height, camera_id = 11, 16, 12, 7
        images = np.random.default_rng(9).integers(0, 256, (length, height, width, 3), dtype=np.uint8)
        hashes = [hashlib.sha256(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).tobytes()).hexdigest() for image in images]
        settings = {'method': 'dis', 'skip_stride': 5}
        identity = {'settings': settings, 'opencv': cv2.__version__, 'size': [width, height], 'camera_id': camera_id}
        encoded = json.dumps(identity, sort_keys=True)
        directory = root / 'temporal/cam7'
        directory.mkdir(parents=True)
        metadata = {**identity, 'raw_frame_ids': list(range(length))}
        (directory / 'sequence_0_10_test.json').write_text(json.dumps(metadata))
        for a, b in [(i, i + 1) for i in range(length - 1)] + [(0, 5), (5, 10)]:
            key = hashlib.sha256((encoded + hashes[a] + hashes[b]).encode()).hexdigest()[:20]
            forward = np.zeros((height, width, 2), np.float32)
            forward[..., 0] = .2 * (b - a)
            np.savez_compressed(directory / f'raw_{a}_{b}_{key}.npz', forward=forward, backward=-forward)
        uv = np.array([[[[5 + .2 * (t - 5), 5] for t in range(length)]]], np.float32)
        obs = dict(status=np.ones((1, 1, length), np.int16), uv=uv, seed_frame=np.array(5),
                   visible=np.ones((1, 1, length), bool), temporal_fb_error=np.zeros((1, 1, length)))
        run = SimpleNamespace(run_id='cachetest', method='B3', cache_dir=root / 'case', obs=obs,
                              cameras=[dict(camera_id=7, size=[width, height], raw_frame_ids=list(range(length)))])
        run.index = lambda point_id: 0
        run.image = lambda camera_index, frame: images[frame]
        return run, images, directory

    def test_read_only_verified_replay_matches_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            run, _, cache = self.make_cache(Path(directory))
            before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.iterdir()}
            result = flow_diagnostics(run, 1, camera_index=0)
            self.assertTrue(result['cache_replay']['available'])
            self.assertAlmostEqual(result['cache_replay']['forward']['fb_error_px'], 0, places=6)
            self.assertLess(result['cache_replay']['direct_vs_composed']['direct_vs_composed_error_px'], 1e-5)
            self.assertFalse(result['recorded']['cycle_error_available'])
            self.assertIsNone(result['recorded']['first_failure_forward'])
            self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.iterdir()})
            json.dumps(result, allow_nan=False)

    def test_source_image_change_refuses_cached_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            run, images, _ = self.make_cache(Path(directory))
            images[5] = 0
            result = flow_diagnostics(run, 1, camera_index=0)
            self.assertFalse(result['cache_replay']['available'])
            self.assertIn('fingerprint', result['cache_replay']['forward']['missing_reason'])

    def test_missing_cache_and_recorded_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            run, _, _ = self.make_cache(Path(directory))
            run.cache_dir = Path(directory) / 'absent/case'
            run.obs['status'][0, 0, 8:] = [3, 6, 6]
            result = flow_diagnostics(run, 1, camera_index=0)
            self.assertFalse(result['cache_replay']['available'])
            self.assertEqual(result['recorded']['first_failure_forward']['frame'], 8)
            self.assertEqual(result['recorded']['first_failure_forward']['status'], 3)


if __name__ == '__main__':
    unittest.main()
