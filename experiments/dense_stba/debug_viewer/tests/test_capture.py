import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.dense_stba.debug_viewer import capture
from experiments.dense_stba.debug_viewer.adapter import DatasetStore, sha256
try:
    from .test_adapter import make_fixture
except ImportError:
    from test_adapter import make_fixture


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        make_fixture(self.root)
        self.store = DatasetStore(self.root)
        self.run = self.store.load('results/toy')
        self.workspace = self.root / 'viewer_workspace'
        self.allowed = patch.object(capture, 'WORKSPACE_ROOT', self.workspace)
        self.allowed.start()
        # Explicit toy caller arrays; the writer itself must never recover them
        # from final result arrays or run any estimator.
        self.ids = self.run.point_ids[:2].copy()
        self.controls = np.zeros((2, 4, 3))
        self.controls[:, :, 2] = 4.
        self.knots = np.array([-.1] * 4 + [.5] * 4)
        self.beta = np.array([0., .03, -.02])

    def tearDown(self):
        self.allowed.stop()
        self.store.close()
        self.temp.cleanup()

    def save(self, **kwargs):
        inputs = dict(controls=self.controls, knots=self.knots, beta=self.beta,
            point_ids=self.ids, provenance='Unit-test caller supplied a synthetic initialization at capture time')
        inputs.update(kwargs)
        return capture.capture_initialization(self.run, **inputs)

    def test_round_trip_is_independent_bound_readonly_and_does_not_modify_experiment(self):
        paths = [p for p in self.root.rglob('*') if p.is_file()]
        before = {str(p): sha256(p) for p in paths}
        result = self.save()
        self.assertTrue(Path(result['json_path']).is_relative_to(self.workspace))
        restored = capture.load_initialization(self.run, result['json_path'])
        self.assertTrue(restored['available'])
        np.testing.assert_array_equal(restored['arrays']['controls'], self.controls)
        np.testing.assert_array_equal(restored['arrays']['beta'], self.beta)
        self.assertFalse(restored['arrays']['controls'].flags.writeable)
        self.assertFalse(restored['meta']['reconstruction_run'])
        self.assertEqual(restored['meta']['sources']['observations_sha256'], self.run.method_meta['observations_sha256'])
        self.assertEqual(before, {str(p): sha256(p) for p in paths})
        another = self.save()
        self.assertNotEqual(result['json_path'], another['json_path'])
        self.assertTrue(Path(result['json_path']).exists())

    def test_optional_triangulation_preserves_missing_samples_without_inventing_xyz(self):
        xyz = np.ones((2, 3, 3));xyz[1, 2] = np.nan
        times = np.array([0., .1, .2])
        result = self.save(triangulated_xyz=xyz, triangulated_times=times)
        arrays = capture.load_initialization(self.run, result['json_path'])['arrays']
        self.assertFalse(arrays['triangulated_valid'][1, 2])
        self.assertTrue(np.isnan(arrays['triangulated_xyz'][1, 2]).all())
        np.testing.assert_array_equal(arrays['triangulated_times'], times)
        with self.assertRaisesRegex(ValueError, 'finite XYZ'):
            self.save(triangulated_xyz=xyz, triangulated_times=times, triangulated_valid=np.ones((2, 3), bool))

    def test_shapes_identity_provenance_and_reference_gauge_are_required(self):
        cases = [dict(controls=np.zeros((2, 3, 3))), dict(knots=np.arange(6)),
            dict(beta=[.1, .2, .3]), dict(point_ids=[26, 26]), dict(point_ids=[26, 99999]),
            dict(point_ids=[26., 54.]), dict(provenance=' '), dict(controls=np.full((2, 4, 3), np.nan)),
            dict(triangulated_times=[0., .1])]
        for invalid in cases:
            with self.subTest(invalid=tuple(invalid)):
                with self.assertRaises(ValueError):
                    self.save(**invalid)
        self.assertFalse(self.workspace.exists())

    def test_experiment_directory_and_symlink_escape_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'must stay under'):
            self.save(workspace=self.run.directory)
        self.workspace.mkdir()
        (self.workspace / 'escape').symlink_to(self.run.directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'must stay under'):
            self.save(workspace=self.workspace / 'escape')
        (self.workspace / 'escape').unlink();self.workspace.rmdir()
        self.workspace.symlink_to(self.run.directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'must not be a symlink'):
            self.save()

    def test_corruption_and_different_method_cannot_be_loaded(self):
        result = self.save()
        other = self.store.load('results/toy', 'B4')
        with self.assertRaisesRegex(ValueError, 'another run or method'):
            capture.load_initialization(other, result['json_path'])
        with Path(result['npz_path']).open('ab') as handle:
            handle.write(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            capture.load_initialization(self.run, result['json_path'])

    def test_changed_frozen_sources_are_rejected_at_write_and_read(self):
        result = self.save()
        with self.run.observations_path.open('ab') as handle:
            handle.write(b'changed source')
        with self.assertRaisesRegex(ValueError, 'observations SHA256 mismatch'):
            self.save()
        with self.assertRaisesRegex(ValueError, 'observations SHA256 mismatch'):
            capture.load_initialization(self.run, result['json_path'])

    def test_metadata_beta_and_npz_path_must_agree(self):
        result = self.save();path = Path(result['json_path'])
        metadata = json.loads(path.read_text())
        metadata['beta_seconds'][1] += 1
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'beta disagrees'):
            capture.load_initialization(self.run, path)
        metadata['npz_name'] = '../../B3.npz';path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'NPZ name'):
            capture.load_initialization(self.run, path)


if __name__ == '__main__':
    unittest.main()
