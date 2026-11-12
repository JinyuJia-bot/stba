import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.dense_stba.debug_viewer.adapter import DatasetStore, sha256
from experiments.dense_stba.debug_viewer import export_diagnostics as exporter
from experiments.dense_stba.debug_viewer.tests.test_adapter import make_fixture


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'experiment'
        self.workspace = Path(self.temp.name) / 'viewer_workspace'
        make_fixture(self.root)
        self.store = DatasetStore(self.root)
        self.run = self.store.load('results/toy', 'B3')
        self.patch = patch.object(exporter, 'WORKSPACE', self.workspace)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.store.close()
        self.temp.cleanup()

    def test_export_is_read_only_and_missing_not_fabricated(self):
        before = {str(p): sha256(p) for p in self.root.rglob('*') if p.is_file()}
        destination = self.workspace / 'exports/example'
        result = exporter.export_diagnostics(self.run, 26, camera_index=1, frame=1, out=destination)
        after = {str(p): sha256(p) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(sorted(p.name for p in destination.iterdir()), ['alignment.json', 'coverage.json', 'export.json', 'flow.json', 'inspect.json'])
        self.assertFalse(result['initial_controls_recovered'])
        self.assertTrue(result['no_estimation_performed'])
        self.assertTrue(any('control_initial' in value for value in result['missing']))
        self.assertTrue(any('cycle_error' in value for value in result['missing']))
        self.assertFalse(json.loads((destination / 'flow.json').read_text())['cache_replay']['available'])
        self.assertFalse(json.loads((destination / 'inspect.json').read_text())['gt']['available'])
        for path, digest in result['input_sha256'].items():
            self.assertEqual(sha256(path), digest)
        for artifact in result['artifacts'].values():
            self.assertEqual(sha256(artifact['path']), artifact['sha256'])

    def test_experiment_and_parent_paths_rejected_without_writes(self):
        before = {str(p): sha256(p) for p in self.root.rglob('*') if p.is_file()}
        for out in [self.root / 'results/toy/export', self.root / 'data/export', self.workspace / '../escape', self.workspace]:
            with self.subTest(out=out), self.assertRaises(ValueError):
                exporter.export_diagnostics(self.run, 26, out=out)
        self.assertEqual(before, {str(p): sha256(p) for p in self.root.rglob('*') if p.is_file()})
        self.assertFalse(self.workspace.exists())

    def test_symlink_destinations_cannot_redirect_writes(self):
        self.workspace.mkdir()
        (self.workspace / 'redirect').symlink_to(self.root / 'results/toy', target_is_directory=True)
        with self.assertRaises(ValueError):
            exporter.export_diagnostics(self.run, 26, out=self.workspace / 'redirect/export')
        destination = self.workspace / 'export'
        destination.mkdir()
        source = self.root / 'results/toy/meta.json'
        digest = sha256(source)
        (destination / 'inspect.json').symlink_to(source)
        with self.assertRaises(ValueError):
            exporter.export_diagnostics(self.run, 26, out=destination)
        self.assertEqual(sha256(source), digest)

    def test_unknown_point_does_not_create_output(self):
        destination = self.workspace / 'unknown'
        with self.assertRaises(KeyError):
            exporter.export_diagnostics(self.run, 999999, out=destination)
        self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
