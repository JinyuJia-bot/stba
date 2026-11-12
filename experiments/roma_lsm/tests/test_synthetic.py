"""Independent checks of calibrated synthetic GT and non-raster-warp rendering."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import synthetic as s


class SyntheticTests(unittest.TestCase):
    def test_calibrated_subpixel_translation(self):
        cameras, scene, *_ = s.case_definition('translation')
        truth = s.ground_truth(cameras[0], cameras[1], scene)
        # Independent analytical formula for a fronto-parallel plane at z=4.
        offset = np.array([-180*.237/4, -176*(-.063)/4])
        expected = s.pixel_grid()+offset
        np.testing.assert_allclose(truth['uv'], expected, atol=8e-6)
        self.assertTrue(truth['valid'].any())
        self.assertFalse(truth['occluded'].any())

    def test_affine_homography(self):
        cameras, scene, *_ = s.case_definition('affine_brightness')
        a, b = cameras[:2]
        k0, k1, r1, t1 = map(np.array, [a['K'], b['K'], b['R'], b['t']])
        # Plane z=4, reference R=I, C=0. Homography independently derived.
        h = k1 @ (r1 + np.outer(t1, [0, 0, 1])/4) @ np.linalg.inv(k0)
        homogeneous = np.dstack([s.pixel_grid(), np.ones(s.SIZE[::-1])])
        q = homogeneous @ h.T
        q = q[..., :2]/q[..., 2, None]
        truth = s.ground_truth(a, b, scene)
        np.testing.assert_allclose(truth['uv'], q, atol=1e-5)
        self.assertLess(np.max(np.abs(h[2, :2])), 1e-12)
        self.assertGreater(abs(h[0, 1]), .01)

    def test_ray_distance_is_not_z_depth(self):
        cameras, scene, *_ = s.case_definition('occlusion_weak_noise')
        grid = s.pixel_grid()
        xyz, distance, surface, valid = s.intersect_scene(cameras[0], grid, True)
        center, direction = s.rays(cameras[0], grid)
        np.testing.assert_allclose(np.linalg.norm(direction, axis=-1), 1, atol=1e-12)
        np.testing.assert_allclose(center+distance[..., None]*direction, xyz, atol=1e-12)
        self.assertGreater(np.max(np.abs(distance-xyz[..., 2])), .1)
        self.assertTrue(valid.all())
        self.assertTrue((surface == 1).any())

    def test_true_occlusion_and_visibility(self):
        cameras, scene, *_ = s.case_definition('occlusion_weak_noise')
        truth = s.ground_truth(cameras[0], cameras[1], scene)
        self.assertGreater(truth['occluded'].sum(), 0)
        self.assertFalse((truth['valid'] & truth['occluded']).any())
        self.assertTrue(np.array_equal(truth['valid'] | truth['occluded'], truth['inside']))
        # For visibility, target backprojection must hit the same world point.
        hit, *_ = s.intersect_scene(cameras[1], truth['uv'].astype(float), True)
        error = np.linalg.norm(hit-truth['xyz'], axis=-1)
        self.assertLess(np.max(error[truth['valid']]), 2e-6)
        self.assertGreater(np.min(error[truth['occluded']]), .9)

    def test_continuous_render_and_weak_texture(self):
        cameras, scene, gains, biases, sigmas, _ = s.case_definition('occlusion_weak_noise')
        image, clipping = s.render(cameras[0], scene, samples_per_axis=4)
        self.assertEqual(image.shape, s.SIZE[::-1])
        self.assertEqual(image.dtype, np.float32)
        self.assertTrue(np.isfinite(image).all())
        self.assertEqual(clipping, 0.)
        xyz, _, surface, _ = s.intersect_scene(cameras[0], s.pixel_grid(), True)
        weak = (surface == 0) & s._in_rect(xyz, s.WEAK_RECT, inset=.08)
        texture = (surface == 0) & ~s._in_rect(xyz, s.WEAK_RECT, inset=-.08)
        self.assertGreater(weak.sum(), 100)
        self.assertLess(image[weak].std(), image[texture].std()/20)

    def test_controlled_initialization_error(self):
        cameras, scene, *_ = s.case_definition('translation')
        uv = s.ground_truth(cameras[0], cameras[1], scene)['uv']
        for error in [.5, 2., 6.]:
            initial = s.perturbed_initialization(uv, error, seed=13)
            norm = np.linalg.norm(initial-uv, axis=-1)
            # Correspondence fields are float32; subtracting coordinates near
            # x=192 introduces a few ulps even for an exact float64 displacement.
            tolerance = 2*np.spacing(np.float32(max(s.SIZE)))
            np.testing.assert_allclose(norm, error, atol=tolerance)


if __name__ == '__main__':
    unittest.main()
