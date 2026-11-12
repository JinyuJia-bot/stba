"""Independent LSM checks; analytical image rendering never warps a raster.

Run with the experiment's Python environment via unittest discovery.  The
finite-difference check evaluates the implemented bilinear objective itself;
convergence tests instead render both views from a continuous analytic texture.
"""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lsm import LSMConfig, bilinear, patch_offsets, patch_residual_jacobian, refine_points

torch.set_num_threads(2)


def texture(x, y):
    return (.5 + .12*np.sin(.31*x+.23*y) + .095*np.cos(.19*x-.37*y)
            + .065*np.sin(.43*x+.09*y) + .04*np.cos(.11*x+.47*y))


def analytic_pair(A=None, shift=(.65, -.4), gain=1., bias=0., size=72):
    """Target(q)=gain*T(inv(A)@(q-shift))+bias, with no source resampling."""
    A = np.eye(2) if A is None else np.asarray(A, dtype=float)
    y, x = np.mgrid[:size, :size]
    uv = np.stack([x, y], axis=-1).astype(float)
    source = (uv-np.asarray(shift)) @ np.linalg.inv(A).T
    return texture(x, y), gain*texture(source[..., 0], source[..., 1])+bias


class LSMTests(unittest.TestCase):
    def config(self, **overrides):
        values = dict(model='translation', patch_size=15, max_iterations=45,
                      batch_size=16, photo_rmse_max=.025)
        values.update(overrides)
        return LSMConfig(**values)

    def refine(self, a, b, p, q, **kwargs):
        cfg = kwargs.pop('config', self.config())
        return refine_points(a, b, np.asarray(p, float), np.asarray(q, float),
                             cfg, dtype=torch.float64, device='cpu', **kwargs)

    def test_bilinear_validity_and_derivatives(self):
        y, x = torch.meshgrid(torch.arange(8, dtype=torch.float64),
                              torch.arange(9, dtype=torch.float64), indexing='ij')
        image = .2*x+.3*y+.01*x*y
        xy = torch.tensor([[2.35, 3.12], [8., 7.], [float('nan'), 2.],
                           [-.001, 3.], [2., 7.001]], dtype=torch.float64)
        value, grad, valid = bilinear(image, xy, True)
        self.assertEqual(valid.tolist(), [True, True, False, False, False])
        q = xy[:2]
        torch.testing.assert_close(value[:2], .2*q[:, 0]+.3*q[:, 1]+.01*q.prod(-1))
        torch.testing.assert_close(grad[:2], torch.stack([.2+.01*q[:, 1], .3+.01*q[:, 0]], -1))

    def test_exact_patch_jacobian_translation_and_affine(self):
        generator = torch.Generator().manual_seed(918)
        target = torch.rand(40, 42, generator=generator, dtype=torch.float64)
        patch = torch.rand(2, 25, generator=generator, dtype=torch.float64)
        q0 = torch.tensor([[17.23, 18.37], [24.63, 20.18]], dtype=torch.float64)
        offsets = patch_offsets(5, 'cpu', torch.float64)
        for model in ['translation', 'affine']:
            if model == 'translation':
                theta = torch.tensor([[.14, .12, 1.1, .03], [-.08, .19, .9, -.02]], dtype=torch.float64)
            else:
                theta = torch.tensor([[.14,.12,1.02,.03,-.02,.97,1.1,.03],
                                      [-.08,.19,.98,-.01,.025,1.01,.9,-.02]], dtype=torch.float64)
            _, jacobian, inside, _ = patch_residual_jacobian(target, patch, q0, offsets, theta, model)
            self.assertTrue(inside.all())
            for column in range(theta.shape[1]):
                step = torch.zeros_like(theta); step[:, column] = 1e-6
                plus = patch_residual_jacobian(target, patch, q0, offsets, theta+step, model)[0]
                minus = patch_residual_jacobian(target, patch, q0, offsets, theta-step, model)[0]
                numeric = (plus-minus)/(2e-6)
                torch.testing.assert_close(jacobian[..., column], numeric, rtol=1e-7, atol=3e-8)

    def test_known_subpixel_translation(self):
        source, target = analytic_pair()
        p = np.array([[27., 30.], [38., 38.], [46., 29.]])
        truth = p+[.65, -.4]
        initial = truth+[.5, -.3]
        result = self.refine(source, target, p, initial)
        self.assertTrue(result['refined_mask'].all(), result['refinement_status'])
        self.assertLess(np.linalg.norm(result['uv_refined']-truth, axis=-1).max(), .055)
        self.assertTrue((result['photo_error_after'] < result['photo_error_before']).all())

    def test_affine_and_gain_bias(self):
        A = np.array([[1.025, -.065], [.055, .98]])
        shift = np.array([1.2, -.8]); gain = 1.12; bias = .025
        source, target = analytic_pair(A, shift, gain, bias)
        p = np.array([[28., 31.], [38., 40.], [47., 27.]])
        truth = p @ A.T+shift
        result = self.refine(source, target, p, truth+[.5, -.3],
                             config=self.config(model='affine'))
        self.assertTrue(result['refined_mask'].all(), result['refinement_status'])
        self.assertLess(np.linalg.norm(result['uv_refined']-truth, axis=-1).max(), .08)
        np.testing.assert_allclose(result['A'], np.broadcast_to(A, result['A'].shape), atol=.02)
        np.testing.assert_allclose(result['gain'], gain, atol=.035)
        np.testing.assert_allclose(result['bias'], bias, atol=.02)

    def test_nan_and_boundaries_are_not_refined(self):
        source, target = analytic_pair()
        p = [[30.,30.], [30.,30.], [1.,1.], [float('nan'),30.]]
        q = [[float('nan'),30.], [80.,30.], [20.,20.], [30.,30.]]
        result = self.refine(source, target, p, q)
        np.testing.assert_array_equal(result['refinement_status'], [2, 4, 3, 2])
        self.assertFalse(result['refined_mask'].any())

    def test_flat_texture_is_rejected(self):
        image = np.ones((48,48))*.5
        result = self.refine(image, image, [[24.,24.]], [[24.4,23.7]])
        self.assertEqual(result['refinement_status'][0], 5)
        self.assertFalse(result['refined_mask'][0])

    def test_max_correction_is_a_failure_not_success(self):
        source, target = analytic_pair()
        p = np.array([[30.,30.]])
        result = self.refine(source, target, p, p+[.65,-.4]+[1.4,0.],
                             config=self.config(max_correction=.4))
        self.assertLessEqual(result['refinement_displacement'][0], .4+1e-12)
        self.assertEqual(result['refinement_status'][0], 9)
        self.assertFalse(result['refined_mask'][0])

    def test_nonfinite_image_raises_clear_error(self):
        source, target = analytic_pair(); source[1,1] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.refine(source, target, [[30.,30.]], [[30.,30.]])

    def test_nonfinite_affine_is_rejected_without_solver_crash(self):
        source, target = analytic_pair()
        A = np.eye(2)[None]; A[0,0,0] = np.nan
        result = self.refine(source, target, [[30.,30.]], [[30.,30.]],
                             A0=A, config=self.config(model='affine'))
        self.assertEqual(result['refinement_status'][0], 7)
        self.assertFalse(result['refined_mask'][0])

    def test_evaluation_coverage_does_not_count_fallback_or_hide_rejections(self):
        from run_optimizer_checks import evaluate_run
        truth = np.zeros((3,2))
        raw = np.array([[2.,0.], [2.,0.], [2.,0.]])
        proposal = np.array([[.1,0.], [.1,0.], [.1,0.]])
        success = np.array([True,False,False])
        metrics, _, _ = evaluate_run(raw,proposal,success,np.ones(3,bool),
            truth,np.ones(3,bool),np.array([1,8,5]),np.zeros(3,int),np.zeros(3,bool))
        self.assertEqual(metrics['refined_own_valid_epe']['count'], 1)
        self.assertEqual(metrics['refined_own_valid_epe']['fraction_lt_1'], 1.)
        self.assertEqual(metrics['refined_correct_coverage']['correct_coverage_lt_1p0'], 1/3)
        self.assertEqual(metrics['success_evaluable_coverage'], 1/3)
        self.assertEqual(metrics['paired_raw_epe']['count'], 1)


if __name__ == '__main__':
    unittest.main()
