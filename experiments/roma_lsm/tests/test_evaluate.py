"""Independent denominator/selection tests; do not mirror optimizer behavior."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.roma_lsm.evaluate import evaluate_pair,METHODS


class EvaluationTests(unittest.TestCase):
    def fixture(self,path):
        shape=(2,5)
        roi=np.ones(shape,bool)
        truth={'uv':np.zeros(shape+(2,),np.float32),'valid':roi.copy()}
        truth['valid'][1,4]=False  # Nine method-independent evaluable pixels.
        raw=np.zeros(shape+(2,),np.float32);raw[...,0]=.8
        raw[0,4,0]=4.0  # Model-valid hard pixel removed by screened A0.
        for name in METHODS:
            valid=roi.copy();valid[0,4]=False
            if name=='A1_translation':valid[1,:3]=False
            if name=='A1_affine':valid[1,0]=False
            if name=='A2_affine':valid[1,1]=False
            uv=raw.copy()
            if name.startswith('A1') or name.startswith('A2'):
                uv[...,0]=.4
            if name=='A2_affine':uv[0,0,0]=1.2
            control=name in ['A0','A0g']
            attempted=np.zeros(shape,bool) if control else roi.copy()
            refined=np.zeros(shape,bool) if control else valid.copy()
            if name=='A1_affine':refined[0,0]=False  # valid fallback is NOT success.
            np.savez_compressed(path/f'{name}.npz',uv_raw=raw,uv_refined=uv,
                confidence_raw=np.ones(shape),valid_model=roi,valid_raw=roi,
                valid_refined=valid,refined_mask=refined,attempt_mask=attempted,
                reference_roi=roi,geometry_applicable_mask=roi,
                refinement_status=np.where(refined,1,8))
        (path/'meta.json').write_text(json.dumps({'dataset_metadata_path':None}))
        return truth

    def test_denominator_and_filtering_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);truth=self.fixture(path);m=evaluate_pair(path,truth)
            self.assertEqual(m['original_gt_evaluable_roi_pixels'],9)
            # Unscreened hard pixel remains in evaluation; A0 is not mislabeled raw-all.
            raw=m['unfiltered_raw_model']['endpoint_error_px']
            self.assertEqual(raw['count'],9)
            self.assertAlmostEqual(raw['correct_coverage_lt_1_px'],8/9)
            screened=m['methods']['A0']['scopes']['own']['endpoint_error_px']
            self.assertEqual(screened['count'],8)
            self.assertAlmostEqual(screened['valid_coverage_of_original_evaluable_roi'],8/9)
            self.assertAlmostEqual(screened['fraction_lt_1_px'],1.)
            self.assertAlmostEqual(screened['correct_coverage_lt_1_px'],8/9)
            self.assertLess(m['common_all']['pixels'],m['common_primary']['pixels'])
            counts=[m['methods'][n]['scopes']['common_primary']['endpoint_error_px']['count']
                    for n in ['A0','A1_affine','A2_affine']]
            self.assertEqual(len(set(counts)),1)
            self.assertNotIn('common_primary',m['methods']['A1_translation']['scopes'])

    def test_fallback_and_worsening(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);truth=self.fixture(path);m=evaluate_pair(path,truth)
            aff=m['methods']['A1_affine']
            self.assertEqual(aff['successful_refinement_pixels'],7)
            self.assertEqual(aff['failed_attempted_pixels'],3)
            self.assertEqual(aff['valid_but_unrefined_pixels'],1)
            self.assertEqual(m['methods']['A0']['successful_refinement_pixels'],0)
            self.assertIsNone(m['methods']['A0']['success_fraction_of_attempted'])
            self.assertEqual(m['methods']['A2_affine']['accepted_refinement_worsened_vs_raw']['worsened_pixels'],1)

    def test_real_has_no_accuracy_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);self.fixture(path);m=evaluate_pair(path)
            self.assertFalse(m['ground_truth_available'])
            self.assertNotIn('endpoint_error_px',m['methods']['A0']['scopes']['own'])
            self.assertNotIn('endpoint_error_px',m['unfiltered_raw_model'])

    def test_different_roi_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);truth=self.fixture(path)
            with np.load(path/'A2_affine.npz') as archive:
                data={k:archive[k] for k in archive.files}
            data['reference_roi'][0,0]=False
            np.savez_compressed(path/'A2_affine.npz',**data)
            with self.assertRaisesRegex(ValueError,'share reference_roi'):
                evaluate_pair(path,truth)


if __name__=='__main__':
    unittest.main()
