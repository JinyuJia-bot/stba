"""Regression checks for independent metrics, coverage and clock sign."""
from pathlib import Path
import json
import hashlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from experiments.dense_stba.evaluate import evaluate_case,beta_comparison,reprojection_metrics,load_verified_observations,shared_holdout_metrics
from experiments.dense_stba.synthetic_dynamic import load_truth


class EvaluationTests(unittest.TestCase):
    def test_shared_holdout_uses_identical_raw_measurements_and_windows(self):
        arrays={};metas={}
        for i,name in enumerate(['B0','B1','B3','B4']):
            predicted=np.zeros((2,1,4,2));predicted[...,0]=i
            observed=np.zeros_like(predicted)
            if name=='B4':observed[...,0]=10
            arrays[name]=dict(predicted_uv=predicted,observed_uv=observed,holdout_visible=np.ones((2,1,4),bool))
            metas[name]={}
        arrays['B3']['predicted_uv'][0,0,0]=np.nan
        arrays['B4']['predicted_uv'][1,0,1]=np.nan
        metas['B4']['window']=[1,3]
        result=shared_holdout_metrics(arrays,metas)
        self.assertEqual(result['primary']['observations'],7)
        self.assertEqual(result['all_methods_fixed_raw']['observations'],3)
        self.assertAlmostEqual(result['primary']['metrics']['B3']['rmse'],2.)
        self.assertAlmostEqual(result['all_methods_fixed_raw']['metrics']['B4']['rmse'],3.)

    def test_modified_or_mismatched_frontend_cache_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='dense-stba-cache-test-') as tmp:
            path=Path(tmp)/'observations.npz';uv=np.zeros((2,3,4,2));ids=np.arange(2);seed=np.zeros((2,2))
            np.savez(path,uv=uv,point_ids=ids,seed_uv=seed)
            metadata=dict(observations_path=str(path),observations_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            result=dict(observed_uv=uv.copy(),point_ids=ids,seed_uv=seed)
            _,provenance=load_verified_observations(result,metadata,tmp)
            self.assertTrue(provenance['verified'])
            result['observed_uv'][0,0,0,0]=1
            with self.assertRaisesRegex(ValueError,'differs'):
                load_verified_observations(result,metadata,tmp)
            np.savez(path,uv=uv+1,point_ids=ids,seed_uv=seed)
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                load_verified_observations(result,metadata,tmp)

    def test_external_reference_integer_shift_sign(self):
        manifest=dict(reference_camera_id=7,cameras=[dict(camera_id=7,fps=60,integer_raw_frame_shift=0),
            dict(camera_id=9,fps=60,integer_raw_frame_shift=-2)])
        external=dict(cameras=[dict(camera_id=7,raw_beta_seconds=0),dict(camera_id=9,raw_beta_seconds=.04)])
        result=beta_comparison([0,.006666666666666667],manifest,external=external)
        self.assertAlmostEqual(result['cameras'][1]['recovered_total_beta_seconds'],.04)
        self.assertAlmostEqual(result['cameras'][1]['difference_from_published_ms'],0)
        self.assertEqual(result['comparison_kind'],'published_reference_NOT_independent_GT')

    def test_train_holdout_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'overlap'):
            reprojection_metrics(dict(predicted_uv=np.zeros((1,1,1,2)),observed_uv=np.zeros((1,1,1,2)),
                train_visible=np.ones((1,1,1),bool),holdout_visible=np.ones((1,1,1),bool)))

    def test_rejections_reduce_fixed_denominator_coverage(self):
        root=Path(__file__).resolve().parents[1]
        manifest_path=root/'data/synthetic/fractional/manifest.json'
        truth=load_truth(manifest_path.parent);manifest=truth.manifest
        seed=np.array([[80.,60.],[96.,72.],[110.,85.],[120.,65.]])
        times=np.arange(8,24,2)/30
        xyz=truth.evaluate_material_trajectory(seed,times).transpose(1,0,2)
        observed=np.stack([truth.project_material(seed,j,truth.physical_timestamps[j])['uv'].transpose(1,0,2) for j in range(3)],axis=1)
        holdout=np.zeros(observed.shape[:-1],bool);holdout[:,:,::5]=True;train=~holdout
        common=dict(point_ids=np.arange(4),seed_uv=seed,trajectory_times=times,view_support=np.full((4,len(times)),3),
            beta=truth.beta_seconds.copy(),initial_beta=np.zeros(3),predicted_uv=observed,observed_uv=observed,
            train_visible=train,holdout_visible=holdout,motion_mask=np.ones(4,bool))
        with tempfile.TemporaryDirectory(prefix='dense-stba-eval-') as tmp:
            directory=Path(tmp)
            (directory/'meta.json').write_text(json.dumps(dict(manifest_path=str(manifest_path))))
            for method in ['B0','B3']:
                valid=np.ones((4,len(times)),bool)
                position=xyz.copy()
                if method=='B0':position[:,:,0]+=.02
                else:valid[2:]=False
                np.savez_compressed(directory/f'{method}.npz',**common,xyz=position,valid=valid,trajectory_valid=valid.any(-1))
                (directory/f'{method}.json').write_text(json.dumps(dict(method=method,oracle_beta=False)))
            result=evaluate_case(directory)
        b0=result['methods']['B0']['scopes'];b3=result['methods']['B3']['scopes']
        self.assertEqual(result['original_gt_evaluable_samples'],4*len(times))
        self.assertAlmostEqual(b3['own']['position_error_m']['valid_coverage_of_original_evaluable_samples'],.5)
        self.assertAlmostEqual(b3['own']['position_error_m']['correct_coverage_lt_10mm'],.5)
        self.assertAlmostEqual(b0['common_all']['position_error_m']['rmse'],.02)
        self.assertAlmostEqual(b3['common_all']['position_error_m']['rmse'],0.)
        self.assertEqual(result['methods']['B3']['support']['recovered_with_valid_samples'],2)
        self.assertAlmostEqual(result['methods']['B3']['reprojection']['holdout']['rmse'],0.)
        self.assertEqual(result['common_all']['valid_samples'],2*len(times))


if __name__=='__main__':unittest.main()
