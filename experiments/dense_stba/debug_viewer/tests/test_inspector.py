import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import cv2
import numpy as np
from experiments.dense_stba.debug_viewer.adapter import DatasetStore,sha256
from experiments.dense_stba.debug_viewer.inspector import inspect_point,query_pixel,working_to_original,raw_roma_at
try:
    from .test_adapter import make_fixture
except ImportError:
    from test_adapter import make_fixture


class InspectorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);make_fixture(self.root)
        self.store=DatasetStore(self.root);self.run=self.store.load('results/toy')
    def tearDown(self):self.store.close();self.temp.cleanup()
    def test_point_status_time_residual_spline_and_missing_initial_controls(self):
        result=inspect_point(self.run,26,1,include_gt=False);json.dumps(result,allow_nan=False)
        self.assertTrue(result['reconstructed']);self.assertEqual(result['point_id'],26)
        self.assertAlmostEqual(result['cameras'][1]['actual_time'],.12)
        self.assertAlmostEqual(result['cameras'][1]['prediction']['residual_px'],0)
        self.assertEqual(len(result['trajectory']['continuous']['times']),200)
        current=result['trajectory']['current']
        np.testing.assert_allclose(current['velocity'],[8,0,0],atol=1e-12)
        np.testing.assert_allclose(current['acceleration'],0,atol=1e-12)
        self.assertTrue(current['valid']);self.assertFalse(current['model_only'])
        self.assertEqual(current['view_support'],3);self.assertTrue(current['within_observation_support'])
        self.assertTrue(any('control_initial' in f for f in result['missing_fields']))
        self.assertFalse(result['gt']['available'])
        raw=result['cameras'][1]['seed_roma'];self.assertTrue(raw['available'])
        np.testing.assert_allclose(raw['target_uv'],self.run.obs['uv'][0,1,1],atol=1e-6)
    def test_unreconstructed_candidate_is_still_inspectable(self):
        result=inspect_point(self.run,80,1,False)
        self.assertFalse(result['reconstructed']);self.assertFalse(result['trajectory']['continuous']['available'])
        self.assertTrue(result['cameras'][1]['candidate']['temporal_visible'])
        self.assertFalse(result['cameras'][1]['candidate']['visible_after_association'])
        self.assertEqual(result['cameras'][1]['association']['edge_status'],4)
        self.assertIsNone(result['cameras'][1]['prediction']['residual_px'])
    def test_seed_click_does_not_invent_unqueried_trajectory(self):
        self.assertEqual(query_pixel(self.run,2.1,2.1)['point_id'],26)
        missing=query_pixel(self.run,3,2)
        self.assertEqual(missing['status'],'not_queried');self.assertIsNone(missing['point_id'])
        self.assertEqual(missing['nearest_point_id'],26)
        self.assertTrue(missing['raw_roma'][1]['available'])
    def test_nonseed_click_uses_moving_track_not_pixel_id(self):
        uv=self.run.obs['uv'][0,0,3]
        result=query_pixel(self.run,*uv,frame=3)
        self.assertEqual(result['status'],'nearest');self.assertEqual(result['point_id'],26)
        self.assertEqual(result['distance_px'],0)
        self.assertFalse(result['raw_roma'][1]['available'])
        self.assertEqual(query_pixel(self.run,0,9,frame=3)['status'],'no_identity')
    def test_raw_work_mapping_applies_distortion_then_half_pixel_resize(self):
        K=np.array([[320.,0,239.5],[0,315.,134.5],[0,0,1.]])
        raw_K=np.array([[2560.,0,1919.5],[0,2520.,1079.5],[0,0,1.]])
        distortion=np.array([.08,-.03,.001,-.002,.01]);uv=np.array([360.2,90.7])
        camera=dict(K=K,raw_K=raw_K,raw_D=distortion,size=[480,270],raw_size=[3840,2160],source_size=[1920,1080])
        actual=working_to_original(camera,uv)
        ray=np.linalg.solve(K,np.r_[uv,1.]);expected=cv2.projectPoints(ray[None],np.zeros(3),np.zeros(3),raw_K,distortion)[0][0,0]
        np.testing.assert_allclose(actual['uv_raw_distorted'],expected,atol=1e-9)
        np.testing.assert_allclose(actual['uv_source_distorted'],(expected+.5)/2-.5,atol=1e-9)
        recovered=cv2.undistortPoints(expected.reshape(1,1,2),raw_K,distortion,P=K)[0,0]
        np.testing.assert_allclose(recovered,uv,atol=2e-6)
    def test_native_roma_fallback_respects_direction_and_half_pixel_convention(self):
        h=w=8;yy,xx=np.mgrid[:h,:w]
        grid=(np.stack([xx,yy],-1)+.5)*2/np.array([w,h])-1
        shift=np.array([.04,0.])
        warp=np.concatenate([np.concatenate([grid,grid+shift],-1),np.concatenate([grid-shift,grid],-1)],1)[None].astype(np.float32)
        path=self.run.roma_cache_dir/'pair_0f0001_1f0001.npz'
        np.savez_compressed(path,native_warp=warp,native_certainty=np.ones((1,h,w*2),np.float32))
        result=raw_roma_at(self.run,[2,2],1,1)
        np.testing.assert_allclose(result['target_uv'],[2.24,2],atol=1e-6)
        self.assertLess(result['forward_backward_error'],1e-6)
        self.assertTrue(result['raw_model_valid'])
    def test_gt_current_uses_selected_frame_and_independent_physical_time(self):
        np.savez(self.run.manifest_path.parent/'gt.npz',placeholder=np.array(1))
        physical=self.run.result['timestamps']+np.array([0.,.07,-.04])[:,None]
        def trajectory(seeds,times):
            times=np.asarray(times)
            return np.stack([times,np.zeros_like(times),np.ones_like(times)*4],-1)[:,None,:]
        def projection(seeds,camera_id,times):
            times=np.asarray(times)
            return dict(uv=np.stack([times*10+camera_id,times*20],-1)[:,None,:],
                valid=np.ones((len(times),1),bool),occluded=np.zeros((len(times),1),bool))
        truth=SimpleNamespace(physical_timestamps=physical,beta_seconds=np.array([0.,.07,-.04]),
            evaluate_material_trajectory=trajectory,project_material=projection)
        with patch('experiments.dense_stba.synthetic_dynamic.load_truth',return_value=truth):
            result=inspect_point(self.run,26,2)
        gt=result['gt'];self.assertTrue(gt['available']);json.dumps(result,allow_nan=False)
        self.assertEqual(gt['current']['time'],result['trajectory']['current']['time'])
        np.testing.assert_allclose(gt['current']['xyz'],[.2,0,4])
        for c in range(3):
            current=gt['cameras'][c]['current']
            self.assertEqual(current['frame'],2);self.assertEqual(current['physical_time'],physical[c,2])
            np.testing.assert_allclose(current['uv'],[physical[c,2]*10+c,physical[c,2]*20])
        self.assertNotEqual(gt['cameras'][1]['current']['physical_time'],result['cameras'][1]['actual_time'])
    def test_continuous_samples_do_not_claim_support_across_long_gap(self):
        path=self.run.directory/'B3.npz'
        with np.load(path) as z:data={k:z[k] for k in z.files}
        data['observation_inlier'][:]=False;data['observation_inlier'][0,:,0]=True;data['observation_inlier'][0,:,3]=True
        np.savez_compressed(path,**data);self.store.close();self.run=self.store.load('results/toy')
        result=inspect_point(self.run,26,1,False);continuous=result['trajectory']['continuous']
        # 0.3-second endpoint gaps exceed the 0.21-second interpolation policy.
        interior=np.asarray(continuous['times']);center=(interior>.08)&(interior<.22)
        self.assertFalse(np.asarray(continuous['valid'])[center].any())
    def test_readonly_inspection_and_explicit_missing_field(self):
        files=[p for p in self.root.rglob('*') if p.is_file()];before={str(p):sha256(p) for p in files}
        inspect_point(self.run,26,2,False);query_pixel(self.run,3,2)
        self.assertEqual(before,{str(p):sha256(p) for p in files})
        path=self.run.directory/'B3.npz'
        with np.load(path) as z:data={k:z[k] for k in z.files if k not in ['control','xyz','velocity']}
        np.savez_compressed(path,**data);self.store.close();run=DatasetStore(self.root).load('results/toy')
        result=inspect_point(run,26,1,False)
        self.assertIn('result.xyz',result['missing_fields']);self.assertFalse(result['trajectory']['continuous']['available'])
        run.close()


if __name__=='__main__':unittest.main()
