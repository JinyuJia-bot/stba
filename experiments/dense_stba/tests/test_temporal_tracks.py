"""Temporal identity, failure propagation, cache, and PLUS-clock regression tests."""
import tempfile
import unittest

import cv2
import numpy as np

from experiments.dense_stba.data_loader import clock_mapping,query_grid
from experiments.dense_stba.temporal_tracks import bilinear_sample,build_flow_cache,track_queries


class TemporalTests(unittest.TestCase):
    def constant_cache(self):
        displacement=np.array([.3,-.2],np.float32)
        f=np.broadcast_to(displacement,(8,32,40,2)).copy()
        return {'forward':f,'backward':-f,'skip_flows':{},'frame_ids':np.arange(9)},displacement

    def test_arbitrary_seed_both_directions_follow_identity(self):
        cache,delta=self.constant_cache()
        seeds=np.array([[15.25,12.8],[23.6,21.3]],np.float32)
        result=track_queries(cache,4,seeds,point_ids=[101,305])
        expected=seeds[:,None]+(np.arange(9)-4)[None,:,None]*delta
        np.testing.assert_allclose(result['tracks'],expected,atol=6e-6)
        self.assertTrue(result['visible'].all())
        np.testing.assert_array_equal(result['point_ids'],[101,305])
        self.assertGreater(np.linalg.norm(result['tracks'][0,-1]-seeds[0]),1.)

    def test_fb_failure_terminates_only_one_time_branch(self):
        cache,_=self.constant_cache();cache['backward'][5,...,0]+=2
        result=track_queries(cache,4,[[15.25,12.8]])
        self.assertTrue(result['visible'][0,:6].all())
        self.assertEqual(result['status'][0,6],3)
        self.assertFalse(result['visible'][0,6:].any())
        self.assertEqual(result['status'][0,7],6)
        self.assertTrue(np.isnan(result['tracks'][0,7:]).all())

    def test_multi_frame_drift_detected_despite_local_fb(self):
        cache,delta=self.constant_cache()
        direct=np.broadcast_to(delta*4+[2.,0.],(32,40,2)).astype(np.float32).copy()
        cache['skip_flows']={(0,4):(direct,-direct)}
        result=track_queries(cache,0,[[15.25,12.8]])
        self.assertEqual(result['status'][0,4],5)
        self.assertAlmostEqual(result['cycle_error'][0,4],2.,places=5)
        self.assertTrue(result['cycle_check_reliable'][0,4])
        self.assertFalse(result['visible'][0,4:].any())

    def test_nonfinite_and_outside_seeds_never_visible(self):
        cache,_=self.constant_cache()
        result=track_queries(cache,4,[[float('nan'),3.],[-.1,5.]])
        self.assertFalse(result['visible'].any())
        self.assertTrue((result['status'][:,4]==7).all())

    def test_exact_subpixel_sampler(self):
        y,x=np.mgrid[:12,:17];field=np.stack([2*x+3*y,4*x-y],axis=-1).astype(np.float32)
        uv=np.array([[3.2,4.7],[16.,11.],[-.1,5.]],np.float32)
        value,valid=bilinear_sample(field,uv)
        np.testing.assert_allclose(value[:2],np.stack([2*uv[:2,0]+3*uv[:2,1],4*uv[:2,0]-uv[:2,1]],-1),atol=3e-6)
        np.testing.assert_array_equal(valid,[True,True,False])

    def test_plus_clock_and_integer_shift_equivalence(self):
        ref=.0030276253819465637
        for sync in [ref,-.009394669905304909,-.04636843875050545]:
            raw=clock_mapping(sync,ref,60,False);aligned=clock_mapping(sync,ref,60,True)
            self.assertAlmostEqual(aligned['residual_beta_seconds']-aligned['raw_frame_shift']/60,raw['raw_beta_seconds'])
            n=np.arange(60)
            absolute=(200+n+aligned['raw_frame_shift'])/60-sync-(200/60-ref)
            np.testing.assert_allclose(absolute,n/60+aligned['residual_beta_seconds'],atol=1e-14)
            self.assertLessEqual(abs(aligned['residual_beta_frames']),.5)
        self.assertEqual(clock_mapping(-.04636843875050545,ref,60,True)['raw_frame_shift'],-3)

    def test_grid_is_not_feature_detection(self):
        uv,ids,mask=query_grid((16,12),4)
        self.assertEqual(len(uv),12)
        np.testing.assert_array_equal(ids,uv[:,1]*16+uv[:,0])
        self.assertEqual(mask.sum(),12)

    def test_actual_dis_and_shared_pair_cache(self):
        rng=np.random.default_rng(123)
        base=cv2.GaussianBlur(rng.integers(0,256,(80,112),dtype=np.uint8),(5,5),.8)
        frames=np.stack([cv2.warpAffine(base,np.array([[1,0,1.25*t],[0,1,.7*t]],np.float32),
                                      (112,80),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT101) for t in range(4)])
        with tempfile.TemporaryDirectory() as tmp:
            first=build_flow_cache(frames[:3],tmp,[10,11,12],0)
            second=build_flow_cache(frames[1:],tmp,[11,12,13],0)
            self.assertEqual(second['metadata']['pair_cache_hits'],1)
            seeds=np.array([[40.,30.],[70.,40.],[50.,55.]],np.float32)
            result=track_queries(first,1,seeds)
            expected=seeds[:,None]+(np.arange(3)-1)[None,:,None]*np.array([1.25,.7])
            self.assertTrue(result['visible'].all())
            self.assertLess(np.median(np.linalg.norm(result['tracks']-expected,axis=-1)),.25)


if __name__=='__main__':unittest.main()
