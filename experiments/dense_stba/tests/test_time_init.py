"""Independent analytic 2D measurements test timing signs and data isolation.

These are numerical unit scenes with known identities, not an automatic frontend
accuracy claim. Truth is used for assertions only, never passed to estimate_beta.
"""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from time_init import (TimeInitConfig,interpolate_tracks,training_split,motion_diagnostics,
                       estimate_beta,triangulate_samples)


def analytic_observations(kind='aperiodic'):
    P,C,T,fps=12,3,64,20.
    K=np.array([[180.,0,79.5],[0,180.,59.5],[0,0,1.]])
    cameras=[dict(camera_id=j,K=K.copy(),R=np.eye(3),t=-np.array(center),size=[160,120])
             for j,center in enumerate([[0,0,0],[.35,0,0],[-.45,0,0]])]
    nominal=np.tile(np.arange(T)/fps,(C,1));beta=np.array([0.,.3,-.5])/fps
    seed_x=np.linspace(-.4,.4,P);seed_y=np.linspace(-.18,.18,P)
    def xyz_at(t):
        t=np.asarray(t);shape=(P,)+(1,)*t.ndim
        x=seed_x.reshape(shape)+np.zeros((P,)+t.shape)
        y=seed_y.reshape(shape)+np.zeros((P,)+t.shape);z=np.full_like(x,4.)
        phase=np.arange(P).reshape(shape)*.17
        if kind=='periodic':y=y+.12*np.sin(2*np.pi*fps/4*t+phase)
        elif kind=='epipolar':x=x+.2*t+.05*np.sin(1.7*t+phase)
        else:
            amplitude=1e-4 if kind=='near_static' else 1.
            x=x+amplitude*(.08*np.sin(1.9*t+phase)+.06*t)
            y=y+amplitude*(.16*np.sin(1.5*t+.2*phase)+.04*np.cos(3.1*t))
            z=z+amplitude*.03*np.sin(1.3*t+.1*phase)
        return np.stack([x,y,z],-1)
    xyz=xyz_at(nominal+beta[:,None]);uv=[]
    for j,cam in enumerate(cameras):
        h=(xyz[:,j]+cam['t'])@cam['K'].T;uv.append(h[...,:2]/h[...,2:])
    uv=np.stack(uv,1);valid=np.ones((P,C,T),bool);weight=np.ones((P,C,T))
    return uv,valid,weight,nominal,cameras,beta,xyz_at


class TimeInitializationTests(unittest.TestCase):
    def test_interpolation_never_extrapolates_or_fills_long_gaps(self):
        times=np.arange(5)/10;uv=np.stack([times,2*times],-1)[None].repeat(3,0)
        valid=np.array([[1,1,0,1,1],[0,1,0,0,0],[0,0,0,0,0]],bool)
        uv[~valid]=np.nan
        query=np.array([-.01,0,.05,.1,.2,.3,.35,.4,.401])
        interpolated,support=interpolate_tracks(uv,valid,times,query,.15)
        expected=np.array([[0,1,1,1,0,1,1,1,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]],bool)
        np.testing.assert_array_equal(support,expected)
        self.assertTrue(np.isnan(interpolated[~support]).all())
        expected_uv=np.stack([query,2*query],-1)
        np.testing.assert_allclose(interpolated[0,support[0]],expected_uv[support[0]],atol=1e-14)
        # A valid final sample does not require a later observation.
        self.assertTrue(support[0,-2])
        with self.assertRaisesRegex(ValueError,'increase'):
            interpolate_tracks(uv,valid,np.array([0,.1,.1,.3,.4]),query,.15)

    def test_training_split_is_disjoint_and_preserves_missing_data(self):
        visible=np.ones((4,3,17),bool);visible[0,1,2]=False;visible[2,0,5:8]=False
        original=visible.copy();train,holdout=training_split(visible,5,2)
        np.testing.assert_array_equal(visible,original)
        np.testing.assert_array_equal(train|holdout,visible)
        self.assertFalse((train&holdout).any())
        self.assertTrue(np.all(np.where(holdout)[-1]%5==2))

    def test_plus_beta_sign_is_recovered_from_projected_motion(self):
        uv,visible,weight,times,cams,beta,_=analytic_observations()
        train,_=training_split(visible,5,2)
        result=estimate_beta(uv,train,weight,times,cams,TimeInitConfig(search_bound_frames=2.))
        np.testing.assert_allclose(result['beta'],beta,atol=.015/20)
        self.assertTrue(result['reliable'].all(),result['profiles'])
        self.assertGreater(result['beta'][1],0);self.assertLess(result['beta'][2],0)
        for profile in result['profiles']:
            self.assertGreater(profile['common_original_samples'],100)
            self.assertEqual(profile['status'],'observable_candidate')

    def test_aligned_triangulation_uses_query_time_minus_beta(self):
        uv,visible,weight,times,cams,beta,xyz_at=analytic_observations()
        train,_=training_split(visible,5,2);samples=times[0,4:-4]
        right=triangulate_samples(uv,train,weight,times,cams,beta,samples)
        wrong=triangulate_samples(uv,train,weight,times,cams,np.zeros(3),samples)
        self.assertTrue(right['valid'].all())
        self.assertTrue((right['view_support']==3).all())
        truth=xyz_at(samples)
        correct_rmse=np.sqrt(np.mean((right['xyz']-truth)**2))
        wrong_rmse=np.sqrt(np.mean((wrong['xyz']-truth)**2))
        self.assertLess(correct_rmse,.0005)
        self.assertGreater(wrong_rmse,correct_rmse*3)

    def test_heldout_coordinates_and_weights_do_not_affect_initialization(self):
        uv,visible,weight,times,cams,beta,_=analytic_observations()
        train,heldout=training_split(visible,5,2)
        changed_uv=uv.copy();changed_uv[heldout]=1e5
        changed_weight=weight.copy();changed_weight[heldout]=1e8
        cfg=TimeInitConfig(search_bound_frames=2.)
        a=estimate_beta(uv,train,weight,times,cams,cfg)
        b=estimate_beta(changed_uv,train,changed_weight,times,cams,cfg)
        np.testing.assert_array_equal(a['beta'],b['beta'])
        np.testing.assert_array_equal(a['moving_tracks'],b['moving_tracks'])
        for pa,pb in zip(a['profiles'],b['profiles']):self.assertEqual(pa['costs'],pb['costs'])
        tri_a=triangulate_samples(uv,train,weight,times,cams,a['beta'],times[0])
        tri_b=triangulate_samples(changed_uv,train,changed_weight,times,cams,b['beta'],times[0])
        np.testing.assert_array_equal(tri_a['xyz'],tri_b['xyz'])
        np.testing.assert_array_equal(tri_a['valid'],tri_b['valid'])
        np.testing.assert_array_equal(tri_a['view_support'],tri_b['view_support'])

    def test_near_static_cannot_estimate_time_offsets(self):
        uv,visible,weight,times,cams,_,_=analytic_observations('near_static')
        result=estimate_beta(uv,visible,weight,times,cams,TimeInitConfig(search_bound_frames=2.))
        self.assertFalse(result['moving_tracks'].any())
        self.assertFalse(result['reliable'][1:].any())
        np.testing.assert_array_equal(result['beta'],np.zeros(3))
        self.assertTrue(all(p['status']=='insufficient_dynamic_support' for p in result['profiles']))

    def test_motion_along_epipolar_lines_has_flat_time_profile(self):
        uv,visible,weight,times,cams,_,_=analytic_observations('epipolar')
        result=estimate_beta(uv,visible,weight,times,cams,TimeInitConfig(search_bound_frames=2.))
        self.assertTrue(result['moving_tracks'].all())
        self.assertFalse(result['reliable'][1:].any())
        self.assertTrue(all(p['status'] in ['flat_profile','search_boundary','ambiguous_separated_minima'] for p in result['profiles']))

    def test_separated_periodic_minima_are_not_declared_reliable(self):
        # Four-frame periodic motion gives equivalent beta basins separated by
        # four frames. A sharp local minimum is not unique synchronization.
        uv,visible,weight,times,cams,_,_=analytic_observations('periodic')
        result=estimate_beta(uv,visible,weight,times,cams,TimeInitConfig(search_bound_frames=6.))
        self.assertFalse(result['reliable'][1:].any(),result['profiles'])
        self.assertTrue(all(p['status']=='ambiguous_separated_minima' for p in result['profiles']))


if __name__=='__main__':unittest.main()
