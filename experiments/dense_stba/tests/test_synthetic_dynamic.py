"""Independent calibrated dynamic renderer, identity and timing checks."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import synthetic_dynamic as s


def params(case='fractional'):
    return dict(motion_scale=s.CASES[case]['motion_scale'],
                occlusion=s.CASES[case]['occlusion'],texture_seed=20261111)


class DynamicGeometryTests(unittest.TestCase):
    def test_pose_is_valid_rotation_and_accelerating_3d_motion(self):
        times=np.linspace(-.05,1.1,51);R,C=s.motion_pose(times,params())
        np.testing.assert_allclose(R@np.swapaxes(R,-1,-2),np.broadcast_to(np.eye(3),R.shape),atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(R),1.,atol=1e-12)
        self.assertGreater(np.ptp(C[:,0]),.3)
        self.assertGreater(np.ptp(C[:,1]),.25)
        self.assertGreater(np.ptp(C[:,2]),.5)
        self.assertGreater(np.max(np.abs(np.diff(C,n=2,axis=0))),.001)

    def test_material_projection_ray_intersection_roundtrip(self):
        material=np.array([[-.6,.4],[0.,0.],[.3,-.2],[.7,.6]])
        times=np.array([-.02,.34,.7,1.05])
        world=s.material_world(material,np.zeros(len(material)),times,params())
        for cam in s.cameras():
            for index,time in enumerate(times):
                uv,z=s.project(cam,world[index]);hit=s.intersect_scene(cam,uv,time,params())
                self.assertTrue((z>0).all());self.assertTrue(hit['valid'].all())
                np.testing.assert_allclose(hit['xyz'],world[index],atol=2e-14)
                np.testing.assert_allclose(hit['material_uv'],material,atol=2e-14)

    def test_synchronous_epipolar_geometry_and_asynchronous_difference(self):
        a,b=s.cameras()[:2]
        K0,K1,R0,R1,t0,t1=map(np.asarray,[a['K'],b['K'],a['R'],b['R'],a['t'],b['t']])
        R=R1@R0.T;t=t1-R@t0;x,y,z=t
        cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
        F=np.linalg.inv(K1).T@cross@R@np.linalg.inv(K0)
        uv=np.array([[-.4,-.2],[0.,0.],[.35,.3]])
        world=s.material_world(uv,np.zeros(3),[.2,.21],params())
        p,_=s.project(a,world[0]);q,_=s.project(b,world[0]);async_q,_=s.project(b,world[1])
        line=np.column_stack([p,np.ones(3)])@F.T
        synchronous=(np.column_stack([q,np.ones(3)])*line).sum(-1)/np.linalg.norm(line[:,:2],axis=-1)
        asynchronous=(np.column_stack([async_q,np.ones(3)])*line).sum(-1)/np.linalg.norm(line[:,:2],axis=-1)
        np.testing.assert_allclose(synchronous,0.,atol=1e-12)
        self.assertGreater(np.mean(np.abs(asynchronous)),.05)

    def test_independent_continuous_render_finite_and_temporally_smooth(self):
        camera=s.cameras((96,72))[0]
        a,clip=s.render(camera,.4,params());b,_=s.render(camera,.40001,params())
        self.assertEqual(a.shape,(72,96,3));self.assertTrue(np.isfinite(a).all())
        self.assertEqual(clip,0.);self.assertLess(np.mean(np.abs(a-b)),1e-4)
        self.assertGreater(np.std(a),.06)

    def test_static_occluder_and_material_motion_are_distinct(self):
        material=np.array([[.2,.1],[-.7,.4]])
        surface=np.array([1,0]);times=np.array([.2,.5,.8])
        xyz=s.material_world(material,surface,times,params('noisy_occlusion'))
        np.testing.assert_allclose(xyz[:,0],np.broadcast_to([.2,.1,3.],(3,3)),atol=1e-12)
        self.assertGreater(np.linalg.norm(xyz[-1,1]-xyz[0,1]),.1)
        cam=s.cameras()[0];uv,z=s.project(cam,xyz[0]);hit=s.intersect_scene(cam,uv,.2,params('noisy_occlusion'))
        self.assertEqual(hit['surface_id'][0],1)
        np.testing.assert_allclose(hit['xyz'][0],xyz[0,0],atol=1e-12)

    def test_temporal_visibility_detects_true_dynamic_occlusion(self):
        nominal=np.arange(32)/30
        physical=nominal[None]+np.array([0.,.3,-.5])[:,None]/30
        truth=s.DynamicTruth(Path('/unused'),dict(cameras=s.cameras(),seed_frame=16),
            params('noisy_occlusion'),np.array([0.,.3,-.5])/30,physical)
        p=s.pixel_grid((192,144))[::4,::4].reshape(-1,2)
        sample=truth.project_material(p,1,physical[1])
        self.assertGreater(sample['occluded'].sum(),100)
        self.assertFalse((sample['valid']&sample['occluded']).any())
        self.assertTrue(np.array_equal(sample['valid']|sample['occluded'],sample['inside']))
        self.assertGreater((sample['surface_id']==1).sum(),0)
        frame,point=np.argwhere(sample['occluded'])[0]
        hit=s.intersect_scene(s.cameras()[1],sample['uv'][frame,point],physical[1,frame],params('noisy_occlusion'))
        self.assertGreater(np.linalg.norm(hit['xyz']-sample['xyz'][frame,point]),.5)

    def test_near_static_is_small_relative_to_dynamic_motion(self):
        material=np.array([[0.,0.],[.5,-.5]])
        times=np.linspace(0,1.,31);cam=s.cameras()[0]
        dynamic=s.project(cam,s.material_world(material,np.zeros(2),times,params()))[0]
        slow=s.project(cam,s.material_world(material,np.zeros(2),times,params('near_static')))[0]
        dynamic_displacement=np.linalg.norm(dynamic-dynamic[0],axis=-1).max()
        slow_displacement=np.linalg.norm(slow-slow[0],axis=-1).max()
        self.assertGreater(dynamic_displacement,20.)
        self.assertLess(slow_displacement,.1)
        self.assertLess(np.linalg.norm(np.diff(dynamic,axis=0),axis=-1).max(),4.)

    def test_gt_finite_derivatives_against_independent_center_formulas(self):
        # Zero material coordinates follow center(t); their derivatives have a
        # closed form independent of renderer projection and spline derivatives.
        times=np.array([.12,.37,.68,.91]);eps=1e-4;p=params()
        center=s.material_world([[0.,0.]],[0],times,p)[:,0]
        plus=s.material_world([[0.,0.]],[0],times+eps,p)[:,0]
        minus=s.material_world([[0.,0.]],[0],times-eps,p)[:,0]
        terms=[[(.30,.8,0),(.055,1.7,.3)],[(.24,.73,.4),(.04,1.35,0)],[(.32,.72,.2)]]
        velocity=np.zeros((len(times),3));acceleration=velocity.copy()
        for axis,entries in enumerate(terms):
            for amplitude,frequency,phase in entries:
                omega=2*np.pi*frequency
                velocity[:,axis]+=amplitude*omega*np.cos(omega*times+phase)
                acceleration[:,axis]-=amplitude*omega**2*np.sin(omega*times+phase)
        np.testing.assert_allclose((plus-minus)/(2*eps),velocity,atol=1e-6)
        np.testing.assert_allclose((plus-2*center+minus)/eps**2,acceleration,atol=3e-6)


class DynamicFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(prefix='dense-stba-synthetic-')
        cls.directory=Path(cls.temporary.name)
        cls.manifest=s.generate_case('fractional',cls.directory,frames=8,fps=30.,size=(32,24))
        cls.truth=s.load_truth(cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_public_inputs_do_not_contain_true_time_offsets_or_material_identity(self):
        def keys(obj):
            if isinstance(obj,dict):
                return set(obj)|set().union(*(keys(v) for v in obj.values()))
            if isinstance(obj,list):
                return set().union(*(keys(v) for v in obj))
            return set()
        fields=keys(self.manifest)
        self.assertFalse(fields & {'beta','beta_seconds','beta_frames','physical_timestamps',
                                  'seed_material_uv','motion_scale','motion_and_scene'})
        for cam in self.manifest['cameras']:
            sequence=np.load(cam['frames_path'])
            self.assertEqual(sequence.shape,(8,24,32,3));self.assertEqual(sequence.dtype,np.uint8)
            np.testing.assert_allclose(cam['nominal_timestamps'],np.arange(8)/30)
        np.testing.assert_allclose(self.truth.physical_timestamps,
            np.arange(8)[None]/30+np.array([0.,.3,-.5])[:,None]/30,atol=1e-16)

    def test_seed_identity_and_truth_api(self):
        p=np.array([[8.,8.],[16.,12.],[25.,18.]])
        t=self.truth.physical_timestamps[0,self.manifest['seed_frame']]
        sample=self.truth.project_material(p,0,[t])
        np.testing.assert_allclose(sample['uv'][0],p,atol=1e-12)
        self.assertTrue(sample['valid'].all())
        xyz=self.truth.evaluate_material_trajectory(p,[t,t+.1])
        self.assertEqual(xyz.shape,(2,3,3))
        self.assertGreater(np.linalg.norm(xyz[1]-xyz[0],axis=-1).min(),.05)
        grid=s.pixel_grid((32,24)).reshape(-1,2)
        material=self.truth.material_from_reference(grid)
        with np.load(self.directory/'gt.npz') as archive:
            np.testing.assert_allclose(material['material_uv'],archive['seed_material_uv'].reshape(-1,2),atol=1e-12)


if __name__=='__main__':
    unittest.main()
