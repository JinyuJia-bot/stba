"""Numerical/analytic optimizer checks; these unit scenes are explicit oracle IDs."""
from pathlib import Path
import sys
import unittest
from dataclasses import replace
import numpy as np
import torch
from scipy.interpolate import BSpline

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trajectory_spline import make_knots,basis_matrix,evaluate_spline,acceleration_gram,fit_control_points
from stba_optimizer import STBAConfig,observation_model,optimize_stba


def scene(static=False):
    K=np.array([[160.,0.,79.5],[0.,155.,59.5],[0.,0.,1.]])
    cams=[dict(K=K.copy(),R=np.eye(3),t=-np.array(C),size=[160,120]) for C in [[0,0,0],[.35,.03,0],[-.55,.12,.04]]]
    xx,yy=np.meshgrid(np.linspace(-.3,.3,4),[-.18,.18]);seed=np.stack([xx.ravel(),yy.ravel()],-1)
    def truth(t):
        t=np.asarray(t);shape=(len(seed),)+(1,)*t.ndim
        x=seed[:,0].reshape(shape)+np.zeros((len(seed),)+t.shape)
        y=seed[:,1].reshape(shape)+np.zeros((len(seed),)+t.shape)
        z=np.ones_like(x)*4
        if not static:
            x=x+(.18+.03*seed[:,1].reshape(shape))*t+.045*t**2-.025*t**3
            y=y+(.06+.04*seed[:,0].reshape(shape))*t+.08*t**2-.031*t**3
            z=z+.07*t+.04*t**2-.008*t**3
        return np.stack([x,y,z],-1)
    times=np.tile(np.arange(18)/10,(3,1));beta=np.array([0.,.03,-.05]);knots=make_knots(-.25,2.,.3)
    xyz=truth(times+beta[:,None]);uv=[]
    for c,cam in enumerate(cams):
        xc=xyz[:,c]+cam['t'];h=xc@K.T;uv.append(h[...,:2]/h[...,2:])
    uv=np.stack(uv,1);visible=np.ones(uv.shape[:-1],bool);weight=np.ones_like(visible,dtype=float)
    sample_times=np.linspace(0,1.7,18);initial_xyz=truth(sample_times)
    if not static:
        initial_xyz=initial_xyz+np.array([.006,-.004,.012])[None,None]
    fitted=fit_control_points(sample_times,initial_xyz,np.ones(initial_xyz.shape[:2],bool),knots,regularization=1e-8)
    return cams,times,beta,knots,uv,visible,weight,fitted['control'],truth


class SplineOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_basis_matches_independent_scipy_and_partition(self):
        knots=make_knots(-.2,1.1,.17);n=len(knots)-4
        times=np.r_[-.2,np.linspace(-.19,1.09,31),1.1]
        reference=BSpline(knots,np.eye(n),3)
        for derivative in range(3):
            actual=basis_matrix(times,knots,derivative)
            np.testing.assert_allclose(actual,reference(times,nu=derivative),atol=2e-12,rtol=2e-12)
            np.testing.assert_allclose(actual.sum(-1),1 if derivative==0 else 0,atol=2e-12)
        np.testing.assert_array_equal(basis_matrix(np.array([-.3,1.2]),knots),0)

    def test_time_derivatives_autograd_and_acceleration_integral(self):
        knots=make_knots(0,1,.2);n=len(knots)-4
        t=torch.tensor([.117,.351,.823],dtype=torch.float64,requires_grad=True)
        B=basis_matrix(t,knots)
        weights=torch.arange(n,dtype=torch.float64)**2
        first=torch.autograd.grad((B*weights).sum(),t,create_graph=True)[0]
        second=torch.autograd.grad(first.sum(),t)[0]
        torch.testing.assert_close(first,(basis_matrix(t,knots,1)*weights).sum(-1),atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(second,(basis_matrix(t,knots,2)*weights).sum(-1),atol=1e-9,rtol=1e-10)
        samples=np.linspace(0,1,25);xyz=np.stack([samples**2,2*samples**2,samples**3],-1)[None]
        fitted=fit_control_points(samples,xyz,np.ones((1,25),bool),knots,regularization=0)
        control=fitted['control'];Q=acceleration_gram(knots)
        integral=np.einsum('pkd,kl,pld->',control,Q,control)
        self.assertAlmostEqual(integral,32.,places=5) # int (2² + 4² + (6t)²) dt
        queried=evaluate_spline(control,np.array([-.1,.4,1.1]),knots)
        self.assertTrue(np.isnan(queried[:,[0,2]]).all())

    def test_projection_control_and_beta_jacobians(self):
        cams,times,beta,knots,uv,visible,weight,control,_=scene()
        control=torch.tensor(control[:2],dtype=torch.float64);beta=torch.tensor(beta,dtype=torch.float64)
        times=times[:,:3]+.013
        model=observation_model(control,beta,times,knots,cams,True)
        eps=1e-6
        for c in [0,1,2]:
            step=torch.zeros_like(beta);step[c]=eps
            numeric=(observation_model(control,beta+step,times,knots,cams)['uv']-
                     observation_model(control,beta-step,times,knots,cams)['uv'])/(2*eps)
            torch.testing.assert_close(numeric[:,c],model['J_time'][:,c],atol=2e-7,rtol=2e-7)
            self.assertLess(float(numeric[:,[j for j in range(3) if j!=c]].abs().max()),1e-10)
        for k,d in [(1,0),(3,1),(4,2)]:
            step=torch.zeros_like(control);step[:,k,d]=eps
            numeric=(observation_model(control+step,beta,times,knots,cams)['uv']-
                     observation_model(control-step,beta,times,knots,cams)['uv'])/(2*eps)
            torch.testing.assert_close(numeric,model['J_control'][...,3*k+d],atol=2e-7,rtol=2e-7)

    def test_joint_and_fixed_share_initialization_and_recover_timing(self):
        cams,times,beta,knots,uv,visible,weight,control,truth=scene()
        config=STBAConfig(max_iterations=45,batch_size=3,lambda_motion=1e-5,objective_tolerance=1e-10)
        kwargs=dict(uv=uv,visible=visible,weight=weight,times=times,cams=cams,knots=knots,
                    control_init=control,beta_init=np.zeros(3),config=config,beta_bounds_seconds=[-.15,.15])
        before_control=control.copy();before_visible=visible.copy()
        joint=optimize_stba(**kwargs,mode='joint')
        fixed=optimize_stba(**kwargs,mode='fixedbeta')
        np.testing.assert_array_equal(control,before_control);np.testing.assert_array_equal(visible,before_visible)
        np.testing.assert_array_equal(joint['control_initial'],fixed['control_initial'])
        np.testing.assert_array_equal(fixed['beta'],np.zeros(3));self.assertEqual(joint['beta'][0],0)
        np.testing.assert_allclose(joint['beta'],beta,atol=4e-4)
        self.assertLess(joint['reprojection_rmse_final'],fixed['reprojection_rmse_final']*.05)
        self.assertLess(joint['objective_final'],joint['objective_initial'])
        t=np.linspace(.1,1.6,20);recovered=evaluate_spline(joint['control'],t,knots)
        self.assertLess(float(np.sqrt(np.mean((recovered-truth(t))**2))),.001)
        self.assertEqual(joint['timing']['raw_training_observations'],int(visible.sum()))
        self.assertTrue(joint['trajectory_valid'].all())

    def test_near_static_reports_no_time_information(self):
        cams,times,beta,knots,uv,visible,weight,control,_=scene(static=True)
        initial=np.array([0.,.02,-.03]);mask=np.zeros(len(control),bool)
        result=optimize_stba(uv,visible,weight,times,cams,knots,control,initial,
            STBAConfig(max_iterations=10),beta_bounds_seconds=[-.15,.15],beta_track_mask=mask)
        np.testing.assert_array_equal(result['beta'],initial)
        np.testing.assert_array_equal(mask,np.zeros(len(control),bool))
        self.assertEqual(result['beta_status'][1:],['insufficient_motion','insufficient_motion'])
        self.assertFalse(result['beta_reliable'].any());self.assertTrue(result['trajectory_valid'].all())

    def test_known_beta_trajectory_fit_uses_raw_training_observations_only(self):
        cams,times,beta,knots,uv,visible,weight,control,truth=scene()
        visible[:,:,2::5]=False
        altered=uv.copy();altered[~visible]=1e5  # Held-out values must never enter fitting.
        cfg=STBAConfig(max_iterations=35,lambda_motion=0.)
        original=optimize_stba(uv,visible,weight,times,cams,knots,control,beta,cfg,mode='fixedbeta')
        changed=optimize_stba(altered,visible,weight,times,cams,knots,control,beta,cfg,mode='fixedbeta')
        np.testing.assert_array_equal(original['control'],changed['control'])
        np.testing.assert_array_equal(original['beta'],beta)
        self.assertEqual(original['timing']['raw_training_observations'],int(visible.sum()))
        self.assertFalse(original['observation_used'][~visible].any())
        self.assertLess(original['reprojection_rmse_final'],1e-5)
        query=np.linspace(.1,1.6,25)
        xyz=evaluate_spline(original['control'],query,knots)
        self.assertLess(float(np.sqrt(np.mean((xyz-truth(query))**2))),.001)

    def test_gauge_bounds_and_knots_are_enforced(self):
        cams,times,beta,knots,uv,visible,weight,control,_=scene()
        args=(uv,visible,weight,times,cams,knots,control)
        with self.assertRaisesRegex(ValueError,'Reference beta'):
            optimize_stba(*args,np.ones(3)*.01,beta_bounds_seconds=[-.15,.15])
        with self.assertRaisesRegex(ValueError,'knot domain'):
            optimize_stba(*args,np.zeros(3),beta_bounds_seconds=[-2,2])
        with self.assertRaisesRegex(ValueError,'Explicit beta_bounds'):
            optimize_stba(*args,np.zeros(3))

    def test_global_beta_is_invariant_to_chunk_partition(self):
        cams,times,beta,knots,uv,visible,weight,control,_=scene()
        config=STBAConfig(max_iterations=25,batch_size=2,lambda_motion=1e-5)
        args=(uv,visible,weight,times,cams,knots,control,np.zeros(3))
        small=optimize_stba(*args,config=config,beta_bounds_seconds=[-.15,.15])
        large=optimize_stba(*args,config=replace(config,batch_size=128),beta_bounds_seconds=[-.15,.15])
        np.testing.assert_allclose(small['beta'],large['beta'],atol=2e-7,rtol=2e-7)
        np.testing.assert_allclose(small['control'],large['control'],atol=2e-6,rtol=2e-7)


if __name__=='__main__':unittest.main()
