"""Independent analytic plane rendering; CPU checks, no external data or weights."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry import world_rays, project, camera_center
from lsm import bilinear, patch_offsets
from multiview import (MultiViewConfig, initialize_shared_depth, frozen_view_weights,
    shared_patch_residual_jacobian, refine_shared_depth, WEIGHT_MODES)


def plane_scene(shift_second=0.):
    """Render intensity from continuous world-plane texture, not patch resampling."""
    width,height=96,80
    K=np.array([[100.,0.,47.5],[0.,95.,39.5],[0.,0.,1.]])
    centers=[np.zeros(3),np.array([.15,0.,0.]),np.array([.32,.02,0.])]
    cams=[dict(K=K.copy(),R=np.eye(3),t=-C,size=[width,height],coordinate_system='undistorted') for C in centers]
    plane_z=3.2; yy,xx=np.mgrid[:height,:width]; images=[]
    for j,C in enumerate(centers):
        # Optional target-only corruption is a controlled photometric mismatch.
        u=xx-(shift_second if j==2 else 0.)
        X=C[0]+(u-K[0,2])/K[0,0]*plane_z
        Y=C[1]+(yy-K[1,2])/K[1,1]*plane_z
        texture=.45+.12*np.sin(4.7*X+.3)+.11*np.cos(6.3*Y+.2)
        texture+=.08*np.sin(3.6*X+5.8*Y)+.04*np.cos(12.8*X-9.1*Y)
        images.append(texture*[1.,1.08,.93][j]+[0.,.02,-.01][j])
    x,y=np.meshgrid([27.,38.,55.,66.],[24.,36.,49.,61.])
    p=np.stack([x.ravel(),y.ravel()],axis=-1)
    rays=world_rays(cams[0],p)
    true_depth=plane_z/rays[:,2]
    xyz=true_depth[:,None]*rays
    q_true=np.stack([project(c,xyz) for c in cams[1:]],axis=1)
    wrong_xyz=(3.35/rays[:,2])[:,None]*rays
    q_init=np.stack([project(c,wrong_xyz) for c in cams[1:]],axis=1)
    A=np.tile(np.eye(2),(len(p),2,1,1))
    quality=dict(confidence=np.tile([.9,.85],(len(p),1)),
                 fb_error=np.tile([.1,.2],(len(p),1)),
                 photo_error=np.tile([.02,.025],(len(p),1)))
    return images,cams,p,q_init,A,quality,q_true,true_depth


class MultiViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg=MultiViewConfig(max_iterations=30,batch_size=64,step_tolerance=1e-4,
                               texture_std_min=.004,texture_eigen_min=1e-9)

    def run_scene(self,scene,**kwargs):
        images,cams,p,q,A,quality,_,_=scene
        return refine_shared_depth(images[0],images[1:],cams[0],cams[1:],p,q,A,quality,
            config=kwargs.pop('config',self.cfg),mode=kwargs.pop('mode','static'),dtype=torch.float64,**kwargs)

    def test_initial_range_uses_both_cameras_and_is_not_z(self):
        _,cams,p,q,A,quality,q_true,true_depth=plane_scene()
        initial=initialize_shared_depth(cams[0],cams[1:],p,q_true)
        self.assertTrue(initial['valid'].all())
        np.testing.assert_allclose(initial['depth'],true_depth,atol=2e-13)
        np.testing.assert_allclose(initial['pair_ray_distance'],true_depth[:,None]*np.ones((1,2)),atol=2e-13)
        self.assertGreater(float(np.max(true_depth-3.2)),.05)
        # A conflicting second depth changes the unweighted initializer.
        q_mixed=q_true.copy();q_mixed[:,1]=q[:,1]
        other=initialize_shared_depth(cams[0],cams[1:],p,q_mixed)
        np.testing.assert_allclose(other['depth'],.5*(true_depth+3.35/initial['rays'][:,2]),atol=2e-13)
        q_mixed[0,1]=p[0]  # parallel rays cannot supply a valid two-view depth.
        invalid=initialize_shared_depth(cams[0],cams[1:],p,q_mixed)
        self.assertFalse(invalid['valid'][0])

    def test_shared_residual_jacobian_finite_difference(self):
        images,cams,p,q,A,quality,q_true,true_depth=plane_scene()
        tensor=lambda v:torch.as_tensor(v,dtype=torch.float64)
        p=tensor(p[:3]+.173);rays=world_rays(cams[0],p)
        offsets=patch_offsets(7,'cpu',torch.float64)
        ref,_=bilinear(tensor(images[0]),p[:,None]+offsets)
        theta=torch.zeros((3,5),dtype=torch.float64)
        theta[:,0]=torch.log(tensor(true_depth[:3])*1.017)
        theta[:,1]=1.07;theta[:,2]=.013;theta[:,3]=.91;theta[:,4]=-.023
        args=([tensor(im) for im in images[1:]],cams[1:],tensor(camera_center(cams[0])),rays,ref,offsets,tensor(A[:3]))
        residual,J,inside,_,_=shared_patch_residual_jacobian(*args,theta)
        self.assertTrue(inside.all())
        epsilon=1e-6
        for k in range(5):
            step=torch.zeros_like(theta);step[:,k]=epsilon
            numeric=(shared_patch_residual_jacobian(*args,theta+step)[0]-
                     shared_patch_residual_jacobian(*args,theta-step)[0])/(2*epsilon)
            torch.testing.assert_close(J[:,:,:,k],numeric,atol=2e-9,rtol=2e-7)

    def test_frozen_weight_components_and_baselines(self):
        _,cams,_,_,_,quality,_,_=plane_scene()
        cfg=self.cfg
        weights=frozen_view_weights(cams[0],cams[1:],quality,'quality_distance',cfg)
        B=np.array([.15,np.hypot(.32,.02)])
        distance=np.clip(np.exp(-(B/np.median(B))**2),.2,1)
        score=np.clip(quality['confidence']*np.exp(-.5*(quality['fb_error']/1.5)**2)*
                      np.exp(-.5*(quality['photo_error']/.05)**2),.05,1)
        np.testing.assert_allclose(weights['baselines'],B)
        np.testing.assert_allclose(weights['distance_component'],distance)
        np.testing.assert_allclose(weights['quality_component'],score)
        np.testing.assert_allclose(weights['weights_unnormalized'],score*distance)
        np.testing.assert_allclose(weights['weights'].sum(-1),2)
        quality['confidence'][0,1]=np.nan
        self.assertFalse(frozen_view_weights(cams[0],cams[1:],quality,'uniform',cfg)['quality_valid'][0])

    def test_shared_depth_improves_analytic_plane(self):
        scene=plane_scene();out=self.run_scene(scene)
        valid=out['valid_refined'];q_true,true_depth=scene[-2:]
        self.assertGreaterEqual(int(valid.sum()),14,str(np.unique(out['refinement_status'],return_counts=True)))
        before=np.linalg.norm(out['uv_raw'][valid]-q_true[valid],axis=-1)
        after=np.linalg.norm(out['uv_refined'][valid]-q_true[valid],axis=-1)
        self.assertLess(float(np.median(after)),.04)
        self.assertLess(float(np.median(after)),float(np.median(before))/4)
        self.assertLess(float(np.sqrt(np.mean((out['depth'][valid]-true_depth[valid])**2))),.02)
        self.assertTrue((out['objective_after'][valid]<=out['objective_before'][valid]).all())
        np.testing.assert_array_equal(out['A_fixed'],scene[4])

    def test_four_arms_share_initialization_but_weights_affect_depth(self):
        scene=plane_scene(shift_second=.8)
        images,cams,p,q,A,quality,_,_=scene
        quality['confidence'][:,1]=.1;quality['fb_error'][:,1]=2.;quality['photo_error'][:,1]=.1
        initial=initialize_shared_depth(cams[0],cams[1:],p,q)
        results={mode:self.run_scene(scene,weight_mode=mode,initialization=initial) for mode in WEIGHT_MODES}
        for result in results.values():
            np.testing.assert_array_equal(result['depth_initial'],initial['depth'])
            self.assertTrue(result['timing']['weights_frozen'])
            self.assertGreaterEqual(result['valid_refined'].sum(),10)
        common=results['uniform']['valid_refined']&results['quality']['valid_refined']
        self.assertGreater(np.median(np.abs(results['uniform']['depth'][common]-results['quality']['depth'][common])),.01)
        self.assertLess(results['quality']['weights_frozen'][0,1],results['uniform']['weights_frozen'][0,1])

    def test_asynchronous_and_failure_gates(self):
        scene=plane_scene()
        with self.assertRaisesRegex(ValueError,'static masks'):
            self.run_scene(scene,mode='asynchronous_unknown')
        masks=[np.ones_like(image) for image in scene[0]]
        p=scene[2];masks[0][int(p[0,1]),int(p[0,0])]=0
        eligible=np.ones(len(p),bool);eligible[1]=False
        out=self.run_scene(scene,mode='asynchronous_unknown',reference_mask=masks[0],target_masks=masks[1:],eligible=eligible)
        self.assertEqual(out['refinement_status'][0],14)
        self.assertEqual(out['refinement_status'][1],17)
        self.assertFalse(out['valid_refined'][:2].any())
        # A malformed correspondence and a constant patch must not crash an
        # otherwise valid batch (including the Schur/eigen diagnostic solves).
        scene[3][0,0]=np.nan
        out=self.run_scene(scene)
        self.assertEqual(out['refinement_status'][0],2)
        self.assertFalse(out['valid_refined'][0])
        images=list(scene[0]);images[0]=np.full_like(images[0],.5)
        out=self.run_scene((images,*scene[1:]))
        self.assertTrue(np.all(out['refinement_status'][1:]==5))

    def test_batch_invariance(self):
        scene=plane_scene()
        small=self.run_scene(scene,config=replace(self.cfg,batch_size=3))
        full=self.run_scene(scene)
        np.testing.assert_array_equal(small['valid_refined'],full['valid_refined'])
        np.testing.assert_allclose(small['depth'],full['depth'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(small['uv_refined'],full['uv_refined'],rtol=1e-10,atol=1e-10)


if __name__=='__main__':unittest.main()
