"""Independent analytical tests; no image matching, fitted cameras, or GT filtering."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import coordinates as c
import geometry as g


def rotation_y(angle):
    x,y=np.cos(angle),np.sin(angle)
    return np.array([[x,0,y],[0,1,0],[-y,0,x]])


def rotation_z(angle):
    x,y=np.cos(angle),np.sin(angle)
    return np.array([[x,-y,0],[y,x,0],[0,0,1]])


def fixture():
    a=dict(K=np.array([[420.,.7,158.2],[0,390.,98.8],[0,0,1]]),
           R=rotation_z(.12)@rotation_y(-.05),size=[317,199],coordinate_system='undistorted')
    b=dict(K=np.array([[510.,0,109.6],[0,480.,200.1],[0,0,1]]),
           R=rotation_z(-.08)@rotation_y(.17),size=[221,401],coordinate_system='undistorted')
    a['t']=-a['R']@np.array([.15,-.08,.02])
    b['t']=-b['R']@np.array([.8,.04,.1])
    xyz=np.array([[.3,.1,3.],[-.2,.4,4.5],[1.,-.2,5.],[-.4,-.3,3.8]])
    return a,b,xyz


def independent_project(cam,xyz):
    # Column-vector homogeneous projection, independently of helper's row-vector path.
    projection=cam['K']@np.column_stack([cam['R'],cam['t']])
    homogeneous=projection@np.column_stack([xyz,np.ones(len(xyz))]).T
    return (homogeneous[:2]/homogeneous[2:]).T


class CoordinateTests(unittest.TestCase):
    def test_pixel_centers_and_normalized_edges(self):
        expected=np.array([[-6/7,-4/5],[6/7,4/5],[0,0]])
        uv=np.array([[0.,0.],[6.,4.],[3.,2.]])
        np.testing.assert_allclose(c.pixel_to_normalized(uv,[7,5]),expected,atol=1e-15)
        np.testing.assert_allclose(c.normalized_to_pixel([[-1.,-1.],[1.,1.]],[7,5]),
                                   [[-.5,-.5],[6.5,4.5]],atol=1e-15)

    def test_identity_dense_grid(self):
        size=[7,5];grid=c.pixel_grid(size)
        self.assertEqual(grid.shape,(5,7,2))
        np.testing.assert_array_equal(grid[3,4],[4,3])
        recovered=c.normalized_to_pixel(c.pixel_to_normalized(grid,size),size)
        np.testing.assert_allclose(recovered,grid,atol=1e-15)

    def test_unequal_sizes_subpixel_roundtrip(self):
        a,b=[317,199],[221,401]
        uv=np.array([[10.25,7.75],[0,0],[316,198],[99.4,88.3]])
        expected=np.column_stack([(uv[:,0]+.5)*221/317-.5,(uv[:,1]+.5)*401/199-.5])
        np.testing.assert_allclose(c.resize_pixel(uv,a,b),expected,atol=1e-13)
        np.testing.assert_allclose(c.resize_pixel(expected,b,a),uv,atol=1e-13)
        normalized=c.pixel_to_normalized(uv,a)
        np.testing.assert_allclose(c.normalized_to_pixel(normalized,b),expected,atol=1e-13)
        shift=np.array([.375,-.625])
        np.testing.assert_allclose(c.normalized_to_pixel(c.pixel_to_normalized(uv+shift,a),a)-uv,
                                  np.broadcast_to(shift,uv.shape),atol=1e-13)

    def test_resize_and_crop_calibration(self):
        a,_,xyz=fixture()
        uv=independent_project(a,xyz)
        resized=dict(a,K=c.resize_K(a['K'],a['size'],[190,107]),size=[190,107])
        np.testing.assert_allclose(independent_project(resized,xyz),c.resize_pixel(uv,a['size'],resized['size']),atol=1e-12)
        crop=dict(resized,K=c.crop_K(resized['K'],[12,9]),size=[140,80])
        np.testing.assert_allclose(independent_project(crop,xyz),c.crop_pixel(independent_project(resized,xyz),[12,9]),atol=1e-12)

    def test_no_padding_as_valid_observation(self):
        points=np.array([[-.1,0],[0,0],[6,4],[6.2,4],[np.nan,2],[3,2]])
        np.testing.assert_array_equal(c.inside_image(points,[7,5]),[False,True,True,False,False,True])
        np.testing.assert_array_equal(c.inside_image(points,[7,5],margin=1),[False,False,False,False,False,True])
        with self.assertRaises(ValueError):c.pixel_to_normalized(points,[0,5])

    @unittest.skipIf(c.torch is None,'PyTorch unavailable')
    def test_torch_identity_sampler_and_gradients(self):
        torch=c.torch
        image=torch.arange(35,dtype=torch.float64).reshape(1,1,5,7).square()/100
        grid=c.pixel_grid([7,5],like=image)
        norm=c.pixel_to_normalized(grid,[7,5])[None]
        sampled=torch.nn.functional.grid_sample(image,norm,mode='bilinear',padding_mode='zeros',align_corners=False)
        torch.testing.assert_close(sampled,image,atol=1e-12,rtol=0)
        point=torch.tensor([[10.2,7.3]],dtype=torch.float64,requires_grad=True)
        result=c.normalized_to_pixel(c.pixel_to_normalized(point,[317,199]),[317,199])
        torch.testing.assert_close(result,point,atol=1e-12,rtol=0)
        result.sum().backward();torch.testing.assert_close(point.grad,torch.ones_like(point),atol=1e-12,rtol=0)


class GeometryTests(unittest.TestCase):
    def test_fixed_camera_projection_and_unit_ray(self):
        a,b,xyz=fixture()
        for cam in [a,b]:
            uv,z=g.project(cam,xyz,True)
            np.testing.assert_allclose(uv,independent_project(cam,xyz),atol=1e-12)
            self.assertTrue((z>0).all())
            center=g.camera_center(cam)
            np.testing.assert_allclose(cam['R']@center+cam['t'],0,atol=1e-15)
            ray=g.world_rays(cam,uv)
            np.testing.assert_allclose(np.linalg.norm(ray,axis=-1),1,atol=1e-15)
            np.testing.assert_allclose(ray,(xyz-center)/np.linalg.norm(xyz-center,axis=-1,keepdims=True),atol=1e-14)

    def test_known_3d_f_direction_and_resize(self):
        a,b,xyz=fixture();pa=independent_project(a,xyz);pb=independent_project(b,xyz)
        F=g.fundamental(a,b)
        ap=np.c_[pa,np.ones(len(pa))];bp=np.c_[pb,np.ones(len(pb))]
        np.testing.assert_allclose(np.einsum('ni,ij,nj->n',bp,F,ap),0,atol=1e-11)
        np.testing.assert_allclose(g.epipolar_residual(F,pa,pb),0,atol=1e-11)
        self.assertGreater(np.max(abs(np.einsum('ni,ij,nj->n',ap,F,bp))),.01)
        ba=g.fundamental(b,a)
        self.assertAlmostEqual(abs(float(np.sum(ba*F.T))),1,places=12)
        newa=dict(a,K=c.resize_K(a['K'],a['size'],[101,73]),size=[101,73])
        newb=dict(b,K=c.resize_K(b['K'],b['size'],[199,301]),size=[199,301])
        np.testing.assert_allclose(g.epipolar_residual(g.fundamental(newa,newb),
            c.resize_pixel(pa,a['size'],newa['size']),c.resize_pixel(pb,b['size'],newb['size'])),0,atol=1e-11)

    def test_normalized_epipolar_distance_and_jacobian(self):
        a,b,xyz=fixture();pa=independent_project(a,xyz);pb=independent_project(b,xyz);F=g.fundamental(a,b)
        error,jac=g.epipolar_residual(F,pa,pb,True)
        np.testing.assert_allclose(np.linalg.norm(jac,axis=-1),1,atol=1e-14)
        displaced=pb+.37*jac
        np.testing.assert_allclose(g.epipolar_residual(F,pa,displaced),.37,atol=1e-11)
        epsilon=1e-4
        for axis in range(2):
            step=np.zeros(2);step[axis]=epsilon
            numerical=(g.epipolar_residual(F,pa,displaced+step)-g.epipolar_residual(F,pa,displaced-step))/(2*epsilon)
            np.testing.assert_allclose(numerical,jac[:,axis],atol=5e-9,rtol=0)
        np.testing.assert_allclose(g.epipolar_residual(7.3*F,pa,displaced),.37,atol=1e-11)

    def test_ray_range_projection_derivative(self):
        a,b,xyz=fixture();uv=independent_project(a,xyz);center=g.camera_center(a);rays=g.world_rays(a,uv)
        ranges=np.linalg.norm(xyz-center,axis=-1)
        projected,jac=g.project_ray_depth(b,center,rays,ranges,True)
        np.testing.assert_allclose(projected,independent_project(b,xyz),atol=1e-12)
        eps=1e-5
        numerical=(g.project_ray_depth(b,center,rays,ranges+eps)-g.project_ray_depth(b,center,rays,ranges-eps))/(2*eps)
        np.testing.assert_allclose(jac,numerical,atol=2e-8,rtol=1e-8)
        # Off-axis range is deliberately unequal to optical-axis Z.
        self.assertGreater(np.max(abs(ranges-(xyz@a['R'].T+a['t'])[:,2])),.01)

    def test_triangulation_positive_depth_and_degeneracy(self):
        a,b,xyz=fixture();pa=independent_project(a,xyz);pb=independent_project(b,xyz)
        tri=g.triangulate_pair(a,b,pa,pb)
        self.assertTrue(tri['valid'].all())
        np.testing.assert_allclose(tri['xyz'],xyz,atol=1e-11)
        np.testing.assert_allclose(tri['ray_distance'],np.linalg.norm(xyz-g.camera_center(a),axis=-1),atol=1e-11)
        np.testing.assert_allclose(tri['ray_gap'],0,atol=1e-12)
        pa[1]=np.nan
        invalid=g.triangulate_pair(a,b,pa,pb)
        self.assertFalse(invalid['valid'][1]);self.assertTrue(np.isnan(invalid['xyz'][1]).all())
        simple=dict(K=np.array([[100.,0,40],[0,100.,30],[0,0,1]]),R=np.eye(3),t=np.zeros(3),size=[80,60])
        other=dict(simple,t=np.array([-1.,0,0]))
        parallel=g.triangulate_pair(simple,other,np.array([[40.,30.]]),np.array([[40.,30.]]))
        self.assertFalse(parallel['valid'][0])
        behind=np.array([[.2,0,-3.]])
        tri_back=g.triangulate_pair(simple,other,independent_project(simple,behind),independent_project(other,behind))
        self.assertFalse(tri_back['valid'][0])

    def test_bad_calibration_and_raw_coordinate_rejected(self):
        a,b,_=fixture()
        with self.assertRaises(ValueError):g.fundamental(a,a)
        with self.assertRaises(ValueError):g.fundamental(dict(a,coordinate_system='raw_distorted'),b)

    @unittest.skipIf(c.torch is None,'PyTorch unavailable')
    def test_torch_geometry_parity_and_autograd(self):
        torch=c.torch;a,b,xyz=fixture()
        x=torch.tensor(xyz,dtype=torch.float64)
        uv=g.project(a,x)
        np.testing.assert_allclose(uv.numpy(),independent_project(a,xyz),atol=1e-12)
        center=torch.tensor(g.camera_center(a),dtype=torch.float64)
        rays=g.world_rays(a,uv)
        depth=torch.tensor(np.linalg.norm(xyz-g.camera_center(a),axis=-1),dtype=torch.float64,requires_grad=True)
        p,j=g.project_ray_depth(b,center,rays,depth,True)
        p.sum().backward()
        torch.testing.assert_close(depth.grad,j.sum(-1),atol=1e-12,rtol=1e-12)
        F=torch.tensor(g.fundamental(a,b),dtype=torch.float64)
        target=torch.tensor(independent_project(b,xyz)+np.array([.3,-.2]),dtype=torch.float64,requires_grad=True)
        e,jac=g.epipolar_residual(F,uv,target,True);e.sum().backward()
        torch.testing.assert_close(target.grad,jac,atol=1e-12,rtol=1e-12)
        tri=g.triangulate_pair(a,b,uv,g.project(b,x))
        self.assertTrue(bool(tri['valid'].all()))
        torch.testing.assert_close(tri['xyz'],x,atol=1e-11,rtol=0)


if __name__=='__main__':unittest.main(verbosity=2)
