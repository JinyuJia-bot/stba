import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from scipy.interpolate import BSpline

from experiments.dense_stba.debug_viewer.adapter import DatasetStore,LazyNPZ,sha256,frame_identity,to_jsonable


def make_fixture(root,name='toy'):
    root=Path(root);directory=root/'results'/name;data=root/'data'/name;cache=root/'cache'/name
    for p in [directory,data,cache/'roma']:p.mkdir(parents=True,exist_ok=True)
    w,h,T=12,10,4;K=np.array([[10.,0,5.5],[0,10.,4.5],[0,0,1.]])
    beta=np.array([0.,.02,-.01]);times=np.tile(np.arange(T)/10,(3,1));seed=1
    seeds=np.array([[2,2],[6,4],[8,6]],np.float32);ids=(seeds[:,1]*w+seeds[:,0]).astype(np.int64)
    query=np.zeros((h,w),bool);query[seeds[:,1].astype(int),seeds[:,0].astype(int)]=True
    knot=np.array([-.1]*4+[.5]*4);cp_times=np.array([-.1,.1,.3,.5])
    def xyz_at(t):
        x=(seeds[:,0]-K[0,2])/K[0,0]*4;y=(seeds[:,1]-K[1,2])/K[1,1]*4
        return np.stack([x[:,None]+8*(np.asarray(t)[None]-.1),np.broadcast_to(y[:,None],(3,len(t))),np.full((3,len(t)),4.)],-1)
    control=xyz_at(cp_times);cameras=[];uv=[]
    for c,C in enumerate([[0,0,0],[.4,0,0],[-.3,.03,0]]):
        path=data/f'cam{c}.png';rgb=np.zeros((h,w,3),np.uint8);rgb[:]=[20+c,40,80]
        cv2.imwrite(str(path),rgb[...,::-1])
        cam=dict(camera_id=c,K=K.tolist(),R=np.eye(3).tolist(),t=(-np.array(C)).tolist(),D=[0]*5,
                 size=[w,h],n_frames=T,fps=10,image_paths=[str(path)]*T,raw_frame_ids=list(range(100+c,104+c)))
        cameras.append(cam)
        p=xyz_at(times[c]+beta[c])+np.array(cam['t']);q=p@K.T;uv.append(q[...,:2]/q[...,2:])
    uv=np.stack(uv,1).astype(np.float32);visible=np.ones((3,3,T),bool);visible[2,1:]=False
    status=np.ones_like(visible,np.int16);edge=np.ones((3,3),np.int16);edge[2,1:]=4
    manifest=dict(dataset='fixture',reference_camera_id=0,seed_frame=seed,cameras=cameras)
    manifest_path=data/'manifest.json';manifest_path.write_text(json.dumps(manifest))
    obs=dict(point_ids=ids,seed_uv=seeds,query_mask=query,seed_roi=np.ones((h,w),bool),uv=uv,
        visible=visible,weight=visible.astype(np.float32),status=status,edge_status=edge,
        temporal_fb_error=np.zeros_like(visible,np.float32),roma_confidence=np.full((3,3),.8,np.float32),roma_fb_error=np.zeros((3,3)),
        anchor_error=np.zeros((3,3,1)),anchor_support=np.ones((3,3,1),bool),anchor_frames=np.array([seed]),seed_frame=np.array(seed),
        timestamps=times,lsm_seed_refined=np.zeros((3,3),bool),lsm_seed_status=np.zeros((3,3),np.int16))
    config=dict(association=dict(stride=4,confidence_min=.15,roma_fb_max=2.),holdout_every=5,holdout_phase=2)
    common=dict(name=name,manifest_path=str(manifest_path),manifest_sha256=sha256(manifest_path),methods=['B3','B4'],
        seed_frame=seed,seed_grid_size=3,reference_camera_index=0,config=config,camera_ids=[0,1,2])
    for method in ['B3','B4']:
        chosen={k:v.copy() for k,v in obs.items()};filename='observations_lsm.npz' if method=='B4' else 'observations.npz'
        if method=='B4':chosen['uv'][:,1,seed,0]+=.1
        frozen=directory/filename;np.savez_compressed(frozen,**chosen)
        pred=uv.astype(float);pred[2]=np.nan;control2=control.copy();control2[2]=np.nan
        spline=BSpline(knot,control2,3,axis=1);xyz=spline(times[0]);valid=np.ones((3,T),bool);valid[2]=False
        train=visible.copy();train[:,:,2]=False;holdout=visible&~train
        result=dict(point_ids=ids,seed_uv=seeds,camera_ids=np.arange(3),timestamps=times,beta=beta,initial_beta=np.zeros(3),
            predicted_uv=pred,observed_uv=chosen['uv'],train_visible=train,holdout_visible=holdout,observation_inlier=train,
            trajectory_times=times[0],xyz=xyz,velocity=spline(times[0],nu=1),acceleration=spline(times[0],nu=2),
            valid=valid,model_only_mask=np.zeros_like(valid),view_support=valid.astype(int)*3,
            trajectory_valid=valid.any(-1),support_interval=np.array([[0,.3],[0,.3],[np.nan,np.nan]]),
            control=control2,knots=knot,query_mask=query,seed_roi=np.ones((h,w),bool))
        np.savez_compressed(directory/f'{method}.npz',**result)
        meta=dict(method=method,manifest_path=str(manifest_path),manifest_sha256=sha256(manifest_path),
                  observations_path=str(frozen),observations_sha256=sha256(frozen),config=config,
                  valid_trajectories=2,valid_3D_samples=int(valid.sum()),log=[dict(iteration=0)])
        (directory/f'{method}.json').write_text(json.dumps(meta))
        if method=='B3':common.update(observations_path=str(frozen),observations_sha256=sha256(frozen))
    (directory/'meta.json').write_text(json.dumps(common))
    yy,xx=np.mgrid[:h,:w];grid=np.stack([xx,yy],-1).astype(float)
    for c,cam in enumerate(cameras[1:],1):
        shift=np.array([(8*beta[c]+cam['t'][0])*10/4,cam['t'][1]*10/4])
        np.savez_compressed(cache/'roma'/f'pair_0f0001_{c}f0001.npz',uv_AB=(grid+shift).astype(np.float32),
            uv_BA=(grid-shift).astype(np.float32),confidence_AB=np.full((h,w),.8,np.float32),
            confidence_BA=np.full((h,w),.8,np.float32),valid_AB=np.ones((h,w),bool),valid_BA=np.ones((h,w),bool))
    return directory


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);make_fixture(self.root)
        self.store=DatasetStore(self.root)
    def tearDown(self):self.store.close();self.temp.cleanup()
    def test_catalog_load_and_field_laziness(self):
        entries=self.store.catalog();self.assertEqual(entries[0]['id'],'results/toy')
        run=self.store.load(entries[0]['id'])
        self.assertNotIn('uv',run.obs.cached_fields);self.assertNotIn('predicted_uv',run.result.cached_fields)
        summary=run.summary();self.assertEqual(summary['query_count'],3)
        self.assertNotIn('uv',run.obs.cached_fields)
        self.assertEqual(run.obs.field_info('uv')['shape'],[3,3,4,2])
        self.assertNotIn('uv',run.obs.cached_fields)
        arr=run.obs['uv'];self.assertFalse(arr.flags.writeable)
        with self.assertRaises(ValueError):arr[0,0,0]=0
        self.assertEqual(run.index(26),0)
        with self.assertRaises(KeyError):run.index(9999)
    def test_single_point_row_does_not_materialize_large_field(self):
        run=self.store.load('results/toy');row=run.field_slice('obs','uv',1)
        self.assertEqual(row.shape,(3,4,2));self.assertNotIn('uv',run.obs.cached_fields)
        self.assertEqual(run.obs.row_load_count,1)
        self.assertFalse(row.flags.writeable)
        np.testing.assert_array_equal(row,run.obs['uv'][1])
        scalar=run.field_slice('result','trajectory_valid',0)
        self.assertEqual(scalar.shape,());self.assertTrue(scalar)
    def test_method_specific_frozen_lsm_and_two_run_cache(self):
        a=self.store.load('results/toy','B3');b=self.store.load('results/toy','B4')
        self.assertNotEqual(a.observations_path,b.observations_path)
        self.assertAlmostEqual(float(b.obs['uv'][0,1,1,0]-a.obs['uv'][0,1,1,0]),.1,places=5)
        make_fixture(self.root,'other');self.store.load('results/other')
        self.assertEqual(len(self.store.cached_runs),2);self.assertEqual(a.obs.cache_bytes,0)
    def test_image_rgb_single_frame_and_identity(self):
        run=self.store.load('results/toy');image=run.image(1,2)
        np.testing.assert_array_equal(image[0,0],[21,40,80]);self.assertFalse(image.flags.writeable)
        self.assertEqual(frame_identity(run,1,2)['raw_frame_id'],103)
        with self.assertRaises(IndexError):run.image(0,99)
    def test_reference_seed_identity_is_independent_of_display_frame_and_method(self):
        run=self.store.load('results/toy');identity=run.summary()['reference_seed']
        self.assertEqual(identity['camera_id'],0);self.assertEqual(identity['local_frame'],1)
        self.assertEqual(identity['raw_frame_id'],101)
        self.assertEqual(identity['image_sha256'],sha256(identity['image_path']))
        self.assertEqual(identity['image_hash_kind'],'file_bytes_sha256')
        run.image(1,3)
        self.assertEqual(identity,run.summary()['reference_seed'])
        self.assertEqual(identity,self.store.load('results/toy','B4').summary()['reference_seed'])
    def test_frozen_source_hash_and_traversal_are_rejected(self):
        with self.assertRaises(KeyError):self.store.load('../secret')
        path=self.root/'results/toy/observations.npz'
        with path.open('ab') as handle:handle.write(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):self.store.load('results/toy')
    def test_json_conversion_and_readonly_requests_do_not_write(self):
        paths=list(self.root.rglob('*'));before={str(p):sha256(p) for p in paths if p.is_file()}
        run=self.store.load('results/toy');run.image(0,0);run.summary();run.obs['status']
        self.assertEqual(before,{str(p):sha256(p) for p in paths if p.is_file()})
        self.assertEqual(to_jsonable(dict(a=np.array([1.,np.nan,np.inf]),p=self.root)),dict(a=[1.,None,None],p=str(self.root)))


if __name__=='__main__':unittest.main()
