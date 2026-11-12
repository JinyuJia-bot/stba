"""Independent read-only timing/3D/comparison checks, including unfair pairs."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

import cv2
import numpy as np

from experiments.dense_stba.debug_viewer.alignment import alignment_panel
from experiments.dense_stba.debug_viewer.scene import scene_data,trajectory_at
from experiments.dense_stba.debug_viewer.comparison import compare_runs


class FakeRun(SimpleNamespace):
    def index(self,point_id):return self.id_to_index[int(point_id)]
    def image(self,camera_index,frame):return cv2.cvtColor(cv2.imread(self.cameras[camera_index]['image_paths'][frame]),cv2.COLOR_BGR2RGB)


def make_run(root,name='left',target_shift=0,beta=.02):
    directory=Path(root)/name;directory.mkdir()
    image=Path(root)/'seed.png'
    if not image.exists():cv2.imwrite(str(image),np.arange(16*12*3,dtype=np.uint8).reshape(12,16,3))
    times=np.arange(5)/10;seed=np.array([[4.,4.],[8.,6.],[10.,8.]],np.float32);ids=np.array([68,104,138])
    K=np.array([[10.,0,7.5],[0,10.,5.5],[0,0,1]])
    cams=[]
    for j in range(2):
        raw=np.arange(100,105)+(target_shift if j else 0)
        cams.append(dict(camera_id=7+j,K=K.tolist(),R=np.eye(3).tolist(),t=[-float(j),0,0],
            D=[0]*5,size=[16,12],fps=10.,n_frames=5,raw_frame_ids=raw.tolist(),
            raw_presentation_timestamps_s=(raw/10).tolist(),nominal_timestamps=times.tolist(),
            image_paths=[str(image)]*5,integer_raw_frame_shift=target_shift if j else 0))
    local=np.column_stack([(seed[:,0]-7.5)/10*5,(seed[:,1]-5.5)/10*5,np.full(3,5.)])
    xyz=local[:,None]+np.stack([.2*times,.1*times,np.zeros(5)],-1)[None]
    observed=np.empty((3,2,5,2),np.float32)
    for j,cam in enumerate(cams):
        physical=(np.asarray(cam['raw_frame_ids'])-100)/10
        world=local[:,None]+np.stack([.2*physical,.1*physical,np.zeros(5)],-1)[None]
        xc=world+np.array(cam['t']);q=xc@K.T;observed[:,j]=q[...,:2]/q[...,2,None]
    valid=np.ones((3,5),bool);valid[1,2]=False
    model=np.zeros((3,5),bool);model[1,2]=True
    result=dict(point_ids=ids,seed_uv=seed,xyz=xyz,valid=valid,model_only_mask=model,
        trajectory_times=times,trajectory_valid=np.ones(3,bool),view_support=np.where(valid,2,1),
        beta=np.array([0.,beta]),initial_beta=np.zeros(2),timestamps=np.broadcast_to(times,(2,5)).copy(),
        observed_uv=observed,predicted_uv=observed+np.array([.1,0],np.float32),
        holdout_visible=np.ones((3,2,5),bool),train_visible=np.zeros((3,2,5),bool),
        motion_mask=np.array([True,False,True]),seed_eligible=np.array([True,True,False]),confidence=np.ones(3))
    manifest=dict(dataset='test_scene',reference_camera_id=7,seed_frame=2,cameras=cams,
                  mode='raw',translation_unit='native units')
    manifest_path=directory/'manifest.json';manifest_path.write_text(json.dumps(manifest))
    method=dict(manifest_path=str(manifest_path),config=dict(real_raw_beta_bound_frames=6.,
        association=dict(stride=4),time_init=dict(min_dynamic_tracks=2,min_samples_per_track=2,max_gap_frames=2.1)),
        log=[dict(iteration=0,beta_seconds=[0.,.01],objective=4,accepted=True),
             dict(iteration=1,objective=3),dict(iteration=2,beta_seconds=[0.,beta],objective=2,accepted=True)],
        coarse_profiles=[dict(camera_id=8,beta_grid_frames=[-1.,0.,1.],costs=[2.,1.,2.])],
        observability=dict(status=['reference_gauge','locally_observable'],dynamic_tracks_per_camera=[2,2]))
    return FakeRun(directory=directory,run_id=name,method='B3',meta=dict(manifest_path=str(manifest_path)),
        method_meta=method,manifest_path=manifest_path,manifest=manifest,cameras=cams,
        obs=dict(seed_uv=seed,seed_frame=np.array(2)),result=result,point_ids=ids,
        id_to_index={int(v):i for i,v in enumerate(ids)},cache_dir=directory,source_validation={})


class ViewerPanelsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='dense-studio-panels-');self.root=Path(self.tmp.name)
        self.left=make_run(self.root)
    def tearDown(self):self.tmp.cleanup()

    def test_archived_history_only_and_roi_does_not_refit_beta(self):
        self.left.result['train_visible'][:]=True;self.left.result['holdout_visible'][:]=False
        before=self.left.result['beta'].copy()
        result=alignment_panel(self.left,{'type':'rectangle','bounds':[0,0,6,6]})
        self.assertEqual([r['iteration'] for r in result['iteration_history']['records']],[0,2])
        self.assertEqual(result['iteration_history']['rows_without_beta'],[1])
        self.assertEqual(result['roi_constraints']['query_count'],1)
        self.assertEqual(result['roi_constraints']['dynamic_training_tracks_per_camera'],[1,1])
        self.assertEqual(result['roi_constraints']['initialized_motion_candidates_per_camera'],[1,1])
        self.assertEqual(result['roi_constraints']['final_valid_motion_candidates_per_camera'],[1,1])
        self.assertEqual(result['roi_constraints']['archived_global_optimizer_dynamic_tracks_per_camera'],[2,2])
        self.assertFalse(result['roi_constraints']['meets_track_count_only'][1])
        self.assertFalse(result['coarse_profiles']['is_reoptimized_stba_objective'])
        self.assertFalse(result['objective_profile']['online_optimization'])
        np.testing.assert_array_equal(self.left.result['beta'],before)
        self.left.method_meta['log']=[dict(iteration=0,objective=1)]
        missing=alignment_panel(self.left)
        self.assertEqual(missing['iteration_history']['status'],'missing')
        self.assertEqual(missing['iteration_history']['records'],[])

    def test_synthetic_gt_beta_is_separate_read_only_overlay(self):
        self.left.manifest['dataset']='analytic_dynamic_plane'
        np.savez(self.left.directory/'gt.npz',beta_seconds=np.array([0.,.03]))
        original=self.left.result['beta'].copy()
        result=alignment_panel(self.left)
        self.assertTrue(result['synthetic_gt']['available'])
        self.assertTrue(result['synthetic_gt']['independent'])
        self.assertEqual(result['synthetic_gt']['beta_frames'],[0.,.3])
        self.assertAlmostEqual(result['beta_rows'][1]['error_to_gt_ms'],-10.)
        self.assertNotIn('published_total_beta_seconds',result['beta_rows'][1])
        np.testing.assert_array_equal(original,self.left.result['beta'])

    def test_scene_fixed_camera_centers_and_actual_ray_times(self):
        before={k:v.copy() for k,v in self.left.result.items() if isinstance(v,np.ndarray)}
        for value in self.left.result.values():
            if isinstance(value,np.ndarray):value.setflags(write=False)
        scene=scene_data(self.left,2,point_id=68,max_points=2,include_gt=False)
        np.testing.assert_allclose(scene['cameras'][1]['center'],[1,0,0])
        self.assertEqual(scene['sampling']['displayed_count'],2)
        self.assertIn(68,scene['cloud']['point_ids'])
        self.assertAlmostEqual(scene['reference_physical_time'],.2)
        self.assertAlmostEqual(scene['observation_rays'][1]['estimated_physical_time'],.22)
        self.assertAlmostEqual(scene['observation_rays'][1]['difference_from_cloud_time'],.02)
        self.assertEqual(scene['gt']['status'],'disabled')
        json.dumps(scene,allow_nan=False)
        for key,value in before.items():np.testing.assert_array_equal(self.left.result[key],value)

    def test_scene_roi_excludes_outside_selection_from_cloud_statistics(self):
        result=scene_data(self.left,2,point_id=138,max_points=3,
            roi={'type':'rectangle','bounds':[0,0,6,6]})
        self.assertEqual(result['cloud']['point_ids'],[68])
        self.assertEqual(result['sampling']['total_query_count'],3)
        self.assertEqual(result['sampling']['roi_query_count'],1)
        self.assertEqual(result['sampling']['supported_count'],1)
        self.assertEqual(result['sampling']['displayed_count'],1)
        self.assertTrue(result['selected_outside_roi'])
        self.assertEqual(result['selected_trajectory']['point_id'],138)
        self.assertEqual(len(result['observation_rays']),2)
        self.assertEqual(result['roi']['selected_query_count'],1)

    def test_trajectory_support_never_fills_invalid_gaps_or_extrapolates(self):
        xyz,valid,model,kind=trajectory_at(self.left,[1],[-.1,.15,.2,.45])
        np.testing.assert_array_equal(valid,[[False,False,False,False]])
        np.testing.assert_array_equal(model,[[False,True,True,False]])
        self.assertTrue(np.isnan(xyz[0,0]).all());self.assertTrue(np.isnan(xyz[0,-1]).all())
        self.assertIn('not_a_fitted_continuous_model',kind)

    def test_raw_exposure_pairing_does_not_pair_equal_local_frame_indices(self):
        right=make_run(self.root,'right',target_shift=-1,beta=-.06)
        # Frame 2 is a training frame; examples must remain inspectable while
        # the strict mutually held-out statistics continue to exclude it.
        self.left.result['holdout_visible'][:,:,2]=False;self.left.result['train_visible'][:,:,2]=True
        right.result['holdout_visible'][:,:,3]=False;right.result['train_visible'][:,:,3]=True
        result=compare_runs(self.left,right,point_id=68,frame=2)
        target=next(e for e in result['exposure_pairing']['examples'] if e['camera_id']==8)
        self.assertEqual(target['raw_frame_id'],102)
        self.assertEqual((target['left_local_frame'],target['right_local_frame']),(2,3))
        self.assertFalse(target['mutually_heldout'])
        self.assertTrue(target['left_train']);self.assertTrue(target['right_train'])
        self.assertTrue(target['same_measurement'])
        self.assertAlmostEqual(target['left_estimated_physical_time'],.22)
        self.assertAlmostEqual(target['right_estimated_physical_time'],.24)
        self.assertTrue(result['compatibility']['shared_reference_clock'])

    def test_changed_observations_counted_before_missing_predictions(self):
        right=make_run(self.root,'right')
        right.result['observed_uv'][:,1,:,0]+=1
        right.result['predicted_uv'][:,1]=np.nan
        result=compare_runs(self.left,right)
        self.assertEqual(result['changed_measurements']['count'],15)
        self.assertEqual(result['same_measurement_holdout']['identical_target_observations'],15)
        self.assertEqual(result['same_measurement_holdout']['common_finite_predictions'],15)
        self.assertTrue(result['changed_measurements']['excluded_from_same_measurement_comparison'])

    def test_reference_clock_mismatch_keeps_exposure_pairing_separate(self):
        right=make_run(self.root,'right')
        right.result['timestamps'][0]+=.1
        result=compare_runs(self.left,right)
        self.assertFalse(result['compatibility']['shared_reference_clock'])
        self.assertGreater(result['same_measurement_holdout']['common_finite_predictions'],0)
        self.assertEqual(result['trajectory_pairing']['status'],'unavailable')

    def test_dataset_seed_and_gt_mismatches_are_rejected(self):
        right=make_run(self.root,'right')
        right.manifest['dataset']='different'
        with self.assertRaisesRegex(ValueError,'different datasets'):compare_runs(self.left,right)
        right.manifest['dataset']='test_scene'
        other=self.root/'other.png';cv2.imwrite(str(other),np.zeros((12,16,3),np.uint8))
        right.cameras[0]['image_paths']=[str(other)]*5
        with self.assertRaisesRegex(ValueError,'seed images'):compare_runs(self.left,right)
        right.cameras[0]['image_paths']=list(self.left.cameras[0]['image_paths'])
        for run,value in [(self.left,0),(right,1)]:
            np.savez(run.directory/'gt.npz',identity=np.array([value]))
            (run.directory/'gt_metadata.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'GT identities'):compare_runs(self.left,right)

    def test_support_states_and_query_grid_difference_remain_distinct(self):
        right=make_run(self.root,'right')
        right.result['valid'][0,2]=False;right.result['model_only_mask'][0,2]=True
        # The last seed exists only on the right query grid.
        keep=np.array([0,1])
        self.left.point_ids=self.left.point_ids[keep]
        self.left.id_to_index={int(v):i for i,v in enumerate(self.left.point_ids)}
        for key,value in list(self.left.result.items()):
            if isinstance(value,np.ndarray) and value.shape[:1]==(3,):self.left.result[key]=value[keep]
        self.left.obs['seed_uv']=self.left.obs['seed_uv'][keep]
        result=compare_runs(self.left,right,frame=2)
        self.assertEqual(result['query_grid_pairing']['right_only_count'],1)
        self.assertEqual(result['query_grid_pairing']['right_only_point_ids'],[138])
        support=result['trajectory_pairing']
        self.assertEqual(support['left_only_supported_samples'],1)
        self.assertEqual(support['neither_supported_samples'],1)
        self.assertEqual(support['support_state_point_ids']['left_only_supported'],[68])
        self.assertEqual(support['support_state_point_ids']['neither_supported'],[104])

    def test_saved_time_windows_restrict_heldout_comparison(self):
        right=make_run(self.root,'right')
        self.left.meta['window']=[0,3];right.meta['window']=[2,5]
        result=compare_runs(self.left,right)
        self.assertEqual(result['same_measurement_holdout']['common_finite_predictions'],3*2)
        self.assertEqual(result['same_measurement_holdout']['left']['count'],3*2)


if __name__=='__main__':unittest.main()
