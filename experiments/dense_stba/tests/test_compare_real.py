"""Guard against unfair comparisons caused by changed targets or array ordering."""
import unittest
import numpy as np
from experiments.dense_stba.compare_real import common_reference


class CommonReferenceTests(unittest.TestCase):
    def make_run(self,name,ids,frames):
        ids=np.array(ids,np.int64);frames=np.array(frames,np.int64)
        values=(ids[:,None]+frames[None,:]).astype(np.float32)
        uv=np.stack([values,values*.5],axis=-1)[:,None]
        arrays={'point_ids':ids,'observed_uv':uv,'predicted_uv':uv.astype(float)+.2,
                'holdout_visible':np.ones(uv.shape[:-1],bool),
                'timestamps':((frames-200)/60.)[None]}
        return {'key':name,'arrays':arrays,'reference_index':0,'raw_frame_ids':frames,
                'reference_calibration':{'K':np.eye(3),'R':np.eye(3),'t':np.zeros(3)}}

    def test_seed_and_rawframe_identity_not_array_order(self):
        a=self.make_run('a',[5,3],[200,201,202])
        b=self.make_run('b',[3,5],[202,201,200])
        b['arrays']['predicted_uv'][0,0,0]=np.nan
        result=common_reference([a,b])
        self.assertEqual(result['mutual_original_heldout_observations'],6)
        self.assertEqual(result['common_finite_prediction_heldout_observations'],5)
        self.assertTrue(result['original_observed_uv_bitwise_equal'])
        self.assertAlmostEqual(result['by_run']['a']['rmse'],np.sqrt(.08))
        self.assertEqual(result['by_run']['a']['count'],result['by_run']['b']['count'])

    def test_changed_measurement_rejected_even_without_prediction(self):
        a=self.make_run('a',[5,3],[200,201,202])
        b=self.make_run('b',[3,5],[202,201,200])
        b['arrays']['predicted_uv'][0,0,0]=np.nan
        b['arrays']['observed_uv'][0,0,0,0]+=np.float32(.001)
        with self.assertRaisesRegex(ValueError,'not bitwise equal'):
            common_reference([a,b])

    def test_only_mutually_heldout_targets_are_evaluated(self):
        a=self.make_run('a',[5,3],[200,201,202])
        b=self.make_run('b',[5,3],[200,201,202])
        a['arrays']['holdout_visible'][0,0,0]=False
        b['arrays']['holdout_visible'][1,0,1]=False
        result=common_reference([a,b])
        self.assertEqual(result['common_finite_prediction_heldout_observations'],4)


if __name__=='__main__':unittest.main()
