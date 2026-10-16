"""Check root-error recovery from post-action source recordings."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec=importlib.util.spec_from_file_location('closed_loop_plot',Path(__file__).resolve().parents[1]/'scripts/Delta_Utils/plot_closed_loop_ensemble.py')
plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(plot)


class RootMetricTests(unittest.TestCase):
    def test_frame_alignment_origins_and_quaternion_sign(self):
        ref=np.zeros((3,1,3));ref[:,0,0]=np.arange(3)
        origins=np.array([[10.,20.,0.],[-10.,-20.,0.]])
        positions=np.repeat(ref[None],2,axis=0)+origins[:,None,None,:]
        positions[:,:,:,0]+=.2
        quats=np.tile([1.,0.,0.,0.],(2,3,1,1))
        quats[:,1,0]=[np.sqrt(.5),0.,0.,np.sqrt(.5)]
        quats[1]*=-1
        archive=dict(body_names=np.array(['pelvis']),body_pos_w=positions,body_quat_w=quats)
        reference=dict(body_pos_w=ref,body_quat_w=np.tile([1.,0.,0.,0.],(3,1,1)))
        rows=[dict(env_id=i,diagnostic_curve=[dict(step=2)]) for i in range(2)]
        plot.enrich_root_errors(rows,archive,reference,origins)
        for row in rows:
            point=row['diagnostic_curve'][0]
            self.assertAlmostEqual(point['root_pos_error_m'],.2)
            self.assertAlmostEqual(point['root_orientation_error_rad'],np.pi/2)


if __name__=='__main__':unittest.main()
