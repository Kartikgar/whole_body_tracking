import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/Delta_Utils' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

replay = load('eval_delta_replay_isaac')
plot = load('plot_delta_error_growth')

class ErrorGrowthTests(unittest.TestCase):
    def test_gate_sweep_schedule(self):
        windows = replay.make_windows([1000, 57], 50)
        selected = replay.select_trajectory_windows(windows, 3)
        self.assertEqual([w['start'] for w in selected if w['trajectory']==0], [0, 450, 925])
        self.assertEqual(len(selected), 4)
        expanded = replay.gate_conditions(selected, [0., 2.], True)
        self.assertEqual(len(expanded), 12)
        self.assertEqual([w['condition'] for w in expanded[:3]], ['zero_delta', 'lambda_0', 'lambda_2'])
        self.assertEqual(len({(w['trajectory'], w['start']) for w in expanded}), 4)

    def test_gate_matches_finetune_formula(self):
        import torch
        predictions = torch.tensor([[[2.]*6]*3, [[0.]*6]*3])
        delta, gate, uncertainty = replay.gated_ensemble_action(
            predictions, torch.tensor([0., 2., 0.]), torch.tensor([False, False, True]))
        torch.testing.assert_close(uncertainty, torch.ones(3))
        torch.testing.assert_close(gate, torch.tensor([1., np.exp(-2.), 0.], dtype=torch.float32))
        torch.testing.assert_close(delta, gate[:, None].expand(-1,6))

    def test_disagreement_scaling_and_clipping(self):
        raw=np.array([[[2.]*6], [[-2.]*6]])
        result=replay.ensemble_statistics(raw,300.,60.,1.)[0]
        self.assertEqual(result["normalized_disagreement"],2.)
        self.assertEqual(result["clipped_normalized_disagreement"],1.)
        self.assertEqual(result["force_disagreement_n"],300.)
        self.assertEqual(result["torque_disagreement_nm"],60.)
        self.assertEqual(result["saturation_fraction"],1.)

    def test_half_frame_rounding_and_reset(self):
        points = replay.measurement_schedule(50, 250, .25)
        self.assertEqual(len(points), 21)
        self.assertEqual([p['step'] for p in points[:5]], [0, 13, 25, 38, 50])
        self.assertEqual(points[-1]['elapsed_s'], 5.)

    def test_nondivisible_horizon_includes_endpoint(self):
        self.assertEqual(replay.measurement_schedule(50, 27, .25)[-1]['step'], 27)
        with self.assertRaises(ValueError):
            replay.measurement_schedule(50, 250, .001)

    def test_sustained_crossing_ignores_transient(self):
        values = [0, 1, 0, 1, 1, 1]
        curve = [dict(requested_time_s=i*.25, error=v) for i,v in enumerate(values)]
        self.assertEqual(plot.departure_time(curve, 'error', .5, .5), .75)
        self.assertIsNone(plot.departure_time(curve[:-1], 'error', .5, .5))

if __name__ == '__main__':
    unittest.main()
