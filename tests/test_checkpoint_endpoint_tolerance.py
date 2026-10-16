"""Check batch endpoint stopping thresholds without starting Isaac Sim."""
import ast
from pathlib import Path
import unittest

import torch


path = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/checkpoint_batch_eval.py"
tree = ast.parse(path.read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_failure_masks")
# All fixtures use identity quaternions. Isolate the threshold function from
# simulator imports; rotating gravity by identity leaves gravity unchanged.
scope = {"torch": torch, "quat_rotate_inverse": lambda q, v: v}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
failure_masks = scope["_failure_masks"]


class EndpointToleranceTests(unittest.TestCase):
    def make_states(self, endpoint_height, anchor_height=0.0):
        reference = {"joint_pos": torch.zeros(1, 2), "body_pos_w": torch.zeros(1, 2, 3),
                     "body_quat_w": torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]])}
        state = {key: value.clone() for key, value in reference.items()}
        state["body_pos_w"][0, 1, 2] = endpoint_height
        state["body_pos_w"][0, 0, 2] = anchor_height
        return state, reference

    def test_relaxation_only_changes_endpoint_check(self):
        state, ref = self.make_states(0.5)
        self.assertTrue(failure_masks(state, ref, 0, [1])[-1].item())
        self.assertFalse(failure_masks(state, ref, 0, [1], 1.0)[-1].item())

    def test_relaxed_boundary_is_strict(self):
        for height, failed in [(1.0, False), (1.001, True), (-1.001, True)]:
            state, ref = self.make_states(height)
            self.assertEqual(failure_masks(state, ref, 0, [1], 1.0)[-1].item(), failed)

    def test_pelvis_threshold_is_not_relaxed(self):
        state, ref = self.make_states(0.5, anchor_height=0.251)
        masks = failure_masks(state, ref, 0, [1], 1.0)
        self.assertTrue(masks[1].item())
        self.assertFalse(masks[-1].item())


if __name__ == "__main__":
    unittest.main()
