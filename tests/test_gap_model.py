"""Gap-model ensemble: shapes, uncertainty semantics, persistence and layout checks."""
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/whole_body_tracking"))
from whole_body_tracking.utils.gap_model import GapEnsemble, gap_model_obs_columns, student_t_nll  # noqa: E402

X_NAMES, Y_NAMES = [f"x{i}" for i in range(4)], [f"y{i}" for i in range(3)]


class GapModelTests(unittest.TestCase):
    def test_shapes(self):
        model = GapEnsemble(X_NAMES, Y_NAMES, num_members=3, hidden=(16,))
        loc, log_scale = model(torch.randn(7, 4))
        self.assertEqual(loc.shape, (3, 7, 3)); self.assertEqual(log_scale.shape, (3, 7, 3))
        pred = model.predict(torch.randn(7, 4))
        self.assertEqual(pred["mean"].shape, (7, 3)); self.assertEqual(pred["u"].shape, (7,))

    def test_identical_members_have_zero_epistemic_uncertainty(self):
        model = GapEnsemble(X_NAMES, Y_NAMES, num_members=3, hidden=(16,))
        for member in model.members[1:]:
            member.load_state_dict(model.members[0].state_dict())
        self.assertTrue(torch.allclose(model.predict(torch.randn(5, 4))["u"], torch.zeros(5), atol=1e-6))

    def test_round_trip_and_layout_check(self):
        model = GapEnsemble(X_NAMES, Y_NAMES, num_members=2, hidden=(8,))
        model.set_normalization(torch.randn(50, 4) * 3 + 1, torch.randn(50, 3) * 2)
        x = torch.randn(6, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model.save(path)
            loaded = GapEnsemble.load(path, expected_x_names=X_NAMES)
            self.assertTrue(torch.allclose(model.predict(x)["mean"], loaded.predict(x)["mean"]))
            with self.assertRaises(ValueError):
                GapEnsemble.load(path, expected_x_names=list(reversed(X_NAMES)))

    def test_obs_columns_skip_contacts_and_alias_current_action(self):
        # Finetune delta group: contact forces present (excluded from g), base action named current_action.
        names, dims = ["base_pos_z", "feet_contact_force", "joint_vel", "current_action"], [1, 2, 2, 2]
        x_names = ["base_pos_z[0]", "joint_vel[0]", "joint_vel[1]", "motion_joint_action[0]", "motion_joint_action[1]"]
        self.assertEqual(gap_model_obs_columns(names, dims, x_names), [0, 3, 4, 5, 6])
        with self.assertRaises(ValueError):
            gap_model_obs_columns(["base_pos_z"], [1], x_names)

    def test_uncertainty_grows_outside_training_support(self):
        torch.manual_seed(0)
        x = torch.rand(2000, 4) * 2 - 1  # support: [-1, 1]^4
        y = torch.stack([x[:, 0] ** 2, torch.sin(3 * x[:, 1]), x[:, 2] * x[:, 3]], -1)
        model = GapEnsemble(X_NAMES, Y_NAMES, num_members=4, hidden=(32, 32))
        model.set_normalization(x, y)
        yn = model.normalize_y(y)
        for member in model.members:
            opt = torch.optim.Adam(member.parameters(), lr=3e-3)
            for _ in range(300):
                loc, log_scale = member(model.normalize_x(x))
                loss = student_t_nll(loc, log_scale, yn, 4.0).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        inside = model.predict(torch.rand(500, 4) * 2 - 1)["u"].median()
        outside = model.predict(torch.rand(500, 4) * 2 + 4)["u"].median()  # [4, 6]^4
        self.assertGreater(float(outside), 3 * float(inside))


if __name__ == "__main__":
    unittest.main()
