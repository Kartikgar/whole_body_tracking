"""Exercise the real RSL-RL resume/update path without launching Isaac Sim."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic


def load_runner_class():
    # Only export helpers need Isaac; checkpoint loading and PPO run on CPU.
    exports = types.ModuleType("isaaclab_rl.rsl_rl")
    exports.export_policy_as_onnx = lambda *args, **kwargs: None
    motion_exports = types.ModuleType("whole_body_tracking.utils.exporter")
    motion_exports.attach_onnx_metadata = lambda *args, **kwargs: None
    motion_exports.export_motion_policy_as_onnx = lambda *args, **kwargs: None
    path = Path(__file__).resolve().parents[1] / "source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py"
    spec = importlib.util.spec_from_file_location("resume_test_runner", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {
        "isaaclab_rl.rsl_rl": exports,
        "whole_body_tracking.utils.exporter": motion_exports,
        "wandb": types.ModuleType("wandb"),
    }):
        spec.loader.exec_module(module)
    return module.MotionOnPolicyRunner


Runner = load_runner_class()


class OptimizerResumeTests(unittest.TestCase):
    saved_lr = 0.000050625
    configured_lr = 0.001

    def make_runner(self):
        with contextlib.redirect_stdout(io.StringIO()):
            policy = ActorCritic(2, 2, 2, actor_hidden_dims=[8], critic_hidden_dims=[8])
        runner = Runner.__new__(Runner)
        runner.alg = PPO(policy, learning_rate=self.configured_lr,
                         schedule="adaptive", desired_kl=0.01, device="cpu")
        runner.empirical_normalization = False
        runner._validate_wrench_checkpoint = lambda loaded: None
        return runner

    def save_checkpoint(self, path):
        runner = self.make_runner()
        runner.alg.optimizer.param_groups[0]["lr"] = self.saved_lr
        torch.save({"model_state_dict": runner.alg.policy.state_dict(),
                    "optimizer_state_dict": runner.alg.optimizer.state_dict(),
                    "iter": 42, "infos": {"test": "resume"}}, path)

    def test_scheduler_and_optimizer_resume_together(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            self.save_checkpoint(path)
            runner = self.make_runner()
            infos = runner.load(str(path))
        self.assertEqual(runner.alg.learning_rate, self.saved_lr)
        self.assertEqual(runner.alg.optimizer.param_groups[0]["lr"], self.saved_lr)
        self.assertEqual(runner.current_learning_iteration, 42)
        self.assertEqual(infos, {"test": "resume"})

    def test_weights_only_load_keeps_configured_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            self.save_checkpoint(path)
            runner = self.make_runner()
            runner.load(str(path), load_optimizer=False)
        self.assertEqual(runner.alg.learning_rate, self.configured_lr)
        self.assertEqual(runner.alg.optimizer.param_groups[0]["lr"], self.configured_lr)

    def test_first_adaptive_minibatch_uses_restored_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            self.save_checkpoint(path)
            runner = self.make_runner()
            runner.load(str(path))
        alg = runner.alg
        alg.init_storage("rl", 2, 2, [2], [2], [2])
        obs = torch.zeros(2, 2)
        with torch.inference_mode():
            for _ in range(2):
                alg.act(obs, obs)
                alg.process_env_step(torch.tensor([0.1, 0.2]), torch.zeros(2, dtype=torch.bool), {})
            alg.compute_returns(obs)

        class StopBeforeStep(Exception):
            pass

        rates = []

        def record_rate(optimizer, args, kwargs):
            rates.append(optimizer.param_groups[0]["lr"])
            raise StopBeforeStep

        alg.optimizer.register_step_pre_hook(record_rate)
        with self.assertRaises(StopBeforeStep):
            alg.update()
        self.assertAlmostEqual(rates[0], self.saved_lr * 1.5)


if __name__ == "__main__":
    unittest.main()
