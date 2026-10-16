"""Shared test helpers: load MotionOnPolicyRunner on CPU without Isaac Sim."""
import importlib.util
from pathlib import Path
import types
from unittest.mock import patch


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
