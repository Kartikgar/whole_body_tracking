"""Check orchestration contracts without launching simulators."""
import importlib.util
from pathlib import Path
import unittest
import json
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "finetune_driver", ROOT / "scripts/Delta_Utils/sweep_ensemble_finetune.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class FinetuneDriverTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        folder = Path(self.directory.name)
        config = json.loads((ROOT / "configs/finetune/dance29999_matched.json").read_text())
        for key in ["base_checkpoint", "motion_file", "genesis_initial_state_npz"]:
            file = folder / key
            file.write_bytes(b"fixture")
            config[key] = str(file)
        config["ensemble_checkpoints"] = []
        for index in range(3):
            file = folder / f"member{index}.pt"
            file.write_bytes(b"fixture")
            config["ensemble_checkpoints"].append(str(file))
        config.update(isaac_python=sys.executable, genesis_python=sys.executable)
        self.config_path = folder / "config.json"
        self.config_path.write_text(json.dumps(config))

    def config(self, **overrides):
        return driver.resolve_config(self.config_path, overrides=overrides)

    def test_matched_plan_uses_requested_training_and_evaluation_settings(self):
        config = self.config(lambdas=[12.], injection_probability=.25,
                             export_iterations=[49, 74, 99, 149, 199])
        stages = driver.build_plan(config, "/tmp/ensemble_driver_unit_plan")
        self.assertEqual(len(stages), 11)
        train = stages[0]["command"]
        self.assertIn("--require_dr", train)
        self.assertIn("--preserve_frozen_policy_rng", train)
        for flag, value in [("--max_iterations", "200"),
                            ("--delta_policy_injection_probability", "0.25"),
                            ("--delta_policy_uncertainty_gate_scale", "12.0")]:
            self.assertEqual(train[train.index(flag) + 1], value)
        genesis = [s for s in stages if s["kind"] == "genesis"]
        self.assertEqual([s["finetune_iteration"] + 1 for s in genesis], [50, 75, 100, 150, 200])
        for stage in genesis:
            cmd = stage["command"]
            for flag in ["--no-domain_randomization", "--no-add_noise",
                         "--randomize_startup_qpos", "--initial_state_npz", "--motion_file"]:
                self.assertIn(flag, cmd)
            self.assertEqual(cmd[cmd.index("--endpoint-height-error-tolerance-m") + 1], "1.0")

    def test_invalid_settings_fail_before_launch(self):
        for override in [dict(injection_probability=-.1), dict(injection_probability=1.1),
                         dict(injection_probability=float("nan")), dict(lambdas=[float("inf")]),
                         dict(export_iterations=[200]), dict(export_iterations=[49, 49]),
                         dict(iterations=0), dict(genesis_endpoint_height_error_m=0),
                         dict(finetune_domain_randomization=False)]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.config(**override)


if __name__ == "__main__":
    unittest.main()
