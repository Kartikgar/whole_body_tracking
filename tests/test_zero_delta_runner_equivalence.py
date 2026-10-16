"""Compare the actual standard and zero-injection PPO loops on matched data.

This isolates runner behavior; it does not claim Isaac physics equivalence.
"""
import contextlib
from pathlib import Path
import io
import unittest
from unittest.mock import patch

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))  # run by path or by discovery
from helpers import Runner


class ToyEnv:
    num_envs = 4
    num_actions = 2
    device = "cpu"
    clip_actions = None
    max_episode_length = 5

    def __init__(self, critic):
        self.unwrapped = self
        self.critic = critic
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.state = torch.arange(8, dtype=torch.float).reshape(4, 2) / 10
        self.action_history = []

    def get_observations(self):
        observations = {"critic": self.state.square()} if self.critic else {}
        return self.state.clone(), {"observations": observations}

    def step(self, actions):
        if hasattr(self, "delta_external_actions"):
            assert torch.count_nonzero(self.delta_external_actions) == 0
        self.action_history.append(actions.clone())
        self.state = self.state * .9 + .1 * actions
        self.episode_length_buf += 1
        dones = self.episode_length_buf >= self.max_episode_length
        rewards = -self.state.square().sum(-1)
        self.state[dones] = .1
        self.episode_length_buf[dones] = 0
        obs, extras = self.get_observations()
        extras["time_outs"] = dones.clone()
        return obs, rewards, dones, extras


class ZeroDeltaRunnerEquivalenceTests(unittest.TestCase):
    def test_frozen_loading_can_preserve_training_rng(self):
        cfg = dict(num_steps_per_env=2, save_interval=10, empirical_normalization=True,
                   policy=dict(class_name="ActorCritic", actor_hidden_dims=[8], critic_hidden_dims=[8]),
                   algorithm=dict(class_name="PPO"))

        def rng_after_construction(frozen, preserve):
            import copy
            torch.manual_seed(42)
            config = copy.deepcopy(cfg)
            config.update(delta_policy_checkpoints=["mock"] if frozen else [],
                          preserve_frozen_policy_rng=preserve)
            with patch.object(Runner, "_load_delta_policies", lambda *_: torch.rand(100)), \
                    contextlib.redirect_stdout(io.StringIO()):
                Runner(ToyEnv(False), config, device="cpu")
            return torch.random.get_rng_state()

        expected = rng_after_construction(False, False)
        self.assertTrue(torch.equal(expected, rng_after_construction(True, True)))
        self.assertFalse(torch.equal(expected, rng_after_construction(True, False)))

    def run_loop(self, *, custom, critic):
        torch.manual_seed(123)
        env = ToyEnv(critic)
        cfg = dict(num_steps_per_env=8, save_interval=10,
                   empirical_normalization=True,
                   policy=dict(class_name="ActorCritic", actor_hidden_dims=[8],
                               critic_hidden_dims=[8], activation="elu"),
                   algorithm=dict(class_name="PPO", num_learning_epochs=2,
                                  num_mini_batches=2, learning_rate=1e-5,
                                  schedule="adaptive", desired_kl=.01),
                   delta_policy_injection_probability=0.)
        runner = Runner(env, cfg, device="cpu")
        runner.disable_logs = True
        if custom:
            runner.delta_policy = torch.nn.Identity()
            runner._compute_delta_policy_obs = lambda: env.state
            # Nonzero predictions must be suppressed by the real injection mask.
            runner._compute_delta_actions = lambda _: torch.ones(env.num_envs, 6)
        runner.learn(3, init_at_random_ep_len=True)
        return runner, env

    def test_zero_injection_matches_standard_updates_with_resets(self):
        for critic in [False, True]:
            with self.subTest(privileged_critic=critic):
                with contextlib.redirect_stdout(io.StringIO()):
                    standard, env_a = self.run_loop(custom=False, critic=critic)
                    custom, env_b = self.run_loop(custom=True, critic=critic)
                for a, b in zip(env_a.action_history, env_b.action_history, strict=True):
                    self.assertTrue(torch.equal(a, b))
                for module in ["obs_normalizer", "privileged_obs_normalizer"]:
                    for name, a in getattr(standard, module).state_dict().items():
                        self.assertTrue(torch.equal(a, getattr(custom, module).state_dict()[name]))
                for name, a in standard.alg.policy.state_dict().items():
                    self.assertTrue(torch.equal(a, custom.alg.policy.state_dict()[name]), name)
                self.assertEqual(standard.alg.learning_rate, custom.alg.learning_rate)


if __name__ == "__main__":
    unittest.main()
