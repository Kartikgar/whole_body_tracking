from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import Literal

import torch
import yaml
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
)

from isaaclab.managers import ObservationGroupCfg, ObservationManager
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass


class HighLevelPolicySwitchAction(ActionTerm):
    """High-level switch action that routes between two frozen low-level policies."""

    cfg: HighLevelPolicySwitchActionCfg

    def __init__(self, cfg: HighLevelPolicySwitchActionCfg, env):
        super().__init__(cfg, env)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._selected_policy = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._low_level_action_term = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self._low_level_action_dim = self._low_level_action_term.action_dim
        self._low_level_actions = torch.zeros(self.num_envs, self._low_level_action_dim, device=self.device)

        self._policy_1_last_actions = torch.zeros(self.num_envs, self._low_level_action_dim, device=self.device)
        self._policy_2_last_actions = torch.zeros(self.num_envs, self._low_level_action_dim, device=self.device)
        setattr(self._env, self.cfg.policy_1_last_action_buffer_name, self._policy_1_last_actions)
        setattr(self._env, self.cfg.policy_2_last_action_buffer_name, self._policy_2_last_actions)

        self._policy_1_obs_group_name = "ll_policy_1"
        self._policy_2_obs_group_name = "ll_policy_2"
        self._policy_1_obs_manager = self._build_obs_manager(
            observation_group_cfg=self.cfg.policy_1_observations,
            group_name=self._policy_1_obs_group_name,
            policy_last_action_buffer_name=self.cfg.policy_1_last_action_buffer_name,
        )
        self._policy_2_obs_manager = self._build_obs_manager(
            observation_group_cfg=self.cfg.policy_2_observations,
            group_name=self._policy_2_obs_group_name,
            policy_last_action_buffer_name=self.cfg.policy_2_last_action_buffer_name,
        )

        (
            self._policy_1,
            self._policy_1_normalizer,
            self._policy_1_expected_obs_dim,
            self._policy_1_expected_action_dim,
        ) = self._load_policy(self.cfg.policy_1_checkpoint)
        (
            self._policy_2,
            self._policy_2_normalizer,
            self._policy_2_expected_obs_dim,
            self._policy_2_expected_action_dim,
        ) = self._load_policy(self.cfg.policy_2_checkpoint)

        self._validate_action_dims()

        self._counter = 0

    @property
    def action_dim(self) -> int:
        if self.cfg.switch_action_space == "categorical":
            return 2
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _resolve_agent_cfg_path(self, checkpoint_path: Path) -> Path:
        candidates = [
            checkpoint_path.parent / "params" / "agent.yaml",
            checkpoint_path.parent.parent / "params" / "agent.yaml",
            checkpoint_path.parent / "agent.yaml",
            checkpoint_path.parent.parent / "agent.yaml",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"Could not find agent.yaml near checkpoint {checkpoint_path}. Looked at: {[str(c) for c in candidates]}"
        )

    @staticmethod
    def _resolve_policy_class(class_name: str):
        supported_policy_classes = {
            "ActorCritic": ActorCritic,
            "ActorCriticRecurrent": ActorCriticRecurrent,
            "StudentTeacher": StudentTeacher,
            "StudentTeacherRecurrent": StudentTeacherRecurrent,
        }
        if class_name in supported_policy_classes:
            return supported_policy_classes[class_name]

        # Allow module-qualified names in config, e.g. "rsl_rl.modules.ActorCritic".
        short_name = class_name.rsplit(".", maxsplit=1)[-1]
        if short_name in supported_policy_classes:
            return supported_policy_classes[short_name]

        supported_names = ", ".join(sorted(supported_policy_classes.keys()))
        raise RuntimeError(
            f"Unsupported low-level policy class '{class_name}'. "
            f"Supported classes: {supported_names}."
        )

    @staticmethod
    def _infer_mlp_input_dim(state_dict: dict, prefix: str) -> int | None:
        direct_key = f"{prefix}.0.weight"
        if direct_key in state_dict and getattr(state_dict[direct_key], "ndim", None) == 2:
            return int(state_dict[direct_key].shape[1])
        for key, value in state_dict.items():
            if key.startswith(f"{prefix}.") and key.endswith(".weight") and getattr(value, "ndim", None) == 2:
                return int(value.shape[1])
        return None

    @staticmethod
    def _infer_mlp_output_dim(state_dict: dict, prefix: str) -> int | None:
        indexed_layers: list[tuple[int, str, int]] = []
        for key, value in state_dict.items():
            if not (key.startswith(f"{prefix}.") and key.endswith(".weight")):
                continue
            if getattr(value, "ndim", None) != 2:
                continue
            layer_name = key[len(prefix) + 1 : -len(".weight")]
            layer_head = layer_name.split(".", 1)[0]
            if layer_head.isdigit():
                indexed_layers.append((int(layer_head), key, int(value.shape[0])))
        if not indexed_layers:
            return None
        indexed_layers.sort(key=lambda x: (x[0], x[1]))
        return indexed_layers[-1][2]

    def _build_obs_manager(
        self,
        observation_group_cfg: ObservationGroupCfg,
        group_name: str,
        policy_last_action_buffer_name: str,
    ) -> ObservationManager:
        group_cfg = copy.deepcopy(observation_group_cfg)

        if hasattr(group_cfg, "actions"):

            def _last_actions(dummy_env):
                last_actions = getattr(self._env, policy_last_action_buffer_name)
                if hasattr(self._env, "episode_length_buf"):
                    reset_ids = self._env.episode_length_buf == 0
                    if torch.any(reset_ids):
                        last_actions[reset_ids] = 0.0
                return last_actions

            group_cfg.actions.func = _last_actions
            group_cfg.actions.params = dict()

        return ObservationManager({group_name: group_cfg}, self._env)

    def _load_policy(self, checkpoint: str):
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Low-level policy checkpoint not found: {checkpoint_path}")

        loaded_dict = None
        try:
            loaded_dict = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        except Exception:
            loaded_dict = None

        if isinstance(loaded_dict, dict) and "model_state_dict" in loaded_dict:
            return self._load_rsl_rl_checkpoint(checkpoint_path, loaded_dict)

        # Fallback: TorchScript policy.
        try:
            jit_policy = torch.jit.load(str(checkpoint_path), map_location=self.device).eval()
        except Exception as exc:
            raise RuntimeError(
                f"Unsupported low-level policy checkpoint format: {checkpoint_path}. "
                "Expected an RSL-RL checkpoint with 'model_state_dict' or a TorchScript module."
            ) from exc

        for param in jit_policy.parameters():
            param.requires_grad = False

        return jit_policy, torch.nn.Identity().to(self.device), None, None

    def _load_rsl_rl_checkpoint(self, checkpoint_path: Path, loaded_dict: dict):
        agent_cfg_path = self._resolve_agent_cfg_path(checkpoint_path)
        with open(agent_cfg_path, encoding="utf-8") as f:
            agent_cfg = yaml.safe_load(f)

        state_dict = loaded_dict["model_state_dict"]
        ckpt_actor_obs = self._infer_mlp_input_dim(state_dict, "actor")
        ckpt_critic_obs = self._infer_mlp_input_dim(state_dict, "critic")
        ckpt_action_dim = self._infer_mlp_output_dim(state_dict, "actor")

        if ckpt_actor_obs is None:
            raise RuntimeError(
                f"Could not infer actor input dim from checkpoint: {checkpoint_path}"
            )
        if ckpt_action_dim is None:
            raise RuntimeError(
                f"Could not infer actor output dim from checkpoint: {checkpoint_path}"
            )

        policy_cfg = dict(agent_cfg["policy"])
        policy_class_name = policy_cfg.pop("class_name")
        policy_class = self._resolve_policy_class(policy_class_name)
        num_actor_obs = int(ckpt_actor_obs)
        num_critic_obs = int(ckpt_critic_obs) if ckpt_critic_obs is not None else num_actor_obs
        num_actions = int(ckpt_action_dim)

        policy: ActorCritic | ActorCriticRecurrent | StudentTeacher | StudentTeacherRecurrent = policy_class(
            num_actor_obs,
            num_critic_obs,
            num_actions,
            **policy_cfg,
        ).to(self.device)
        policy.load_state_dict(state_dict, strict=True)
        policy.eval()
        for param in policy.parameters():
            param.requires_grad = False

        empirical_norm = bool(agent_cfg.get("empirical_normalization", False))
        if empirical_norm and "obs_norm_state_dict" in loaded_dict:
            obs_normalizer = EmpiricalNormalization(shape=[num_actor_obs], until=1.0e8).to(self.device)
            obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            obs_normalizer.eval()
        else:
            obs_normalizer = torch.nn.Identity().to(self.device)

        return policy, obs_normalizer, num_actor_obs, num_actions

    def _validate_action_dims(self):
        for policy_name, action_dim in (
            ("policy_1", self._policy_1_expected_action_dim),
            ("policy_2", self._policy_2_expected_action_dim),
        ):
            if action_dim is not None and int(action_dim) != self._low_level_action_dim:
                raise RuntimeError(
                    f"{policy_name} action dim mismatch: checkpoint expects {action_dim}, "
                    f"but low_level_actions expects {self._low_level_action_dim}."
                )

    def _compute_obs_tensor(self, manager: ObservationManager, group_name: str) -> torch.Tensor:
        obs = manager.compute_group(group_name)
        if isinstance(obs, dict):
            active_terms = manager.active_terms[group_name]
            obs = torch.cat([obs[name] for name in active_terms], dim=-1)
        if obs.ndim > 2:
            obs = obs.reshape(obs.shape[0], -1)
        return obs.to(self.device)

    def _infer_actions(self, policy, normalizer, obs: torch.Tensor, expected_obs_dim: int | None, policy_name: str):
        if expected_obs_dim is not None and obs.shape[1] != int(expected_obs_dim):
            raise RuntimeError(
                f"{policy_name} observation dim mismatch: got {obs.shape[1]}, expected {expected_obs_dim}. "
                "Ensure the low-level observation config matches the checkpoint training setup."
            )

        with torch.inference_mode():
            normalized_obs = normalizer(obs)
            if hasattr(policy, "act_inference"):
                actions = policy.act_inference(normalized_obs).detach()
            else:
                actions = policy(normalized_obs)
                if isinstance(actions, tuple):
                    actions = actions[0]
                actions = actions.detach()
        return actions

    def process_actions(self, actions: torch.Tensor):
        if actions.ndim != 2 or actions.shape != self._raw_actions.shape:
            raise RuntimeError(
                f"High-level switch action shape mismatch: got {tuple(actions.shape)}, "
                f"expected {tuple(self._raw_actions.shape)}."
            )

        self._raw_actions[:] = actions

        if self.cfg.switch_action_space == "categorical":
            # Interpret high-level outputs as a 2-way categorical distribution over low-level policies.
            if self.cfg.categorical_input_is_logits:
                probs = torch.softmax(self._raw_actions, dim=-1)
            else:
                probs = torch.clamp(self._raw_actions, min=0.0)
                prob_sum = probs.sum(dim=-1, keepdim=True)
                invalid = prob_sum <= 1.0e-6
                probs = probs / torch.where(invalid, torch.ones_like(prob_sum), prob_sum)
                if torch.any(invalid):
                    probs[invalid.expand_as(probs)] = 0.5
            self._processed_actions[:] = probs
            return

        switch_prob = self._raw_actions[:, 0]
        if self.cfg.use_sigmoid:
            switch_prob = torch.sigmoid(switch_prob)
        switch_prob = torch.clamp(switch_prob, 0.0, 1.0)
        self._processed_actions[:, 0] = switch_prob

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            obs_1 = self._compute_obs_tensor(self._policy_1_obs_manager, self._policy_1_obs_group_name)
            obs_2 = self._compute_obs_tensor(self._policy_2_obs_manager, self._policy_2_obs_group_name)

            low_level_actions_1 = self._infer_actions(
                self._policy_1,
                self._policy_1_normalizer,
                obs_1,
                self._policy_1_expected_obs_dim,
                policy_name="policy_1",
            )
            low_level_actions_2 = self._infer_actions(
                self._policy_2,
                self._policy_2_normalizer,
                obs_2,
                self._policy_2_expected_obs_dim,
                policy_name="policy_2",
            )

            if self.cfg.low_level_action_clip is not None:
                low_level_actions_1 = torch.clamp(
                    low_level_actions_1, -self.cfg.low_level_action_clip, self.cfg.low_level_action_clip
                )
                low_level_actions_2 = torch.clamp(
                    low_level_actions_2, -self.cfg.low_level_action_clip, self.cfg.low_level_action_clip
                )

            self._policy_1_last_actions[:] = low_level_actions_1
            self._policy_2_last_actions[:] = low_level_actions_2

            if self.cfg.switch_action_space == "categorical":
                # Index 0 maps to policy 1, index 1 maps to policy 2.
                select_policy_1 = self._processed_actions[:, 0] >= self._processed_actions[:, 1]
            else:
                select_policy_1 = self._processed_actions[:, 0] > self.cfg.switch_threshold
            self._selected_policy[:] = torch.where(
                select_policy_1,
                torch.ones_like(self._selected_policy),
                torch.full_like(self._selected_policy, 2),
            )
            self._low_level_actions[:] = torch.where(
                select_policy_1.unsqueeze(-1),
                low_level_actions_1,
                low_level_actions_2,
            )

            self._low_level_action_term.process_actions(self._low_level_actions)
            self._counter = 0

        self._low_level_action_term.apply_actions()
        self._counter += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._selected_policy[env_ids] = 0
        self._low_level_actions[env_ids] = 0.0
        self._policy_1_last_actions[env_ids] = 0.0
        self._policy_2_last_actions[env_ids] = 0.0
        self._low_level_action_term.reset(env_ids)


@configclass
class HighLevelPolicySwitchActionCfg(ActionTermCfg):
    """Config for high-level switch action with two frozen low-level policies."""

    class_type: type[ActionTerm] = HighLevelPolicySwitchAction

    low_level_actions: ActionTermCfg = MISSING
    policy_1_checkpoint: str = MISSING
    policy_2_checkpoint: str = MISSING

    policy_1_observations: ObservationGroupCfg = MISSING
    policy_2_observations: ObservationGroupCfg = MISSING

    low_level_decimation: int = 1
    switch_action_space: Literal["bernoulli", "categorical"] = "bernoulli"
    switch_threshold: float = 0.5
    use_sigmoid: bool = True
    categorical_input_is_logits: bool = True
    low_level_action_clip: float | None = None

    policy_1_last_action_buffer_name: str = "low_level_policy_1_last_actions"
    policy_2_last_action_buffer_name: str = "low_level_policy_2_last_actions"
