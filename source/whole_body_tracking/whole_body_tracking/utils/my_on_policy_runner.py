import os
import time
from collections import deque
from pathlib import Path

import torch
import yaml
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
)
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import store_code_state

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
        self.delta_policy_checkpoints = self._normalize_delta_policy_checkpoints(
            self.cfg.get("delta_policy_checkpoints")
        )
        self.delta_policy_obs_group = self.cfg.get("delta_policy_obs_group", "delta_policy")
        self.delta_policy_critic_obs_group = self.cfg.get("delta_policy_critic_obs_group", self.delta_policy_obs_group)
        self.delta_policy_action_buffer_name = self.cfg.get("delta_policy_action_buffer_name", "delta_external_actions")
        self.delta_policy_base_action_buffer_name = self.cfg.get(
            "delta_policy_base_action_buffer_name", "delta_base_actions"
        )
        self.delta_policy_require = bool(self.cfg.get("delta_policy_require", False))
        self.delta_policy_clip_actions = self.cfg.get("delta_policy_clip_actions", self.env.clip_actions)
        self.delta_policy_uncertainty_gate_scale = float(self.cfg.get("delta_policy_uncertainty_gate_scale", 1.0))

        self.delta_policy = None
        self.delta_policies: list[ActorCritic | ActorCriticRecurrent | StudentTeacher | StudentTeacherRecurrent] = []
        self.delta_obs_normalizer = torch.nn.Identity().to(self.device)
        self.delta_obs_normalizers: list[torch.nn.Module] = []
        self.delta_policy_ensemble_size = 0
        self._reset_delta_ensemble_stats()

        if self.delta_policy_checkpoints:
            self._load_delta_policies(self.delta_policy_checkpoints)
        elif self.delta_policy_require:
            raise ValueError("`delta_policy_require=True` but no `delta_policy_checkpoints` were provided.")

    @staticmethod
    def _normalize_delta_policy_checkpoints(checkpoints) -> list[str]:
        if checkpoints is None:
            return []
        if isinstance(checkpoints, str):
            return [checkpoints]
        return [str(path) for path in checkpoints]

    def _reset_delta_ensemble_stats(self):
        """Reset cached ensemble diagnostics with normal (non-inference) tensors."""
        self.delta_policy_last_uncertainty = torch.zeros(self.env.num_envs, device=self.device)
        self.delta_policy_last_gate = torch.ones(self.env.num_envs, device=self.device)

    def _resolve_delta_policy_checkpoint_paths(self, checkpoints: list[str]) -> list[Path]:
        checkpoint_paths: list[Path] = []
        for checkpoint in checkpoints:
            checkpoint_path = Path(checkpoint).expanduser().resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Delta policy checkpoint not found: {checkpoint_path}")
            checkpoint_paths.append(checkpoint_path)
        return checkpoint_paths

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

    def _load_delta_policies(self, checkpoints: list[str]):
        checkpoint_paths = self._resolve_delta_policy_checkpoint_paths(checkpoints)

        self.delta_policies = []
        self.delta_obs_normalizers = []
        self.delta_policy_checkpoints = [str(path) for path in checkpoint_paths]

        expected_obs_group = self.delta_policy_obs_group
        expected_num_actions: int | None = None

        def _infer_mlp_input_dim(state_dict: dict, prefix: str) -> int | None:
            direct_key = f"{prefix}.0.weight"
            if direct_key in state_dict and state_dict[direct_key].ndim == 2:
                return int(state_dict[direct_key].shape[1])
            for key, value in state_dict.items():
                if key.startswith(f"{prefix}.") and key.endswith(".weight") and getattr(value, "ndim", None) == 2:
                    return int(value.shape[1])
            return None

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

        # Keep current-action channel available before first delta-policy obs computation.
        self._set_delta_base_action_buffer(torch.zeros(self.env.num_envs, self.env.num_actions, device=self.env.device))
        obs, extras = self.env.get_observations()
        obs_dict = extras["observations"]
        if expected_obs_group not in obs_dict:
            raise KeyError(
                f"Delta policy observation group '{expected_obs_group}' not found. "
                f"Available groups: {list(obs_dict.keys())}"
            )

        for member_idx, checkpoint_path in enumerate(checkpoint_paths):
            agent_cfg_path = self._resolve_agent_cfg_path(checkpoint_path)
            with open(agent_cfg_path, encoding="utf-8") as f:
                delta_agent_cfg = yaml.safe_load(f)

            loaded_dict = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
            state_dict = loaded_dict["model_state_dict"]

            ckpt_action_dim = _infer_mlp_output_dim(state_dict, "actor")
            ckpt_actor_obs = _infer_mlp_input_dim(state_dict, "actor")
            num_actor_obs = obs_dict[expected_obs_group].shape[1]
            if ckpt_actor_obs is not None and num_actor_obs != ckpt_actor_obs:
                matching_groups = [name for name, value in obs_dict.items() if value.shape[1] == ckpt_actor_obs]
                if len(matching_groups) == 1 and member_idx == 0:
                    expected_obs_group = matching_groups[0]
                    self.delta_policy_obs_group = expected_obs_group
                    num_actor_obs = ckpt_actor_obs
                    print(
                        f"[INFO]: Delta policy obs-group auto-switch: using '{self.delta_policy_obs_group}' "
                        f"to match checkpoint actor input dim {ckpt_actor_obs}."
                    )
                else:
                    group_dims = {name: int(value.shape[1]) for name, value in obs_dict.items()}
                    raise RuntimeError(
                        "Delta policy actor observation dimension mismatch: "
                        f"env group '{expected_obs_group}' has dim {num_actor_obs}, "
                        f"checkpoint expects dim {ckpt_actor_obs}. Available groups: {group_dims}."
                    )

            env_critic_obs = (
                obs_dict[self.delta_policy_critic_obs_group].shape[1]
                if self.delta_policy_critic_obs_group in obs_dict
                else None
            )
            ckpt_critic_obs = _infer_mlp_input_dim(state_dict, "critic")
            if ckpt_critic_obs is not None:
                num_critic_obs = ckpt_critic_obs
                if env_critic_obs is not None and env_critic_obs != ckpt_critic_obs:
                    print(
                        "[INFO]: Delta critic obs dim mismatch between env and checkpoint "
                        f"({env_critic_obs} vs {ckpt_critic_obs}). Using checkpoint dim for frozen policy load."
                    )
            elif env_critic_obs is not None:
                num_critic_obs = env_critic_obs
            else:
                num_critic_obs = num_actor_obs

            num_actions = ckpt_action_dim if ckpt_action_dim is not None else self.env.num_actions
            if expected_num_actions is None:
                expected_num_actions = num_actions
                self._set_delta_action_buffer(torch.zeros(self.env.num_envs, num_actions, device=self.env.device))
            elif num_actions != expected_num_actions:
                raise RuntimeError(
                    "Delta-policy ensemble action dimension mismatch: "
                    f"expected {expected_num_actions}, got {num_actions} for checkpoint '{checkpoint_path}'."
                )
            if ckpt_action_dim is not None and ckpt_action_dim != self.env.num_actions:
                print(
                    "[INFO]: Delta policy action dim mismatch between env and checkpoint "
                    f"({self.env.num_actions} vs {ckpt_action_dim}). Using checkpoint action dim for frozen policy load."
                )

            policy_cfg = dict(delta_agent_cfg["policy"])
            policy_class = eval(policy_cfg.pop("class_name"))
            delta_policy: ActorCritic | ActorCriticRecurrent | StudentTeacher | StudentTeacherRecurrent = policy_class(
                num_actor_obs, num_critic_obs, num_actions, **policy_cfg
            ).to(self.device)
            delta_policy.load_state_dict(state_dict, strict=True)
            delta_policy.eval()
            for param in delta_policy.parameters():
                param.requires_grad = False

            delta_empirical_norm = bool(delta_agent_cfg.get("empirical_normalization", False))
            if delta_empirical_norm and "obs_norm_state_dict" in loaded_dict:
                delta_obs_normalizer = EmpiricalNormalization(shape=[num_actor_obs], until=1.0e8).to(self.device)
                delta_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                delta_obs_normalizer.eval()
            else:
                delta_obs_normalizer = torch.nn.Identity().to(self.device)

            self.delta_policies.append(delta_policy)
            self.delta_obs_normalizers.append(delta_obs_normalizer)

        self.delta_policy = self.delta_policies[0] if self.delta_policies else None
        self.delta_obs_normalizer = self.delta_obs_normalizers[0] if self.delta_obs_normalizers else torch.nn.Identity().to(self.device)
        self.delta_policy_ensemble_size = len(self.delta_policies)
        if self.delta_policy_ensemble_size == 1:
            print(f"[INFO]: Loaded frozen delta policy from '{self.delta_policy_checkpoints[0]}'.")
        elif self.delta_policy_ensemble_size > 1:
            print(
                "[INFO]: Loaded delta-policy ensemble with "
                f"{self.delta_policy_ensemble_size} members; uncertainty gate scale="
                f"{self.delta_policy_uncertainty_gate_scale:.4f}."
            )

    def _compute_delta_policy_obs(self) -> torch.Tensor:
        delta_obs = self.env.unwrapped.observation_manager.compute_group(self.delta_policy_obs_group)
        if isinstance(delta_obs, dict):
            active_terms = self.env.unwrapped.observation_manager.active_terms[self.delta_policy_obs_group]
            delta_obs = torch.cat([delta_obs[name] for name in active_terms], dim=-1)
        return delta_obs

    def _compute_delta_actions(self, delta_obs_or_dict: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor | None:
        if self.delta_policy is None:
            return None
        if isinstance(delta_obs_or_dict, dict):
            if self.delta_policy_obs_group not in delta_obs_or_dict:
                raise KeyError(
                    f"Delta policy observation group '{self.delta_policy_obs_group}' missing from env observations."
                )
            delta_obs = delta_obs_or_dict[self.delta_policy_obs_group].to(self.device)
        else:
            delta_obs = delta_obs_or_dict.to(self.device)
        with torch.inference_mode():
            ensemble_actions = [
                delta_policy.act_inference(delta_obs_normalizer(delta_obs)).detach()
                for delta_policy, delta_obs_normalizer in zip(self.delta_policies, self.delta_obs_normalizers, strict=True)
            ]
        if len(ensemble_actions) == 1:
            delta_actions = ensemble_actions[0]
            self._reset_delta_ensemble_stats()
        else:
            stacked_actions = torch.stack(ensemble_actions, dim=0)
            delta_actions = stacked_actions.mean(dim=0)
            ensemble_std = stacked_actions.std(dim=0, correction=0)
            uncertainty = torch.sqrt(torch.mean(torch.square(ensemble_std), dim=-1))
            gate = torch.exp(-self.delta_policy_uncertainty_gate_scale * uncertainty)
            delta_actions = delta_actions * gate.unsqueeze(-1)
            # Clone so cleanup/logging can safely touch these outside inference_mode.
            self.delta_policy_last_uncertainty = uncertainty.detach().clone()
            self.delta_policy_last_gate = gate.detach().clone()
        if self.delta_policy_clip_actions is not None:
            delta_actions = torch.clamp(delta_actions, -self.delta_policy_clip_actions, self.delta_policy_clip_actions)
        return delta_actions

    def _set_delta_action_buffer(self, delta_actions: torch.Tensor):
        setattr(
            self.env.unwrapped,
            self.delta_policy_action_buffer_name,
            delta_actions.to(self.env.device),
        )

    def _set_delta_base_action_buffer(self, base_actions: torch.Tensor):
        setattr(
            self.env.unwrapped,
            self.delta_policy_base_action_buffer_name,
            base_actions.to(self.env.device),
        )

    def _clear_delta_action_buffer(self):
        if hasattr(self.env.unwrapped, self.delta_policy_action_buffer_name):
            delattr(self.env.unwrapped, self.delta_policy_action_buffer_name)
        if hasattr(self.env.unwrapped, self.delta_policy_base_action_buffer_name):
            delattr(self.env.unwrapped, self.delta_policy_base_action_buffer_name)
        self._reset_delta_ensemble_stats()

    def _log_com_force_metrics(self, it: int):
        if self.writer is None:
            return
        action_manager = getattr(self.env.unwrapped, "action_manager", None)
        if action_manager is None:
            return
        try:
            joint_pos_term = action_manager.get_term("joint_pos")
        except Exception:
            return
        consume_stats = getattr(joint_pos_term, "consume_applied_force_log_stats", None)
        if consume_stats is None:
            return

        force_stats = consume_stats()
        if not force_stats:
            return

        self.writer.add_scalar("DeltaForce/applied_net", force_stats["applied_force_net"], it)
        self.writer.add_scalar("DeltaForce/applied_x", force_stats["applied_force_x"], it)
        self.writer.add_scalar("DeltaForce/applied_y", force_stats["applied_force_y"], it)
        self.writer.add_scalar("DeltaForce/applied_z", force_stats["applied_force_z"], it)

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width=width, pad=pad)
        self._log_com_force_metrics(locs["it"])
        if self.writer is not None and self.delta_policy_ensemble_size > 1:
            self.writer.add_scalar(
                "DeltaEnsemble/epistemic_uncertainty_mean", float(self.delta_policy_last_uncertainty.mean().item()), locs["it"]
            )
            self.writer.add_scalar(
                "DeltaEnsemble/gate_mean", float(self.delta_policy_last_gate.mean().item()), locs["it"]
            )
            self.writer.add_scalar("DeltaEnsemble/num_members", float(self.delta_policy_ensemble_size), locs["it"])

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        if self.delta_policy is None:
            return super().learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=init_at_random_ep_len)

        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard.writer import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs)
                    self._set_delta_base_action_buffer(actions)

                    delta_obs = self._compute_delta_policy_obs()
                    delta_actions = self._compute_delta_actions(delta_obs)
                    if delta_actions is None:
                        raise RuntimeError("Delta policy is required for this runner mode but is not initialized.")
                    self._set_delta_action_buffer(delta_actions)

                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    self.alg.process_env_step(rewards, dones, infos)

                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        self._clear_delta_action_buffer()
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
