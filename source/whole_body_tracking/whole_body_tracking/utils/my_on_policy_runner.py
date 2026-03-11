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
        self.delta_policy_checkpoint = self.cfg.get("delta_policy_checkpoint")
        self.delta_policy_obs_group = self.cfg.get("delta_policy_obs_group", "delta_policy")
        self.delta_policy_critic_obs_group = self.cfg.get("delta_policy_critic_obs_group", self.delta_policy_obs_group)
        self.delta_policy_action_buffer_name = self.cfg.get("delta_policy_action_buffer_name", "delta_external_actions")
        self.delta_policy_require = bool(self.cfg.get("delta_policy_require", False))
        self.delta_policy_clip_actions = self.cfg.get("delta_policy_clip_actions", self.env.clip_actions)

        self.delta_policy = None
        self.delta_obs_normalizer = torch.nn.Identity().to(self.device)

        if self.delta_policy_checkpoint is not None:
            self._load_delta_policy(self.delta_policy_checkpoint)
        elif self.delta_policy_require:
            raise ValueError("`delta_policy_require=True` but no `delta_policy_checkpoint` was provided.")

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

    def _load_delta_policy(self, checkpoint: str):
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Delta policy checkpoint not found: {checkpoint_path}")

        agent_cfg_path = self._resolve_agent_cfg_path(checkpoint_path)
        with open(agent_cfg_path, encoding="utf-8") as f:
            delta_agent_cfg = yaml.safe_load(f)

        obs, extras = self.env.get_observations()
        obs_dict = extras["observations"]
        if self.delta_policy_obs_group not in obs_dict:
            raise KeyError(
                f"Delta policy observation group '{self.delta_policy_obs_group}' not found. "
                f"Available groups: {list(obs_dict.keys())}"
            )
        if self.delta_policy_critic_obs_group not in obs_dict:
            raise KeyError(
                f"Delta policy critic observation group '{self.delta_policy_critic_obs_group}' not found. "
                f"Available groups: {list(obs_dict.keys())}"
            )

        num_actor_obs = obs_dict[self.delta_policy_obs_group].shape[1]
        num_critic_obs = obs_dict[self.delta_policy_critic_obs_group].shape[1]
        num_actions = self.env.num_actions

        policy_cfg = dict(delta_agent_cfg["policy"])
        policy_class = eval(policy_cfg.pop("class_name"))
        self.delta_policy: ActorCritic | ActorCriticRecurrent | StudentTeacher | StudentTeacherRecurrent = (
            policy_class(num_actor_obs, num_critic_obs, num_actions, **policy_cfg).to(self.device)
        )

        loaded_dict = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        self.delta_policy.load_state_dict(loaded_dict["model_state_dict"], strict=True)
        self.delta_policy.eval()
        for param in self.delta_policy.parameters():
            param.requires_grad = False

        delta_empirical_norm = bool(delta_agent_cfg.get("empirical_normalization", False))
        if delta_empirical_norm and "obs_norm_state_dict" in loaded_dict:
            self.delta_obs_normalizer = EmpiricalNormalization(shape=[num_actor_obs], until=1.0e8).to(self.device)
            self.delta_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.delta_obs_normalizer.eval()
        else:
            self.delta_obs_normalizer = torch.nn.Identity().to(self.device)

    def _compute_delta_actions(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor | None:
        if self.delta_policy is None:
            return None
        if self.delta_policy_obs_group not in obs_dict:
            raise KeyError(
                f"Delta policy observation group '{self.delta_policy_obs_group}' missing from env observations."
            )

        delta_obs = obs_dict[self.delta_policy_obs_group].to(self.device)
        with torch.inference_mode():
            delta_actions = self.delta_policy.act_inference(self.delta_obs_normalizer(delta_obs)).detach()
        if self.delta_policy_clip_actions is not None:
            delta_actions = torch.clamp(delta_actions, -self.delta_policy_clip_actions, self.delta_policy_clip_actions)
        return delta_actions

    def _set_delta_action_buffer(self, delta_actions: torch.Tensor):
        setattr(
            self.env.unwrapped,
            self.delta_policy_action_buffer_name,
            delta_actions.to(self.env.device),
        )

    def _clear_delta_action_buffer(self):
        if hasattr(self.env.unwrapped, self.delta_policy_action_buffer_name):
            delattr(self.env.unwrapped, self.delta_policy_action_buffer_name)

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
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, extras = self.env.get_observations()
        obs_dict_for_delta = extras["observations"]
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
                    delta_actions = self._compute_delta_actions(obs_dict_for_delta)
                    if delta_actions is None:
                        raise RuntimeError("Delta policy is required for this runner mode but is not initialized.")
                    self._set_delta_action_buffer(delta_actions)

                    actions = self.alg.act(obs, privileged_obs)
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    obs_dict_for_delta = infos["observations"]

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
