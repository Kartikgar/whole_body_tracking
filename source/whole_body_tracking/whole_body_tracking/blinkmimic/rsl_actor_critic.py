"""RSL-RL actor-critic whose actor is a BlinkMimic policy, so BeyondMimic tasks can play and PPO-finetune it.

The actor is BlinkMimic's ``Policy`` (motion encoder + decoder over 10 frames of proprioception, with its frozen
running-normalization buffers and the reference-action skip). It consumes the ``policy`` observation group of the
Sprout BlinkMimic tasks: ``[obs history (hist * obs_dim, oldest first), goal (K * (15 + 2J))]`` and outputs actions in
BlinkMimic (MJCF) joint order, which those tasks' action term uses (``preserve_order=True``). The critic is a fresh
RSL-RL MLP on the privileged ``critic`` group, and the Gaussian exploration std starts small (``init_noise_std``) so
PPO finetuning begins at the deployed (EMA) policy.

Architecture comes from ``blinkmimic_ckpt``'s training arguments; its EMA weights are loaded at construction unless
``load_actor_weights`` is False (then a finetuned RSL-RL checkpoint supplies them via ``runner.load``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic

from . import networks
from .mdp import obs_dim as blinkmimic_obs_dim
from .networks import Policy, policy_kwargs
from .robot import get_robot


class BlinkMimicActor(nn.Module):
    """Flat observation -> (history, goal) -> BlinkMimic policy action."""

    def __init__(self, pi: Policy, hist: int, obs_dim: int, amp_dtype: torch.dtype | None):
        super().__init__()
        self.pi, self.hist, self.obs_dim, self.amp_dtype = pi, hist, obs_dim, amp_dtype

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        n = self.hist * self.obs_dim
        obs_hist = obs[..., :n].reshape(*obs.shape[:-1], self.hist, self.obs_dim)
        networks.AMP_DTYPE = self.amp_dtype  # module-global switch read by networks.run_mlp
        return self.pi(obs_hist, obs[..., n:])


def load_blinkmimic_policy(ckpt_path: str, robot: str = "sprout", load_weights: bool = True):
    """(Policy, training args) of a BlinkMimic checkpoint; weights are the EMA policy ``pi`` (as evaluate.py)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a_ = ck["args"]
    spec = get_robot(robot)
    J = spec.num_joints
    widths = lambda key, default: tuple(int(x) for x in a_.get(key, default).split(","))  # noqa: E731
    goal_dim = a_["goal_frames"] * (15 + 2 * J)  # = MotionLib.goal_dim
    pi = Policy(blinkmimic_obs_dim(J), goal_dim, J, hist=a_["hist"], enc_hidden=widths("enc_hidden", "2048,1024,512,256"),
                dec_hidden=widths("dec_hidden", "2048,1024,512,256"), **policy_kwargs(a_, spec))  # fmt: skip
    if load_weights:
        pi.load_state_dict(ck["pi"])
    return pi, a_


class BlinkMimicActorCritic(ActorCritic):
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        blinkmimic_ckpt: str | None = None,
        blinkmimic_robot: str = "sprout",
        load_actor_weights: bool = True,
        critic_hidden_dims=(512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 0.05,
        noise_std_type: str = "scalar",
        actor_hidden_dims=None,  # unused: the actor architecture comes from the BlinkMimic checkpoint
        **kwargs,
    ):
        if blinkmimic_ckpt is None:
            raise ValueError("BlinkMimicActorCritic needs `blinkmimic_ckpt` (a BlinkMimic ckpt_*.pt)")
        super().__init__(num_actor_obs, num_critic_obs, num_actions, actor_hidden_dims=[1],
                         critic_hidden_dims=list(critic_hidden_dims), activation=activation,
                         init_noise_std=init_noise_std, noise_std_type=noise_std_type, **kwargs)  # fmt: skip
        pi, a_ = load_blinkmimic_policy(blinkmimic_ckpt, blinkmimic_robot, load_weights=load_actor_weights)
        J = get_robot(blinkmimic_robot).num_joints
        obs_dim = blinkmimic_obs_dim(J)
        expect = a_["hist"] * obs_dim + a_["goal_frames"] * (15 + 2 * J)
        if num_actor_obs != expect or num_actions != J:
            raise ValueError(f"policy obs/action dims {num_actor_obs}/{num_actions} != BlinkMimic {expect}/{J}")
        amp = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
        self.actor = BlinkMimicActor(pi, a_["hist"], obs_dim, amp)
        self.blinkmimic_ckpt = blinkmimic_ckpt
        print(f"[BlinkMimicActorCritic] actor from {blinkmimic_ckpt} (hist {a_['hist']}, goal frames "
              f"{a_['goal_frames']}, amp {a_.get('amp')}, weights {'loaded' if load_actor_weights else 'not loaded'})")

    def load_blinkmimic_weights(self, ckpt: dict) -> None:
        """Load the EMA policy of a BlinkMimic checkpoint dict into the actor (critic and std unchanged)."""
        self.actor.pi.load_state_dict(ckpt["pi"])


def is_blinkmimic_checkpoint(ckpt: dict) -> bool:
    return isinstance(ckpt, dict) and "pi" in ckpt and "args" in ckpt and "model_state_dict" not in ckpt
