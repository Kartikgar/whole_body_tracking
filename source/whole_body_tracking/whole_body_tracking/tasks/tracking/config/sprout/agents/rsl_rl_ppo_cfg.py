"""RSL-RL runner configs for the Sprout BlinkMimic tasks.

The base policy is a BlinkMimic checkpoint (``BLINKMIMIC_CKPT`` by default; override with
``agent.policy.blinkmimic_ckpt=...``). ``play.py`` / ``train.py --checkpoint`` accept either that raw BlinkMimic
checkpoint (actor weights only; critic, std and optimizer start fresh) or an RSL-RL ``model_*.pt`` saved by a
finetuning run.
"""

from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import (
    NEXT_LAB_DATE,
    G1FlatDeltaActPPORunnerCfg,
    G1FlatDeltaAFineTunePPORunnerCfg,
    G1FlatPPORunnerCfg,
)

_REPO = Path(__file__).resolve().parents[8]  # .../whole_body_tracking (repository root)
BLINKMIMIC_CKPT = str(_REPO / "logs/blinkmimic_isaac/sprout_10k_s0/ckpt_010000.pt")


@configclass
class BlinkMimicActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "BlinkMimicActorCritic"
    blinkmimic_ckpt: str = BLINKMIMIC_CKPT
    """BlinkMimic checkpoint giving the actor architecture (and its initial EMA weights)."""
    blinkmimic_robot: str = "sprout"
    init_noise_std: float = 0.05
    """Exploration std in action units (x action scale: ~0.03 rad legs, ~0.1 rad arms): start at the deployed policy."""
    actor_hidden_dims: list[int] = [1]  # unused (architecture from the checkpoint)
    critic_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"


@configclass
class SproutBlinkMimicPPORunnerCfg(G1FlatPPORunnerCfg):
    experiment_name = f"{NEXT_LAB_DATE}/sprout_blinkmimic_policies"
    max_iterations = 1000
    save_interval = 50
    # The BlinkMimic actor normalizes its own inputs with frozen running statistics.
    empirical_normalization = False
    policy = BlinkMimicActorCriticCfg()

    def __post_init__(self):
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.desired_kl = 0.005


@configclass
class SproutDeltaActPPORunnerCfg(G1FlatDeltaActPPORunnerCfg):
    experiment_name = f"{NEXT_LAB_DATE}/sprout_delta_policies"


@configclass
class SproutBlinkMimicDeltaAFineTunePPORunnerCfg(G1FlatDeltaAFineTunePPORunnerCfg):
    experiment_name = f"{NEXT_LAB_DATE}/sprout_blinkmimic_finetuned_policies"
    empirical_normalization = False
    policy = BlinkMimicActorCriticCfg()

    def __post_init__(self):
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.desired_kl = 0.005
