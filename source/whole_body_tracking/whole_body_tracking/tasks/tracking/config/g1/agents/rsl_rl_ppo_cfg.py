from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

NEXT_LAB_DATE = "2026.07.08"

@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = f"{NEXT_LAB_DATE}/g1_base_policies"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class G1FlatDeltaActPPORunnerCfg(G1FlatPPORunnerCfg):
    experiment_name = f"{NEXT_LAB_DATE}/g1_delta_policies"
    max_iterations = 2000
    save_interval = 200

LOW_FREQ_SCALE = 0.5


@configclass
class G1FlatLowFreqPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)


@configclass
class G1FlatDeltaAFineTunePPORunnerCfg(G1FlatPPORunnerCfg):
    experiment_name = f"{NEXT_LAB_DATE}/g1_finetuned_policies"
    max_iterations = 1000
    save_interval = 100

    # Frozen open-loop delta policy checkpoints used to inject delta actions during rollout.
    delta_policy_checkpoints: list[str] | None = None
    delta_policy_obs_group: str = "delta_policy"
    delta_policy_critic_obs_group: str = "critic"
    delta_policy_action_buffer_name: str = "delta_external_actions"
    delta_policy_base_action_buffer_name: str = "delta_base_actions"
    delta_policy_require: bool = True
    delta_policy_clip_actions: float | None = None
    delta_policy_uncertainty_gate_scale: float = 2.5
