import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments (Sprout, BlinkMimic base policy).
##

_AGENTS = f"{agents.__name__}.rsl_rl_ppo_cfg"

for task_id, env_cfg, runner_cfg in (
    ("Tracking-Flat-Sprout-BlinkMimic-v0", flat_env_cfg.SproutBlinkMimicFlatEnvCfg, "SproutBlinkMimicPPORunnerCfg"),
    ("Tracking-Flat-Sprout-DeltaA-OpenLoop-v0", flat_env_cfg.SproutDeltaAOpenLoopEnvCfg, "SproutDeltaActPPORunnerCfg"),
    (
        "Tracking-Flat-Sprout-BlinkMimic-DeltaA-Finetune-v0",
        flat_env_cfg.SproutBlinkMimicDeltaAFineTuneEnvCfg,
        "SproutBlinkMimicDeltaAFineTunePPORunnerCfg",
    ),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": env_cfg, "rsl_rl_cfg_entry_point": f"{_AGENTS}:{runner_cfg}"},
    )
