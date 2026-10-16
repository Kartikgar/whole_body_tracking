"""Measure how finetuning rollouts shift a frozen pelvis-wrench ensemble's inputs and outputs.

Two modes share the same per-step measurements:

* ``finetune``: replicate the ensemble-finetune rollout (base actor, frozen ensemble, gate, injection 1.0).
  On each state the ensemble is evaluated on paired inputs that differ only in the base-action channel
  (deterministic actor mean vs. the sampled PPO action) and, with DR on, in the joint-default offset
  (randomized vs. nominal default in ``joint_pos_rel``). The environment is driven by ``--drive``.
* ``reference``: replay a recorded Genesis dataset in the open-loop wrench task (the delta's training task).
  On each state the ensemble is evaluated on the recorded action and on the recorded action plus Gaussian
  noise with the base actor's exploration std. The environment applies the gated ensemble wrench.

Input shift is reported as squared z-scores against each member's empirical observation normalizer, which was
fit on that member's training inputs (about 1 per dimension in distribution), averaged over members.
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
parser.add_argument("--mode", choices=("finetune", "reference"), required=True)
parser.add_argument("--task", default=None, help="Defaults to the wrench finetune or open-loop task for --mode.")
parser.add_argument("--base_checkpoint", required=True, help="Base actor; supplies the exploration std in both modes.")
parser.add_argument("--motion_file", required=True)
parser.add_argument("--delta_policy_checkpoints", nargs="+", required=True)
parser.add_argument("--gate_scale", type=float, default=8.0)
parser.add_argument("--dr", choices=("on", "off"), default="on", help="Source DR events (finetune mode only).")
parser.add_argument("--drive", choices=("sampled", "deterministic"), default="sampled")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--num_steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--pre_failure_steps", type=int, default=25)
parser.add_argument("--output_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
if args_cli.task is None:
    args_cli.task = ("Tracking-Flat-G1-DeltaWrench-Finetune-v0" if args_cli.mode == "finetune"
                     else "Tracking-Flat-G1-DeltaWrench-OpenLoop-v0")
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import json
import numpy as np
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner

DR_EVENTS = ("physics_material", "add_joint_default_pos", "base_com", "push_robot")
COMPONENTS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


def ensemble_outputs(runner, obs, gate_scale, clip, scale):
    """Per-member outputs and the runner's gated mean, clipped and scaled as the wrench action term does."""
    preds = torch.stack([p.act_inference(n(obs)) for p, n in
                         zip(runner.delta_policies, runner.delta_obs_normalizers, strict=True)])
    uncertainty = preds.std(dim=0, correction=0).square().mean(dim=-1).sqrt()
    gate = torch.exp(-gate_scale * uncertainty)
    gated = preds.mean(dim=0) * gate.unsqueeze(-1)
    return dict(gated=gated, uncertainty=uncertainty, gate=gate,
                saturated=gated.abs() >= clip, wrench=gated.clamp(-clip, clip) * scale)


def block_z2(runner, obs, blocks):
    """Per-block mean squared z-score against each member's training normalizer, averaged over members."""
    z2 = torch.stack([((obs - n._mean) / (n._std + n.eps)).square() for n in runner.delta_obs_normalizers]).mean(0)
    return torch.stack([z2[:, s].mean(-1) for s in blocks.values()] + [z2.mean(-1)], dim=-1)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    out = Path(args_cli.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    env_cfg.commands.motion.motion_file = str(Path(args_cli.motion_file).resolve())
    if args_cli.mode == "finetune" and args_cli.dr == "off":
        for name in DR_EVENTS:
            setattr(env_cfg.events, name, None)
    if args_cli.mode == "reference" and args_cli.dr == "on":
        raise ValueError("Reference mode replicates delta training, which used DR off; pass --dr off")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    unwrapped = env.unwrapped
    train_cfg = agent_cfg.to_dict()
    train_cfg.update(delta_policy_checkpoints=list(args_cli.delta_policy_checkpoints),
                     delta_policy_uncertainty_gate_scale=args_cli.gate_scale,
                     delta_policy_injection_probability=1.0, preserve_frozen_policy_rng=True,
                     delta_policy_clip_actions=None)
    if args_cli.mode == "reference":
        train_cfg.update(delta_policy_obs_group="policy", delta_policy_critic_obs_group="policy")
    runner = MotionOnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    group = runner.delta_policy_obs_group
    base_ckpt = torch.load(args_cli.base_checkpoint, map_location=runner.device, weights_only=False)
    base_std = base_ckpt["model_state_dict"]["std"].to(runner.device)
    if args_cli.mode == "finetune":
        runner.load(args_cli.base_checkpoint, load_optimizer=False)
    runner.eval_mode()

    manager = unwrapped.observation_manager
    blocks, start = {}, 0
    for name, dim in zip(manager.active_terms[group], manager.group_obs_term_dim[group]):
        blocks["current_action" if name == "motion_joint_action" else name] = slice(start, start + dim[0])
        start += dim[0]
    term = unwrapped.action_manager.get_term("joint_pos")
    clip = float(term.cfg.action_clip)
    scale = term._wrench_scale
    robot = unwrapped.scene["robot"]
    default_offset = None
    if args_cli.mode == "finetune" and hasattr(robot.data, "default_joint_pos_nominal"):
        default_offset = robot.data.default_joint_pos - robot.data.default_joint_pos_nominal
    variants = (["deterministic", "sampled"] + (["deterministic_nominal_default"] if default_offset is not None else [])
                if args_cli.mode == "finetune" else ["recorded", "recorded_plus_noise"])

    if args_cli.mode == "finetune":
        unwrapped.episode_length_buf = torch.randint_like(unwrapped.episode_length_buf,
                                                          high=int(unwrapped.max_episode_length))
    obs, _ = env.get_observations()
    T, N, V = args_cli.num_steps, args_cli.num_envs, len(variants)
    rec = dict(z2=np.zeros((T, N, V, len(blocks) + 1), np.float32), uncertainty=np.zeros((T, N, V), np.float32),
               gate=np.zeros((T, N, V), np.float32), wrench=np.zeros((T, N, V, 6), np.float32),
               saturated=np.zeros((T, N, V, 6), bool), done=np.zeros((T, N), bool),
               time_out=np.zeros((T, N), bool), base_pos_z=np.zeros((T, N), np.float32))
    terminations = {name: 0 for name in unwrapped.termination_manager.active_terms}
    checked = False
    with torch.inference_mode():
        for t in range(T):
            inputs = {}
            if args_cli.mode == "finetune":
                obs_n = runner.obs_normalizer(obs.to(runner.device))
                det = runner.alg.policy.act_inference(obs_n)
                sampled = runner.alg.policy.act(obs_n)
                for name, action in (("deterministic", det), ("sampled", sampled)):
                    runner._set_delta_base_action_buffer(action)
                    inputs[name] = runner._compute_delta_policy_obs().clone()
                if default_offset is not None:
                    inputs["deterministic_nominal_default"] = inputs["deterministic"].clone()
                    inputs["deterministic_nominal_default"][:, blocks["joint_pos"]] += default_offset
                drive = sampled if args_cli.drive == "sampled" else det
            else:
                inputs["recorded"] = runner._compute_delta_policy_obs().clone()
                inputs["recorded_plus_noise"] = inputs["recorded"].clone()
                inputs["recorded_plus_noise"][:, blocks["current_action"]] += torch.randn_like(
                    inputs["recorded"][:, blocks["current_action"]]) * base_std
            results = {name: ensemble_outputs(runner, x, args_cli.gate_scale, clip, scale) for name, x in inputs.items()}
            drive_key = ("sampled" if args_cli.drive == "sampled" else "deterministic") if args_cli.mode == "finetune" else "recorded"
            if not checked:
                if args_cli.mode == "finetune":
                    runner._set_delta_base_action_buffer(drive)
                reference = runner._compute_delta_actions(inputs[drive_key])
                if not torch.allclose(reference, results[drive_key]["gated"], atol=1e-6):
                    raise RuntimeError("Diagnostic gate computation does not match MotionOnPolicyRunner")
                checked = True
            for v, name in enumerate(variants):
                r = results[name]
                rec["z2"][t, :, v] = block_z2(runner, inputs[name], blocks).cpu().numpy()
                rec["uncertainty"][t, :, v] = r["uncertainty"].cpu().numpy()
                rec["gate"][t, :, v] = r["gate"].cpu().numpy()
                rec["wrench"][t, :, v] = r["wrench"].cpu().numpy()
                rec["saturated"][t, :, v] = r["saturated"].cpu().numpy()
            rec["base_pos_z"][t] = robot.data.root_pos_w[:, 2].cpu().numpy()
            if args_cli.mode == "finetune":
                runner._set_delta_base_action_buffer(drive)
                runner._set_delta_action_buffer(results[drive_key]["gated"])
                obs, _, dones, infos = env.step(drive.to(unwrapped.device))
            else:
                obs, _, dones, infos = env.step(results["recorded"]["gated"].to(unwrapped.device))
            rec["done"][t] = dones.bool().cpu().numpy()
            rec["time_out"][t] = infos["time_outs"].bool().cpu().numpy()
            for name in terminations if t >= 1 else ():  # step 0 is the warmup excluded in summarize
                terminations[name] += int(unwrapped.termination_manager.get_term(name).sum())

    np.savez_compressed(out / "per_step.npz", variants=np.array(variants), blocks=np.array([*blocks, "total"]), **rec)
    summary = summarize(rec, variants, [*blocks, "total"], args_cli.pre_failure_steps)
    summary.update(config=dict(vars(args_cli), task=args_cli.task, blocks={k: [s.start, s.stop] for k, s in blocks.items()},
                               base_std_mean=float(base_std.mean()), action_clip=clip, wrench_scale=scale.tolist(),
                               delta_obs_group=group),
                   termination_counts=terminations)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("variants", "pre_failure")}, indent=1)[:4000])
    env.close()


def describe(x):
    return dict(mean=float(np.mean(x)), p10=float(np.percentile(x, 10)), p50=float(np.percentile(x, 50)),
                p90=float(np.percentile(x, 90)), p99=float(np.percentile(x, 99))) if x.size else None


def variant_stats(rec, v, blocks, mask):
    w = rec["wrench"][:, :, v][mask]
    return dict(samples=int(mask.sum()),
                z2={b: describe(rec["z2"][:, :, v, i][mask]) for i, b in enumerate(blocks)},
                uncertainty=describe(rec["uncertainty"][:, :, v][mask]), gate=describe(rec["gate"][:, :, v][mask]),
                wrench_mean={c: float(w[:, i].mean()) for i, c in enumerate(COMPONENTS)},
                wrench_abs_mean={c: float(np.abs(w[:, i]).mean()) for i, c in enumerate(COMPONENTS)},
                force_norm=describe(np.linalg.norm(w[:, :3], axis=-1)),
                torque_norm=describe(np.linalg.norm(w[:, 3:], axis=-1)),
                saturation={c: float(rec["saturated"][:, :, v, i][mask].mean()) for i, c in enumerate(COMPONENTS)})


def summarize(rec, variants, blocks, pre_failure_steps, warmup_steps=1):
    """Exclude the first step(s): the initial state precedes the first motion-command reset and always fails."""
    T, N = rec["done"].shape
    valid = np.zeros((T, N), bool)
    valid[warmup_steps:] = True
    failure = rec["done"] & ~rec["time_out"] & valid
    pre_failure = np.zeros((T, N), bool)
    for t, n in zip(*np.nonzero(failure)):
        start = max(warmup_steps, t - pre_failure_steps + 1)
        earlier = np.nonzero(rec["done"][start:t, n])[0]  # do not cross into the previous episode
        pre_failure[(start + earlier[-1] + 1 if earlier.size else start):t + 1, n] = True
    summary = dict(variants={name: variant_stats(rec, v, blocks, valid) for v, name in enumerate(variants)},
                   valid_steps=int(valid.sum()), warmup_steps=warmup_steps,
                   failures=int(failure.sum()), time_outs=int((rec["time_out"] & valid).sum()))
    paired = {}
    for v, name in enumerate(variants[1:], start=1):
        dw = (rec["wrench"][:, :, v] - rec["wrench"][:, :, 0])[valid]
        paired[f"{name}_minus_{variants[0]}"] = dict(
            force_change_norm=describe(np.linalg.norm(dw[:, :3], axis=-1)),
            torque_change_norm=describe(np.linalg.norm(dw[:, 3:], axis=-1)),
            mean_change={c: float(dw[:, i].mean()) for i, c in enumerate(COMPONENTS)},
            gate_change=describe((rec["gate"][:, :, v] - rec["gate"][:, :, 0])[valid]),
            uncertainty_change=describe((rec["uncertainty"][:, :, v] - rec["uncertainty"][:, :, 0])[valid]))
    summary["paired"] = paired
    summary["pre_failure"] = ({name: dict(within=variant_stats(rec, v, blocks, pre_failure),
                                          outside=variant_stats(rec, v, blocks, valid & ~pre_failure))
                               for v, name in enumerate(variants)} if pre_failure.any() else None)
    return summary


if __name__ == "__main__":
    main()
    simulation_app.close()
