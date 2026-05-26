# Project Understanding

This document records a working understanding of the `whole_body_tracking` project, with emphasis on the current Delta-A sim2real pipeline and the intended deployment story.

## 1) Repository Role

This repository is the Isaac Lab training side of the BeyondMimic motion-tracking stack.

At a high level, it provides:

- motion preprocessing and replay utilities
- G1 humanoid tracking environments in Isaac Lab
- PPO training and playback scripts through RSL-RL
- Delta-A open-loop pretraining and Delta-A-assisted finetuning
- ONNX export and Genesis sim2sim evaluation utilities

The main custom code lives under:

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/`
- `source/whole_body_tracking/whole_body_tracking/robots/`
- `source/whole_body_tracking/whole_body_tracking/utils/`
- `scripts/rsl_rl/`

## 2) Core Training Workflow

The main workflow appears to be:

1. Prepare retargeted motion data as `.npz`.
2. Train a base tracking policy in Isaac Lab.
3. Optionally train an open-loop Delta-A policy that models a source-to-target dynamics gap.
4. Finetune a base tracking policy while injecting the frozen delta policy during rollout.
5. Export the finetuned policy and evaluate in Genesis as a target simulator.

Relevant scripts:

- `scripts/csv_to_npz.py`
- `scripts/replay_npz.py`
- `scripts/rsl_rl/train.py`
- `scripts/rsl_rl/play.py`
- `scripts/rsl_rl/evaluate_sim2sim_genesis.py`

## 3) Tracking Task Structure

The tracking task is manager-based and decomposed in the standard Isaac Lab style.

Important modules:

- `tasks/tracking/mdp/commands.py`
  - motion loading
  - motion sampling
  - reference-state queries
- `tasks/tracking/mdp/observations.py`
  - tracking observations
  - delta-policy observations
- `tasks/tracking/mdp/rewards.py`
  - tracking rewards and delta penalties
- `tasks/tracking/mdp/delta_actions.py`
  - open-loop delta action composition
  - finetune-time external delta composition
- `tasks/tracking/config/g1/flat_env_cfg.py`
  - G1 tracking task variants

## 4) Delta-A: Current Intended Meaning

The intended role of the delta policy in this project is:

- the delta policy is a training-time helper that makes the source simulator behave more like a target environment with shifted dynamics
- the target dynamics mismatch should live in the target environment, not in the nominal source simulator
- during deployment in the target environment, the delta policy is not meant to be used
- the deployed controller is the finetuned base policy alone

This means the conceptual goal of Delta-A finetuning is **not** necessarily to make the base policy numerically copy the delta outputs.

Instead, the goal is:

`T_source_with_delta(s, a_base) ~= T_target(s, a_base)`

If that equivalence holds closely enough, then a base policy trained in the delta-assisted source environment should transfer to the target environment without requiring the delta policy at deployment.

## 5) Important Distinction

There are two different ways to think about the delta policy:

1. **Controller composition view**
   - The executed action during finetuning is `base_action + delta_action`.
   - In this view, one may worry that the base policy depends on the delta branch.

2. **Dynamics emulation view**
   - The delta policy is treated as a mechanism to emulate the target environment inside the source simulator.
   - In this view, the finetuned base policy only needs to become good under the effective target-like dynamics induced by the delta helper.

The current project understanding is that the second view is the intended one.

## 6) Practical Risk in the Current Delta-A Setup

Even if the intended interpretation is correct, successful transfer still depends on one critical condition:

the delta-assisted source dynamics must match the real target dynamics closely enough over the state-action distribution visited by the finetuned base policy.

This is stronger than simply observing that:

- the open-loop delta policy trains well
- the finetuning reward increases
- the combined controller performs well in Isaac

Those facts only show that the assisted source setup is internally workable. They do not by themselves prove that the effective transition map matches the true target environment closely enough for deployment.

## 7) Current Working Hypothesis for Sim2Sim Failure

If a finetuned policy performs well in Isaac with Delta-A assistance but poorly in Genesis, plausible explanations include:

1. The delta-assisted source dynamics are not actually equivalent to the Genesis target dynamics.
2. The target gain perturbation used in Genesis does not match the target shift that Delta-A was intended to emulate.
3. Contact, solver, actuator, or timing differences dominate beyond a simple Kp/Kd mismatch.
4. The finetuned base policy only experienced a target-like environment through an imperfect learned helper, so transfer fails when the true target simulator differs from that helper-induced dynamics.

Under this interpretation, poor Genesis performance is not immediate evidence that Delta-A is conceptually wrong. It may instead indicate that the learned delta helper is not an accurate enough surrogate for the target simulator.

## 8) What Would Validate the Delta-A Assumption

The most direct validation experiment would be to compare:

- `source simulator + delta helper + base action`
- `target simulator + base action`

starting from matched states and using matched base-policy actions.

If the next-state trajectories are close, then the delta helper is serving its intended role.
If they diverge significantly, then the learned delta policy is acting more like a helpful auxiliary controller than a true target-dynamics emulator.

## 9) Deployment Interpretation

Current understanding of deployment:

- open-loop delta policy: training artifact
- frozen delta policy during finetuning: training artifact
- finetuned base policy: deployment policy

This is the interpretation that should guide future debugging, documentation, and evaluation.
