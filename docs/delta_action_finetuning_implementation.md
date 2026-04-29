# Delta-Action Finetuning Implementation (Detailed)

This document explains the **current implementation** of Delta-Action finetuning in this repository, with focus on:

1. How the frozen delta policy checkpoint is loaded.
2. How observations/inputs are prepared for both policies (base finetuned policy and frozen delta policy).
3. How both policies are combined each environment step.


## 1) High-Level Architecture

Delta-Action finetuning in this repo uses two policies at runtime:

1. **Base policy** (trainable, updated by PPO during finetuning).
2. **Delta policy** (frozen, loaded from an open-loop checkpoint).

At each step, the environment receives one of two compositions:

`combined_action = base_policy_action + external_delta_scale * frozen_delta_action`

Then the normal joint action post-processing applies:

`processed_action = combined_action * action_scale + action_offset`

The composition is implemented by `ExternalDeltaJointPositionAction`:

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/delta_actions.py`

When `--delta_action_space com_force` is used at the train/play launcher, a finetune-specific variant is used instead:

1. `base_policy_action` stays in joint space.
2. `frozen_delta_action` becomes a 3D COM force action.
3. The action term applies the processed base joint target and the processed COM force in the same step.

That variant is implemented by `ExternalDeltaComForceAction` in the same file.


## 2) Where Delta Finetuning Is Registered

Task registration:

- `Tracking-Flat-G1-DeltaA-Finetune-v0` is registered in  
  `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/__init__.py`

It points to:

1. Env config: `G1FlatDeltaAFineTuneEnvCfg`
2. Runner config: `G1FlatDeltaAFineTunePPORunnerCfg`

Relevant files:

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`


## 3) Finetune Config: What Changes vs Base Tracking

`G1FlatDeltaAFineTuneEnvCfg` modifies the base G1 tracking env in these key ways:

1. Action term is replaced by `ExternalDeltaJointPositionActionCfg`:
   - `external_action_buffer_name = "delta_external_actions"`
   - `external_action_scale = 1.0`
   - `require_external_action = True`

2. Adds extra observation group for frozen policy:
   - `self.observations.delta_policy = DeltaPolicyObsCfg()`

3. Reward adjustments:
   - `penalty_minimal_action_norm = None`
   - `motion_body_pos_global = None`
   - `motion_body_ori_global = None`

4. Episode length is explicitly kept at `10.0` seconds.

Runner-side delta settings (`G1FlatDeltaAFineTunePPORunnerCfg`):

1. `delta_policy_checkpoint: str | None = None`
2. `delta_policy_obs_group = "delta_policy"`
3. `delta_policy_critic_obs_group = "critic"`
4. `delta_policy_action_buffer_name = "delta_external_actions"`
5. `delta_policy_require = True`
6. `delta_policy_clip_actions = None`


## 4) Inputs to Both Policies: Exact Observation Groups

During finetuning, there are three relevant observation groups in `extras["observations"]`:

1. `policy` (base actor input)
2. `critic` (base critic input)
3. `delta_policy` (frozen delta actor input)

### 4.1 Base Policy Actor Input (`policy`)

Inherited from `TrackingEnvCfg.ObservationsCfg.PolicyCfg`:

1. `command`
2. `motion_anchor_pos_b`
3. `motion_anchor_ori_b`
4. `base_lin_vel`
5. `base_ang_vel`
6. `joint_pos`
7. `joint_vel`
8. `actions` (`mdp.last_action`)

This group has corruption enabled (`enable_corruption = True`) in base config.

### 4.2 Base Policy Critic Input (`critic`)

Inherited from `TrackingEnvCfg.ObservationsCfg.PrivilegedCfg`:

1. `command`
2. `motion_anchor_pos_b`
3. `motion_anchor_ori_b`
4. `body_pos`
5. `body_ori`
6. `base_lin_vel`
7. `base_ang_vel`
8. `joint_pos`
9. `joint_vel`
10. `actions`

### 4.3 Frozen Delta Actor Input (`delta_policy`)

Defined in `DeltaPolicyObsCfg` to match open-loop delta policy actor layout/scales:

1. `base_pos_z` (scale 1.0)
2. `feet_contact_force` (scale 0.01)
3. `base_lin_vel` (scale 2.0)
4. `base_ang_vel` (scale 0.25)
5. `projected_gravity` (scale 1.0)
6. `joint_pos` (scale 1.0)
7. `joint_vel` (scale 0.05)
8. `actions` = `mdp.external_delta_action("delta_external_actions")` (scale 1.0)
9. `current_action` = `mdp.current_action("delta_base_actions")` (scale 1.0)

Important behavior:

1. `actions` is the delta-policy autoregressive channel from the external delta buffer.
2. `current_action` carries the same-step action from the policy being finetuned, injected by runner into `delta_base_actions`.

Relevant functions:

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`

### 4.4 Normalization and Tensor Path to Both Policies

At runtime, `env.get_observations()` returns:

1. `obs`: flattened tensor for the active actor observation group used by PPO (`policy` group).
2. `extras["observations"]`: dictionary containing all groups (`policy`, `critic`, `delta_policy`, ...).

In `MotionOnPolicyRunner.learn(...)`:

1. `privileged_obs` is taken from `extras["observations"][self.privileged_obs_type]` when available, otherwise from `obs`.
2. Base PPO actor/critic inputs are normalized through `self.obs_normalizer` and `self.privileged_obs_normalizer` (same behavior as standard RSL-RL flow).
3. Frozen delta policy input is normalized only by `self.delta_obs_normalizer` loaded from the delta checkpoint metadata/config (or identity if unavailable).


## 5) How the Frozen Delta Policy Is Loaded

The loading logic lives in `MotionOnPolicyRunner._load_delta_policy(...)`:

- `source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py`

Detailed flow:

1. Resolve checkpoint path (`Path(...).expanduser().resolve()`), require file exists.
2. Find nearby `agent.yaml` (tries multiple candidate paths around checkpoint dir).
3. Parse `agent.yaml` and extract `policy` config (`class_name`, hidden dims, etc.).
4. Load checkpoint dict via `torch.load(...)`.
5. Read `model_state_dict`.
6. Infer expected actor/critic input dims from checkpoint weights:
   - Looks for `actor.0.weight` / `critic.0.weight`.
   - Falls back to first matching linear weight under each prefix.
7. Get live env observation dictionary (`obs, extras = env.get_observations()`).
8. Validate/select actor obs group:
   - Starts with configured `delta_policy_obs_group`.
   - If dim mismatch and exactly one obs group matches checkpoint actor dim, auto-switches to that group.
   - Otherwise raises an explicit mismatch error with available group dims.
9. Determine critic input dim:
   - Prefers checkpoint critic dim (for strict state_dict compatibility).
   - If env critic dim differs, logs info and still uses checkpoint dim.
10. Instantiate policy class from `class_name` with:
    - `num_actor_obs`
    - `num_critic_obs`
    - `num_actions = env.num_actions`
11. `load_state_dict(..., strict=True)`.
12. Set `eval()` and `requires_grad=False` for all delta-policy params.
13. Build delta obs normalizer:
    - If `empirical_normalization` in delta `agent.yaml` is true and checkpoint contains `obs_norm_state_dict`, load `EmpiricalNormalization`.
    - Else use identity normalizer.


## 6) Rollout Step: Exact Order During Finetuning Training

Training loop is in `MotionOnPolicyRunner.learn(...)`.

For each rollout step:

1. Compute base policy action first:
   - `actions = self.alg.act(obs, privileged_obs)`

2. Inject same-step base action for delta-policy `current_action` term:
   - `setattr(env.unwrapped, delta_policy_base_action_buffer_name, actions)`

3. Recompute only the delta-policy observation group from current env state:
   - `delta_obs = observation_manager.compute_group(delta_policy_obs_group)`

4. Compute frozen delta action:
   - `delta_actions = delta_policy.act_inference(delta_obs_normalizer(delta_obs))`
   - Optional clamp if `delta_policy_clip_actions` is set.

5. Write delta action buffer onto env object:
   - `setattr(env.unwrapped, delta_policy_action_buffer_name, delta_actions)`

6. Step environment with base action:
   - `obs, rewards, dones, infos = env.step(actions)`

7. Environment action term composes final command:
   - `combined = base_raw + external_scale * external_delta`
   - `processed = combined * scale + offset`

8. Continue PPO bookkeeping (`process_env_step`, `compute_returns`, `update`).

At the end of training:

1. Clear delta buffer attribute from env (`_clear_delta_action_buffer()`).


## 7) Time Alignment Detail (Important)

`DeltaPolicyObsCfg` uses:

1. `actions` from `external_delta_action` (delta channel).
2. `current_action` from `current_action("delta_base_actions")` (base-policy channel).

So the recurrence is:

1. Step `t` delta-policy input contains:
   - previous delta action (via external delta buffer), and
   - same-step base-policy action (via `delta_base_actions` set by runner before delta inference).
2. Runner computes frozen delta action for step `t`.
3. Runner writes frozen delta action to external delta buffer.
4. Environment composes base action and frozen delta action at step `t`.
5. Next step repeats with a newly rolled-out base action.

This keeps an autoregressive policy-action channel while still using external delta buffer only for control composition.

In COM-force finetune mode, the same observation/buffer flow is reused, but the external delta buffer holds `[Fx, Fy, Fz]` instead of joint deltas.


## 8) Training Entry-Point Wiring

`scripts/rsl_rl/train.py` uses:

1. `MotionOnPolicyRunner` (not vanilla `OnPolicyRunner`)
2. CLI arg `--delta_policy_checkpoint` propagated via `scripts/rsl_rl/cli_args.py`

Resume behavior in current script:

1. If `--resume True`, checkpoint is loaded from:
   - `--checkpoint /abs/path/to/model.pt`, or
   - `--wandb_path entity/project/run[/model_x.pt]`


## 9) Play-Time (Inference) Wiring

`scripts/rsl_rl/play.py` now also uses `MotionOnPolicyRunner`.

In the play loop:

1. Roll base policy action.
2. Inject same-step base action via `ppo_runner._set_delta_base_action_buffer(...)`.
3. Recompute delta-policy group and infer frozen delta action.
4. Write delta buffer via `ppo_runner._set_delta_action_buffer(...)`.
5. Step env with base policy action.
6. Clear buffers at end.

This is required for Delta-A finetune checkpoints because env action processing expects external delta actions.


## 10) Failure Modes and What They Mean

### 10.1 Actor/critic size mismatch when loading delta checkpoint

Symptom:

`RuntimeError: size mismatch for actor.0.weight` or `critic.0.weight`.

Meaning:

1. Delta checkpoint architecture/input dims do not match the currently selected obs groups.
2. Current loader auto-handles many cases:
   - actor group auto-switch by matching checkpoint dim.
   - critic dim taken from checkpoint if needed.

If still failing, the checkpoint likely comes from a different robot/task or different action dimensionality.

### 10.2 Missing delta buffer during action processing

Symptom:

`Expected external delta action buffer 'delta_external_actions' on the env.`

Meaning:

1. `require_external_action=True` but runner did not set the buffer.
2. Use `MotionOnPolicyRunner` path (train/play scripts in this repo now do).

### 10.3 Motion file missing `action`/`actions`

Symptom:

`motion_joint_action was requested` runtime error.

Meaning:

1. This is relevant to configs that request `mdp.motion_joint_action` from dataset.
2. Finetune `DeltaPolicyObsCfg` uses `current_action` from `delta_base_actions`, so this specific requirement is removed there.


## 11) Minimal Mental Model

At each step:

1. Frozen delta policy predicts a correction from `delta_policy` obs.
2. Base finetuned policy predicts its own action from standard `policy` obs.
3. Joint-delta finetune mode:
   - env action term adds both actions before applying standard scaling/offset.
4. COM-force finetune mode:
   - env action term applies the base joint target normally and applies the frozen delta output as a COM force.
5. PPO updates only the base policy; frozen delta policy stays fixed.


## 12) Key Files (Quick Index)

1. Runner implementation:
   - `source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py`
2. Finetune env and obs/action configs:
   - `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
3. Finetune runner cfg fields:
   - `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
4. Action composition terms:
   - `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/delta_actions.py`
5. Observation helper terms:
   - `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
6. Motion command joint-action source:
   - `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
7. Train/play entry points:
   - `scripts/rsl_rl/train.py`
   - `scripts/rsl_rl/play.py`
