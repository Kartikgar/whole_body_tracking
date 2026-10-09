# Purpose and research workflow

[Documentation index](README.md)

## What this project investigates

HumSim2Real extends BeyondMimic whole-body tracking for the Unitree G1. The research objective is to learn corrections to source-simulator dynamics and understand how well those corrections generalize across humanoid motions, especially from safer motions to riskier ones.

The repository supports two tracking behaviors: pretrained NVIDIA SONIC and motion-specific BeyondMimic checkpoints. Their rollouts support difficulty/safety ranking, executed-behavior relevance analysis, and selecting motion pairs for transfer experiments. Genesis provides a target simulation environment in which base policies can be evaluated and their state/action trajectories recorded.

## The role of a delta model

A delta model observes the current robot state and base joint action. It adds a joint correction or pelvis wrench in Isaac so that the resulting transition resembles the recorded target transition:

`T_Isaac_with_delta(s, a_base) ≈ T_target(s, a_base)`

The target can differ through simulator physics, contact handling, actuators, or explicit physical-property changes. The learned delta is a dynamics-emulation helper. Its current intended deployment role is training-time assistance: optionally finetune a base tracking policy in the delta-assisted source environment, then deploy the base policy alone in the target environment.

## End-to-end workflow

1. **Prepare reference motions.** Retargeted LAFAN1 NPZ clips include joint/body reference states. PgS2R-mini splits parent motions into approximately 20-second segments.
2. **Evaluate tracking behaviors.** Run SONIC or parent-motion checkpoints over clips. Record parallel rollouts, completion, tracking and safety measurements.
3. **Choose transfer candidates.** Rank difficulty and safety separately. Compare executed window features and inspect captions/videos to select behaviorally relevant motion pairs.
4. **Record target state/action data.** Run exported base policies in Genesis. Preserve valid lengths and raw joint actions, including early failures.
5. **Train delta models.** Replay recorded joint actions in Isaac while learning corrections that reproduce target states. Combine trajectory datasets to study how target-motion examples affect generalization.
6. **Measure generalization before finetuning.** Evaluate one delta checkpoint on a supplied validation dataset from multiple initial states over fixed replay windows. Compare with zero delta under identical settings.
7. **Optionally finetune and deploy.** Freeze the learned delta, train the base policy, export the base actor, and measure target-simulator performance.

Current transfer experiments use jump-only, walk-only, jump/dance and walk/dance compositions, plus a dance-trained reference model. These are experiment choices; the evaluator accepts arbitrary compatible checkpoints and datasets.

## Three evaluations answer different questions

- **Full-clip tracking:** can a closed-loop base policy finish the motion, and how safe is its execution?
- **Delta replay:** given a recorded initial state and fixed recorded joint-action sequence, how accurately does the corrected source reproduce the target states?
- **Finetuned deployment:** does a base policy trained with delta assistance improve closed-loop target tracking without the helper?

Completion of a replay window means valid numerical execution. It does not imply a fall-free or successful tracking rollout. Low replay error supports dynamics matching on the tested distribution; it does not by itself establish deployment transfer.

## Where to go next

Use [training commands](training_and_evaluation_commands.md) for execution, [batch ranking](batch_evaluation_and_ranking.md) for motion selection, and [delta replay](delta_replay_evaluation.md) for controlled model comparisons. Consult [model interfaces](model_interfaces.md) before changing observations or action representations.
