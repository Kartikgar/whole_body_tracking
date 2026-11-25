# Purpose and research workflow

[Documentation index](README.md)

## What this project investigates

HumSim2Real extends BeyondMimic whole-body tracking for the Unitree G1. The research objective is to learn corrections to source-simulator dynamics and understand how well those corrections generalize across humanoid motions, especially from safer motions to riskier ones.

The repository supports two tracking behaviors: pretrained NVIDIA SONIC and motion-specific BeyondMimic checkpoints. Their rollouts support difficulty/safety ranking, executed-behavior relevance analysis, and selecting motion pairs for transfer experiments. Genesis provides a target simulation environment in which base policies can be evaluated and their state/action trajectories recorded.

## Target: iterative, failure-free sim-to-real through a motion curriculum

On hardware, every fall during data collection is costly. The goal is to learn the sim-to-real gap from data gathered
**without failures**. In the sim-to-sim study Genesis plays the real robot, and a Genesis fall counts as a hardware failure.
The approach is a curriculum over motions m1 → m2 → … → m5, ordered from safe to risky:

1. Collect data safely on m_k.
2. Train or update the delta.
3. Finetune the base policy for m_{k+1} in Isaac + delta.
4. Deploy on m_{k+1} without falls, then repeat with k + 1.

Deployment inside a motion is iterative too. Segments of m_{k+1} differ in difficulty, so they are deployed in order of
estimated safety (difficulty/safety metrics, delta uncertainty). Data from the segments deployed so far updates the delta.
The policy is then re-finetuned before the next, harder segments.

**Evidence so far.** A delta trained on a sequence of easier motions (m1_m2_m3) transfers to harder ones after finetuning:
m4 Fast Walking and m6 Running. That covers one jump of the curriculum. Still open:
- each single step (m1 → m2, m2 → m3, m3 → m4);
- the full chain;
- the segment-level deployment loop.

The figure of merit is the number of failures incurred along the way as well as final tracking.

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
