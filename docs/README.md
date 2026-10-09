# Project documentation

This project studies G1 humanoid motion tracking and whether learned residual dynamics models transfer between motions. Isaac Lab is the training/source simulator; Genesis is the principal target simulator used to collect state/action trajectories and test transfer.

## Start here

1. [Purpose and research workflow](project_understanding.md): what the delta model learns, why transfer matters, and what is deployed.
2. [Setup and repository map](setup_and_repository.md): environments, directories, and script entrypoints.
3. [Training and evaluation commands](training_and_evaluation_commands.md): base tracking, joint/wrench delta training, and optional finetuning.
4. [Model interfaces](model_interfaces.md): observations, actions, reference data and export contracts.

## Choose a workflow

- **Evaluate many motions:** [Batch evaluation, safety and difficulty](batch_evaluation_and_ranking.md).
- **Find similar executed behaviors:** [Motion relevance](motion_relevance.md).
- **Export a selected segment:** [Segment videos and ONNX artifacts](segment_artifacts.md).
- **Prepare and combine data:** [Motion and trajectory datasets](datasets.md).
- **Record target dynamics:** [Genesis simulation and recording](genesis_workflow.md); [physical-property overrides](genesis_experiments.md).
- **Test a supplied delta checkpoint:** [Delta replay evaluation](delta_replay_evaluation.md), including independent runs, aggregation and Matplotlib plots.
- **Understand the wrench model:** [Pelvis-wrench dynamics emulation](pelvis_wrench_delta.md).
- **Understand resets and sampling:** [Episode length and motion sampling](episode_length_resets_motion_sampling.md).
- **Read VLM descriptions:** [Motion captioning](motion_captioning.md).

## Detailed visual references

These HTML pages contain additional diagrams and implementation explanations. Current CLI commands and contracts are maintained in the guides above.

- [Playback and evaluation workflows](play_evaluation_workflows.html)
- [Genesis evaluator](genesis_sim2sim_evaluator.html)
- [Tracking metrics](tracking_metrics.html)
- [Motion alignment transform](motion_alignment_transform.html)

## Historical implementation notes

[Archive index](archive/README.md) preserves earlier design notes and command recipes. It is background material, not the source of current defaults.

## Reading conventions

Commands run from `whole_body_tracking/` unless stated otherwise. `path/to/...` and `<run>` are placeholders. Batch YAML paths are relative to the YAML file. Saved `params/env.pkl`, `params/agent.pkl`, manifests and resolved run configs describe an existing experiment; source defaults can change without changing old results. Local recordings, models and results generally live in ignored `data/`, `artifacts/` and `logs/` directories.
