# HumSim2Real: G1 tracking and transferable delta dynamics

This project extends BeyondMimic motion tracking to study whether delta dynamics models learned from one humanoid motion can transfer to other, riskier motions. Isaac Lab is the source/training simulator; Genesis provides target dynamics and recorded state/action trajectories.

## Start here

- [Documentation index](docs/README.md): all workflows and references.
- [Purpose and research workflow](docs/project_understanding.md): what is learned, evaluated and deployed.
- [Setup and repository map](docs/setup_and_repository.md): environments and main entrypoints.
- [Training and evaluation commands](docs/training_and_evaluation_commands.md): executable recipes.

## Workflow

1. Prepare G1 reference motions and train base tracking policies, or use pretrained SONIC.
2. Batch-evaluate motions; rank safety and difficulty and compare executed behavior relevance.
3. Export selected segment policies and record their target-simulator states/actions in Genesis.
4. Train joint-delta or pelvis-wrench models to reproduce target trajectories in Isaac.
5. Test generalization on held-out recordings through matched replay windows and a zero-delta baseline.
6. Optionally finetune a base policy with the delta frozen, then deploy the base policy alone.

The delta helper emulates a dynamics gap during source training/evaluation. In the intended deployment workflow, it is not part of the deployed target controller. Replay error and full-clip tracking success answer different questions.

## Guides by task

- [Motion datasets and compositions](docs/datasets.md)
- [Batch evaluation, safety and difficulty ranking](docs/batch_evaluation_and_ranking.md)
- [Motion relevance](docs/motion_relevance.md)
- [Segment demo videos and ONNX export](docs/segment_artifacts.md)
- [Genesis evaluation and recording](docs/genesis_workflow.md)
- [Delta model inputs and outputs](docs/model_interfaces.md)
- [Pelvis-wrench dynamics emulation](docs/pelvis_wrench_delta.md)
- [Single-checkpoint replay evaluation, aggregation and plots](docs/delta_replay_evaluation.md)

## Installation and upstream provenance

The tracking stack derives from [BeyondMimic](https://beyondmimic.github.io/) ([paper](https://arxiv.org/abs/2508.08241), [upstream repository](https://github.com/HybridRobotics/whole_body_tracking)). This workspace contains an IsaacLab checkout alongside `whole_body_tracking/`. The upstream baseline uses Isaac Sim 4.5, Isaac Lab 2.1, Python 3.10 and Linux.

Configure an Isaac Lab Python environment following its [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). From `whole_body_tracking/`, install the extension with that interpreter:

```bash
python -m pip install -e source/whole_body_tracking
```

The G1 description assets belong under `source/whole_body_tracking/whole_body_tracking/assets/`. The upstream asset archive is available at [Unitree descriptions](https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz). Keep its directory layout when extracting. Genesis and VLM captioning use separate dependency environments; see [setup](docs/setup_and_repository.md).

References include the [retargeted LAFAN1 dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset). Respect dataset, robot asset and model licenses. SONIC setup stores downloaded license files beside its artifacts; attribution is in [SONIC NOTICE](source/whole_body_tracking/whole_body_tracking/sonic/NOTICE.md).

Local models, datasets and experimental outputs are generally ignored by Git. Retain saved configs/manifests with results; source defaults can differ from settings used in an existing run. Historical implementation notes are preserved in the [documentation archive](docs/archive/README.md).
