# SONIC integration provenance

SONIC model: NVIDIA GEAR-SONIC default release, Hugging Face revision
`6733128a3d8a523b1418b06bca3cdf61c8b0987f`.

Observation and control conventions follow NVIDIA GR00T-WholeBodyControl,
revision `b042411fae38ee4d1af9aac82a37a1f8d14d6dd0`:

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/state_logger.cpp`
- `gear_sonic/envs/manager_env/robots/g1.py`

NVIDIA source is licensed under Apache 2.0. Model weights are licensed under
the NVIDIA Open Model License. The setup script downloads upstream licenses
alongside the artifacts. Robot assets originate from the upstream
robot_description directory; upstream identifies the G1 URDF/config as adapted
from BeyondMimic. No model weights or robot meshes are embedded in this package.
