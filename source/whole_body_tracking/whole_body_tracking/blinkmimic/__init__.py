"""BlinkMimic (world-model BPTT motion tracking) ported to Isaac Lab.

Vendored from https://github.com/fuyichen0725/blinkmimic (MIT, see LICENSE_blinkmimic), commit d264658, with the
local robot extensions (feet_bodies / contact_z / mirror_sign). Simulator-independent modules are copied unchanged
(rotations, motion, mdp, kinematics, networks, buffer, objective, mirror, sensing). Changed for Isaac Lab:

- robot.py: RobotSpec without mjlab (``robot_cfg`` returns an Isaac Lab ArticulationCfg).
- dr.py: DomainRandCfg and XiLayout only; the Isaac domain randomizer lives in env_isaac.py.
- optim.py: a local Muon (same arithmetic as torch 2.9's torch.optim.Muon), since Isaac Sim pins torch 2.5.
- env_isaac.py: TrackingSim on an Isaac Lab articulation (replaces env.py on mjlab / MuJoCo Warp).
- sprout.py: the Sprout RobotSpec for Isaac (replaces sprout/sprout_constants.py).

Import only after the Isaac Sim app is launched (env_isaac and sprout import isaaclab).
"""
