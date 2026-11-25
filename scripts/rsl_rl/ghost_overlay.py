"""Translucent "ghost" robots that replay logged target-simulator (e.g. Genesis) rollouts inside a live Isaac batch eval.

``add_ghost_to_scene`` adds a second G1 articulation per env (no collisions, no gravity, tinted translucent material).
``make_ghost_writer`` returns a callback that, once per control step and before the frame is rendered, writes ALL ghosts'
root poses and joint positions in one batched call from a ghost NPZ (see ``scripts/rsl_rl/make_ghost_npz.py``):
    root_pos (N, T, 3) [env-origin frame], root_quat (N, T, 4) wxyz, joint_pos (N, T, J), joint_names (J,), valid_lengths (N,)
Frame t is the state after t control steps (frame 0 = start state), so after Isaac step s the ghost shows frame s + 1; a
terminated ghost holds its last frame. Ghost env i replays NPZ row i, so the NPZ must come from the same start states.
"""
import numpy as np
import torch

import isaaclab.sim as sim_utils

GHOST_MATERIAL = "/World/Looks/GhostMaterial"


def _tinted_urdf_spawner(color, opacity):
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_urdf

    def spawn(prim_path, cfg, translation=None, orientation=None):
        import omni.usd
        from pxr import Usd

        prim = spawn_from_urdf(prim_path, cfg, translation=translation, orientation=orientation)
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(GHOST_MATERIAL).IsValid():
            mat = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color), opacity=opacity)
            mat.func(GHOST_MATERIAL, mat)
        # The converted URDF uses instanced meshes, which ignore material overrides: de-instance this subtree first
        # (spawned once for the env_0 template; the cloner copies the result to the other envs).
        while True:
            inst = [p for p in Usd.PrimRange(prim) if p.IsInstance()]
            if not inst:
                break
            for p in inst:
                p.SetInstanceable(False)
        sim_utils.bind_visual_material(prim.GetPath(), GHOST_MATERIAL, stage=stage)
        # Collision settings applied at spawn cannot reach instanced meshes; disable every collider of the now
        # de-instanced ghost explicitly, or the ghosts push the Isaac robots around.
        from pxr import UsdPhysics
        n_off = 0
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
                n_off += 1
        print(f"[ghost] {prim.GetPath()}: material bound, {n_off} colliders disabled", flush=True)
        return prim

    return spawn


def add_ghost_to_scene(env_cfg, opacity=0.45, color=(0.25, 0.55, 1.0)):
    import carb

    carb.settings.get_settings().set("/rtx/translucency/enabled", True)  # off by default: opacity < 1 renders invisible
    robot = env_cfg.scene.robot
    spawn = robot.spawn.replace(
        func=_tinted_urdf_spawner(color, opacity),
        activate_contact_sensors=False,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        rigid_props=robot.spawn.rigid_props.replace(disable_gravity=True),
        articulation_props=robot.spawn.articulation_props.replace(enabled_self_collisions=False),
    )
    env_cfg.scene.ghost = robot.replace(prim_path="{ENV_REGEX_NS}/Ghost", spawn=spawn)


def make_ghost_writer(base_env, ghost_npz):
    ghost = base_env.scene["ghost"]
    dev = ghost.device
    d = np.load(ghost_npz)
    names = [str(n) for n in d["joint_names"]]
    jmap = [names.index(j) for j in ghost.data.joint_names]
    n_env = base_env.num_envs
    if d["root_pos"].shape[0] < n_env:
        raise ValueError(f"ghost NPZ has {d['root_pos'].shape[0]} rollouts, scene has {n_env} envs")
    root_pos = torch.tensor(d["root_pos"][:n_env], device=dev)
    root_quat = torch.tensor(d["root_quat"][:n_env], device=dev)
    joint_pos = torch.tensor(d["joint_pos"][:n_env][..., jmap], device=dev)
    last = torch.tensor(np.minimum(d["valid_lengths"][:n_env], root_pos.shape[1]) - 1, device=dev, dtype=torch.long)
    rows = torch.arange(n_env, device=dev)
    origins = base_env.scene.env_origins
    zeros_v = torch.zeros(n_env, 6, device=dev)
    zeros_j = torch.zeros(n_env, joint_pos.shape[-1], device=dev)

    @torch.inference_mode()  # the articulation's data buffers are inference tensors (created during env steps)
    def write(frame):
        k = torch.minimum(torch.full_like(last, frame), last)
        ghost.write_root_pose_to_sim(torch.cat([root_pos[rows, k] + origins, root_quat[rows, k]], -1))
        ghost.write_root_velocity_to_sim(zeros_v)
        ghost.write_joint_state_to_sim(joint_pos[rows, k], zeros_j)

    write(0)
    return lambda step: write(step + 1)


ROBOT_MATERIAL = "/World/Looks/TranslucentRobotMaterial"


def make_robot_translucent(env_cfg, opacity=0.35, color=(0.92, 0.92, 0.92)):
    """Render the simulated robot translucent (e.g. so wrench arrows inside the body stay visible).

    Only the instanced ``visuals`` prims are de-instanced and re-coloured; ``collisions`` prims stay instanced and
    untouched, so contact geometry and physics are unchanged.
    """
    import carb

    from isaaclab.sim.spawners.from_files.from_files import spawn_from_urdf

    carb.settings.get_settings().set("/rtx/translucency/enabled", True)

    def spawn(prim_path, cfg, translation=None, orientation=None):
        import omni.usd
        from pxr import Usd

        prim = spawn_from_urdf(prim_path, cfg, translation=translation, orientation=orientation)
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(ROBOT_MATERIAL).IsValid():
            mat = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color), opacity=opacity)
            mat.func(ROBOT_MATERIAL, mat)
        visuals = [p for p in Usd.PrimRange(prim) if p.GetName() == "visuals"]
        for v in visuals:
            while True:
                inst = [p for p in Usd.PrimRange(v) if p.IsInstance()]
                if not inst:
                    break
                for p in inst:
                    p.SetInstanceable(False)
            sim_utils.bind_visual_material(v.GetPath(), ROBOT_MATERIAL, stage=stage)
        print(f"[robot] {prim.GetPath()}: {len(visuals)} visual subtrees made translucent (opacity {opacity})", flush=True)
        return prim

    env_cfg.scene.robot = env_cfg.scene.robot.replace(spawn=env_cfg.scene.robot.spawn.replace(func=spawn))
