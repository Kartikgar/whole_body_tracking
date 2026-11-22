"""Live keyboard teleop of Sprout in MuJoCo with Fauna's SDK example walk policy (a virtual joystick).

    # genesis env (mujoco + glfw + onnxruntime); needs a display
    python scripts/sprout/sprout_keyboard_teleop.py

Hold keys like a stick; the command eases toward the held value (as a real stick would) and back to zero on release.
  W / S  or  Up / Down      forward / backward        (axis 0, +1 = forward, ~0.8 m/s)
  A / D                     step left / right         (axis 1, +1 = left)
  Q / E  or  Left / Right   turn left / right         (axis 2, +1 = turn left / CCW, ~75 deg/s)
  Shift                     half speed
  Space                     zero the command         R  reset robot       P  pause       Esc  quit
  Mouse: left-drag rotate, right-drag move up/down, scroll zoom. The camera follows the torso.

The policy runs at --control_hz (50 Hz default) on physics at 2 ms, synced to wall-clock time. See
walk_sdk_policy_mujoco.py for the observation wiring (hardware joint order, imu_site frame, policy kp/kd).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from walk_sdk_policy_mujoco import SdkPolicy, quat_mul, spec  # noqa: E402

# key -> (axis, sign)
KEYMAP = {
    glfw.KEY_W: (0, +1), glfw.KEY_UP: (0, +1), glfw.KEY_S: (0, -1), glfw.KEY_DOWN: (0, -1),
    glfw.KEY_A: (1, +1), glfw.KEY_D: (1, -1),
    glfw.KEY_Q: (2, +1), glfw.KEY_LEFT: (2, +1), glfw.KEY_E: (2, -1), glfw.KEY_RIGHT: (2, -1),
}  # fmt: skip


class SproutSim:
    def __init__(self, physics_dt: float, control_hz: float):
        self.m = mujoco.MjModel.from_xml_path(str(spec.SPROUT_SCENE_PATH))
        self.m.opt.timestep = physics_dt
        self.d = mujoco.MjData(self.m)
        self.decim = int(round(1.0 / (control_hz * physics_dt)))
        hw = list(spec.HARDWARE_JOINT_ORDER)
        self.qadr = np.array([self.m.joint(n).qposadr[0] for n in hw])
        self.dadr = np.array([self.m.joint(n).dofadr[0] for n in hw])
        self.act = np.array([self.m.actuator(n).id for n in hw])
        self.site_q = self.m.site_quat[self.m.site("imu_site").id].copy()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.site_q)
        self.R_site = R.reshape(3, 3)
        self.stand = np.array([spec.DEFAULT_STAND_POSE[n] for n in hw])
        self.policy = SdkPolicy()
        self.est = np.zeros(5)
        self.reset()

    def reset(self):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = (0.0, 0.0, spec.STAND_ROOT_HEIGHT)
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        d.qpos[self.qadr] = self.stand
        d.ctrl[self.act] = self.stand
        mujoco.mj_forward(m, d)
        self.policy.reset()
        self.k = 0

    def step(self, axes: np.ndarray):
        m, d = self.m, self.d
        if self.k % self.decim == 0:
            imu_q = quat_mul(d.qpos[3:7], self.site_q)
            gyro = self.R_site.T @ d.qvel[3:6]  # free-joint angular velocity is in the torso frame
            tgt, kp, kd, self.est = self.policy(d.qpos[self.qadr], d.qvel[self.dadr], imu_q, gyro, axes)
            d.ctrl[self.act] = tgt
            m.actuator_gainprm[self.act, 0] = kp
            m.actuator_biasprm[self.act, 1] = -kp
            m.actuator_biasprm[self.act, 2] = -kd
        mujoco.mj_step(m, d)
        self.k += 1

    def tilt_deg(self) -> float:
        w, x, y, z = self.d.qpos[3:7]
        return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))


class Teleop:
    def __init__(self, a):
        self.a = a
        self.sim = SproutSim(a.physics_dt, a.control_hz)
        self.held: set[int] = set()
        self.shift = False
        self.axes = np.zeros(3, np.float32)
        self.paused = False
        self.mouse = dict(left=False, right=False, x=0.0, y=0.0)

        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 4)
        self.win = glfw.create_window(a.width, a.height, "Sprout keyboard teleop (Fauna SDK walk policy)", None, None)
        glfw.make_context_current(self.win)
        glfw.swap_interval(1)
        m = self.sim.m
        self.scn = mujoco.MjvScene(m, maxgeom=10000)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = m.body(spec.ROOT_BODY).id
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.4, 135.0, -15.0
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[1] = 0  # hide collision geoms
        self.ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_mouse_button_callback(self.win, self.on_button)
        glfw.set_cursor_pos_callback(self.win, self.on_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)

    # --- input -------------------------------------------------------------------------------------------------------
    def on_key(self, win, key, scancode, action, mods):
        if key in (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT):
            self.shift = action != glfw.RELEASE
            return
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_SPACE:
                self.held.clear()
                self.axes[:] = 0
            elif key == glfw.KEY_R:
                self.sim.reset()
                self.axes[:] = 0
            elif key == glfw.KEY_P:
                self.paused = not self.paused
            elif key in KEYMAP:
                self.held.add(key)
        elif action == glfw.RELEASE:
            self.held.discard(key)

    def on_button(self, win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse["left"] = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_RIGHT:
            self.mouse["right"] = action == glfw.PRESS
        self.mouse["x"], self.mouse["y"] = glfw.get_cursor_pos(win)

    def on_move(self, win, x, y):
        dx, dy = x - self.mouse["x"], y - self.mouse["y"]
        self.mouse["x"], self.mouse["y"] = x, y
        if not (self.mouse["left"] or self.mouse["right"]):
            return
        _, h = glfw.get_window_size(win)
        act = mujoco.mjtMouse.mjMOUSE_ROTATE_V if self.mouse["left"] else mujoco.mjtMouse.mjMOUSE_MOVE_V
        mujoco.mjv_moveCamera(self.sim.m, act, dx / h, dy / h, self.scn, self.cam)

    def on_scroll(self, win, xoff, yoff):
        mujoco.mjv_moveCamera(self.sim.m, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoff, self.scn, self.cam)

    def target_axes(self) -> np.ndarray:
        t = np.zeros(3, np.float32)
        for key in self.held:
            ax, s = KEYMAP[key]
            t[ax] += s
        t = np.clip(t, -1, 1) * (0.5 if self.shift else 1.0) * np.array(self.a.max_axes, np.float32)
        return t

    # --- loop --------------------------------------------------------------------------------------------------------
    def run(self):
        sim, a = self.sim, self.a
        dt = sim.m.opt.timestep
        wall0, sim_t = time.perf_counter(), 0.0
        last = time.perf_counter()
        while not glfw.window_should_close(self.win):
            now = time.perf_counter()
            frame_dt = now - last
            last = now
            # ease toward the held command at a.ramp per second (stick-like)
            tgt = self.target_axes()
            step = a.ramp * frame_dt
            self.axes += np.clip(tgt - self.axes, -step, step)
            if self.paused:
                wall0 = now - sim_t
            else:
                # advance physics to wall clock (cap the catch-up so a stall doesn't run away)
                target_t = min(now - wall0, sim_t + 0.1)
                while sim_t + dt <= target_t:
                    sim.step(self.axes)
                    sim_t += dt
                wall0 = now - sim_t
            self.render(sim_t)
            glfw.poll_events()
        glfw.terminate()

    def render(self, sim_t: float):
        sim = self.sim
        w, h = glfw.get_framebuffer_size(self.win)
        vp = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(sim.m, sim.d, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        mujoco.mjr_render(vp, self.scn, self.ctx)
        a0, a1, a2 = self.axes
        est = sim.est
        tilt = sim.tilt_deg()
        status = "PAUSED" if self.paused else ("FALLEN - press R" if tilt > 60 else "running")
        left = "\n".join(["status", "time", "forward (a0)", "left (a1)", "turn left (a2)", "policy est v",
                          "height / tilt"])
        right = "\n".join([status, f"{sim_t:6.1f} s", f"{a0:+.2f}", f"{a1:+.2f}", f"{a2:+.2f}",
                           f"vx {est[0]:+.2f}  vy {est[1]:+.2f} m/s", f"{sim.d.qpos[2]:.3f} m / {tilt:.0f} deg"])
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, vp, left, right, self.ctx)
        keys = "W/S fwd-back   A/D step   Q/E turn   Shift half   Space stop   R reset   P pause   Esc quit"
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, vp, keys, None, self.ctx)
        glfw.swap_buffers(self.win)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control_hz", type=float, default=50.0)
    ap.add_argument("--physics_dt", type=float, default=0.002)
    ap.add_argument("--ramp", type=float, default=3.0, help="command slew rate (axis units per second)")
    ap.add_argument("--max_axes", type=float, nargs=3, default=(1.0, 1.0, 0.6),
                    help="full-key value per axis (fwd, left, turn); turn 0.6 ~ 45 deg/s")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    Teleop(ap.parse_args()).run()


if __name__ == "__main__":
    main()
