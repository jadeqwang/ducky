#!/usr/bin/env python3
"""Render the Chicken Banana dance in CPU MuJoCo, for tuning choreography.

The choreography itself lives in ``scripts/dance_sequencer.py`` and is IMPORTED
here, not copied — the same function drives this render and the real robot, so
what you tune is what ships. Edit ``choreography()`` there, re-render here.

No trained policy: this drives the stock ``policies/alpha_stand.onnx``, which
holds balance while the head and body pose commands do the dancing. See
``docs/chicken_banana.md``.

    MUJOCO_GL=glfw uv run python video/render_chicken_banana.py

EGL and osmesa are both broken on the dev box; glfw works when a display is
present. Rendering is cheap — the cost is env setup, so prefer one long clip
over several short ones.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

from bam.model import load_model  # noqa: E402
from dance_sequencer import choreography  # noqa: E402  — single source of truth
from infer_policy import MICRODUCK_XML, PolicyInference  # noqa: E402

DECIMATION = 4
CURRENT_LIMIT_A = 1.75  # infer_policy.py default; XL330 firmware saturation

JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)


def build():
    """Model + data + policy, matching infer_policy's deployment rehearsal setup."""
    model = mujoco.MjModel.from_xml_path(MICRODUCK_XML)
    # infer_policy.py sets this inside main(); without it the XML default of
    # 0.002 runs the policy at 125 Hz instead of the trained 50 Hz and it falls.
    model.opt.timestep = 0.005
    kt = load_model(motor_name="xl330", model="m6").kt.value
    torque = kt * CURRENT_LIMIT_A
    model.actuator_forcerange[:, 0] = -torque
    model.actuator_forcerange[:, 1] = torque
    model.actuator_forcelimited[:] = 1

    data = mujoco.MjData(model)
    policy = PolicyInference(
        model, data,
        standing_onnx_path=str(_ROOT / "policies" / "alpha_stand.onnx"),
        new_cmd_obs=True, use_projected_gravity=True, action_scale=1.0,
    )
    policy.current_policy = "standing"
    policy.ort_session = policy.standing_session
    policy.input_name = policy.ort_session.get_inputs()[0].name
    policy.output_name = policy.ort_session.get_outputs()[0].name
    return model, data, policy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bpm", type=float, default=130.0)
    ap.add_argument("--phrases", type=int, default=8,
                    help="two-beat phrases to render (default: 8)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "chicken_banana.mp4"))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--head-alpha", type=float, default=1.0,
                    help="simulate robotd's head command low-pass. The runtime "
                         "default is 0.2, which costs ~40%% of peak angular speed; "
                         "1.0 is no filtering (default here, matching raw choreography)")
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "glfw"

    beat = 60.0 / args.bpm
    model, data, policy = build()
    dt = DECIMATION * model.opt.timestep

    qidx = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in JOINT_NAMES]
    root = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")]

    data.qpos[root + 2] = 0.125
    data.qpos[root + 3:root + 7] = [1, 0, 0, 0]
    for i, qi in enumerate(qidx):
        data.qpos[qi] = policy.default_pose[i]
    data.ctrl[:] = policy.default_pose
    mujoco.mj_forward(model, data)

    policy.head_offset = np.zeros(4, dtype=np.float32)
    policy.body_cmd = np.zeros(6, dtype=np.float32)

    def tick():
        policy._update_command()
        policy.apply_action(policy.infer())
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)

    for _ in range(150):          # let it settle into a stand
        tick()

    renderer = mujoco.Renderer(model, args.height, args.width)
    # Eye level, deliberately: from above, any downward head pose hides the face,
    # and the face is the whole joke. See docs/chicken_banana.md.
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 0.42, -2, 132
    cam.lookat[:] = [0, 0, 0.107]

    head_ema = np.zeros(4, dtype=np.float32)
    frames, zs = [], []
    t = 0.0
    while t < args.phrases * 2 * beat:
        neck, head, yaw, roll, body_pitch = choreography(t, beat)
        target = np.array([neck, head, yaw, roll], dtype=np.float32) * args.scale
        head_ema += args.head_alpha * (target - head_ema)
        policy.head_offset = head_ema.copy()
        # body_cmd is [x, y, z, roll, pitch, yaw]; z stays 0 (see the doc).
        body = np.zeros(6, dtype=np.float32)
        body[4] = body_pitch * args.scale
        policy.body_cmd = body

        tick()
        t += dt
        cam.lookat[0], cam.lookat[1] = data.qpos[root + 0], data.qpos[root + 1]
        renderer.update_scene(data, cam)
        frames.append(renderer.render())
        zs.append(data.qpos[root + 2])

    imageio.mimsave(args.out, frames, fps=int(round(1 / dt)), quality=8)
    print(f"wrote {args.out}  {len(frames) * dt:.1f}s @ {1/dt:.0f}fps")
    print(f"trunk z {min(zs)*1000:.0f}-{max(zs)*1000:.0f}mm  "
          f"FELL={'YES' if min(zs) < 0.09 else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
