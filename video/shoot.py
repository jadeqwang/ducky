#!/usr/bin/env python3
"""Film a scripted Microduck scene and write an mp4.

    uv run video/shoot.py --out video/micro.mp4

Drives the official ONNX policies through a beat sheet in scene_micro.xml
(the "MICRO" letters + a rubber duck) and renders offscreen with MuJoCo's
Renderer. No viewer window, so it runs headless.

Beats are CLOSED-LOOP: each one advances when a predicate on the duck's actual
pose is satisfied, with a timeout as a backstop. Open-loop timings drift badly
here because the walking policy's real displacement never matches the commanded
velocity -- it slips, staggers, and takes a moment to spin up.

Reuses PolicyInference from scripts/infer_policy.py rather than rebuilding the
61-D observation, which is easy to get subtly wrong (see AGENTS.md: the obs
layout is a hard invariant shared by every policy).
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
import wave

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from infer_policy import PolicyInference, BALL_OFFSET_X, BALL_OFFSET_ABS_Y  # noqa: E402


def _quiet_set_vel_cmd(self, lin_vel_x=0.0, lin_vel_y=0.0, ang_vel_z=0.0):
    """set_vel_cmd without the per-call print — we call it every control tick."""
    self.vel_cmd = np.array([lin_vel_x, lin_vel_y, ang_vel_z], dtype=np.float32)
    self._update_policy_session()
    self._update_command()


PolicyInference.set_vel_cmd = _quiet_set_vel_cmd

SCENE = os.path.join(REPO, "src/mjlab_microduck/robot/microduck/scene_micro.xml")
POLICIES = os.path.join(REPO, "policies")
DEFAULT_QUACK = os.path.join(REPO, "video", "duck-quack.wav")

TIMESTEP = 0.005      # matches infer_policy.py
DECIMATION = 4        # -> 50 Hz control
CTRL_DT = TIMESTEP * DECIMATION


def synthesize_quack(path, sample_rate=48_000):
    """Write a short, deterministic toy-duck quack as 16-bit mono WAV.

    Two noisy, downward-swept voiced pulses approximate the abrupt attack and
    nasal resonance of a rubber-duck quack. This built-in sound keeps rendering
    self-contained; --quack-wav can substitute the official Microduck chirp.
    """
    duration = 0.72
    t = np.arange(int(duration * sample_rate), dtype=np.float64) / sample_rate
    rng = np.random.default_rng(42)
    sound = np.zeros_like(t)
    for start, length, gain, f0, f1 in (
        (0.00, 0.29, 1.00, 260.0, 150.0),
        (0.34, 0.25, 0.72, 235.0, 135.0),
    ):
        mask = (t >= start) & (t < start + length)
        u = (t[mask] - start) / length
        freq = f0 + (f1 - f0) * u
        phase = 2.0 * math.pi * np.cumsum(freq) / sample_rate
        # Odd harmonics make the source buzzy; band-like nasal resonances and
        # a little breath noise make it read as a quack rather than a beep.
        voiced = (np.sin(phase) + 0.46 * np.sin(3 * phase)
                  + 0.22 * np.sin(5 * phase))
        nasal = 0.30 * np.sin(2.0 * math.pi * 780.0 * (t[mask] - start))
        noise = rng.normal(0.0, 0.16, mask.sum())
        attack = np.minimum(u / 0.025, 1.0)
        release = np.maximum(1.0 - u, 0.0) ** 1.7
        sound[mask] += gain * attack * release * (0.64 * voiced + nasal + noise)
    sound = np.tanh(1.7 * sound)
    peak = max(float(np.max(np.abs(sound))), 1e-9)
    pcm = np.asarray(sound / peak * 0.88 * 32767.0, dtype="<i2")
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def mux_quack(silent_video, quack_wav, offset, duration, output):
    """Mux a quack into a full-length audio track, copying the video stream.

    Do not use an input timestamp offset here. Some MP4 players ignore its edit
    list and treat the short audio stream as ending near t=0. A real delay plus
    padding produces a conventional duration-matched track in every player.
    """
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    delay_ms = max(0, round(offset * 1000.0))
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", silent_video, "-i", quack_wav,
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
        "-af", f"adelay={delay_ms}:all=1,apad",
        "-t", f"{duration:.3f}", "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", output,
    ]
    subprocess.run(cmd, check=True)

# Where the "I" stands in scene_micro.xml (letter pitch 0.23, see make_letters.py).
PITCH = 0.23
I_POS = np.array([0.0, PITCH])

# Kick from the FRONT (camera side, x < 0), straight through the I's position.
# The robot therefore faces +X for the kick.  This deliberately decouples the
# kick heading from the final pose: after the punt it turns to PAYOFF_YAW, steps
# into the gap, and can still perform the established head-only camera look.
KICK_YAW = 0.0
PAYOFF_YAW = math.radians(-125.0)

# Launch away from the front camera, angled toward screen-left so the tumbling
# I clears the row rather than disappearing edge-on behind the C. This is the
# exact opposite of the old back-side punt direction.
PUNT_YAW = math.radians(28.0)

# To kick the I, the duck's trunk must sit so the I lands where the kick policy
# expects its ball: trunk + yaw_rotate(BALL_OFFSET_X, -BALL_OFFSET_ABS_Y) for a
# right-foot kick.
# ...but biased 1 cm CLOSER than the trained ball offset, which was calibrated
# against a small sphere. Measured, the tolerance here is sharply asymmetric:
# standing 2 cm short is a clean whiff (the boot passes ~6 cm from the letter's
# axis), while 1-1.5 cm long still connects cleanly. Being early costs nothing;
# being late costs the shot.
KICK_STANDOFF = BALL_OFFSET_X - 0.010
_c, _s = math.cos(KICK_YAW), math.sin(KICK_YAW)
KICK_SPOT = I_POS - np.array([
    _c * KICK_STANDOFF - _s * (-BALL_OFFSET_ABS_Y),
    _s * KICK_STANDOFF + _c * (-BALL_OFFSET_ABS_Y),
])

# Alternate "stomp" take.  The duck launches from directly in front of the I
# so its ballistic path crosses the top face rather than the narrow kick face.
# There is no jump policy in the shipped policy set; the standing policy poses
# the crouch/landing while one root-velocity impulse supplies the flight.
JUMP_YAW = 0.0
JUMP_SPOT = I_POS - np.array([0.14, 0.0])

# Staging marks for the opening sprint. The duck enters from screen right
# (-Y), runs across the front of the word at x=-0.45, and wipes out past the M
# before taking a direct diagonal route to the front kick mark.
# Marks are kept short because the gait only makes ~0.12 m/s of real ground
# speed at a 0.25 command (~0.47x commanded); long runs are just long takes.
START = np.array([-0.52, -0.55])
SPRINT_LANE = np.array([-0.52, 0.95])   # aim point that keeps the run parallel
                                        # to the word instead of veering into it
SPRINT_END_Y = 0.30
# In film_word, the duck's body first enters the right edge of frame at about
# y=0.40 m.  Its cast shadow arrives earlier, so the otherwise empty-looking
# tail of the recovery-to-approach cut can be skipped independently of the
# visible walk that follows.
WORD_CAM_BODY_ENTRY_Y = 0.40
# The camera the duck looks into for the payoff, so head_to_cam() can aim at
# something real. Must match scene_micro.xml's film_close.
CLOSE_CAM = np.array([-0.55, 0.26])


def yaw_of(quat):
    qw, qx, qy, qz = quat
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Shoot:
    def __init__(self, model, data, policy, args):
        self.m, self.d, self.p, self.args = model, data, policy, args
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.qadr = int(model.jnt_qposadr[jid])
        self.vadr = int(model.jnt_dofadr[jid])
        # --- movable-I bookkeeping (see punt()/stomp()) ---
        ijid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "letter_I_free")
        self.i_vadr = int(model.jnt_dofadr[ijid])
        self.i_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "letter_I")
        self.i_geoms = {g for g in range(model.ngeom)
                        if model.geom_bodyid[g] == self.i_bid}
        # Everything that is NOT the robot: floor, the other letters, and the
        # rubber duck. A contact between the I and anything outside this set is
        # the robot connecting with it.
        props = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                 for n in ("letter_M", "letter_C", "letter_R", "letter_O",
                           "rubber_duck")}
        self.other_geoms = {g for g in range(model.ngeom)
                            if model.geom_bodyid[g] in props} | {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")}
        self.foot_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                           "right_foot_collision")
        self.punted = False
        self.jumped = False
        self.stomped = False
        self.mouth_gid = None
        self.mouth_closed_quat = None
        if args.cinematic_mouth:
            for gid in range(model.ngeom):
                mesh_id = int(model.geom_dataid[gid])
                mesh_name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
                             if mesh_id >= 0 else None)
                if mesh_name == "jaw" and model.geom_group[gid] == 2:
                    self.mouth_gid = gid
                    self.mouth_closed_quat = model.geom_quat[gid].copy()
                    break
            if self.mouth_gid is None:
                raise RuntimeError("cinematic mouth requested, but visual jaw mesh not found")
        self._foot_dist = None
        self._min_foot = 9.0
        self._latched = set()

    # ---- duck state -------------------------------------------------
    @property
    def pos(self):
        return self.d.qpos[self.qadr:self.qadr + 2].copy()

    @property
    def height(self):
        return float(self.d.qpos[self.qadr + 2])

    @property
    def yaw(self):
        return yaw_of(self.d.qpos[self.qadr + 3:self.qadr + 7])

    def upright(self):
        """Trunk z-axis still pointing up-ish -- i.e. not face-planted."""
        return self.p.get_projected_gravity()[2] < -0.75

    def place(self, x, y, yaw):
        q = self.d.qpos
        q[self.qadr + 0] = x
        q[self.qadr + 1] = y
        q[self.qadr + 2] = 0.125
        q[self.qadr + 3:self.qadr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        for i, idx in enumerate(self.p.joint_qpos_indices):
            q[idx] = self.p.default_pose[i]
        self.d.ctrl[:] = self.p.default_pose
        mujoco.mj_forward(self.m, self.d)

    def latch(self, key):
        """True exactly once per key. For one-shot events inside a per-tick."""
        if key in self._latched:
            return False
        self._latched.add(key)
        return True

    def sprint_accel(self, a):
        """Sustained forward acceleration, in m/s^2, applied per control tick.

        SUSTAINED, not a single impulse. The gait tops out around 0.25 m/s of
        real ground speed, which does not read as running at all. Pushing the
        trunk at ~0.8 m/s^2 for a couple of seconds gets it to ~0.73 m/s -- 3x
        its own top speed, legs visibly scrambling to keep up, which is the
        joke. Scaled by CTRL_DT: an early version added the whole figure every
        frame, summing to ~12 m/s, and fired the duck across the set.

        Note this does NOT make it fall. The policy is robust enough to shed
        even 0.73 m/s on a reverse command and stay upright, which is why the
        wipeout needs trip() rather than just braking.
        """
        self.d.qvel[self.vadr + 0] += a * CTRL_DT * math.cos(self.yaw)
        self.d.qvel[self.vadr + 1] += a * CTRL_DT * math.sin(self.yaw)

    def trip(self, v, pitch):
        """One-shot stumble: a forward nudge plus a nose-down pitch rate.

        Braking from a sprint is not enough to put this policy down, and a
        forward impulse big enough to do it (~0.9 m/s) skates the duck a metre
        out of frame. Catching a toe is what actually fells a runner, so this
        adds the rotation directly: a small forward nudge with a nose-down
        pitch about the body's lateral axis. It face-plants close to where it
        was running instead of being launched.
        """
        if not self.latch("trip"):
            return
        y = self.yaw
        self.d.qvel[self.vadr + 0] += v * math.cos(y)
        self.d.qvel[self.vadr + 1] += v * math.sin(y)
        # -pitch about the lateral axis (sin y, -cos y, 0) sends the nose down.
        self.d.qvel[self.vadr + 3] += -pitch * math.sin(y)
        self.d.qvel[self.vadr + 4] += pitch * math.cos(y)

    def jump(self):
        """Launch once toward the top of the I for the alternate stomp take."""
        if not self.latch("jump"):
            return
        y = self.yaw
        self.d.qvel[self.vadr + 0] += self.args.jump_vx * math.cos(y)
        self.d.qvel[self.vadr + 1] += self.args.jump_vx * math.sin(y)
        self.d.qvel[self.vadr + 2] += self.args.jump_vz
        self.jumped = True

    def stomp(self):
        """Topple the I as the descending duck reaches its top face.

        The collision remains physical after this one angular nudge.  Applying
        it at the top-face crossing makes either a feet-first or body-first
        landing knock the prop away from the incoming duck instead of requiring
        a particular foot contact from a policy that was never trained to jump.
        """
        if self.stomped or not self.jumped:
            return
        close = float(np.linalg.norm(self.pos - I_POS)) < 0.075
        descending = self.d.qvel[self.vadr + 2] < 0.0
        top_face = self.height < 0.34
        if not (close and descending and top_face):
            return
        # Fall along X, away from whichever side the duck arrived on.  A tiny
        # translation prevents the serif from balancing under the duck.
        direction = 1.0 if self.pos[0] <= I_POS[0] else -1.0
        v = self.i_vadr
        self.d.qvel[v + 0] += 0.20 * direction
        self.d.qvel[v + 4] += -self.args.stomp_spin * direction
        self.stomped = True
        print(f"    stomp! I falling toward {'+X' if direction > 0 else '-X'}")

    def head_to_cam(self, cam_pos):
        """Head yaw that points the beak at a camera, from wherever the body is.

        Deliberately measured against the duck's ACTUAL heading each tick
        rather than the staged PAYOFF_YAW: the kick and subsequent body turn
        introduce variable drift, so a baked constant aims the payoff look off
        into space. Clamped to the trained joint cap.
        """
        bearing = math.atan2(cam_pos[1] - self.pos[1], cam_pos[0] - self.pos[0])
        return float(np.clip(wrap(bearing - self.yaw), -1.4, 1.4))

    def fallen(self):
        return self.p.get_projected_gravity()[2] > -0.35

    # ---- the punt ---------------------------------------------------
    def punt(self, t):
        """Add a launch impulse to the I on the frame the boot connects.

        This is a deliberate cheat and it is the only one in the shoot. The
        kick CANNOT punt a letter on physics alone: a 0.20 m letter struck at
        ankle height rotates instead of translating, so it topples and lands
        about 19 cm away. Measured, that 19 cm is insensitive to the letter's
        mass (15 g vs 40 g), its floor friction (a 35x sweep changed nothing --
        it never slides), and its centre-of-mass height. It is just the letter
        falling over. Getting a punt needs an UPWARD component the swing has no
        way to produce, so the shot adds one.

        Triggered on the swing itself rather than a wall clock, so it stays in
        sync with whatever the kick policy actually does, and applied along the
        duck's own heading so it follows the staging.
        """
        if self.punted or self.args.punt_vx <= 0:
            return
        # Fire at the boot's CLOSEST APPROACH to the letter, geometrically,
        # rather than on a physical contact. Both alternatives were tried here
        # and both failed:
        #   - Contact between the letter and any robot geom fires on the FIRST
        #     frame, because the duck is already resting against the letter
        #     when it takes its stance; the swing has not started yet.
        #   - Contact with the kicking foot specifically is only about two
        #     control frames wide, and a 2 cm staging error erases it while the
        #     shin still topples the letter -- a silent whiff.
        # Closest approach is smooth, always occurs, and is in sync with the
        # swing. The t gate clears the settled stance; the swing bottoms out
        # around t=0.12, and the 0.075 m ceiling means a badly mis-staged duck
        # that never gets near the letter does not launch it from thin air.
        d = float(np.linalg.norm(self.d.geom_xpos[self.foot_geom][:2] - I_POS))
        self._min_foot = min(self._min_foot, d)
        prev, self._foot_dist = self._foot_dist, d
        if t < 0.05 or prev is None:
            return
        if not (prev < 0.075 and d > prev):
            return
        # Launch along a FIXED staged direction, not the duck's instantaneous
        # yaw: the kick policy rotates the duck a variable 10-35 deg mid-swing,
        # and letting that steer the letter can aim it into the neighboring
        # letters. PUNT_YAW sends it behind and diagonally clear of the row.
        yaw = PUNT_YAW
        v = self.i_vadr
        self.d.qvel[v + 0] += self.args.punt_vx * math.cos(yaw)
        self.d.qvel[v + 1] += self.args.punt_vx * math.sin(yaw)
        self.d.qvel[v + 2] += self.args.punt_vz
        # Topspin about the horizontal axis perpendicular to travel, so the
        # letter tumbles end over end instead of sailing off like a plank.
        self.d.qvel[v + 3] += -self.args.punt_spin * math.sin(yaw)
        self.d.qvel[v + 4] += self.args.punt_spin * math.cos(yaw)
        self.punted = True
        print(f"    punt! vx={self.args.punt_vx} vz={self.args.punt_vz} "
              f"along yaw={math.degrees(yaw):.0f}deg")

    # ---- command helpers --------------------------------------------
    # Training ranges (microduck_velocity_env_cfg.py): lin_vel_x (-0.4, 0.4),
    # lin_vel_y (-0.3, 0.3), ang_vel_z (-1.0, 1.0). Commanding yaw outside
    # ±1.0 puts the policy out of distribution and it turns in place instead of
    # walking, so YAW_CAP stays well inside the trained range.
    # Turning is slow: ~20 deg/s achieved at the trained maximum, so a 180 deg
    # spin takes ~9 s. Budget beat timeouts accordingly.
    # The gait has a COMMAND DEADBAND: anything below roughly 0.2 produces no
    # motion at all. A plain P-controller therefore stalls with ~10 deg of yaw
    # error or ~7 cm of position error left, because its output has decayed
    # into the deadband. Every command below floors its magnitude once the
    # error is outside the dead zone, so the last bit of error still closes.
    YAW_GAIN = 1.4
    YAW_CAP = 1.0
    YAW_DEAD = 0.05      # rad — inside this, stop commanding
    YAW_FLOOR = 0.45     # rad/s — smallest command that actually turns

    POS_DEAD = 0.015     # m
    POS_FLOOR = 0.18     # m/s

    @staticmethod
    def _floored(value, floor, lo, hi):
        v = float(np.clip(value, lo, hi))
        return math.copysign(max(abs(v), floor), v)

    def turn_cmd(self, target_yaw):
        err = wrap(target_yaw - self.yaw)
        if abs(err) < self.YAW_DEAD:
            return 0.0
        return self._floored(self.YAW_GAIN * err, self.YAW_FLOOR,
                             -self.YAW_CAP, self.YAW_CAP)

    def face(self, target_yaw):
        """Spin in place toward a heading."""
        self.p.set_vel_cmd(0.0, 0.0, self.turn_cmd(target_yaw))

    def goto(self, target, speed=0.30):
        """Steer toward a waypoint.

        Bearing-following rather than fixed-heading holding: holding a heading
        corrects orientation but not cross-track error, and this gait drifts
        ~0.25 m sideways over 12 s. Steering at the target closes that loop.
        Forward speed scales with heading alignment so the duck turns first and
        walks second instead of crabbing sideways.
        """
        delta = np.asarray(target, dtype=float) - self.pos
        bearing = math.atan2(delta[1], delta[0])
        align = max(0.0, math.cos(wrap(bearing - self.yaw)))
        self.p.set_vel_cmd(speed * align, 0.0, self.turn_cmd(bearing))

    def dash(self, target, speed):
        """goto() at a fixed speed: steer, but never throttle back.

        goto() scales forward speed by heading alignment so the duck turns
        before it walks. That is right for placement and wrong for the
        entrance, where easing off the throttle to correct course is exactly
        what a sprint must not do. Without any steering at all the sprint
        drifts ~0.3 m off line over its length and ends up in the letters.
        """
        delta = np.asarray(target, dtype=float) - self.pos
        bearing = math.atan2(delta[1], delta[0])
        self.p.set_vel_cmd(speed, 0.0, self.turn_cmd(bearing))

    def creep(self, target, target_yaw, dead=None):
        """Close on a mark while holding a heading, using body-frame velocity.

        goto() steers by bearing, which spins the duck away from the heading it
        needs for the kick (and flips 180 deg on the slightest overshoot). Here
        the position error is rotated into the body frame and driven with
        forward + lateral velocity instead, so the duck strafes onto its mark
        facing the right way. lin_vel_y is a trained command (+-0.3).
        """
        dead = self.POS_DEAD if dead is None else dead
        err = np.asarray(target, dtype=float) - self.pos
        c, s_ = math.cos(self.yaw), math.sin(self.yaw)
        fwd = c * err[0] + s_ * err[1]
        lat = -s_ * err[0] + c * err[1]
        vx = 0.0 if abs(fwd) < dead else self._floored(
            2.2 * fwd, self.POS_FLOOR, -0.16, 0.26)
        vy = 0.0 if abs(lat) < dead else self._floored(
            2.2 * lat, self.POS_FLOOR, -0.18, 0.18)
        self.p.set_vel_cmd(vx, vy, self.turn_cmd(target_yaw))

    def dist_to(self, target):
        return float(np.linalg.norm(np.asarray(target, dtype=float) - self.pos))

    def head(self, neck=0.0, pitch=0.0, yaw=0.0, roll=0.0):
        self.p.head_offset[:] = [neck, pitch, yaw, roll]
        self.p._update_command()

    def body_height(self, z=0.0):
        """Set the standing policy's commanded body-height offset."""
        self.p.body_cmd[:] = 0.0
        self.p.body_cmd[2] = float(np.clip(z, -0.03, 0.03))
        self.p._update_command()

    def mouth(self, opening=0.0):
        """Rotate only the rendered lower beak; physics remains the 14-DoF model."""
        if self.mouth_gid is None:
            return
        angle = float(np.clip(opening, 0.0, 0.60))
        hinge_quat = np.array([math.cos(angle / 2), 0.0,
                               math.sin(angle / 2), 0.0])
        mujoco.mju_mulQuat(self.m.geom_quat[self.mouth_gid],
                           hinge_quat, self.mouth_closed_quat)


def build_beats(s):
    """The beat sheet. Each: name, camera, on_enter, per-tick, done, timeout."""
    A = s.args

    opening = [
        dict(
            # Running start. The command goes well past the trained +-0.4 and a
            # forward impulse is added on top: the point is a duck moving
            # faster than it can stop, which is also faster than the gait is
            # trained for, so the stumble comes for free.
            name="sprint", cam="film_run", timeout=12.0,
            enter=lambda: s.place(START[0], START[1], math.pi / 2),
            tick=lambda t: (s.dash(SPRINT_LANE, A.sprint_vel),
                            s.sprint_accel(A.sprint_accel) if t > 0.8 else None),
            done=lambda t: s.pos[1] > SPRINT_END_Y,
        ),
        dict(
            # Catch a toe. Measured: this policy sheds a 0.73 m/s sprint on a
            # plain reverse command and stays up, so the fall has to come from
            # the stumble itself -- see Shoot.trip. Trunk height drops
            # 0.116 -> ~0.04 and projected gravity z flips sign.
            name="wipeout", cam="film_fall", timeout=4.0,
            enter=lambda: s.trip(A.trip_vel, A.trip_pitch),
            tick=lambda t: s.p.set_vel_cmd(0.0, 0.0, 0.0),
            done=lambda t: t > 0.8 and s.fallen(),
        ),
        dict(
            # Get up immediately. There is NO standup policy in the shipped set -- but the
            # plain standing policy scrambles the duck back onto its feet from
            # face-down in about 3 s, which is what this beat is. (alpha_sitstand
            # does not: it stays down.) Zero command keeps the standing session
            # selected; anything above switch_threshold hands back to walking.
            name="standup", cam="film_fall", timeout=9.0,
            tick=lambda t: s.p.set_vel_cmd(0.0, 0.0, 0.0),
            done=lambda t: t > 1.0 and s.upright() and s.height > 0.105,
        ),
        dict(
            # Stay on the camera side of the title and head directly from the
            # recovery position to the front kick mark. The old cut first
            # rounded the M and walked a full metre behind the letters. This is
            # mostly repetitive walking before the duck returns to the action,
            # so compress it hard in the final cut.
            name="approach_front", cam="film_word", timeout=16.0,
            playback_rate=lambda: (100.0 if s.pos[1] > WORD_CAM_BODY_ENTRY_Y
                                   else 4.0),
            tick=lambda t: s.goto(JUMP_SPOT if A.stomp_i else KICK_SPOT, speed=0.36),
            done=lambda t: s.dist_to(JUMP_SPOT if A.stomp_i else KICK_SPOT) < 0.09,
        ),
        dict(
            # Square up BEFORE the fine placement -- turning drifts the duck
            # ~0.12 m, which would otherwise knock it straight back off its mark.
            # Compress the necessary footwork in the final cut.
            name="square_up", cam="film_low", timeout=12.0,
            playback_rate=2.0,
            tick=lambda t: s.face(JUMP_YAW if A.stomp_i else KICK_YAW),
            done=lambda t: (
                abs(wrap(s.yaw - (JUMP_YAW if A.stomp_i else KICK_YAW))) < 0.15
                and t > 0.8
            ),
        ),
    ]

    if A.stomp_i:
        action = [
            dict(
                name="creep_to_jump", cam="film_low", timeout=12.0,
                tick=lambda t: s.creep(JUMP_SPOT, JUMP_YAW, dead=0.007),
                done=lambda t: (s.dist_to(JUMP_SPOT) < 0.012
                                and abs(wrap(s.yaw - JUMP_YAW)) < 0.06),
            ),
            dict(
                # A short, readable crouch before the impulse. The standing
                # policy was trained for this 30 mm body-height command.
                name="coil", cam="film_low", timeout=1.4,
                tick=lambda t: (s.p.set_vel_cmd(0.0, 0.0, 0.0),
                                s.body_height(-0.03)),
                done=lambda t: t > 0.9,
            ),
            dict(
                name="jump_on_i", cam="film_low", timeout=1.5,
                playback_rate=1.0,
                enter=lambda: (s.body_height(0.03), s.jump()),
                tick=lambda t: s.stomp(),
                done=lambda t: s.stomped,
            ),
            dict(
                # Let the duck and letter finish the same contact event, then
                # hand the standing policy enough time to recover its feet.
                name="stomp_landing", cam="film_low", timeout=5.0,
                playback_rate=1.0,
                tick=lambda t: (s.body_height(0.0),
                                s.p.set_vel_cmd(0.0, 0.0, 0.0)),
                done=lambda t: t > 1.0 and s.upright() and s.height > 0.10,
            ),
        ]
    else:
        action = [
            dict(
            # Fine: strafe onto the mark while staying square to the letter.
            # dead=0.007 overrides the usual 15 mm position dead zone: the mark
            # has to be hit to ~1 cm and the default deadband stops correcting
            # well before that.
            name="creep", cam="film_low", timeout=12.0,
            tick=lambda t: s.creep(KICK_SPOT, KICK_YAW, dead=0.007),
            # Tight: the kick swings open-loop from wherever the duck is
            # standing, and the kick policy itself rotates the duck a further
            # ~10 deg mid-swing, so aim errors at trigger time are unrecoverable.
            done=lambda t: (s.dist_to(KICK_SPOT) < 0.012
                            and abs(wrap(s.yaw - KICK_YAW)) < 0.06),
            ),
            dict(
            # PLANT. Do not delete this beat: the kick policies were trained to
            # run "from a standing start with an all-zero command", and firing
            # one straight out of the creep -- while the duck still has stride
            # momentum -- changes the swing enough to miss. Measured: triggered
            # mid-stride the boot's closest approach to the letter is 8.8 cm;
            # triggered after a second of standing, from the SAME mark, it is
            # 3.7 cm. That is the difference between a punt and a whiff. Keep
            # the full physical settle. Show its final second at real time so
            # the cut visibly decelerates into the kick instead of snapping
            # straight from compressed setup to contact.
            name="settle", cam="film_low", timeout=1.6,
            playback_rate=1.0,
            tick=lambda t: s.p.set_vel_cmd(0.0, 0.0, 0.0),
            done=lambda t: t > 1.0,
            ),
            dict(
            # s.punt(t) runs every tick and fires once, at the bottom of the
            # swing -- see Shoot.punt for why that is not a contact test. End
            # this beat at that exact instant: waiting for the kick policy's
            # full timer left a conspicuous pause after the I was already gone.
            name="kick", cam="film_low", timeout=A.kick_duration + 0.6,
            enter=lambda: s.p.trigger_behavior("kick_right"),
            tick=lambda t: s.punt(t),
            done=lambda t: s.punted,
            ),
            dict(
            # Stay at 1x after contact long enough to see the I launch and the
            # kicking leg follow through. The next beat ends the kick behavior
            # and compresses the much longer body turn.
            name="punt_followthrough", cam="film_low", timeout=1.0,
            playback_rate=1.0,
            done=lambda t: t >= 0.8,
            ),
        ]

    payoff = [
        dict(
            # The front kick faces away from the lens, beyond the head joint's
            # +-80 degree look range. Turn the BODY only after the I is gone;
            # this restores the established final silhouette and leaves the
            # later head turn as a distinct gesture. The walking policy only
            # achieves ~20 deg/s at its trained yaw limit, so play this beat at
            # 2x to show roughly half as many steps without an OOD command.
            name="turn_for_payoff", cam="film_word", timeout=12.0,
            playback_rate=2.0,
            tick=lambda t: (s.p._end_behavior() if s.p.behavior_mode else None,
                            s.face(PAYOFF_YAW)),
            done=lambda t: abs(wrap(s.yaw - PAYOFF_YAW)) < 0.15 and t > 0.8,
        ),
        dict(
            # Into the gap at the payoff heading. creep() holds orientation
            # while correcting the position drift introduced by the turn.
            name="step_in", cam="film_word", timeout=12.0,
            tick=lambda t: s.creep(I_POS, PAYOFF_YAW),
            done=lambda t: s.dist_to(I_POS) < 0.05,
        ),
        dict(
            # The payoff: body stays where it is, HEAD comes round to the lens.
            name="head_turn", cam="film_close", timeout=1.6,
            tick=lambda t: (s.p.set_vel_cmd(0.0, 0.0, 0.0),
                            s.head(yaw=s.head_to_cam(CLOSE_CAM) * min(t / 1.0, 1.0),
                                   pitch=-0.14 * min(t / 1.0, 1.0))),
            done=lambda t: t > 1.2,
        ),
        dict(
            # Preserve the full head nod, but give the cinematic lower beak one
            # open-close cycle exactly over the bundled 0.425 s quack sample.
            name="quack", cam="film_close", timeout=1.5,
            tick=lambda t: (s.p.set_vel_cmd(0.0, 0.0, 0.0),
                            s.head(yaw=s.head_to_cam(CLOSE_CAM),
                                   pitch=-0.14 + (0.42 * math.sin(2 * math.pi * 1.6 * t)
                                                  if t < 1.3 else 0.0)),
                            s.mouth(0.52 * math.sin(math.pi * t / 0.425)
                                    if t < 0.425 else 0.0)),
            done=lambda t: t > 1.3,
        ),
        dict(
            # Pull back to the word master long enough to read the full title
            # with the duck standing in the I's place.
            name="hold", cam="film_final", timeout=1.2,
            tick=lambda t: (s.p.set_vel_cmd(0.0, 0.0, 0.0),
                            s.head(yaw=s.head_to_cam(CLOSE_CAM), pitch=-0.12)),
            done=lambda t: t >= 1.0,
        ),
    ]
    return opening + action + payoff


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "video/micro.mp4"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sprint-vel", type=float, default=0.55,
                    help="entrance command speed; past the trained 0.4 on purpose")
    ap.add_argument("--sprint-accel", type=float, default=0.8,
                    help="sustained forward acceleration during the run (m/s^2)")
    ap.add_argument("--trip-vel", type=float, default=0.30,
                    help="forward nudge at the stumble (m/s)")
    ap.add_argument("--trip-pitch", type=float, default=7.0,
                    help="nose-down pitch rate at the stumble (rad/s)")
    ap.add_argument("--punt-vx", type=float, default=1.85,
                    help="launch speed added to the I on contact (m/s); 0 = physics only, "
                         "which only topples it ~19 cm. See Shoot.punt()")
    ap.add_argument("--punt-vz", type=float, default=0.95, help="launch rise (m/s)")
    ap.add_argument("--punt-spin", type=float, default=9.0, help="launch topspin (rad/s)")
    ap.add_argument("--kick-duration", type=float, default=3.0)
    ap.add_argument("--stomp-i", action="store_true",
                    help="jump onto and topple the I instead of punting it")
    ap.add_argument("--jump-vx", type=float, default=0.36,
                    help="forward launch speed for --stomp-i (m/s)")
    ap.add_argument("--jump-vz", type=float, default=2.05,
                    help="upward launch speed for --stomp-i (m/s)")
    ap.add_argument("--stomp-spin", type=float, default=4.0,
                    help="I topple angular speed at stomp contact (rad/s)")
    ap.add_argument("--quack", action=argparse.BooleanOptionalAction, default=True,
                    help="mux a quack at the final head nod (default: enabled)")
    ap.add_argument("--quack-wav", default=None,
                    help="optional WAV to use instead of the bundled real-duck quack")
    ap.add_argument("--cinematic-mouth", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="animate the filming-only lower beak during the quack "
                         "(default: enabled)")
    ap.add_argument("--current-limit", type=float, default=1.75,
                    help="XL330 current limit [A]; 0 disables torque clipping")
    ap.add_argument("--cam", default=None, help="override every beat's camera")
    ap.add_argument("--stills", action="store_true",
                    help="write one PNG per beat instead of an mp4 (fast framing check)")
    args = ap.parse_args()

    np.random.seed(args.seed)

    model = mujoco.MjModel.from_xml_path(SCENE)
    model.opt.timestep = TIMESTEP
    data = mujoco.MjData(model)

    def onnx(name):
        p = os.path.join(POLICIES, name)
        return p if os.path.exists(p) else None

    # XL330 current saturation, as infer_policy.py applies by default. The
    # policies were trained against this torque ceiling.
    if args.current_limit > 0:
        from bam.model import load_model
        kt = load_model(motor_name="xl330", model="m6").kt.value
        tl = kt * args.current_limit
        model.actuator_forcerange[:, 0] = -tl
        model.actuator_forcerange[:, 1] = tl
        model.actuator_forcelimited[:] = 1

    policy = PolicyInference(
        model, data,
        walking_onnx_path=onnx("alpha_walking.onnx"),
        standing_onnx_path=onnx("alpha_stand.onnx"),
        kick_right_onnx_path=onnx("ball_kick_right.onnx"),
        kick_left_onnx_path=onnx("ball_kick_left.onnx"),
        # MUST be True. The constructor defaults to False, which feeds the
        # policy the raw accelerometer instead of projected gravity; that reads
        # ~correct while standing still and diverges as soon as the duck
        # accelerates, so the gait collapses after about a second.
        use_projected_gravity=True,
        new_cmd_obs=True,
        kick_duration=args.kick_duration,
    )
    policy.vel_max_x, policy.vel_min_x = 0.3, -0.3
    policy.vel_max_y, policy.vel_min_y = 0.2, -0.2
    policy.vel_max_ang = 1.5

    s = Shoot(model, data, policy, args)
    s.place(START[0], START[1], math.pi / 2)

    renderer = mujoco.Renderer(model, args.height, args.width)
    frames, frame_dt = [], 1.0 / args.fps
    next_frame = 0.0
    sim_t = 0.0
    quack_time = None

    def grab(cam):
        renderer.update_scene(data, camera=(args.cam or cam))
        frames.append(renderer.render())

    for beat in build_beats(s):
        if beat["name"] == "quack":
            quack_time = len(frames) / args.fps
        if beat.get("enter"):
            beat["enter"]()
        t0, ticks = sim_t, 0
        playback_rate = beat.get("playback_rate", 1.0)
        keep_accumulator = 0.0
        while True:
            t = sim_t - t0
            # Keep the passive cinematic hinge closed except where a beat
            # explicitly poses it (currently the quack).
            s.mouth(0.0)
            if beat.get("tick"):
                beat["tick"](t)
            if beat["done"](t):
                break
            if t > beat["timeout"]:
                print(f"  ! '{beat['name']}' hit its {beat['timeout']:.1f}s timeout "
                      f"(pos={s.pos.round(3)} yaw={math.degrees(s.yaw):.0f}deg)")
                break

            # A "freeze" beat stops running the policy and lets the servos
            # hold their last target, so the duck lies where it landed instead
            # of instantly fighting its way upright. Physics still steps.
            if not beat.get("freeze"):
                action = policy.infer()
                policy.apply_action(action)
            for _ in range(DECIMATION):
                mujoco.mj_step(model, data)
            sim_t += CTRL_DT
            policy.update_behavior(CTRL_DT)
            ticks += 1

            if not args.stills and sim_t >= next_frame:
                next_frame += frame_dt
                rate = float(playback_rate() if callable(playback_rate)
                             else playback_rate)
                keep_accumulator += 1.0 / rate
                if keep_accumulator >= 1.0:
                    grab(beat["cam"])
                    keep_accumulator -= 1.0

        if args.stills:
            grab(beat["cam"])
        print(f"  {beat['name']:<16} {sim_t - t0:5.2f}s  pos={s.pos.round(3)} "
              f"yaw={math.degrees(s.yaw):6.1f}deg  h={s.height:.3f}")

    if args.stomp_i:
        print(f"  jump fired: {s.jumped}   stomp fired: {s.stomped}")
    else:
        print(f"  punt fired: {s.punted}   closest boot approach to the I: "
              f"{s._min_foot*100:.1f} cm")

    # Did the kick actually connect? The letter should be metres away, not
    # centimetres -- a near miss still looks like a hit in the beat log.
    for name, home in (("letter_I", (0.0, PITCH)), ("letter_M", (0.0, 2 * PITCH)),
                       ("letter_C", (0.0, 0.0)), ("letter_R", (0.0, -PITCH)),
                       ("letter_O", (0.0, -2 * PITCH)),
                       ("rubber_duck", (0.0, -3.15 * PITCH))):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        moved = np.linalg.norm(data.xpos[bid][:2] - np.array(home))
        verb = "STOMPED" if args.stomp_i else "KICKED"
        tag = f"  <-- {verb}" if name == "letter_I" and moved > 0.15 else ""
        if name == "letter_I" or moved > 0.02:
            print(f"  {name}: moved {moved*100:5.1f} cm{tag}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.stills:
        base = os.path.splitext(args.out)[0]
        for i, f in enumerate(frames):
            imageio.imwrite(f"{base}_{i:02d}.png", f)
        print(f"\n{len(frames)} stills -> {base}_NN.png")
    else:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if args.quack:
            with tempfile.TemporaryDirectory(prefix="microduck-video-", dir=out_dir) as tmp:
                silent = os.path.join(tmp, "silent.mp4")
                audio = args.quack_wav or DEFAULT_QUACK
                muxed = os.path.join(tmp, "with-quack.mp4")
                if not os.path.isfile(audio):
                    if args.quack_wav:
                        raise FileNotFoundError(f"quack WAV not found: {audio}")
                    audio = os.path.join(tmp, "quack.wav")
                    synthesize_quack(audio)
                imageio.mimwrite(silent, frames, fps=args.fps, quality=8,
                                 macro_block_size=1)
                duration = len(frames) / args.fps
                mux_quack(silent, audio, quack_time or 0.0, duration, muxed)
                os.replace(muxed, args.out)
            audio_note = args.quack_wav or "bundled real-duck quack"
            print(f"\n{len(frames)} frames ({len(frames)/args.fps:.1f}s) -> "
                  f"{args.out}  [quack at {quack_time:.2f}s: {audio_note}]")
        else:
            imageio.mimwrite(args.out, frames, fps=args.fps, quality=8,
                             macro_block_size=1)
            print(f"\n{len(frames)} frames ({len(frames)/args.fps:.1f}s) -> {args.out}")


if __name__ == "__main__":
    main()
