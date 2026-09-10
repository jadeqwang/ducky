# Chicken Banana — a beat-locked dance for Microduck

**Status (2026-09-10):** working in sim, untested on hardware (nobody has hardware).
**Headline finding: this does not need a trained policy.**

Goal: make the duck dance to "Chicken Banana" (130 BPM, 4/4 — beat 462 ms,
two-beat phrase 923 ms) for a small human audience.

---

## The one thing to know

The original plan was a trained policy: one network, a binary command flag
selecting one of two poses, a host sequencer toggling the flag on the beat —
structurally the `sitstand` task. That plan is sound, and it is **the wrong tool
for this job**.

A head-and-torso dance needs no policy at all. The stock **`alpha_stand.onnx`**
(an official policy, already in `policies/`, and the slot the robot boots into)
holds balance while a sequencer streams pose commands at 50 Hz. Measured across
four different moves, the trunk never moved more than ~1 mm and it never fell.

Training only buys what commands cannot reach: **weight shifts, stepping in time,
squats — anything where the legs move to the beat.** Those need balance the
standing policy was not trained to hold. Nothing in this dance does.

Deliverable: [`scripts/dance_sequencer.py`](../scripts/dance_sequencer.py).

---

## Measurements

All from CPU MuJoCo via `scripts/infer_policy.py`'s `PolicyInference`, driving
the shipped ONNX policies. **Fidelity caveat, applies to everything below:**
`scene.xml` uses plain MuJoCo position actuators with a 0.64 Nm torque clip, not
the BAM voltage model training used — and the README says actuator fidelity is
most of the sim2real gap. No domain randomization, no observation noise, single
seed. These are best-case numbers.

### Head bandwidth — the number that decided the project

Stepping the head-pose command under `alpha_stand`:

| | value |
|---|---|
| `head_pitch` to 90% of command | **160–180 ms** |
| return to 10% | 200–220 ms |
| full forward-and-back cycle | **~380 ms** |
| trunk z during the move | 116 → 116 mm (unmoved) |

A two-beat phrase is 923 ms. A bob needs 380. There is enormous headroom — you
could bob on every beat (462 ms) and still have margin. **The constraint on this
dance is legibility, not physics.**

`neck_pitch` tracks only **47–59%** of commanded amplitude and stays there. That
is the documented head droop (see the `head_pose_bias` comment in
`microduck_velocity_env_cfg.py`) — a steady-state bias, not a bandwidth limit.
Command bigger than you want. `head_pitch` tracks 85–109%.

### Body pose works too, and that was a surprise

`alpha_stand` accepts **body-pose commands on z/roll/pitch**, not just head:
commanding body pitch produced **21.4° of trunk pitch** peak-to-peak with no
fall. `microduck_standup_env_cfg.py:466` trains `body_pose_tracking` with
`axis_weights=(0,0,1,1,1,0)`, ramped in from iteration 2500 — and the shipped
policy clearly got far enough. That doubles the available dance vocabulary for
free.

### Sit↔stand timing — and a correction

The original plan would have copied the `sitstand` recipe, so we timed it.

| | 90% of travel | 98% | peak \|vz\| | training cap |
|---|---|---|---|---|
| descent (stand→sit) | 0.70 s | 0.74 s | 0.39 m/s | 0.05 m/s |
| rise (sit→stand) | 0.32 s | 0.44 s | 0.34 m/s | 0.08 m/s |

Measured rest heights match the config almost exactly (115.9 vs `STAND_Z` 0.115;
59.5 vs `SIT_Z` 0.060), which validates the harness.

**Correction to an earlier claim in this project:** the speed caps
(`MAX_DESCENT_SPEED`, `MAX_RISE_SPEED`) were initially read as imposing hard
floors of 1.10 s / 0.69 s, making a 0.92 s transition impossible. That was
wrong. They are *penalties*, not constraints — the trained policy pays them and
exceeds them 5–8× instantaneously. Both transitions already fit inside a
two-beat phrase.

The descent profile is instructive: the first 25 mm travel at 0.0274 m/s —
exactly the `POSTURE_RAMP_S = 2.0` design rate (55 mm / 2 s) — and then the fold
passes static stability and the last 30 mm drops in ~150 ms. The ramp governs
the controlled half; physics takes the rest.

Delay sensitivity (training used 3–6 ticks): timing is robust (0.70→0.78 s
descent), *stability* is not — at 3 ticks max tilt hit 42°, at 6 ticks the rise
failed outright. Treat as "degrades", not calibrated: stacking a delay buffer on
non-BAM actuators is not what training saw.

### The reference video

The source is a photo-collage animation — a chicken's head on a banana body,
2D cutouts moved over flat colour fields, hard cuts every ~4 s. Motion tracked
by foreground centroid across four moving shots:

- **Horizontal, not vertical.** ~66 px side-to-side vs ~6.6 px up-down — a
  **10:1 ratio**, consistent across shots.
- **Not on the beat.** Power at 2.17 Hz (130 BPM) is only **1–5%** of motion
  energy; **38–66%** sits below 1 Hz, peaking at 0.24–0.53 Hz — one sway per 4–8
  beats. The rhythmic feel comes from the *editing*, not the movement.
- The head stays in profile with the face level. It never rotates enough to hide
  the face.

*Method caveat:* centroid tracking of the whole foreground blob; in tight
close-ups some motion reflects frame cropping. Horizontal-dominance and
absence-of-beat-power hold across four independent shots; exact frequencies are
less trustworthy (a ~2-cycle window cannot resolve 0.5 Hz).

**Design consequence:** take the *spatial* character from the reference
(horizontal thrust, level face, no tilt) and the *timing* from the music
(beat-locked). This is a deliberate departure. The reference gets away with drift
because cuts supply its rhythm; you have one continuous duck, and a physical
object moving off-tempo reads as broken.

---

## Choreography iterations

| | outcome |
|---|---|
| **v1** | Head craned so far back the face was invisible for most of the chicken beat. Read as malfunction. |
| **v2** | Chicken fixed; banana still hid the face (63° droop + camera above eye level). |
| **v3** | Face always visible — but 1.20:1 horizontal:vertical, 3.7% beat energy, 22°/s mean speed. Too slow, and an invented `head_roll` read as a head tilt. |
| **v4** | **Current.** 18.7% beat energy, 50°/s mean, `head_roll` removed entirely. One hit per beat. |

The rule that emerged, and the most transferable thing here:

> **The duck's entire expressiveness is the one dark lens. Any pose that hides
> the face kills the joke, however energetic it is.** The camera matters as much
> as the pose — from above, any downward head pose hides the face. Shoot at the
> duck's eye level.

### What failed, and why it was not a tuning problem

**Flattening the head's arc into horizontal travel.** Attempts got 1.20:1 →
1.26:1 against the reference's 10:1. The cause is kinematic, not parametric:
`head_pitch` sits *between* the neck and the face, so counter-rotating it to hold
the face level also swings the face backward, cancelling most of the neck's
forward push. With a two-link neck, holding orientation constant means the links
move as a parallelogram and the head travels on the **arc of the first link** —
forward *and down*, never purely forward.

**The reference's 10:1 is an artifact of 2D compositing and is not reproducible
on a physical neck.** A body-z lift to cancel the drop bought 0.06 of ratio and
was dropped (it is also the one command whose trained range is uncertain).

Note that the `big` variant had both the *most* travel (52.6 mm) and the *best*
ratio (1.80:1) precisely because it did **not** counter-rotate. Fighting for a
level face costs the thrust that makes the move read at all. Accept the arc.

---

## Deployment path

`docs/policy-manifest.md` and `docs/design/policy-channel-design.md` in
`pollen-robotics/microduck` (**not** this repo — that was a false memory worth
recording; there is also no `publish` entry point in this clone, only upstream).

**A `posture_flag` or `phase` policy cannot be published as a skill.**
`publish/manifest.py:146` hardcodes `"encoding": "constant"`, and the daemon
refuses non-constant encodings on `policy add`. A skill's twist is frozen for its
whole window (`control.rs`: `SkillPhase::Holding => d.command`), and `sit_toggle`
is latched and self-refusing mid-rise. So the *originally planned* architecture —
a host toggling a posture flag at ~1 Hz — has no supported path.

But this dance does not need one. Two **continuous public channels** exist:

| method | params | note |
|---|---|---|
| `robot.head` | `neck_pitch, head_pitch, head_yaw, head_roll` (rad) | continuous notification |
| `robot.pose` | `z, roll, pitch, active` | `active:false` snaps back to nominal |

JSON-RPC 2.0, NDJSON, over `/run/robotd.sock`. Served over BLE and WebRTC too.

Things that will bite:

- **`head_alpha` defaults to 0.2.** Simulated through the runtime's exact EMA,
  that costs **~40% of peak angular speed** (177 → 103 °/s) — it eats precisely
  the snap that makes this read as dancing. Set it to ~0.6.
  (Beat-*power* rises under the EMA, 18.7 → 21.7%; that is a *share* of total
  energy, not more punch. Trust the speed numbers.)
- **Deadman is 500 ms.** Stream continuously at 20–50 Hz. Do not send only on
  the beat.
- **Body block order is `z, roll, pitch`**, not `z, pitch, roll`.

---

## Harness bugs worth not repeating

Three mistakes made during this work, each of which produced confident garbage:

1. **`infer_policy.py:882` sets `model.opt.timestep = 0.005` inside `main()`.**
   Reusing `PolicyInference` without that inherits the XML's 0.002 and runs the
   policy at **125 Hz instead of the trained 50 Hz**. The duck collapsed and it
   looked like a policy failure. Also set the current limit (`--current-limit`
   defaults to 1.75 A, i.e. **on**).
2. **Open-loop `ctrl = DEFAULT_POSE` does not hold the robot up.** A head
   step-response test ran on a duck that had silently sagged from 125 mm to
   51 mm. This is exactly the settle-test trap AGENTS.md warns about: *check
   tilt and height, not just that the number looks plausible.*
3. **`atan2`-based roll wraps across ±180°**, reporting 355° of "head tilt" that
   did not exist.

General lesson: **render frames and look at them.** The face-visibility bug was
invisible in every numeric metric and obvious in a contact sheet.

---

## Open questions

- **Head command semantics: deltas or absolute?** The obs doc says head values
  ride in the command block (obs 51–55) and are not added to policy output, and
  in training that block is deltas from home — so deltas is almost certainly
  right. But `HeadParams`' docstring says "joint targets". If the head sits ~20°
  off on hardware, that is this; `--absolute-head` is the fix.
- **No audio sync.** The sequencer holds its own clock and will drift over a
  track. Alignment is by ear via `--start-delay`. Closing that loop is real work.
- **Peak neck command exceeds the joint's range** (+48.7° vs +40° available),
  deliberately relying on the documented undershoot. It is inside the *trained
  command* range (curriculum reaches ±63°), so the policy sees nothing unusual —
  the joint simply saturates. This is the regime v4 was tuned in. Not clamped, so
  as not to change the approved motion.
- **Everything timing-related is sim-only.** Expect real servos slower and
  softer; expect to re-tune amplitude and snap on hardware.

## If you want legs in it

*Then* build the trained environment. Copy `microduck_sitstand_env_cfg.py` —
it is the right template and carries the whole hand-ported DR / noise / delay /
NaN-guard stack plus contact-solver hardening (`nconmax=200`, iters 30/50).
Minimal changes: two pose-override dicts instead of `SITTING_TARGET_OVERRIDES`;
`ramp_s` from 2.0 down to ~0.3–0.5; dwell retimed around 0.92 s; delete
`descent_speed`/`rise_speed` and their curricula (keep `gentle_motion`); lower
`body_ang_vel`/`angular_momentum` (motion-blockers, and this is a dynamic task);
widen or drop `posture_stillness`. Symmetry stays **off** — the poses are
asymmetric.

Doing the command-driven version first is not wasted: you learn which poses read
well and which the servos cannot hit, which otherwise costs training runs to
discover.
