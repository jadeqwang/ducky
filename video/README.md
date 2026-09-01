# Filming scripted scenes in sim

> **Audio update:** Pollen has now published the real Microduck's seedable
> voice synthesizer. See [`QUACK.md`](QUACK.md) for how to render an authentic
> quack and synchronize it with a sim video.

```bash
uv run video/shoot.py --out video/micro.mp4      # render the mp4
uv run video/shoot.py --stomp-i --out video/micro-stomp.mp4  # jump onto the I
uv run video/shoot.py --no-quack                 # render silently
uv run video/shoot.py --quack-wav voice.wav      # use a custom/official voice
uv run video/shoot.py --no-cinematic-mouth       # keep the rendered beak closed
uv run video/shoot.py --out /tmp/b.mp4 --stills  # one PNG per beat (fast framing check)
uv run video/shoot.py --cam film_word            # force one camera for every beat
uv run video/make_letters.py                     # regenerate the MICRO letters
```

Renders offscreen, so no viewer window and no display needed. The current cut
is about 24 s. MP4 renders include a synchronized real mallard quack by default;
its source and CC BY-SA attribution are in
[`duck-quack-SOURCE.md`](duck-quack-SOURCE.md). `--quack-wav` substitutes any
preferred WAV, including an official Microduck `chirp` rendered as described
in [`QUACK.md`](QUACK.md).

## The shot

The duck sprints in across the front of a "MICRO" title, catches a toe, goes
down face-first, picks itself up, approaches the word from the front, punts the
**I** away through the row, steps into the gap, and turns its head to the lens.

`--stomp-i` makes an alternate take: after the same opening, the duck crouches,
jumps onto the top of the **I**, knocks it over, recovers, and rejoins the same
step-in/head-turn/quack payoff. The shipped policies do not include a learned
jump, so the standing policy poses the crouch and landing while a single root
velocity impulse supplies the ballistic flight. A small angular nudge at the
descending top-face crossing guarantees that either landing direction topples
the I; all subsequent contact and recovery remain physical.

## The letters

`props_micro.xml` is **generated** — edit `make_letters.py` and re-run it. The
face is a heavy serif with real elliptical bowls and stroke contrast (thick
verticals, thin horizontals), which is what makes it read as type rather than
as boxes. Bowls are built as ~20 tangential box segments each, which is why
this is code and not hand-written XML.

Two things about the letters that are load-bearing:

- **Only the `I` is a free body. M/C/R/O and the rubber duck are welded to the
  world.** A free-standing letter is not stable: the C is asymmetric about its
  opening and topples onto its own back within a second of the sim starting,
  every take, untouched. The rounded rubber duck similarly rolls onto its side.
  That also breaks the payoff, which needs the title props to stay readable.
- **The `I` is deeper along X than the other letters** (`DEPTH_I`). This is free
  — every camera that matters looks straight down +X, so depth is invisible from
  the front — and it is the difference between a boot and a whiff. See below.

## How to add objects

MJCF, exactly the pattern `ball.xml` uses for the kick ball:

1. **Write a props file** with free-floating bodies — one `<body>` per object, each with its own `<freejoint>`, geoms built from primitives (`box`, `sphere`, `ellipsoid`, `cylinder`, `capsule`) or a `<mesh>` asset. See `../src/mjlab_microduck/robot/microduck/props_micro.xml`.
2. **Write a scene** that `<include>`s the robot and the props, plus lights and cameras. See `scene_micro.xml`.
3. **Point `SCENE` in `shoot.py`** at it.

Five things that will bite:

- **Put scene/prop XML in `src/mjlab_microduck/robot/microduck/`.** `robot_allcollisions.xml` declares `meshdir="assets"` relative to that directory; include it from elsewhere and mesh loading breaks.
- **No `<keyframe>` in a scene with props.** Every freejoint adds 7 to `nq`, which invalidates `scene.xml`'s keyframe `qpos` sizes. Set the duck's pose programmatically instead (`Shoot.place`).
- **Don't name a prop's freejoint `ball_free`.** `infer_policy.py`'s `_place_ball()` teleports a joint by that name in front of the kicking foot on every kick.
- **A freejoint is not free.** Give a prop one only if it needs to move; anything with a narrow or asymmetric footprint will fall over on its own before the take starts.
- **Mass barely matters for kicks.** It does not do what you would expect — see the punt section.

Sizing reference: the duck stands **0.2511 m** at the `STAND` keyframe.

## How the duck is driven

`shoot.py` reuses `PolicyInference` from `scripts/infer_policy.py` rather than rebuilding the 61-D observation by hand — that layout is a hard invariant shared by every policy (see `AGENTS.md`).

Beats are **closed-loop**: each advances on a predicate over the duck's actual pose, with a timeout backstop. Open-loop timings drift badly because achieved motion never matches the command. A beat may also set `freeze: True`, which stops running the policy for its duration and lets the servos hold their last target — physics still steps. That is what keeps the duck lying on the floor after the wipeout instead of instantly scrambling upright.

### Gait facts measured on the shipped `alpha_walking.onnx`

These drove every control decision in `shoot.py`, and are worth knowing before choreographing anything new:

| Property | Measured |
|---|---|
| Forward speed | **≈0.47× commanded** — 0.25 cmd → 0.12 m/s real |
| Turn rate | **≈20°/s** at the trained max, so a 180° spin takes ~9 s |
| Command deadband | Commands below **~0.2 produce no motion at all** |
| Lateral drift | ~0.25 m sideways per 12 s of walking |
| Turning drift | ~0.12 m of translation per in-place turn |
| Sprint ceiling | a sustained trunk push reaches **~0.73 m/s**, ~3× the gait's own top speed |

Consequences baked into the script:

- **A plain P-controller stalls.** Its output decays into the deadband, leaving ~10° of yaw or ~7 cm of position error forever. Every command floors its magnitude once outside a dead zone (`YAW_FLOOR`, `POS_FLOOR`).
- **Yaw commands stay inside ±1.0**, the trained range. Commanding 1.5 puts the policy out of distribution and it turns in place instead of walking.
- **`goto()` steers by bearing** (self-correcting for cross-track drift); **`creep()` drives body-frame velocity** to strafe onto a mark while holding a heading; **`dash()` steers without ever throttling back, for the sprint.** Use `creep` for anything where the final heading matters — `goto` rotates the duck toward its target and flips 180° on the slightest overshoot.
- **Square up before fine placement**, since turning drifts ~0.12 m off the mark.

### Kick and payoff staging

The kick now happens from the **front** of the title: the duck stands at x < 0,
faces +X, and punts the I away through the row. The final payoff is still a
head-only turn to camera, but `head_yaw` is capped at ±1.4 rad (~80°), so the
duck first turns its body after the kick and then steps into the gap at the
payoff heading. This keeps the kick route direct without sacrificing the final
look.

`head_to_cam()` measures against the duck's *actual* heading every tick, not the
staged payoff heading, because the kick and subsequent body turn introduce
variable yaw drift and a baked constant aims the look off into space.

## The scripted effects

The scripted effects are deliberate cheats, documented in code, and exist where
the available policies or physics genuinely will not do the thing:

- **The punt** (`Shoot.punt`). The kick cannot launch a letter. A 0.20 m letter
  struck at ankle height rotates instead of translating, so it topples and lands
  ~19 cm away — and that 19 cm is insensitive to the letter's mass (15 g vs
  40 g), its floor friction (a 35× sweep changed *nothing*; it never slides) and
  its centre-of-mass height. It is just the letter falling over. A punt needs an
  upward component the swing cannot produce, so the shot adds one, fired at the
  boot's **closest approach** to the letter and launched along a fixed
  `PUNT_YAW`. Tune with `--punt-vx/-vz/-spin`; `--punt-vx 0` disables it and
  gives you the honest 19 cm topple.
- **The trip** (`Shoot.trip`). This policy sheds a 0.73 m/s sprint on a plain
  reverse command and stays upright, so braking does not fell it. A forward
  impulse big enough to (≈0.9 m/s) skates the duck a metre out of frame. Instead
  the stumble adds a small forward nudge plus a nose-down pitch rate — catching a
  toe — and it face-plants near where it was running. `--trip-vel`,
  `--trip-pitch`.
- **The stomp jump** (`Shoot.jump` / `Shoot.stomp`, `--stomp-i`). There is no
  jump policy in the shipped set. A one-shot root velocity creates the leap;
  tune it with `--jump-vx/-vz`. As the descending duck crosses the I's top face,
  `stomp` nudges the letter away from the approach side so it cannot balance
  under the duck; `--stomp-spin` controls that topple.

## Four bugs worth not rediscovering

- **`use_projected_gravity=True` is mandatory.** `PolicyInference.__init__` defaults it to `False`, which feeds the policy the raw accelerometer. That reads ≈ gravity while standing still and diverges as soon as the duck accelerates — so the duck stands fine and then collapses about a second into any walk.
- **Apply the XL330 current limit** (`--current-limit`, default 1.75 A). `infer_policy.py` applies it by default and the policies were trained against that torque ceiling.
- **Trigger a kick only from a settled stand.** The kick policies were trained to run "from a standing start with an all-zero command". Fired straight out of a creep, while the duck still has stride momentum, the swing changes enough to miss: the boot's closest approach to the letter is **8.8 cm** mid-stride versus **3.7 cm** from the same mark after a second of standing. Hence the `settle` beat — do not delete it.
- **The kick mark's tolerance is sharply asymmetric.** Standing 2 cm *short* is a clean whiff; 1–1.5 cm *long* still connects. `KICK_STANDOFF` is therefore biased 1 cm closer than the trained ball offset (which was calibrated against a small sphere, not a flat-faced letter), and the final `creep` overrides the usual 15 mm position deadband to hit the mark to ~1 cm.

## Cameras

All defined in `scene_micro.xml`. `xyaxes="0 -1 0  0 0 1"` looks along +X and
puts +Y on screen **left**, so the word reads M I C R O left to right.

| Camera | Used for |
|---|---|
| `film_run` | the sprint — close enough that the duck is bigger than the letters behind it |
| `film_fall` | the wipeout, sprawl and standup |
| `film_high` | unused high 3/4 master, retained for alternate edits |
| `film_low` | low hero angle on the kick |
| `film_word` | **the payoff frame** — the whole word with the duck standing in the I's gap |
| `film_close` | the head turn and quack. `CLOSE_CAM` in `shoot.py` must match its position |
| `film` | unused wide master, kept for `--cam` overrides |

## Known rough edges

- The post-fall approach remains entirely on the camera side of the title. If a
  retake drifts across x=0 before reaching the mark, add a front-side waypoint
  rather than restoring the old route behind the letters.
- `approach_front` runs at 100× while only the duck's shadow is in the
  `film_word` frame, then returns to 4× as soon as the body enters. The policy
  still gets its full closed-loop setup time in simulation.
- The final one-second plant, kick, and 0.8-second punt follow-through play at
  1× so the I's launch is readable; the longer payoff turn then runs at 2×.
- No jaw actuator exists in the 14-DoF physics model (the beak is the real
  robot's 15th motor), so `shoot.py` animates the lower visual beak during the
  head nod as a filming-only effect. It is enabled by default and synchronized
  with the quack audio; `--no-cinematic-mouth` disables it.
- MuJoCo has no native audio path. Sound must be synchronized and muxed into the rendered movie, but `imageio-ffmpeg` supplies an FFmpeg executable, so a system-wide install is unnecessary.
- `--seed` exists but the gait is only weakly stochastic; retakes look very similar. Vary `--sprint-accel` / `--trip-pitch` for different wipeouts.
