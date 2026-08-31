# Running the Microduck sim interactively

Driving the duck live in a MuJoCo viewer with the keyboard. For rendering scripted videos instead, see [`video/README.md`](video/README.md).

Environment is already set up (`uv sync` done, `uv` on `$PATH`, official policies in `policies/`). All commands run from the repo root:

```bash
cd /home/jade/Documents/ducky
```

## Quickstart

Everything loaded — walk, stand, sit/stand, ground pick, both kicks, roulade:

```bash
uv run scripts/infer_policy.py \
    --walking policies/alpha_walking.onnx \
    --standing policies/alpha_stand.onnx \
    --sitstand policies/alpha_sitstand.onnx \
    --ground-pick policies/alpha_ground_pick.onnx \
    --kick-left policies/ball_kick_left.onnx \
    --kick-right policies/ball_kick_right.onnx \
    --roulade policies/roulade.onnx \
    --new-cmd-obs
```

Just walking:

```bash
uv run scripts/infer_policy.py --walking policies/alpha_walking.onnx --new-cmd-obs
```

Roller skating (different robot model — passive wheels under the feet):

```bash
uv run scripts/infer_policy.py --roller --walking policies/roller.onnx --new-cmd-obs
```

In a Claude Code session, prefix with `!` to run it in the terminal directly:
`! uv run scripts/infer_policy.py --walking policies/alpha_walking.onnx --new-cmd-obs`

## Three things that will trip you up

- **`--new-cmd-obs` is required** for the official policies. They use the unified 13-D command block (61-D observation: 48 proprioception + `[twist(3), head_pose(4), body_pose(6)]`). Without the flag you get the legacy 51-D layout and a shape mismatch.
- **Keys are read from the terminal, not the viewer window.** The viewer's own keypresses fire MuJoCo's built-in visualization shortcuts, so the script reads raw stdin instead. Keep focus on the terminal, and note it **needs a real TTY** — piped or non-interactive runs print `WARNING: stdin is not a TTY — keyboard control disabled` and the duck just stands there.
- **Loading either kick policy silently switches the scene** to `scene_ball.xml` (that is where the ball comes from). Drop `--kick-left`/`--kick-right` if you want the plain floor.

## Controls

Keys are modal: **H** and **B** swap what the arrows and A/E do.

### Velocity mode (default)

| Key | Action |
|---|---|
| ↑ / ↓ | forward / backward |
| ← / → | strafe left / right (**roller:** turn left / right, incremental) |
| A / E | turn left / right |
| Space | stop — zero all commands |
| T | toggle policy inference (paused = motors hold last target) |
| P | random shove, for robustness testing |
| Q | quit |

Note the arrows **snap to the limit** rather than accelerating gradually — ↑ sets forward to `vel_max_x` outright. The in-app help says "increase lin_vel_x", which reads as incremental; only the roller turn keys actually step. Defaults: `lin_vel_x ±0.3`, `lin_vel_y ±0.2`, `ang_vel_z ±1.5`.

> `ang_vel_z ±1.5` is **outside the trained range** of ±1.0 (`microduck_velocity_env_cfg.py`). Holding a hard turn puts the policy out of distribution and it tends to spin in place instead of walking.

### Behaviours

| Key | Action | Needs |
|---|---|---|
| G | ground pick — crouch and touch the floor with the beak | `--ground-pick` |
| Y | toggle sit ↔ stand | `--sitstand` (or `--sit` / `--slope`) |
| K | kick with **left** foot | `--kick-left` |
| L | kick with **right** foot | `--kick-right` |
| R | roulade — forward roll over the head | `--roulade` |

Kicks and roulade are episodic: they run from a standing start, then hand control back automatically (`--kick-duration`, default 3 s; `--roulade-duration`, default 2 s).

### Head mode — press **H**

| Key | Axis |
|---|---|
| ↑ / ↓ | head pitch |
| ← / → | head yaw |
| A / E | head roll |
| Z / S | neck pitch |
| Space | reset head to zero |

Trained joint caps: neck/head pitch ±1.1, head yaw ±1.4, head roll ±0.31 rad.

### Body-pose mode — press **B**

| Key | Axis | Max |
|---|---|---|
| ↑ / ↓ | Δz (height) ±10 mm | ±30 mm |
| ← / → | Δpitch ±10° | ±30° |
| A / E | Δroll ±10° | ±30° |
| Z / S | Δyaw ±10° (`--new-cmd-obs` only) | ±30° |
| Space | reset body pose to zero |

## Useful flags

| Flag | Default | What it does |
|---|---|---|
| `--debug` | off | print observations and actions each step |
| `--save-csv FILE` | — | log every observation + action to CSV |
| `--record FILE` | — | save observations to a pickle on Ctrl+C |
| `--action-scale` | 1.0 | scale policy output |
| `--delay MIN MAX` | off | simulate actuator command delay |
| `--current-limit` | 1.75 | XL330 current limit [A] → torque ceiling. **Leave it on** — the policies were trained against this saturation |
| `--foot-friction` | model | override foot sliding friction (emulate the grippy real sole) |
| `--switch-threshold` | 0.05 | command magnitude at which walking ↔ standing swaps |
| `--lin-vel-x/y`, `--ang-vel-z` | 0 | initial velocity command |

## Running on a custom scene

**There is no `--scene` flag.** The path is a module-level constant at `scripts/infer_policy.py:21`, and the only branches are roller and ball:

```python
MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene.xml"
```

To walk around the MICRO letter set interactively, point that line at `scene_micro.xml`. (Adding a real flag is a small patch worth doing if this comes up often.)

## Other ways to run

- **Your own trained checkpoint** (GPU, mjlab side, no wandb account needed):
  ```bash
  uv run play Mjlab-Velocity-Flat-MicroDuck --checkpoint-file logs/rsl_rl/velocity/<run>/model_<N>.pt
  ```
  Note `play` only *watches* — the environment samples its own commands and the mouse just orbits the camera. It is not a driving interface.
- **Zero install** — the [official browser sandbox](https://huggingface.co/spaces/pollen-robotics/microduck-simulator) runs the same policies at 50 Hz with gamepad support.
- **Training** (defaults to wandb, which blocks on login):
  ```bash
  uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 --agent.logger tensorboard
  ```
  Smoke test first, per `AGENTS.md`: `--env.scene.num-envs 64 --agent.max-iterations 5`.
