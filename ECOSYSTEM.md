# Microduck Ecosystem Survey

**Surveyed 2026-08-31.** Microduck launched 2026-08-27, so nearly everything below is days old and moving fast. Re-check before assuming any gap still exists.

Purpose of this file: avoid rebuilding things that already exist. Scope of the survey was ~75 GitHub repos (`q=microduck`) plus the Hugging Face Hub (9 models, 9 spaces, 4 datasets).

## The one structural fact

**Nobody outside Pollen has hardware.** Pre-orders opened 2026-08-27; units ship ~December 2026. Every "hardware" tool in the ecosystem is written against published design docs and mock transports — none has been exercised on a shipped unit.

Consequence: **sim2real validation is currently impossible for everyone**, which is the actually-hard part of this stack. Anything claiming hardware validation today is claiming too much. Both community awesome-lists label such entries *sim-only*.

## Official

| Thing | Where | Notes |
|---|---|---|
| Robot software | [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) | 4638★. `robotd` 50 Hz loop, `mediad` camera/WebRTC, `padd` gamepad, updater. No simulator. |
| RL training envs | [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) | 1107★. mjlab (MuJoCo Warp) + PPO, BAM actuators, backlash, DR. **This is what we cloned.** |
| **Pretrained policies** | [pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies) | 9 ONNX, Apache-2.0. **Went live 2026-08-31.** |
| Browser sandbox | [Space: microduck-simulator](https://huggingface.co/spaces/pollen-robotics/microduck-simulator) | 244♥. MuJoCo WASM + onnxruntime-web, real policies at 50 Hz, gamepad. Zero install. |
| GStreamer plugins | [microduck-gst-plugins](https://github.com/pollen-robotics/microduck-gst-plugins) | Prebuilt aarch64 Rockchip MPP encoders. |
| Product / store / press | [pollen-robotics.com/microduck](https://pollen-robotics.com/microduck) | $399. |

> **Both awesome-lists are stale on one point.** They state policy distribution "is not yet upstream" and that community catalogs are browse-only, citing roadmap milestone M8. The `microduck-policies` Hub repo was published 2026-08-31 — the 9 official ONNX policies are downloadable now.

The 9 shipped policies, all `obs[1,61] -> actions[1,14]` (verified locally):

```
alpha_walking  alpha_stand  alpha_sitstand  alpha_ground_pick
ball_kick_left  ball_kick_right  roulade  roller  roller_crouch
```

Note what is **not** shipped: no `VelStand` (walk + fall-recovery in one policy) and no `StandUp` (get up off the floor). Those tasks exist in `microduck_rl` but you must train them yourself.

## Saturated — do not rebuild

### Alternative simulator ports (4 independent efforts)
- [kabilankb/isaaclab-microduck](https://github.com/kabilankb/isaaclab-microduck) — Isaac Lab 3.0 / Newton MJWarp. Locomotion, running, ball-kick, two-duck rally. A/B'd vs mjlab.
- [5usu/IsaacLab (microduck-port)](https://github.com/5usu/IsaacLab/tree/5usu/microduck-port) — Isaac Lab extension, BAM/backlash actuators, RSL-RL, walking/kicking/parkour.
- [Macmachi/microduck-rl-genesis](https://github.com/Macmachi/microduck-rl-genesis) — Genesis port for **AMD/ROCm**. Actuator model validated bit-exact vs upstream. Ships flat/rough/backlash ONNX.
- [APX103/mjx_microduck](https://github.com/APX103/mjx_microduck) — MJX (JAX) + Brax PPO from scratch, with ONNX export.

### Training workspaces / scaffolding (5+)
Wrappers around `microduck_rl` with bootstrap scripts, pinned submodules, smoke tests, eval:
[jvpflum/microduck-lab](https://github.com/jvpflum/microduck-lab) (DGX Spark) ·
[x10zyn/microduck-sim-playground](https://github.com/x10zyn/microduck-sim-playground) ·
[lgtkgtv/microduck_sim](https://github.com/lgtkgtv/microduck_sim) (custom PPO, 6-phase curriculum) ·
[AlexandreEDMOND/microduck-rl-lab](https://github.com/AlexandreEDMOND/microduck-rl-lab) (retrains 5 skills into an obstacle course) ·
[DollhouseRobotics/microduck-miniverse](https://github.com/DollhouseRobotics/microduck-miniverse) (policies as checksummed bundles)

⚠️ **Our own setup falls in this category.** It is scaffolding, not a contribution — fine to have, not worth publishing.

### Agent / MCP control layers (6+)
- [joeynyc/microduck-mcp](https://github.com/joeynyc/microduck-mcp) — mock/sim/Unix-socket/SSH transports behind one toolset + safety layer.
- [aj-dev-smith/microduck-mcp](https://github.com/aj-dev-smith/microduck-mcp) — CPU MuJoCo sim, rendered camera frames as tool output, agent-experience debug page.
- [rokbenko/quackd](https://github.com/rokbenko/quackd) — 83★. LLM goal-planner, `.duck` task files, safety rules, bundled sim, on PyPI.
- [rangerchaz/meckie-duck-gateway](https://github.com/rangerchaz/meckie-duck-gateway) — WebRTC/JSON-RPC session re-exposed as local HTTP.
- [OpenCastor](https://docs.opencastor.com/robots/microduck/) · [Strands Robots](https://strands-labs.github.io/robots/policies/microduck/) — third-party frameworks.
- [agentculture/microduck-cli](https://github.com/agentculture/microduck-cli) — agent-agnostic CLI.

### Policy catalogs & conformance (3)
[uDuck Registry](https://uduck-registry.pages.dev/) ([repo](https://github.com/ob1-s/uduck-registry)) ·
[microduck-miniverse](https://github.com/DollhouseRobotics/microduck-miniverse) ·
[craigm26/microduck-policy-golden-vectors](https://huggingface.co/datasets/craigm26/microduck-policy-golden-vectors) — SHA256-pinned obs→action vectors for testing custom runners.

### Community skills / trained policies
backflip ([Lulzx](https://github.com/Lulzx/microduck-backflip), safety-gated) ·
courier/pick-carry-place ([selinayfilizp](https://github.com/selinayfilizp/microduck-courier)) ·
vertical jump ([Liyucheng1997](https://github.com/Liyucheng1997/318_lab-microduck-simulator), [live](https://duck.liyucheng.me)) ·
sidekick dance ([pezzonovante7](https://github.com/pezzonovante7/microduck-sidekick-dance), task only) ·
step-up + head-brake ([bihaokun](https://github.com/bihaokun/microduck-step-up-policy)) ·
flamingo ([RemiFabre](https://huggingface.co/RemiFabre/microduck-flamingo-cycle), 9♥) ·
polite bow & moonwalk ([fffiloni](https://huggingface.co/fffiloni/microduck-polite-bow-b1d864)) ·
electric slide ([Histochemichael](https://huggingface.co/datasets/Histochemichael/microduck-electric-slide-motion), 25-duck show) ·
running ([HannesVonEssen](https://huggingface.co/HannesVonEssen/microduck-running))

### Ports, apps, hardware
[craigm26/duckkit](https://github.com/craigm26/duckkit) (pure Swift runtime) ·
[IronSpiderMan/MicroDuckModels](https://github.com/IronSpiderMan/MicroDuckModels) (Three.js 3D viewer) ·
[multimodalart/microduck-ar](https://huggingface.co/spaces/multimodalart/microduck-ar) (WebXR/AR) ·
[fanhao375/microduck-replica](https://github.com/fanhao375/microduck-replica) (92★, reverse-engineered CAD + electronics) ·
[SaberOnGo/open-microduck](https://github.com/SaberOnGo/open-microduck) ·
[ScrapMeta/microduck-diy](https://github.com/ScrapMeta/microduck-diy)

**Hardware is not open source** — only the software is Apache-2.0. The replica projects reverse-engineer CAD from the MJCF and Rust sources.

### Curated lists (2, overlapping)
- [joeynyc/awesome-microduck](https://github.com/joeynyc/awesome-microduck) — 32★, better structured, has an "Ecosystem Status" honesty section.
- [ob1-s/awesome-microduck](https://github.com/ob1-s/awesome-microduck) — broader on community software.

Both are actively maintained. **Check these before starting anything** — cheaper than re-running this survey.

## Thin areas (as of 2026-08-31)

Genuine gaps, in rough order of how underserved they look:

1. **Perception.** Almost everything is locomotion. The camera and 8×8 ToF depth sensor are barely used outside Pollen's own [pet_detect](https://github.com/apirrone/microduck_pet_detect) and [maploc_rs](https://github.com/apirrone/microduck_maploc_rs) (both by Open Duck Mini's creator). No community vision-conditioned policies.
2. **Cinematics / scripted scenes.** No tooling for staged multi-object scenes, choreographed rollouts, or rendered video. Everything renders either the bare checker-floor scene or a browser sandbox. ← *this is what we're building*
3. **Reward-design / DR ablation studies.** `AGENTS.md` contains hard-won reward rules and DR conventions; nobody has published systematic ablations backing them up.
4. **Multi-duck interaction.** One entry only (kabilankb's two-duck rally).
5. **Non-locomotion manipulation.** The beak gripper is a 15th motor absent from the 14-DoF sim models; almost no work on grasping beyond `ground_pick`.

## Local setup notes

Our clone lives at `/home/jade/Documents/ducky` (branch `develop`), env via `uv sync`, verified on RTX 3090.

- `train` defaults to `--agent.logger wandb` → **blocks on login**. Use `--agent.logger tensorboard`.
- `play` accepts `--checkpoint-file`, so no wandb account is needed to view a policy.
- Smoke test before any long run (per `AGENTS.md`): `--env.scene.num-envs 64 --agent.max-iterations 5`.
- Official policies downloaded to `policies/`.
- Scene XMLs must live in `src/mjlab_microduck/robot/microduck/` — `meshdir="assets"` is resolved relative to that directory.
