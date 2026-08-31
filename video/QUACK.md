# Adding a Microduck quack to sim videos

**Updated 2026-08-31.** The note in `video/README.md` saying audio must be
handled manually in post is now incomplete. MuJoCo still has no acoustic or
audio-output simulation, but Pollen has published the real Microduck's
open-source voice synthesizer. We can use the authentic synthesized voice and
mux it into the rendered video automatically.

## What upstream now provides

The official [`pollen-robotics/microduck`](https://github.com/pollen-robotics/microduck)
repository contains a Rust `sounds` crate. The real robot uses it for
`robotctl quack` and derives a stable voice personality from the robot's SoC
serial. It synthesizes sounds rather than distributing prerecorded WAV files.

Available sound tags are:

```text
alarm  greet  inquire  peck  chirp  coo  wheee
```

`chirp` is the ordinary mouth-triggered quack and has 12 variants. A numeric
seed controls the duck's voice personality; a variant adds small variation
within that same voice. Output is 48 kHz, 16-bit mono WAV.

Relevant upstream sources:

- [Microduck voice documentation](https://github.com/pollen-robotics/microduck/blob/main/docs/robot/cheatsheet.md#the-voice)
- [`sounds` crate](https://github.com/pollen-robotics/microduck/tree/main/sounds)
- [Sound tags and WAV renderer](https://github.com/pollen-robotics/microduck/blob/main/sounds/src/lib.rs)
- [`sounds` command-line interface](https://github.com/pollen-robotics/microduck/blob/main/sounds/src/main.rs)

## Render a quack

From a checkout of the official runtime repository:

```bash
cargo run -p sounds -- render chirp quack.wav --seed 42 --variant 0
```

Changing `--seed` creates a different duck personality. Keep the seed fixed if
the duck should retain the same voice across videos. Valid `chirp` variants are
currently `0` through `11`.

Do not use the CLI's default hardware-derived seed on a workstation if
reproducibility matters: without an explicit seed it derives one from the local
machine identity.

## Recommended integration with `shoot.py`

This is not sound propagation inside MuJoCo. It is deterministic soundtrack
generation synchronized to the simulated performance:

1. Keep rendering frames exactly as `video/shoot.py` does now.
2. Record the output-video timestamp when the `quack` beat begins (or when its
   head-nod trigger fires).
3. Generate or reuse a cached official `chirp` WAV for the requested voice seed
   and variant.
4. Build an audio track matching the video duration, placing the WAV at the
   recorded timestamp.
5. Mux that audio track with the silent MP4.

A useful eventual CLI would be:

```bash
uv run video/shoot.py \
  --out video/micro.mp4 \
  --quack \
  --voice-seed 42 \
  --quack-variant 3
```

The project already depends on `imageio-ffmpeg`. Its packaged FFmpeg executable
can do the final mux, so a system-wide `ffmpeg` installation is not required.
In Python, obtain its path with:

```python
import imageio_ffmpeg

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
```

For the simplest first implementation, render the silent MP4 to a temporary
path, create a temporary WAV containing silence plus the positioned chirp, and
invoke the packaged FFmpeg executable with the video stream copied and audio
encoded as AAC. Replace the requested output only after muxing succeeds.

## Animation caveat

The local simulation models expose 14 actuated joints and no movable beak. The
speaker mesh in the MJCF is visual geometry only. Keep the existing head nod as
the visible quack gesture unless a separate cinematic model with a beak joint
is introduced.

This limitation currently resembles the hardware behavior: upstream still has
an open request for [`robotctl quack` to open the mouth](https://github.com/pollen-robotics/microduck/issues/153).
Adding a cinematic-only beak joint must not change the 14-action policy or the
61D observation contract.

## Bottom line

There is still no native audio path in MuJoCo, but there is now an authentic,
seedable Microduck quack generator. The right implementation is to synchronize
the official `chirp` synth with the sim's quack beat and automatically mux it
into the rendered movie.
