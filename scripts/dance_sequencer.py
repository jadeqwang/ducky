#!/usr/bin/env python3
"""Beat-locked dance sequencer for Microduck — drives the *stock* standing policy.

No trained dance policy is involved. `alpha_stand.onnx` (an official policy, and
the robot's default stand slot) holds balance; this script streams head and body
pose commands at the control rate, and the dance falls out.

    robot.head  {neck_pitch, head_pitch, head_yaw, head_roll}   radians
    robot.pose  {z, roll, pitch, active}                        m / radians

Both are *continuous intents*: JSON-RPC 2.0 notifications (no id, no reply), one
object per line (NDJSON), over robotd's unix socket. See the microduck repo,
docs/design/robotd-design.md.

STDLIB ONLY, deliberately: this runs ON the robot, where the training venv does
not exist. `scp` it across and run it with the system python.

Usage
-----
    # print the wire traffic without a robot (works anywhere)
    python3 dance_sequencer.py --dry-run --duration 4

    # on the robot
    python3 dance_sequencer.py --bpm 130

Before running on hardware
--------------------------
Raise `head_alpha` from its 0.2 default, or the runtime's low-pass eats the
snap that makes this read as dancing (measured: ~40% of peak angular speed):

    robotctl configure control.head_alpha 0.6

The choreography ("v4")
-----------------------
One thrust per beat, poses alternating every two beats. Tuned in sim against
the CHICKEN BANANA reference (130 BPM), with three findings baked in:

  * The head must never rotate far enough to hide the face. The duck's whole
    expressiveness is the one dark lens; a pose that hides it reads as
    malfunction however energetic it is. Nothing here exceeds ~15 deg of head
    pitch excursion.
  * No head_roll. An ear-to-shoulder tilt was an earlier invention of ours and
    appears nowhere in the reference; it reads as "broken neck", not "dance".
  * Every beat gets a hit. An earlier version let the second pose be a slow
    melt with no beat component and it read as sluggish. A physical object
    moving off-tempo reads as broken, where a 2D cutout drifting slowly reads
    as stylish -- so this deliberately does NOT copy the reference's timing,
    which measured only 1-5% of its motion energy at the beat frequency.

Deliberately dropped: the body-z lift. It was meant to flatten the head's arc
into horizontal travel and did essentially nothing (1.20:1 -> 1.26:1, against
the reference's ~10:1), because a two-link neck holding the face level moves
the head on an arc by construction. It is also the one command whose trained
range is uncertain, so it is not worth the out-of-distribution risk.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import sys
import time

# Home pose for the two pitch joints, radians (microduck_constants.HOME_FRAME).
# Only used with --absolute-head; see the note on head command semantics below.
HOME_NECK_PITCH = 0.3491
HOME_HEAD_PITCH = 0.3491

DEFAULT_SOCKET = "/run/robotd.sock"
CONTROL_HZ = 50.0


def ease_out(x: float) -> float:
    """Fast departure, soft arrival — the shape of a snap."""
    x = max(0.0, min(1.0, x))
    return 1.0 - (1.0 - x) ** 3


def choreography(t: float, beat: float) -> tuple[float, float, float, float, float]:
    """v4. Returns (neck_pitch, head_pitch, head_yaw, head_roll, body_pitch).

    Head values are DELTAS from the home pose (see --absolute-head).

    The thrust: neck bends forward while head_pitch counter-rotates back, so the
    head travels forward with the face held roughly level. Amplitudes are
    commanded large on purpose — the neck servo tracks only ~50-60% of command
    under load (the documented head droop), so asking for 1.05 rad yields
    something closer to 0.55.
    """
    beat_phase = (t % beat) / beat
    pose_index = int(t // (2.0 * beat))
    chicken = (pose_index % 2) == 0

    # Snap out over ~11% of a beat, then decay back — one hit per beat.
    hit = ease_out(min(1.0, beat_phase / 0.11)) * math.exp(-beat_phase * 2.4)
    amp = 1.05 if chicken else 0.72

    neck_pitch = amp * hit
    head_pitch = -0.78 * amp * hit          # counter-rotate: keep the face level

    # Pose character, established once per two-beat phrase and then held.
    phrase_phase = t % (2.0 * beat)
    settle = ease_out(min(1.0, phrase_phase / 0.13))
    head_yaw = (0.34 if chicken else -0.30) * settle
    head_roll = 0.0                          # never tilt; see module docstring
    body_pitch = (-0.02 if chicken else 0.15) * settle

    return neck_pitch, head_pitch, head_yaw, head_roll, body_pitch


class Link:
    """robotd's NDJSON JSON-RPC socket, or stdout in --dry-run."""

    def __init__(self, path: str | None, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.sock: socket.socket | None = None
        if dry_run:
            return
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.connect(path)
        except OSError as exc:
            raise SystemExit(
                f"could not connect to {path}: {exc}\n"
                "Is robotd running? Use --dry-run to test the choreography "
                "without a robot."
            ) from exc

    def notify(self, method: str, params: dict) -> None:
        """A notification: no `id`, no reply. Continuous intents are sent this way."""
        line = json.dumps(
            {"jsonrpc": "2.0", "method": method, "params": params},
            separators=(",", ":"),
        )
        if self.dry_run:
            sys.stdout.write(line + "\n")
            return
        assert self.sock is not None
        self.sock.sendall(line.encode("utf-8") + b"\n")

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stream a beat-locked dance to a Microduck running the stock stand policy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--bpm", type=float, default=130.0, help="tempo (default: 130)")
    ap.add_argument("--socket", default=DEFAULT_SOCKET,
                    help=f"robotd socket (default: {DEFAULT_SOCKET})")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the NDJSON to stdout instead of sending; needs no robot")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds to dance (default: until Ctrl-C)")
    ap.add_argument("--rate", type=float, default=CONTROL_HZ,
                    help=f"send rate in Hz (default: {CONTROL_HZ:g}). Must stay well "
                         "above 2 Hz: the intent deadman zeroes commands after 500 ms")
    ap.add_argument("--start-delay", type=float, default=0.0,
                    help="seconds to hold the neutral pose before the first beat, so you "
                         "can line the downbeat up with the music by ear")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale every commanded amplitude (default: 1.0). Start around "
                         "0.5 on real hardware and work up")
    ap.add_argument("--absolute-head", action="store_true",
                    help="send head values as absolute joint targets rather than deltas "
                         "from home. See the semantics note below — default (deltas) is "
                         "almost certainly correct; use this if the head sits ~20 deg off")
    args = ap.parse_args()

    if args.bpm <= 0:
        ap.error("--bpm must be positive")
    if args.rate < 10:
        ap.error("--rate below 10 Hz risks tripping the 500 ms deadman")

    beat = 60.0 / args.bpm
    dt = 1.0 / args.rate

    if not args.dry_run:
        sys.stderr.write(
            f"dancing at {args.bpm:g} BPM (beat {beat*1000:.0f} ms), "
            f"streaming {args.rate:g} Hz to {args.socket}\n"
            "reminder: raise control.head_alpha (default 0.2 blunts the snap)\n"
        )

    link = Link(None if args.dry_run else args.socket, args.dry_run)

    stopping = False

    def on_signal(signum, frame):  # noqa: ARG001
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    def send(neck, head, yaw, roll, body_pitch, active=True):
        if args.absolute_head:
            neck += HOME_NECK_PITCH
            head += HOME_HEAD_PITCH
        link.notify("robot.head", {
            "neck_pitch": round(neck, 5), "head_pitch": round(head, 5),
            "head_yaw": round(yaw, 5), "head_roll": round(roll, 5),
        })
        # PoseParams field order is z, roll, pitch. z stays 0 — see module docstring.
        link.notify("robot.pose", {
            "z": 0.0, "roll": 0.0, "pitch": round(body_pitch, 5), "active": active,
        })

    start = time.monotonic()
    next_tick = start
    try:
        while not stopping:
            now = time.monotonic()
            elapsed = now - start
            if args.duration is not None and elapsed >= args.duration + args.start_delay:
                break

            t = elapsed - args.start_delay
            if t < 0.0:
                send(0.0, 0.0, 0.0, 0.0, 0.0)      # hold neutral, keep deadman fed
            else:
                neck, head, yaw, roll, body_pitch = choreography(t, beat)
                s = args.scale
                send(neck * s, head * s, yaw * s, roll * s, body_pitch * s)

            next_tick += dt
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()        # fell behind; resync rather than spiral
    finally:
        # Return to neutral and release body-pose mode. `active: false` snaps the
        # body back to nominal rather than gliding, which is the intended exit.
        for _ in range(3):
            send(0.0, 0.0, 0.0, 0.0, 0.0, active=False)
            time.sleep(dt)
        link.close()
        if not args.dry_run:
            sys.stderr.write("\nstopped; head and body returned to neutral\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
