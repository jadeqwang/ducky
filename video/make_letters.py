#!/usr/bin/env python3
"""Generate props_micro.xml — the "MICRO" letters in a Pixar-style serif face.

    uv run video/make_letters.py

Writes ../src/mjlab_microduck/robot/microduck/props_micro.xml.

Why a generator: the serif face needs real elliptical bowls (C, O, R), which
are 20-odd tangential box segments each with a rotation and a thickness that
varies around the curve. Hand-authoring that XML is miserable and unreviewable;
the geometry is much easier to read as the code that produces it.

The face, approximating the Pixar wordmark:
  - heavy weight, cap height 0.20 m (~80% of the duck's 0.2511 m stand height)
  - STROKE CONTRAST: verticals thick, horizontals thin. This is the single
    biggest thing that separates a serif face from the plain slab boxes this
    file used to hold, so the bowls modulate thickness by |cos(theta)|.
  - wedge serifs at every stroke terminal

Everything lives in the Y-Z plane (thin along X) so it reads as text from a
camera on -X. The film camera puts +Y on screen LEFT, so the word runs from
+Y to -Y and letter pitch is negative going right.
"""

import math
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "../src/mjlab_microduck/robot/microduck/props_micro.xml")

CAP = 0.10          # half cap-height; letters span z in [-CAP, +CAP]
DEPTH = 0.015       # half-thickness along X (into the screen)
DEPTH_I = 0.026     # half-thickness of the I only — see letter_I()
THICK = 0.0135      # half-width of a thick (vertical) stroke
THIN = 0.0070       # half-width of a thin (horizontal) stroke
SERIF_W = 0.029     # half-width of a serif across the stroke
SERIF_H = 0.0075    # half-height of a serif along the stroke
PITCH = 0.23        # letter-to-letter spacing along Y


def box(name, y, z, hy, hz, rot=0.0, depth=None):
    """A box in the Y-Z plane. rot is a rotation about X, in radians."""
    e = f' euler="{rot:.4f} 0 0"' if abs(rot) > 1e-9 else ""
    return (f'            <geom class="letter" name="{name}" '
            f'pos="0 {y:.4f} {z:.4f}" '
            f'size="{depth or DEPTH} {hy:.4f} {hz:.4f}"{e}/>')


def bar(name, y0, z0, y1, z1, half_w):
    """A stroke between two points, drawn as one rotated box.

    The box's LOCAL Z runs along the stroke, so a rotation `a` about X sends
    local z to (-sin a, cos a) in (y, z) and we solve that for the direction.
    """
    dy, dz = y1 - y0, z1 - z0
    length = math.hypot(dy, dz)
    a = math.atan2(-dy, dz)
    return box(name, (y0 + y1) / 2, (z0 + z1) / 2, half_w, length / 2 + half_w * 0.4, a)


def serif(name, y, z, half_w=SERIF_W, half_h=SERIF_H, rot=0.0, depth=None):
    return box(name, y, z, half_w, half_h, rot, depth)


def arc(prefix, cy, cz, ry, rz, t0, t1, n, thick=THICK, thin=THIN, flare=1.0):
    """An elliptical arc as n tangential box segments.

    Segments are placed between consecutive sample points, so the ellipse's
    varying arc length is handled for free. Stroke half-width interpolates
    thick<->thin by |cos(theta)|: theta=0/pi are the left and right flanks of
    the bowl (vertical strokes, thick), theta=+-pi/2 are top and bottom (thin).
    """
    geoms = []
    pts = []
    for i in range(n + 1):
        t = math.radians(t0 + (t1 - t0) * i / n)
        pts.append((cy + ry * math.cos(t), cz + rz * math.sin(t), t))
    for i in range(n):
        y0, z0, ta = pts[i]
        y1, z1, tb = pts[i + 1]
        tm = (ta + tb) / 2
        w = thin + (thick - thin) * abs(math.cos(tm))
        if i in (0, n - 1):
            w *= flare          # flared terminals, as a serif face has
        dy, dz = y1 - y0, z1 - z0
        length = math.hypot(dy, dz)
        a = math.atan2(-dy, dz)
        geoms.append(box(f"{prefix}{i}", (y0 + y1) / 2, (z0 + z1) / 2,
                         w, length / 2 * 1.25, a))
    return geoms


# ----------------------------------------------------------------- letters
def letter_M():
    g = []
    g.append(box("M_sl", 0.070, 0.0, THICK, CAP))
    g.append(box("M_sr", -0.070, 0.0, THICK, CAP))
    # Diagonals from the inner top of each stem down to the middle of the
    # baseline. They meet just above the baseline, as a serif M's vertex does.
    g.append(bar("M_dl", 0.058, 0.093, 0.0, -0.050, THIN * 1.35))
    g.append(bar("M_dr", -0.058, 0.093, 0.0, -0.050, THIN * 1.35))
    for nm, y in (("M_fl", 0.070), ("M_fr", -0.070)):
        g.append(serif(nm, y, -CAP + SERIF_H))
    # Top serifs sit outboard only: the diagonals already fill the inner side.
    g.append(serif("M_tl", 0.070 + 0.010, CAP - SERIF_H, half_w=SERIF_W * 0.72))
    g.append(serif("M_tr", -0.070 - 0.010, CAP - SERIF_H, half_w=SERIF_W * 0.72))
    return g


def letter_I():
    # The hero letter: the one the duck punts. It is DEEPER along X than the
    # rest (DEPTH_I), which is free: every camera that matters looks straight
    # down +X, so extra depth is invisible from the front. It is also the
    # difference between a boot and a whiff. Measured: the kicking foot's
    # closest approach is 3.9 cm from the letter's axis at 3.3 cm height, so a
    # face sitting 1.5 cm out (the standard letter depth) gets grazed for two
    # control frames or missed outright, and standing the duck closer just has
    # it lean on the letter instead of swinging at it.
    g = [box("I_stem", 0.0, 0.0, THICK * 1.15, CAP, depth=DEPTH_I)]
    g.append(serif("I_top", 0.0, CAP - SERIF_H, half_w=SERIF_W * 1.05,
                   depth=DEPTH_I))
    g.append(serif("I_foot", 0.0, -CAP + SERIF_H, half_w=SERIF_W * 1.05,
                   depth=DEPTH_I))
    return g


def letter_C():
    # Opens toward -Y (screen right), so the gap in the ring sits at theta=180.
    g = arc("C_a", 0.0, 0.0, 0.075, CAP - THIN, -143, 143, 22, flare=1.5)
    return g


def letter_R():
    g = [box("R_stem", 0.065, 0.0, THICK, CAP)]
    # Bowl: the half of an ellipse that bulges toward -Y (screen RIGHT), from
    # the stem top round to mid-height. theta 90 -> 270 passes through 180,
    # which is the -Y flank; sweeping 90 -> -90 instead mirrors the letter.
    g += arc("R_b", 0.065, 0.048, 0.085, 0.052, 90, 270, 14)
    # Leg, from under the bowl out to the baseline.
    g.append(bar("R_leg", 0.030, -0.004, -0.075, -CAP + 0.012, THIN * 1.5))
    g.append(serif("R_foot", 0.065, -CAP + SERIF_H))
    g.append(serif("R_top", 0.065 + 0.008, CAP - SERIF_H, half_w=SERIF_W * 0.7))
    g.append(serif("R_lf", -0.078, -CAP + SERIF_H, half_w=SERIF_W * 0.7))
    return g


def letter_O():
    return arc("O_a", 0.0, 0.0, 0.075, CAP - THIN, 0, 360, 28)


# name, builder, y position, density, free?
# ONLY the I gets a freejoint. The others are welded to the world, because a
# free-standing letter is not actually stable: the C is asymmetric about its
# opening and topples onto its own back within a second of the sim starting,
# every take, without anything touching it (measured: 15.2 cm of "movement"
# identical across four unrelated punt settings). A toppled C also breaks the
# payoff -- the whole joke needs M_CRO to still read as a word. Nothing in the
# shot requires them to move, so the simplest fix is that they cannot.
LETTERS = [("M", letter_M, 2 * PITCH, 180, False),
           ("I", letter_I, 1 * PITCH, 45, True),
           ("C", letter_C, 0 * PITCH, 180, False),
           ("R", letter_R, -1 * PITCH, 180, False),
           ("O", letter_O, -2 * PITCH, 180, False)]

HEADER = '''<mujoco model="microduck_props_micro">
    <!-- Props for the "MICRO" title shot.  GENERATED by video/make_letters.py —
         edit that, not this file.

         Same pattern as ball.xml: free-floating bodies with their own
         <freejoint>, included into a scene next to the robot. Kept in this
         directory because robot_allcollisions.xml declares meshdir="assets"
         relative to it.

         Sizing (duck stands 0.2511 m at the STAND keyframe):
           letters       0.20 m tall  = ~80% of duck height, 0.23 m pitch
           rubber duck   0.20 m tall  = same as the letters

         The face is a heavy serif with stroke contrast and wedge serifs (see
         make_letters.py). Letters live in the Y-Z plane, thin along X, so they
         read as text from a camera on -X; +Y is screen LEFT, so the word runs
         from +Y to -Y.

         Letter body origins sit at z=0.10 — the letter's mid-height — so each
         rests on the floor.

         density=180 puts a letter at ~40 g. The I is 70 (~15 g), the mass the
         kick policy was trained against.  NOTE: mass is nearly irrelevant to
         how far the kicked I travels — a 0.20 m letter struck at ankle height
         topples rather than sliding, and the ~19 cm it moves is just the fall.
         shoot.py adds an explicit punt impulse on contact; see its PUNT block.

         Only the I is a free body; see LETTERS in make_letters.py.

         Freejoints are deliberately NOT named ball_free: infer_policy.py's
         _place_ball() teleports a joint by that name in front of the kicking
         foot, which would yank a letter out of the word on every kick. -->

    <default>
        <default class="letter">
            <geom type="box" rgba="0.95 0.83 0.25 1" density="180"
                  friction="0.7 0.01 0.001"/>
        </default>
    </default>

    <worldbody>
'''

# The rubber duck now stands in line with the word, one pitch after the O, as
# if it were the next character.
DUCK = f'''
        <!-- ============ rubber duck ============
             In line with the word, one letter-pitch after the O, facing the
             film camera (-X) so it reads as the last character of "MICRO_".
             0.20 m tall. Light and hollow, like the real thing. -->
        <body name="rubber_duck" pos="0 {-3 * PITCH:.3f} 0.0" euler="0 0 3.1416">
            <freejoint name="rubber_duck_free"/>
            <geom type="ellipsoid" name="rd_body" pos="0 0 0.060"
                  size="0.060 0.050 0.055" rgba="1.0 0.85 0.10 1" density="150"
                  friction="0.9 0.02 0.002"/>
            <geom type="sphere" name="rd_head" pos="0.030 0 0.155"
                  size="0.045" rgba="1.0 0.85 0.10 1" density="150"/>
            <geom type="box" name="rd_beak" pos="0.088 0 0.150"
                  size="0.028 0.012 0.009" rgba="1.0 0.45 0.05 1" density="150"/>
            <geom type="box" name="rd_tail" pos="-0.070 0 0.098"
                  size="0.030 0.010 0.022" rgba="1.0 0.85 0.10 1" density="150"
                  euler="0 -0.6 0"/>
            <geom type="sphere" name="rd_eye_l" pos="0.055 0.028 0.170"
                  size="0.008" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
            <geom type="sphere" name="rd_eye_r" pos="0.055 -0.028 0.170"
                  size="0.008" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
        </body>
    </worldbody>
</mujoco>
'''


def main():
    parts = [HEADER]
    for name, fn, y, density, free in LETTERS:
        note = ""
        if name == "I":
            note = ("\n             The one the duck punts; the duck then stands "
                    "in the gap.")
        parts.append(f'        <!-- ================= {name} ================={note} -->')
        parts.append(f'        <body name="letter_{name}" pos="0 {y:.3f} {CAP:.3f}">')
        if free:
            parts.append(f'            <freejoint name="letter_{name}_free"/>')
        geoms = fn()
        if density != 180:
            geoms = [g.replace('class="letter"',
                               f'class="letter" density="{density}"') for g in geoms]
        parts.extend(geoms)
        parts.append("        </body>\n")
    parts.append(DUCK)
    xml = "\n".join(parts)
    with open(OUT, "w") as f:
        f.write(xml)
    n = sum(len(fn()) for _, fn, _, _, _ in LETTERS)
    print(f"wrote {os.path.normpath(OUT)}  ({n} letter geoms)")


if __name__ == "__main__":
    main()
