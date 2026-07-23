"""Sanity check for maa.simulate — SoA event-stream contract.

Plumbing (Claude-written). Same hand-computed 3-pixel scenario as before,
now checking the sparse struct-of-arrays output instead of dense polarity
maps. This test is the CONTRACT your simulate() rewrite must satisfy:

    simulate(frames, fps, threshold_c) -> (x, y, t, p)

    four 1-D NumPy arrays of equal length N (one entry per event):
      x : column index      — integer dtype (uint16 recommended)
      y : row index         — integer dtype (uint16 recommended)
      t : timestamp, sec    — float dtype   (float64 recommended)
      p : polarity, +1/-1   — signed int    (int8 recommended)

    timestamp convention: an event detected on the transition from
    frame k to frame k+1 is stamped t = (k + 1) / fps — the time of
    the frame where the crossing was OBSERVED.

    ordering: nondecreasing t. Order within one transition is yours.

Red until you rewrite simulate(); that's the point.

    python tests/test_simulate.py
"""
import numpy as np

from maa.simulate import simulate

# Three 1x3 frames — pixels A, B, C (intensity in [0,1]):
#   A: 0.3 -> 0.3 -> 0.3   constant            -> never fires
#   B: 0.2 -> 0.4 -> 0.4   doubles once, then steady
#   C: 0.4 -> 0.2 -> 0.2   halves once, then steady
frames = np.array([
    [[0.3, 0.2, 0.4]],
    [[0.3, 0.4, 0.2]],
    [[0.3, 0.4, 0.2]],
])

# A doubling is log(2) ~= 0.69 in log-space, comfortably above this threshold.
C = 0.5

# fps=2.0 deliberately: makes t = 0.5, which catches both "forgot to divide
# by fps" (t=1) and "multiplied instead" (t=2). fps=1.0 would hide either bug.
FPS = 2.0

# Worked out by hand:
#   transition 0 (frames 0->1, observed at frame 1, t = 1/2.0 = 0.5):
#     A steady -> nothing;  B doubled -> ON at (x=1,y=0);  C halved -> OFF at (x=2,y=0)
#   transition 1 (frames 1->2): nothing changed -> NO events. This is the
#     reset test: B and C are still bright/dark vs frame 0, but their
#     reference moved when they fired, so they must not fire again.
EXPECTED = [  # (x, y, t, p), sorted by (t, x)
    (1, 0, 0.5, +1),
    (2, 0, 0.5, -1),
]


def fail(msg):
    print(f"\nFAIL  — {msg}")
    raise SystemExit(1)


result = simulate(frames, fps=FPS, threshold_c=C)

# --- shape of the return value -------------------------------------------
if isinstance(result, np.ndarray) and result.ndim == 3:
    fail("got a dense (T-1, H, W) polarity map — that's the old contract.\n"
         "New contract: return four 1-D arrays (x, y, t, p), one entry per event.")
if not (isinstance(result, (tuple, list)) and len(result) == 4):
    fail(f"expected a 4-tuple (x, y, t, p); got {type(result).__name__}.")

x, y, t, p = (np.asarray(a) for a in result)

for name, arr in (("x", x), ("y", y), ("t", t), ("p", p)):
    if arr.ndim != 1:
        fail(f"{name} should be 1-D, got shape {arr.shape}.")
lengths = {"x": len(x), "y": len(y), "t": len(t), "p": len(p)}
if len(set(lengths.values())) != 1:
    fail(f"the four arrays must have equal length, got {lengths}.")

# --- dtypes ---------------------------------------------------------------
if x.dtype.kind not in "ui" or y.dtype.kind not in "ui":
    fail(f"x and y should be integer dtypes (got x:{x.dtype}, y:{y.dtype}) — "
         "they are array indices; floats can't index.")
if t.dtype.kind != "f":
    fail(f"t should be a float dtype (got {t.dtype}) — timestamps in seconds.")
if p.dtype.kind != "i":
    fail(f"p should be a SIGNED integer dtype (got {p.dtype}) — it holds -1.")

# --- ordering -------------------------------------------------------------
if len(t) and np.any(np.diff(t) < 0):
    fail("t must be nondecreasing — events come out in time order.")

# --- values ---------------------------------------------------------------
order = np.lexsort((x, t))          # sort by (t, x); within-transition order is free
got = list(zip(x[order].tolist(), y[order].tolist(),
               t[order].tolist(), p[order].tolist()))

print(f"got {len(got)} events: {got}")
print(f"expected            : {EXPECTED}")

if len(got) != len(EXPECTED):
    fail(f"expected exactly {len(EXPECTED)} events, got {len(got)}. "
         "3 or more? Suspect the reset. 1 or fewer? Suspect the threshold test.")
for (gx, gy, gt, gp), (ex, ey, et, ep) in zip(got, EXPECTED):
    if (gx, gy, gp) != (ex, ey, ep) or not np.isclose(gt, et):
        fail(f"event mismatch: got {(gx, gy, gt, gp)}, expected {(ex, ey, et, ep)}.")

print("\nPASS  — sparse (x, y, t, p) stream: ON on brighten, OFF on darken, "
      "timestamps in seconds, no re-fire after reset.")
