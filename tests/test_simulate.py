"""Sanity check for maa.simulate — hand-made frames with known events.

Plumbing (Claude-written). Builds a tiny synthetic clip whose correct event
output is worked out by hand, runs it through YOUR simulate(), and checks the
result. No real video needed.

    python tests/test_simulate.py
"""
import numpy as np

from maa.simulate import simulate

# Three 1x3 frames — three pixels, call them A, B, C (values are intensity in [0,1]):
#   A: 0.3 -> 0.3 -> 0.3   constant            -> never fires
#   B: 0.2 -> 0.4 -> 0.4   doubles once, then steady
#   C: 0.4 -> 0.2 -> 0.2   halves once, then steady
frames = np.array([
    [[0.3, 0.2, 0.4]],
    [[0.3, 0.4, 0.2]],
    [[0.3, 0.4, 0.2]],
])

# A doubling is log(2) ~= 0.69 in log-space, comfortably above this threshold;
# a halving is -0.69, comfortably below.
C = 0.5

events = simulate(frames, fps=1.0, threshold_c=C)

# Worked out by hand:
#   transition 0 (frame0 -> frame1): A steady = 0,  B doubled = +1,  C halved = -1
#   transition 1 (frame1 -> frame2): nothing changed -> all 0
# That second row is the real test of the RESET: B and C are still bright/dark,
# but their reference was moved when they fired, so they must NOT fire again.
expected = np.array([
    [[0, 1, -1]],
    [[0, 0,  0]],
], dtype=np.int8)

print("shape   :", events.shape, " (expected (2, 1, 3))")
print("dtype   :", events.dtype)
print("got     :", events.tolist())
print("expected:", expected.tolist())

passed = events.shape == expected.shape and np.array_equal(events, expected)
if passed:
    print("\nPASS  — ON on brighten, OFF on darken, no event on steady, no re-fire after reset.")
else:
    print("\nFAIL  — output doesn't match. Suspects: line ordering, the reset, or the thresholds.")
    raise SystemExit(1)
