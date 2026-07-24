"""Contract tests for maa.simulate — sparse (x, y, t, p) event stream.

Run with:  pytest        (or: pytest tests/test_simulate.py)

The scenario and the expected events below are the SPEC — hand-computed by
the owner. Everything else here is plumbing (Claude-written): the same
assertions as before, reorganised into pytest test functions so failures
localise and the suite runs under one `pytest` invocation.

Contract simulate() must satisfy:
    simulate(frames, fps, threshold_c) -> (x, y, t, p)
    four equal-length 1-D arrays; x, y integer; t float (seconds); p signed
    int (+1/-1); an event on the frame k -> k+1 transition is stamped
    t = (k+1)/fps; output nondecreasing in t.
"""
import numpy as np
import pytest

from maa.simulate import simulate

# --- the spec (owner's) ---------------------------------------------------
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
#   transition 1 (frames 1->2): nothing changed -> NO events. The reset test:
#     B and C are still bright/dark vs frame 0, but their reference moved when
#     they fired, so they must not fire again.
EXPECTED = [  # (x, y, t, p), sorted by (t, x)
    (1, 0, 0.5, +1),
    (2, 0, 0.5, -1),
]


# --- plumbing (Claude-written) --------------------------------------------
@pytest.fixture
def events():
    """simulate() on the spec scenario, unpacked to four numpy arrays."""
    result = simulate(frames, fps=FPS, threshold_c=C)
    assert isinstance(result, (tuple, list)) and len(result) == 4, (
        "simulate() must return a 4-tuple (x, y, t, p); "
        f"got {type(result).__name__}"
        + (f" of length {len(result)}" if hasattr(result, "__len__") else "")
        + ". A dense (T-1, H, W) map is the old contract."
    )
    return tuple(np.asarray(a) for a in result)


def test_returns_four_1d_arrays_of_equal_length(events):
    x, y, t, p = events
    for name, arr in (("x", x), ("y", y), ("t", t), ("p", p)):
        assert arr.ndim == 1, f"{name} should be 1-D, got shape {arr.shape}"
    assert len(x) == len(y) == len(t) == len(p), (
        f"arrays must be equal length, got "
        f"x:{len(x)} y:{len(y)} t:{len(t)} p:{len(p)}"
    )


def test_dtypes(events):
    x, y, t, p = events
    assert x.dtype.kind in "ui", f"x must be integer (indices), got {x.dtype}"
    assert y.dtype.kind in "ui", f"y must be integer (indices), got {y.dtype}"
    assert t.dtype.kind == "f", f"t must be float (seconds), got {t.dtype}"
    assert p.dtype.kind == "i", f"p must be signed int (holds -1), got {p.dtype}"


def test_time_is_nondecreasing(events):
    _, _, t, _ = events
    assert np.all(np.diff(t) >= 0), "events must come out in nondecreasing t order"


def test_matches_hand_computed_events(events):
    x, y, t, p = events
    order = np.lexsort((x, t))  # sort by (t, x); within-transition order is free
    got = list(zip(x[order].tolist(), y[order].tolist(),
                   t[order].tolist(), p[order].tolist()))
    # exact-count check doubles as the reset test: 3+ events => reset failed;
    # 1 or fewer => threshold test failed.
    assert len(got) == len(EXPECTED), (
        f"expected {len(EXPECTED)} events, got {len(got)}: {got}"
    )
    for (gx, gy, gt, gp), (ex, ey, et, ep) in zip(got, EXPECTED):
        assert (gx, gy, gp) == (ex, ey, ep) and np.isclose(gt, et), (
            f"event mismatch: got {(gx, gy, gt, gp)}, expected {(ex, ey, et, ep)}"
        )
