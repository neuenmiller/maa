"""Contract tests for maa.reconstruct — events back to log-intensity frames.

Run with:  pytest        (or: pytest tests/test_reconstruct.py)

STATUS: scaffolding only. The structure, fixtures and helpers below are
plumbing (Claude-written). Every expected value is still blank, marked
OWNER, because working them out is the point — see CLAUDE.md. Each blank
is one hand-computable number or array; fill them in and delete the skip.

Contract reconstruct() must satisfy (the draft signature; edit here first
if you changed it, then the tests below follow):

    reconstruct(xs, ys, ts, ps, *, sensor_shape, duration, threshold_c,
                out_fps, alpha=0.0) -> (n_frames, H, W) float

    log intensity, n_frames = round(duration * out_fps), frame f sampled at
    t = (f+1) / out_fps; alpha=0 is pure integration, alpha>0 adds the leak.
    Every frame is computed from the whole event history, so no state carries
    between frames.

    An event stamped exactly at a frame's sample time belongs to that frame.
    Events outside [0, duration] are dropped at both ends. A duration and
    out_fps whose product is not a whole number warns and rounds.
"""
import numpy as np
import pytest

reconstruct = pytest.importorskip(
    "maa.reconstruct", reason="maa/reconstruct.py not implemented yet"
).__dict__.get("reconstruct")

pytestmark = pytest.mark.skipif(
    reconstruct is None, reason="reconstruct() not defined yet"
)


# --- plumbing: a tiny event-stream builder --------------------------------
def stream(*events):
    """(x, y, t, p) tuples -> the four parallel arrays reconstruct() takes.

    Sorted by t, because every caller in the pipeline hands over a
    time-ordered stream and reconstruct() should be able to rely on it.
    """
    events = sorted(events, key=lambda e: e[2])
    xs, ys, ts, ps = (np.array(col) for col in zip(*events)) if events else (
        np.array([], dtype=np.int64), np.array([], dtype=np.int64),
        np.array([], dtype=float), np.array([], dtype=np.int8),
    )
    return xs, ys, ts, ps


def run(events, *, shape=(1, 3), duration=1.0, c=0.2, out_fps=4.0, alpha=0.0):
    """reconstruct() on one scenario, with the demo's keyword conventions."""
    return np.asarray(reconstruct(
        *stream(*events), sensor_shape=shape, duration=duration,
        threshold_c=c, out_fps=out_fps, alpha=alpha,
    ))


# --- shape and contract (these need no expected values) -------------------
def test_returns_expected_shape():
    out = run([(0, 0, 0.30, +1)], shape=(1, 3), duration=1.0, out_fps=4.0)
    assert out.shape == (4, 1, 3), (
        f"expected (round(duration*out_fps), H, W) = (4, 1, 3); got {out.shape}"
    )


def test_empty_stream_is_all_zero():
    out = run([])
    assert np.allclose(out, 0.0), (
        "with no events every pixel must stay at zero forever"
    )


def test_untouched_pixels_stay_zero():
    out = run([(0, 0, 0.30, +1)], shape=(1, 3))
    assert np.allclose(out[:, 0, 1:], 0.0), (
        "pixels that never received an event carry no information and must "
        "not move"
    )


def test_does_not_mutate_inputs():
    xs, ys, ts, ps = stream((0, 0, 0.30, +1), (1, 0, 0.60, -1))
    before = [a.copy() for a in (xs, ys, ts, ps)]
    reconstruct(xs, ys, ts, ps, sensor_shape=(1, 3), duration=1.0,
                threshold_c=0.2, out_fps=4.0)
    for a, b, name in zip((xs, ys, ts, ps), before, "xytp"):
        assert np.array_equal(a, b), f"reconstruct() modified the caller's {name} array"


# --- OWNER: expected values still to be worked out ------------------------
# Scenario A — one ON event, alpha = 0.
#   sensor 1x3, C = 0.2, initial = 0.0, out_fps = 4 (frames at t = 0, .25, .5, .75)
#   one event: pixel (x=0, y=0) ON at t = 0.30
#   Question each frame answers: has the event happened yet at this sample time?
def test_single_event_steps_once():
    """One ON event at t=0.30; frames at t = .25 .5 .75 1.0, alpha=0.

    Frame 0 samples before the event happened, so it is still empty. Every
    later frame includes it, and with alpha=0 nothing fades, so the value
    holds at +C rather than growing.
    """
    out = run([(0, 0, 0.30, +1)], shape=(1, 3), duration=1.0, c=0.2, out_fps=4.0)
    EXPECTED_A = [0.0, 0.2, 0.2, 0.2]
    assert np.allclose(out[:, 0, 0], EXPECTED_A)
    assert np.allclose(out[:, 0, 1:], 0.0), "no event reached the other two pixels"


# Scenario B — ON then OFF at the same pixel, alpha = 0.
#   the cancellation that makes moving edges reconstruct correctly
def test_on_then_off_cancels():
    """ON at t=0.30 then OFF at t=0.60 on one pixel, alpha=0.

    Frame 1 sees the ON alone and reads +C. Frames 2 and 3 see both and
    cancel to zero, which is why a moving edge reconstructs in its current
    position instead of smearing a trail behind it.

    Catches three mutations: a dropped polarity sign climbs to +0.4 rather
    than cancelling, and a plain `+=` in place of the scatter-add reads
    -0.2, because frames 2 and 3 each apply two events to the same pixel.
    """
    out = run([(0, 0, 0.30, +1), (0, 0, 0.60, -1)], shape=(1, 3), c=0.2, out_fps=4.0)
    EXPECTED_B = [0.0, 0.2, 0.0, 0.0]
    assert np.allclose(out[:, 0, 0], EXPECTED_B)


# Scenario C — an event landing exactly on a sample instant.
#   Does the frame at t = 0.25 include an event stamped t = 0.25, or not?
#   Either convention is defensible; the test pins whichever you choose.
def test_event_exactly_on_frame_boundary():
    """An event stamped exactly on a sample instant belongs to that frame.

    Not a corner case in this pipeline, the normal case: `simulate` stamps
    events at (k+1)/fps and demo.py reconstructs with out_fps == fps, so
    every event lands exactly on a sample instant. A `<` comparison here
    delays every event in the stream by one frame.
    """
    out = run([(0, 0, 0.25, +1)], shape=(1, 3), c=0.2, out_fps=4.0)
    EXPECTED_C = [0.2, 0.2, 0.2, 0.2]
    assert np.allclose(out[:, 0, 0], EXPECTED_C)


# Scenario D — the leak, alpha > 0, no events.
#   a pixel starting away from `initial` must decay toward it by
#   exp(-alpha*dt); pick an alpha where the arithmetic is clean
def test_leak_halves_every_frame():
    """alpha chosen so one frame interval is exactly one half-life.

    alpha = ln(2) * out_fps gives exp(-alpha * 1/out_fps) = 1/2, so the
    event's surviving weight halves each frame. Placing the event on frame
    0's sample instant means it enters at full strength, and every expected
    value is a power of one half, exact in binary floating point.

    This is the only case that exercises `alpha` at all: with the decay
    dropped it would read [1, 1, 1, 1].
    """
    out = run([(0, 0, 0.25, +1)], shape=(1, 3), c=1.0, out_fps=4.0,
              alpha=np.log(2) * 4)
    EXPECTED_D = [1.0, 0.5, 0.25, 0.125]
    assert np.allclose(out[:, 0, 0], EXPECTED_D)


# Scenario E — x and y must not be transposed.
def test_coordinates_are_not_transposed():
    """One event at x=2, y=1 on a non-square 3x4 sensor.

    Every other case puts its events at (0, 0), where a transposition is
    invisible. Here the correct target is [1, 2] and a swapped one is
    [2, 1]; both are in bounds on a 3-row, 4-column sensor, so the bug
    lands on the wrong pixel instead of raising IndexError. x must stay at
    or below 2 for that to hold.

    The event sits on frame 0's sample instant, so all four frames carry
    the value and the placement is asserted in every one of them.
    """
    out = run([(2, 1, 0.25, +1)], shape=(3, 4), c=0.2, out_fps=4.0)

    assert np.allclose(out[:, 1, 2], [0.2, 0.2, 0.2, 0.2]), "value is not at [y=1, x=2]"
    assert np.allclose(out[:, 2, 1], 0.0), "value landed at [2, 1] -- x and y are swapped"

    # nothing anywhere else
    expected = np.zeros((4, 3, 4))
    expected[:, 1, 2] = 0.2
    assert np.allclose(out, expected)


# --- the four cases the table did not cover -------------------------------
def test_coincident_events_both_apply():
    """Two events on one pixel at the same instant both count.

    `noise` produces these: background and hot-pixel timestamps are drawn
    from a continuous distribution, but jitter and the refractory sort can
    leave duplicates. Order is irrelevant because addition commutes.

    This is the strongest scatter-add test in the suite. Both events are ON,
    so a plain `+=` answers 0.2 where the spec says 0.4, rather than the
    cancelling pair in test_on_then_off_cancels where the error is subtler.
    """
    out = run([(0, 0, 0.30, +1), (0, 0, 0.30, +1)], shape=(1, 3), c=0.2, out_fps=4.0)
    assert np.allclose(out[:, 0, 0], [0.0, 0.4, 0.4, 0.4])


def test_event_after_duration_is_dropped():
    """An event past the last sample instant never appears.

    Jitter pushes events off the end of the recording; demo.py already
    counts and reports them. The library stays silent about it, matching
    how `noise` behaves.
    """
    out = run([(0, 0, 1.50, +1)], shape=(1, 3), duration=1.0, c=0.2, out_fps=4.0)
    assert np.allclose(out, 0.0), "an event beyond duration must not be integrated"


def test_event_before_zero_is_dropped():
    """An event stamped before the recording started never appears.

    Same cause as the case above, other end: jitter is Gaussian and can
    push an early event negative. The recording window is [0, duration],
    and anything outside it is out of window at both ends.
    """
    out = run([(0, 0, -0.10, +1)], shape=(1, 3), duration=1.0, c=0.2, out_fps=4.0)
    assert np.allclose(out, 0.0), "an event before t=0 must not be integrated"


def test_non_integer_frame_count_warns_and_rounds():
    """duration * out_fps must be whole; if not, warn and round.

    0.7 * 4 = 2.8, so the caller asked for something impossible. Rounding
    to 3 frames puts the last sample at 0.75s against a 0.7s recording,
    which is worth saying out loud rather than doing quietly.

    The tolerance matters: demo.py computes duration = T/fps and passes
    out_fps = fps, and (T/fps)*fps is not always exactly T in floating
    point. At 25 fps with 7 frames it is 7.000000000000001. That must not
    warn, so the check needs slack of about 1e-9 rather than exact
    equality.
    """
    with pytest.warns(UserWarning):
        out = run([], shape=(1, 3), duration=0.7, out_fps=4.0)
    assert len(out) == 3, "should still return a rounded frame count, not raise"


def test_float_noise_in_frame_count_does_not_warn():
    """The companion to the case above: legitimate input must stay silent."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning becomes a failure
        out = run([], shape=(1, 3), duration=7 / 25.0, out_fps=25.0)
    assert len(out) == 7