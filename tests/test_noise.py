"""Contract tests for maa.noise.jitter — timestamp jitter.

Run with:  pytest tests/test_noise.py

The CASES are the owner's spec (which properties must hold); the pytest
scaffolding is plumbing (Claude-written).

    jitter(xs, ys, ts, ps, rng, sigma) -> (xs, ys, ts, ps)

    Adds zero-mean Gaussian noise of width `sigma` to every timestamp.
    Only t changes; x, y, p pass through untouched. Pure — the caller's
    input arrays are never modified. Does NOT sort: the single sort lives
    in the noise() orchestrator, so nothing here asserts time order.
"""
import numpy as np
import pytest

from maa.noise import jitter

# --- knobs: Claude-chosen, owner to review --------------------------------
SIGMA = 0.01       # seconds; small next to a typical frame gap
SEED = 20260725
N_STAT = 10_000    # events used for the residual-distribution check


@pytest.fixture
def rng():
    """A fresh seeded generator per test — deterministic, no cross-test bleed."""
    return np.random.default_rng(SEED)


@pytest.fixture
def stream():
    """Four events with distinctive x/y/p, so "untouched" is provable."""
    xs = np.array([3, 7, 1, 9], dtype=np.uint16)
    ys = np.array([2, 5, 8, 4], dtype=np.uint16)
    ts = np.array([0.10, 0.20, 0.30, 0.40])
    ps = np.array([1, -1, 1, -1], dtype=np.int8)
    return xs, ys, ts, ps


def test_xyp_pass_through_untouched(stream, rng):
    """Row 1: jitter perturbs time only — coordinates and polarity survive."""
    xs, ys, ts, ps = stream
    out_x, out_y, _, out_p = jitter(xs, ys, ts, ps, rng, SIGMA)
    assert np.array_equal(out_x, xs), "x must pass through unchanged"
    assert np.array_equal(out_y, ys), "y must pass through unchanged"
    assert np.array_equal(out_p, ps), "p must pass through unchanged"


def test_event_count_preserved(stream, rng):
    """Row 2: jitter moves events in time; it never adds or drops any."""
    xs, ys, ts, ps = stream
    out = jitter(xs, ys, ts, ps, rng, SIGMA)
    for name, arr in zip("xytp", out):
        assert len(arr) == len(ts), (
            f"{name} has {len(arr)} events, input had {len(ts)}"
        )


def test_input_arrays_not_mutated(stream, rng):
    """Row 3: purity. The caller's clean stream must survive the call.

    This is the `ts += noise` bug: in-place addition would overwrite the
    caller's timestamps, destroying the clean ground truth that simulate()
    produced. `ts + noise` returns a new array and leaves the input alone.
    """
    xs, ys, ts, ps = stream
    before = ts.copy()
    jitter(xs, ys, ts, ps, rng, SIGMA)
    assert np.array_equal(ts, before), (
        "jitter mutated the caller's ts array in place — it must return a new one"
    )


def test_timestamps_actually_move(stream, rng):
    """Row 4: with sigma > 0, the timestamps must genuinely change."""
    xs, ys, ts, ps = stream
    _, _, out_t, _ = jitter(xs, ys, ts, ps, rng, SIGMA)
    assert not np.array_equal(out_t, ts), "sigma > 0 but no timestamp moved"


@pytest.mark.parametrize("seed", [0, 1, 20260725])
def test_same_seed_gives_identical_output(stream, seed):
    """Row 5: reproducibility. Same seed in, same stream out, every time.

    Catches a jitter that reaches for global np.random instead of the rng
    it was handed — that would be unseeded, so two runs would diverge.
    """
    xs, ys, ts, ps = stream
    first = jitter(xs, ys, ts, ps, np.random.default_rng(seed), SIGMA)
    second = jitter(xs, ys, ts, ps, np.random.default_rng(seed), SIGMA)
    for name, a, b in zip("xytp", first, second):
        assert np.array_equal(a, b), f"{name} differed between two same-seed runs"


def test_sigma_zero_is_a_no_op(stream, rng):
    """Row 6: zero noise width means zero noise — timestamps exactly unchanged."""
    xs, ys, ts, ps = stream
    _, _, out_t, _ = jitter(xs, ys, ts, ps, rng, 0.0)
    assert np.array_equal(out_t, ts), "sigma=0 must leave timestamps exactly as-is"


def test_empty_stream(rng):
    """Row 7: zero events in, zero events out, no crash."""
    empty = (
        np.array([], dtype=np.uint16),
        np.array([], dtype=np.uint16),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.int8),
    )
    out = jitter(*empty, rng, SIGMA)
    for name, arr in zip("xytp", out):
        assert len(arr) == 0, f"{name} should be empty, got {len(arr)} entries"


def test_residuals_are_zero_mean_gaussian(rng):
    """Row 8: the noise itself is N(0, sigma).

    Checked on the RESIDUALS (out_t - in_t), not on out_t: the output
    timestamps inherit the input's spread, so only the difference isolates
    the noise. Mean ~0 catches a uniform draw (which would sit at +0.5*range);
    std ~sigma catches a wrong scale.

    Tolerances: the mean of N samples has standard error sigma/sqrt(N), so
    4x that is a generous band; std gets 5% relative. The seed is fixed, so
    this test is deterministic — it cannot flake.
    """
    ts = np.linspace(0.0, 1.0, N_STAT)
    zeros = np.zeros(N_STAT, dtype=np.uint16)
    ps = np.ones(N_STAT, dtype=np.int8)

    _, _, out_t, _ = jitter(zeros, zeros, ts, ps, rng, SIGMA)
    residuals = out_t - ts

    standard_error = SIGMA / np.sqrt(N_STAT)
    assert abs(residuals.mean()) < 4 * standard_error, (
        f"residual mean {residuals.mean():.2e} is too far from 0 — "
        "is the noise zero-mean (normal) rather than uniform?"
    )
    assert np.isclose(residuals.std(), SIGMA, rtol=0.05), (
        f"residual std {residuals.std():.4f} != sigma {SIGMA} — wrong noise scale"
    )
