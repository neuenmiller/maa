"""Contract tests for maa.noise — one section per effect.

Run with:  pytest tests/test_noise.py

The CASES are the owner's spec (which properties must hold); the pytest
scaffolding is plumbing (Claude-written).

    jitter(xs, ys, ts, ps, rng, sigma) -> (xs, ys, ts, ps)

    Adds zero-mean Gaussian noise of width `sigma` to every timestamp.
    Only t changes; x, y, p pass through untouched. Pure — the caller's
    input arrays are never modified. Does NOT sort: the single sort lives
    in the noise() orchestrator, so nothing here asserts time order.

    background(xs, ys, ts, ps, rng, *, bg_rate, sensor_shape, duration)
        -> (xs, ys, ts, ps)

    Appends spurious "leak" events: a Poisson count of about
    bg_rate * H * W * duration, scattered uniformly over the sensor and
    the time window, all ON (+1). Incoming events pass through unchanged
    at the front of the stream. Also does NOT sort.
"""
import numpy as np
import pytest

from maa.noise import background, jitter

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


# =========================================================================
# background — spurious leak events
# =========================================================================

# Row 1 knobs: sized so the 4-sigma band is a tight ~4% (expected 10_000).
# Every factor is deliberately != 1 and the sensor is non-square, so dropping
# any one of them changes the count. Mutation-checked: rate=1.0 or duration=1.0
# would make those factors invisible (x * 1 is a no-op), which is exactly the
# blind spot this arrangement avoids.
BG_RATE = 0.5             # events per pixel per second
BG_SENSOR = (80, 125)     # (H, W) -> 10_000 pixels, non-square
BG_DURATION = 2.0         # seconds
BG_EXPECTED = BG_RATE * BG_SENSOR[0] * BG_SENSOR[1] * BG_DURATION   # 10_000
BG_BAND = 4 * np.sqrt(BG_EXPECTED)                                  # 4 sigma = 400

# A small, busy sensor for the bounds/polarity rows: many events on few
# pixels, so the edges of the coordinate range actually get exercised.
SMALL_SENSOR = (4, 5)     # deliberately non-square: catches an H/W swap
SMALL_RATE = 50.0
SMALL_DURATION = 2.0

# A realistic sensor. W = 1280 exceeds uint8's 255, which is the whole point:
# every other row here runs on a toy sensor that a too-narrow coordinate dtype
# would survive.
REAL_SENSOR = (720, 1280)
REAL_RATE = 0.01          # ~9_200 events; enough to reach the far columns
REAL_DURATION = 1.0


def test_background_count_matches_rate_times_pixels_times_duration(stream, rng):
    """Row 1: the model itself — how many events get generated.

    The only row that checks the FORMULA; every other row would still pass
    with a completely wrong count. Poisson(lambda) has std sqrt(lambda), so
    at lambda=10_000 a 4-sigma band is +-400 (~4%). The seed is fixed, so
    this is deterministic and cannot flake; the band only has to be tighter
    than the errors worth catching, which are multiplicative (a forgotten
    `duration`, H+W instead of H*W) and miss by factors, not percents.
    """
    xs, ys, ts, ps = stream
    out_x, _, _, _ = background(xs, ys, ts, ps, rng, bg_rate=BG_RATE, sensor_shape=BG_SENSOR, duration=BG_DURATION)
    n_new = len(out_x) - len(xs)
    assert abs(n_new - BG_EXPECTED) < BG_BAND, (
        f"generated {n_new} events, expected {BG_EXPECTED:.0f} +- {BG_BAND:.0f} "
        "— check rate * H * W * duration"
    )


def test_background_coordinates_stay_in_bounds(stream, rng):
    """Rows 2-3 (space): every generated pixel is on the sensor — and the
    whole sensor gets used.

    The upper-bound check alone is not enough: swapping H and W on a
    non-square sensor draws x from the NARROWER range, which still satisfies
    "max < W". So each axis must also reach its last index. With ~2000 events
    over 4x5 pixels, missing an index by chance has probability ~1e-190, and
    the seed is fixed regardless.
    """
    xs, ys, ts, ps = stream
    H, W = SMALL_SENSOR
    out_x, out_y, _, _ = background(
        xs, ys, ts, ps, rng, bg_rate=SMALL_RATE, sensor_shape=SMALL_SENSOR, duration=SMALL_DURATION
    )
    new_x, new_y = out_x[len(xs):], out_y[len(ys):]
    assert len(new_x) > 0, "no events generated — this row proves nothing"
    assert new_x.min() == 0 and new_x.max() == W - 1, (
        f"x range is [{new_x.min()}, {new_x.max()}], expected exactly [0, {W - 1}] "
        "— too narrow suggests H and W are swapped"
    )
    assert new_y.min() == 0 and new_y.max() == H - 1, (
        f"y range is [{new_y.min()}, {new_y.max()}], expected exactly [0, {H - 1}] "
        "— too narrow suggests H and W are swapped"
    )


def test_background_timestamps_stay_in_window(stream, rng):
    """Rows 2-3 (time): every generated timestamp lies inside the clip."""
    xs, ys, ts, ps = stream
    out_x, _, out_t, _ = background(
        xs, ys, ts, ps, rng, bg_rate=SMALL_RATE, sensor_shape=SMALL_SENSOR, duration=SMALL_DURATION
    )
    new_t = out_t[len(xs):]
    assert len(new_t) > 0, "no events generated — this row proves nothing"
    assert new_t.min() >= 0.0 and new_t.max() < SMALL_DURATION, (
        f"t out of range: [{new_t.min():.3f}, {new_t.max():.3f}] "
        f"not within [0, {SMALL_DURATION})"
    )


def test_background_events_are_all_on(stream, rng):
    """Row 4: leak events are ON (+1) — the reset-transistor drift is one-way."""
    xs, ys, ts, ps = stream
    out_x, _, _, out_p = background(
        xs, ys, ts, ps, rng, bg_rate=SMALL_RATE, sensor_shape=SMALL_SENSOR, duration=SMALL_DURATION
    )
    new_p = out_p[len(xs):]
    assert len(new_p) > 0, "no events generated — this row proves nothing"
    assert np.all(new_p == 1), (
        f"expected every generated polarity to be +1, found {np.unique(new_p)}"
    )


def test_background_original_events_survive_at_the_front(stream, rng):
    """Row 5: the real stream passes through intact, ahead of the new chunk."""
    xs, ys, ts, ps = stream
    n = len(xs)
    out_x, out_y, out_t, out_p = background(
        xs, ys, ts, ps, rng, bg_rate=BG_RATE, sensor_shape=BG_SENSOR, duration=BG_DURATION
    )
    assert np.array_equal(out_x[:n], xs), "original x corrupted"
    assert np.array_equal(out_y[:n], ys), "original y corrupted"
    assert np.array_equal(out_t[:n], ts), "original t corrupted"
    assert np.array_equal(out_p[:n], ps), "original p corrupted"


def test_background_input_arrays_not_mutated(stream, rng):
    """Row 6: purity — the caller's clean stream survives the call."""
    xs, ys, ts, ps = stream
    before = tuple(a.copy() for a in (xs, ys, ts, ps))
    background(xs, ys, ts, ps, rng, bg_rate=BG_RATE, sensor_shape=BG_SENSOR, duration=BG_DURATION)
    for name, now, then in zip("xytp", (xs, ys, ts, ps), before):
        assert np.array_equal(now, then), f"background mutated the caller's {name}"


def test_background_output_dtypes(stream, rng):
    """Row 7: the contract dtypes survive concatenation.

    Concatenating mismatched dtypes silently promotes the whole column
    (uint8 + uint16 -> uint16; int8 + uint8 -> int16), so this row also
    guards the generated chunk's dtypes, not just the inputs'.
    """
    xs, ys, ts, ps = stream
    out_x, out_y, out_t, out_p = background(
        xs, ys, ts, ps, rng, bg_rate=BG_RATE, sensor_shape=BG_SENSOR, duration=BG_DURATION
    )
    assert out_x.dtype == np.uint16, f"x should be uint16, got {out_x.dtype}"
    assert out_y.dtype == np.uint16, f"y should be uint16, got {out_y.dtype}"
    assert out_t.dtype == np.float64, f"t should be float64, got {out_t.dtype}"
    assert out_p.dtype == np.int8, f"p should be int8, got {out_p.dtype}"


def test_background_rate_zero_generates_nothing(stream, rng):
    """Row 8: no leak rate, no leak events — the stream passes through."""
    xs, ys, ts, ps = stream
    out_x, out_y, out_t, out_p = background(
        xs, ys, ts, ps, rng, bg_rate=0.0, sensor_shape=BG_SENSOR, duration=BG_DURATION
    )
    assert len(out_x) == len(xs), (
        f"rate=0 should add no events, got {len(out_x) - len(xs)}"
    )
    for name, got, want in zip("xytp", (out_x, out_y, out_t, out_p), (xs, ys, ts, ps)):
        assert np.array_equal(got, want), f"rate=0 changed {name}"


def test_background_count_is_drawn_not_computed(stream):
    """Row 9: the count is a Poisson DRAW, not the expected value itself.

    Row 1 cannot see this: replacing rng.poisson(expected) with int(expected)
    yields exactly 10_000 every run — dead centre of row 1's band, so it
    passes. Variance is invisible in a single observation; only comparing
    across seeds reveals it. At lambda=10_000 the draw has std 100, so three
    distinct seeds landing on the same count is effectively impossible.
    """
    xs, ys, ts, ps = stream
    counts = {
        len(background(xs, ys, ts, ps, np.random.default_rng(seed),
                       bg_rate=BG_RATE, sensor_shape=BG_SENSOR,
                       duration=BG_DURATION)[0]) - len(xs)
        for seed in (1, 2, 3)
    }
    assert len(counts) > 1, (
        f"every seed produced the same count {counts} — the count looks computed "
        "rather than drawn; is rng.poisson() being used?"
    )


def test_background_handles_realistic_sensor_dimensions(stream, rng):
    """Row 10: coordinates must survive a real sensor, not just toy ones.

    Every other row runs on a sensor small enough to fit in a uint8 (max 255),
    so a too-narrow coordinate dtype passes them all and then dies on first
    contact with real footage (`ValueError: high is out of bounds for uint8`).
    Row 7 cannot catch it either: concatenating a uint8 chunk onto a uint16
    input stream promotes the result back to uint16, so the output dtype looks
    correct. Only a sensor wider than 255 exercises the range.
    """
    xs, ys, ts, ps = stream
    H, W = REAL_SENSOR
    out_x, out_y, _, _ = background(
        xs, ys, ts, ps, rng, bg_rate=REAL_RATE, sensor_shape=REAL_SENSOR, duration=REAL_DURATION
    )
    new_x, new_y = out_x[len(xs):], out_y[len(ys):]
    assert len(new_x) > 0, "no events generated — this row proves nothing"
    assert new_x.max() > 255, (
        f"x only reached {new_x.max()} on a {W}-wide sensor — the coordinate "
        "range beyond uint8 is never exercised, so this row proves nothing"
    )
    assert new_x.max() < W and new_y.max() < H, (
        f"coordinates out of bounds: x max {new_x.max()} (< {W}), "
        f"y max {new_y.max()} (< {H})"
    )
