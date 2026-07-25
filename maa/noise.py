"""Sensor noise model.

Real event cameras are noisy: background-activity events fire without any
real brightness change, hot pixels chatter, and the contrast threshold
varies per pixel. This module perturbs a clean event stream to look more
like silicon.
"""

# TODO: background-activity (leak) noise as a Poisson process per pixel.
# TODO: per-pixel threshold jitter (fixed-pattern noise).
# TODO: hot/dead pixel masks.
# TODO: refractory period after each event.

import numpy as np

def background(xs, ys, ts, ps, rng, bg_rate, sensor_shape, duration):
    H, W = sensor_shape
    expected_events = (H * W) * bg_rate * duration
    total_events = rng.poisson(expected_events)
    
    xs_new = rng.integers(0, W, total_events, dtype=np.uint16)
    ys_new = rng.integers(0, H, total_events, dtype=np.uint16)
    ts_new = rng.uniform(0, duration, total_events)
    ps_new = np.full(total_events, 1, dtype=np.int8)

    xs_out = np.concatenate([xs, xs_new])
    ys_out = np.concatenate([ys, ys_new])
    ts_out = np.concatenate([ts, ts_new])
    ps_out = np.concatenate([ps, ps_new])

    return xs_out, ys_out, ts_out, ps_out

def hot_p(xs, ys, ts, ps, rng, hot_pixels):
    raise NotImplementedError

def jitter(xs, ys, ts, ps, rng, sigma):
    return xs, ys, ts + rng.normal(0, sigma, len(ts)), ps

def refractory(xs, ys, ts, ps, refractory_dt):
    raise NotImplementedError

def noise(xs, ys, ts, ps, bg_rate, hot_pixels, rng, sigma, refractory_dt, sensor_shape, duration):
    xs, ys, ts, ps = background(xs, ys, ts, ps, rng, bg_rate, sensor_shape, duration)
    xs, ys, ts, ps = hot_p(xs, ys, ts, ps, rng, hot_pixels)
    xs, ys, ts, ps = jitter(xs, ys, ts, ps, rng, sigma)
    order = np.argsort(ts)
    xs, ys, ts, ps = refractory(xs[order], ys[order], ts[order], ps[order], refractory_dt)
    return xs, ys, ts, ps