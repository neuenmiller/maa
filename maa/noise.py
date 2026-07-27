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

# Parameter order convention — take the subset you need, never reorder.
#
#   effects + orchestrator (stream in, stream out):
#     xs, ys, ts, ps, rng, bg_rate, xs_hot, ys_hot, hot_rates, sigma,
#     refractory_dt, sensor_shape, duration
#
#   factory (config in, sensor description out):
#     rng, bg_rate, sensor_shape, hot_fraction, hot_multiple, pareto_a

import numpy as np

def background(xs, ys, ts, ps, rng, *, bg_rate, sensor_shape, duration):
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

def make_hot_p(rng, bg_rate, sensor_shape, hot_fraction = 0.007, hot_multiple = 50, pareto_a = 3):
    hot_min = hot_multiple * bg_rate
    H, W = sensor_shape
    
    total_hot_p = round(H * W * hot_fraction)
    
    flat = rng.choice(H * W, total_hot_p, replace=False)
    ys_hot, xs_hot = np.unravel_index(flat, (H, W))
    hot_rates = hot_min * (1 + rng.pareto(pareto_a, total_hot_p))
    
    return xs_hot.astype(np.uint16), ys_hot.astype(np.uint16), hot_rates

def hot_p(xs, ys, ts, ps, rng, *, xs_hot, ys_hot, hot_rates, duration):
    counts = rng.poisson(hot_rates * duration)
    total_events = counts.sum()

    xs_new = np.repeat(xs_hot, counts)
    ys_new = np.repeat(ys_hot, counts)
    ts_new = rng.uniform(0, duration, total_events)
    ps_new = np.full(total_events, 1, dtype=np.int8)

    xs_out = np.concatenate([xs, xs_new])
    ys_out = np.concatenate([ys, ys_new])
    ts_out = np.concatenate([ts, ts_new])
    ps_out = np.concatenate([ps, ps_new])

    return xs_out, ys_out, ts_out, ps_out

def jitter(xs, ys, ts, ps, rng, sigma):
    return xs, ys, ts + rng.normal(0, sigma, len(ts)), ps

def refractory(xs, ys, ts, ps, refractory_dt):
    raise NotImplementedError

def noise(xs, ys, ts, ps, rng, bg_rate, xs_hot, ys_hot, hot_rates, sigma, refractory_dt, sensor_shape, duration):   
    xs, ys, ts, ps = background(xs, ys, ts, ps, rng, bg_rate=bg_rate,sensor_shape=sensor_shape, duration=duration)
    xs, ys, ts, ps = hot_p(xs, ys, ts, ps, rng, xs_hot=xs_hot, ys_hot=ys_hot, hot_rates=hot_rates, duration=duration)
    xs, ys, ts, ps = jitter(xs, ys, ts, ps, rng, sigma)
    order = np.argsort(ts)
    xs, ys, ts, ps = refractory(xs[order], ys[order], ts[order], ps[order], refractory_dt)
    return xs, ys, ts, ps