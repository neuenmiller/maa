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
