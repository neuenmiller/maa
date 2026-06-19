"""Event simulation from video.

Converts a high-fps intensity video into a stream of brightness-change
events. Each pixel emits an event when its log-intensity crosses a
contrast threshold C since its last event, carrying (x, y, t, polarity).

This is the frame-interpolated approach: events are derived from
differences between (optionally upsampled) frames, not from a true
continuous photocurrent. See README Limitations.
"""

# TODO: define the Event representation (structured array vs. dataclass).
# TODO: implement simulate(frames, fps, threshold_c) -> events.
# TODO: optional frame interpolation to approximate sub-frame timing.
# TODO: per-pixel log-intensity state + threshold crossing -> polarity.
