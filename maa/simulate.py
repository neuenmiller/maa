"""Event simulation from video.

Converts a high-fps intensity video into a stream of brightness-change
events. Each pixel emits an event when its log-intensity crosses a
contrast threshold C since its last event, carrying (x, y, t, polarity).

This is the frame-interpolated approach: events are derived from
differences between (optionally upsampled) frames, not from a true
continuous photocurrent. See README Limitations.
"""

import numpy as np

def simulate(frames, fps, threshold_c):
    eps = 1e-6
    L = np.log(frames + eps)
    T, H, W = frames.shape

    latest_ref = L[0].copy()
    out = np.zeros((T - 1, H, W), dtype=np.int8)

    for t in range(1, T):
        frame = L[t]
        polarity = np.zeros_like(frame, dtype=np.int8)
        diff = frame - latest_ref
        polarity[diff >= threshold_c] = 1
        polarity[diff <= -threshold_c] = -1
        fired = polarity != 0
        out[t-1] = polarity
        latest_ref[fired] = frame[fired]

    return out