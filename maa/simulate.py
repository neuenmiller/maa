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
    x_series = []
    y_series = []
    t_series = []
    p_series = []

    latest_ref = L[0].copy()

    for i in range(1, T):
        frame = L[i]
        polarity = np.zeros_like(frame, dtype=np.int8)

        diff = frame - latest_ref
        polarity[diff >= threshold_c] = 1
        polarity[diff <= -threshold_c] = -1
        fired = polarity != 0
        latest_ref[fired] = frame[fired]

        ys, xs = np.nonzero(polarity)
        ps = polarity[ys, xs]

        x_series.append(xs)
        y_series.append(ys)
        t_series.append(np.full(len(xs), i/fps))
        p_series.append(ps)          
    
    return np.concatenate(x_series), np.concatenate(y_series), np.concatenate(t_series), np.concatenate(p_series)