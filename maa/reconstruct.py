"""Intensity reconstruction.

Inverts the simulator: integrates an event stream back into intensity
frames so you can eyeball whether the events actually carry the scene.
Useful as a sanity check and for the side-by-side GIFs in results/.

"""

import numpy as np
import warnings

def reconstruct(xs, ys, ts, ps, *, sensor_shape, duration, threshold_c, out_fps, alpha=0.0):
    """
    
    Notes: <= because we want to include everything when it happens; 
    T = (f + 1) / out_fps to prevent empty first frame; 
    alpha = 0 reduces this to naive integration with no leak; 
    there is no state between frame, everything is recomputed for the next
    frame; 
    sorting is not required
    
    """

    if abs((duration * out_fps) - round(duration * out_fps)) > 1e-9:
           warnings.warn(f"Output frame quantity (duration * out_fps) is non-whole number at {duration * out_fps}  frames, rounded to {round(duration * out_fps)}", stacklevel=2)
    n_frame = int(round(duration * out_fps))
    buffer = np.zeros((n_frame, *sensor_shape))
    for f in range(n_frame):
        
        T = (f + 1) / out_fps
        mask = (0 <= ts) & (ts <= T) & (ts <= duration) 
        weight = ps[mask] * threshold_c * np.exp(-alpha * (T - ts[mask]))
        np.add.at(buffer[f], (ys[mask], xs[mask]), weight)

    return buffer 


