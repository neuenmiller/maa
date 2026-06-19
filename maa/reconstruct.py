"""Intensity reconstruction.

Inverts the simulator: integrates an event stream back into intensity
frames so you can eyeball whether the events actually carry the scene.
Useful as a sanity check and for the side-by-side GIFs in results/.
"""

# TODO: naive integration of events into a log-intensity buffer.
# TODO: leaky / high-pass variant to fight drift.
# TODO: render reconstructed frames at a chosen output fps.
