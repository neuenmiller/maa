"""maa — a minimal event-camera simulator.

馬/ม้า: the horse. After Muybridge's galloping horse (1878), the first
sequence fast enough to resolve what the eye misses between frames.

Public surface:
    simulate      — turn high-fps video into an event stream
    apply_noise   — add a sensor noise model to events
    make_hot_p    — draw a sensor's fixed hot-pixel list (call once)
    reconstruct   — recover intensity frames back from events

Full loop: video in -> simulate -> events -> apply_noise -> reconstruct
-> frames out.
"""

from .noise import make_hot_p
from .noise import noise as apply_noise
from .reconstruct import reconstruct
from .simulate import simulate

__version__ = "0.0.0"

__all__ = [
    "simulate",
    "apply_noise",
    "make_hot_p",
    "reconstruct",
    "__version__",
]

# On the names, because re-exporting shadows submodules and the three
# modules are not in the same situation.
#
# `simulate` and `reconstruct` each hold exactly one public function with
# the same name as the module, so `maa.simulate` becoming the function
# costs nothing: there is nothing else in those modules to reach. Import
# the module explicitly (`from maa.simulate import simulate`) if you want
# the module object.
#
# `noise` holds five public functions plus a factory, so binding the
# orchestrator to `maa.noise` would hide `background`, `hot_p`, `jitter`,
# `refractory` and `make_hot_p` behind it. It is exported as `apply_noise`
# instead, which is also the name the original TODO in this file asked
# for, and the submodule stays reachable as `maa.noise`.
