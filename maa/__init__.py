"""maa — a minimal event-camera simulator.

馬/ม้า: the horse. After Muybridge's galloping horse (1878), the first
sequence fast enough to resolve what the eye misses between frames.

Public surface (stubs for now):
    simulate    — turn high-fps video into an event stream
    noise       — add a sensor noise model to events
    reconstruct — recover intensity frames back from events
"""

__version__ = "0.0.0"

# TODO: re-export the public API once the modules are implemented, e.g.
#   from .simulate import simulate
#   from .noise import apply_noise
#   from .reconstruct import reconstruct
