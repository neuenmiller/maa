"""First-light demo — video in, green/red event GIF out.

PLUMBING (Claude-written, yours to read and tweak). This reads a high-fps
clip, hands the frames to YOUR ``maa.simulate()``, takes the per-pixel
polarity it returns, and paints it — green where a pixel got brighter (ON),
red where it got darker (OFF) — into ``results/demo.gif``.

The event logic in ``maa/simulate.py`` is YOURS to write. This file only does
the boring ends: decode video -> grayscale frames, and polarity -> coloured
GIF. The two meet in exactly two places — the ``simulate(...)`` call in
``main()`` and the ``to_polarity_frames()`` adapter below. If you pick a
different event representation, those are the only spots that change.

    # check the plumbing with no clip and no simulate() yet:
    python examples/demo.py --selftest

    # the real thing, once you have a clip and a simulate():
    python examples/demo.py --input data/clip.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "results" / "demo.gif"

# Classic DVS colours for the two event polarities.
GREEN = np.array([0, 200, 0], dtype=np.uint8)   # ON  — pixel got brighter
RED = np.array([220, 0, 0], dtype=np.uint8)     # OFF — pixel got darker


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """RGB(A) uint8 frame -> float grayscale in [0, 1] (Rec. 601 luma)."""
    f = frame.astype(np.float64)
    if f.ndim == 2:                              # already single-channel
        g = f
    else:
        g = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    return g / 255.0


def downscale(g: np.ndarray, k: int) -> np.ndarray:
    """Box-average a grayscale frame down by an integer factor.

    Averaging rather than decimating (``g[::k, ::k]``) matters here: plain
    decimation aliases high-frequency detail into flicker between frames, and
    simulate() would faithfully turn that flicker into events. Box-averaging
    low-passes first, so the events come from real motion.
    """
    if k <= 1:
        return g
    h, w = (g.shape[0] // k) * k, (g.shape[1] // k) * k
    return g[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def parse_crop(spec: str | None):
    """``"w:h:x:y"`` -> ``(w, h, x, y)``, in ffmpeg's ``crop`` filter order.

    Same order as ffmpeg on purpose: the box is usually found by eye with
    ``ffmpeg -vf crop=...`` first, and retyping it in a different order is a
    reliable way to lose an afternoon.
    """
    if spec is None:
        return None
    try:
        w, h, x, y = (int(v) for v in spec.split(":"))
    except ValueError:
        raise SystemExit(f"--crop wants w:h:x:y (ffmpeg order); got {spec!r}.")
    return w, h, x, y


def crop_frame(g: np.ndarray, box) -> np.ndarray:
    """Crop a grayscale frame to a ``(w, h, x, y)`` box, clipped to the frame.

    Cropping happens before ``downscale`` so the box is in source pixels and
    stays valid when --scale changes. It matters for framing that broadcast
    footage makes hostile: a scoreboard overlay is static, so it emits no
    events and just sits there as dead grey, while a panning rail or a banner
    arch is a huge high-contrast edge that swamps the subject.
    """
    if box is None:
        return g
    w, h, x, y = box
    H, W = g.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x0 + w, W), min(y0 + h, H)
    if x1 <= x0 or y1 <= y0:
        raise SystemExit(f"--crop {w}:{h}:{x}:{y} selects nothing from a {W}x{H} frame.")
    return g[y0:y1, x0:x1]


def read_frames(path: Path, max_frames: int | None, start: int = 0, scale: int = 1,
                crop=None):
    """Decode a video into stacked [0,1] grayscale frames (+ its source fps)."""
    reader = imageio.get_reader(str(path))
    fps = reader.get_meta_data().get("fps")
    frames = []
    for i, frame in enumerate(reader):
        if i < start:
            continue
        if max_frames is not None and len(frames) >= max_frames:
            break
        frames.append(downscale(crop_frame(to_grayscale(frame), crop), scale))
    reader.close()
    if len(frames) < 2:
        raise SystemExit(f"{path}: need at least 2 frames, got {len(frames)}.")
    return np.stack(frames), fps


def to_polarity_frames(events, shape, fps) -> np.ndarray:
    """Adapter: whatever ``simulate()`` returns -> (T-1, H, W) int8 maps in
    {-1, 0, +1}.

    Accepts both simulate() contracts:
      - dense (T-1, H, W) polarity maps (the v0 contract) — passed through;
      - the sparse SoA stream, a 4-tuple of 1-D arrays (x, y, t, p) — binned
        back into per-transition maps. Timestamps follow the convention in
        tests/test_simulate.py: the frames k -> k+1 transition is stamped
        t = (k+1)/fps, so transition index = round(t * fps) - 1.
    ``np.sign`` means raw step *counts* still colour correctly.
    """
    T, H, W = shape
    if isinstance(events, (tuple, list)) and len(events) == 4:
        x, y, t, p = (np.asarray(a) for a in events)
        maps = np.zeros((T - 1, H, W), dtype=np.int8)
        k = np.rint(t * fps).astype(np.intp) - 1
        inside = (k >= 0) & (k < T - 1)
        if len(k) and not inside.any():
            raise SystemExit(
                f"every event bins to transitions {k.min()}..{k.max()}, all outside "
                f"0..{T - 2} — check the t = (k+1)/fps convention."
            )
        # Noise legitimately puts events outside the rendered window: background
        # activity is uniform over [0, duration] but the first renderable
        # transition is t = 1/fps, and jitter can nudge events past the last one.
        # Those events are real, just not visible in this frame-binned view.
        outside = int((~inside).sum())
        if outside:
            print(f"  {outside} events outside the rendered window, not drawn")
        k, y, x, p = k[inside], y[inside], x[inside], p[inside]
        maps[k, y.astype(np.intp), x.astype(np.intp)] = np.sign(p)
        return maps
    maps = np.asarray(events)
    if maps.ndim != 3:
        raise SystemExit(
            "demo expects (T-1, H, W) polarity maps or an (x, y, t, p) tuple; "
            f"got shape {maps.shape}. Adjust to_polarity_frames()."
        )
    return np.sign(maps).astype(np.int8)


def to_log_display(L: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Log-intensity frame -> RGB uint8 grayscale, windowed to [lo, hi].

    Reconstruction comes back in log units, unbounded in principle: drift can push
    a pixel far past white. Windowing to the *source* clip's log range keeps the
    reconstruction panel on the same scale as the original, so "too bright" looks
    too bright instead of being silently renormalised away.
    """
    x = np.clip((L - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    g = (x * 255.0).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def side_by_side(*panels: np.ndarray) -> np.ndarray:
    """Stack equal-height RGB panels left to right with a thin separator."""
    h = panels[0].shape[0]
    sep = np.full((h, 2, 3), 40, dtype=np.uint8)
    out = []
    for i, panel in enumerate(panels):
        if i:
            out.append(sep)
        out.append(panel)
    return np.concatenate(out, axis=1)


def colorize(polarity: np.ndarray, base: np.ndarray | None) -> np.ndarray:
    """One polarity map (H, W in {-1,0,+1}) -> RGB uint8 frame.

    Background is the underlying frame, dimmed, so motion keeps its context;
    events are painted over it in green (ON) / red (OFF).
    """
    h, w = polarity.shape
    if base is None:
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        dim = np.clip(base * 0.35 * 255.0, 0, 255).astype(np.uint8)
        rgb = np.repeat(dim[..., None], 3, axis=2)
    rgb[polarity > 0] = GREEN
    rgb[polarity < 0] = RED
    return rgb


def synthetic_polarity(n=24, h=120, w=160) -> np.ndarray:
    """A vertical bar sweeping right, for --selftest: green leading edge, red
    trailing. Exercises the colour + GIF path with no clip and no simulate()."""
    maps = np.zeros((n, h, w), dtype=np.int8)
    for i in range(n):
        x = int((i / n) * (w - 1))
        maps[i, :, x] = 1                        # leading edge brightens (ON)
        if x - 6 >= 0:
            maps[i, :, x - 6] = -1               # trailing edge darkens (OFF)
    return maps


def apply_noise(events, frames_shape, fps, args):
    """Pipe a clean event stream through maa.noise -> a noisy event stream.

    This function is the *caller* in the library/application split: it holds
    the facts about this particular recording (how big the sensor is, how long
    the clip runs) and hands them to the library, which knows nothing about
    clips. See maa/noise.py's parameter-order convention.

    The hot-pixel list is built here, ONCE, rather than inside noise(): a
    camera's defective pixels are fixed for its lifetime, so re-drawing them
    per call would make them just another flavour of background activity —
    and would confound any sweep, since the sensor would change along with
    the knob under study.
    """
    from maa.noise import make_hot_p, noise

    T, H, W = frames_shape
    sensor_shape = (H, W)
    duration = T / fps
    rng = np.random.default_rng(args.seed)

    xs_hot, ys_hot, hot_rates = make_hot_p(
        rng, args.bg_rate, sensor_shape, hot_fraction=args.hot_fraction
    )
    print(f"  {len(xs_hot)} hot pixels ({args.hot_fraction:.3%} of {H*W}), "
          f"rates {hot_rates.min():.0f}-{hot_rates.max():.0f} Hz")

    before = len(events[0])
    noisy = noise(
        *events, rng,
        bg_rate=args.bg_rate, xs_hot=xs_hot, ys_hot=ys_hot, hot_rates=hot_rates,
        sigma=args.jitter_sigma, refractory_dt=args.refractory_dt,
        sensor_shape=sensor_shape, duration=duration,
    )
    print(f"  events: {before} clean -> {len(noisy[0])} noisy "
          f"({len(noisy[0]) / max(before, 1):.1f}x)")
    return noisy


def log_window(frames: np.ndarray) -> tuple[float, float]:
    """The source clip's own log-intensity range, used as the display window.

    Matches simulate.py's eps so the two agree on what "black" means.
    """
    L = np.log(frames + 1e-6)
    return float(L.min()), float(L.max())


def reconstruction_panels(events, frames: np.ndarray, in_fps: float, args) -> list:
    """Call YOUR maa.reconstruct() and turn its log frames into RGB panels.

    Sampling is deliberately locked to the source frame rate rather than
    --out-fps: the events panel has one image per frame transition, so asking
    the reconstruction for anything else would leave the two panels drifting
    apart on the timeline. --out-fps stays what it always was, the GIF's
    playback speed.

    reconstruct() samples frame f at t = (f+1)/out_fps and transition k is
    stamped t = (k+1)/fps, so with out_fps == fps the two share an index
    exactly. There is one more reconstructed frame than there are transitions
    (T against T-1), so the trailing one is dropped, not the leading one.
    """
    try:
        from maa.reconstruct import reconstruct
    except (ImportError, AttributeError):
        raise SystemExit(
            "maa.reconstruct.reconstruct() isn't implemented yet - that part is yours.\n"
            "Write the integrator in maa/reconstruct.py, then re-run with --reconstruct.\n"
            "Everything else in this demo works without it."
        )

    T, H, W = frames.shape
    L = np.asarray(reconstruct(
        *events,
        sensor_shape=(H, W),
        duration=T / in_fps,
        threshold_c=args.threshold,
        out_fps=in_fps,
        alpha=args.alpha,
    ))

    if L.ndim != 3 or L.shape[1:] != (H, W):
        raise SystemExit(
            f"reconstruct() returned shape {L.shape}; demo expects (n_frames, {H}, {W}). "
            "If you chose a different contract, adjust reconstruction_panels()."
        )
    if len(L) == T:
        L = L[:-1]
    elif len(L) != T - 1:
        raise SystemExit(
            f"reconstruct() returned {len(L)} frames; demo expects {T} or {T - 1} "
            f"for a {T}-frame clip at out_fps={in_fps}."
        )

    # reconstruct() returns log intensity on its own scale, because events
    # carry no absolute reference (paper eq. 9: L = L^E + L(p,0) + mu, and
    # L(p,0) is unknowable). Untouched pixels therefore sit at 0, which is
    # intensity 1.0 -- above white for a clip normalised to [0, 1]. Recentring
    # on the clip's own log range is a display choice and belongs here, not in
    # the library.
    lo, hi = log_window(frames)
    offset = (lo + hi) / 2
    L = L + offset
    clipped = float(((L < lo) | (L > hi)).mean())
    print(f"  reconstruction: shifted by {offset:+.2f} for display, "
          f"log range [{L.min():.2f}, {L.max():.2f}], "
          f"window [{lo:.2f}, {hi:.2f}], {clipped:.1%} clipped")
    return [to_log_display(frame, lo, hi) for frame in L]


def write_gif(path: Path, rgb_frames, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), list(rgb_frames), fps=fps)
    print(f"wrote {len(rgb_frames)} frames -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="video -> event GIF (first-light demo)")
    ap.add_argument("--input", type=Path, help="high-fps clip, e.g. data/clip.mp4")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--threshold", type=float, default=0.2,
                    help="contrast threshold C, in log units")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames read (handy while iterating)")
    ap.add_argument("--start-frame", type=int, default=0,
                    help="skip this many frames before reading")
    ap.add_argument("--crop", type=str, default=None,
                    help="crop the source to w:h:x:y (ffmpeg order) before "
                         "downscaling; handy for cutting broadcast overlays "
                         "and background clutter out of the frame")
    ap.add_argument("--scale", type=int, default=1,
                    help="box-average downscale factor (4 turns 1080p into 270p)")
    ap.add_argument("--out-fps", type=float, default=25.0,
                    help="playback fps of the output GIF")
    ap.add_argument("--selftest", action="store_true",
                    help="run the colour+GIF plumbing on a synthetic pattern; "
                         "no clip or simulate() needed")

    # --- optional sensor noise (maa.noise). Off by default: the clean stream
    # --- is the correctness oracle, so it stays the default path.
    ns = ap.add_argument_group("sensor noise (opt-in)")
    ns.add_argument("--noise", action="store_true",
                    help="pipe events through maa.noise before rendering")
    ns.add_argument("--seed", type=int, default=0,
                    help="RNG seed; same seed gives the same noise every run")
    ns.add_argument("--bg-rate", type=float, default=0.1,
                    help="background activity, events per pixel per second")
    ns.add_argument("--hot-fraction", type=float, default=0.007,
                    help="fraction of pixels that are hot (~0.7%% on real DVS)")
    ns.add_argument("--jitter-sigma", type=float, default=0.002,
                    help="timestamp jitter, seconds (keep well under 1/fps)")
    ns.add_argument("--refractory-dt", type=float, default=0.005,
                    help="per-pixel dead time after an event, seconds")

    # --- optional reconstruction (maa.reconstruct). Renders a three-panel GIF:
    # --- original | events | reconstruction, all on one timeline.
    rc = ap.add_argument_group("reconstruction (opt-in)")
    rc.add_argument("--reconstruct", action="store_true",
                    help="integrate events back to intensity and show them side by side")
    rc.add_argument("--alpha", type=float, default=0.0,
                    help="leak rate, 1/s; 0 is pure integration, try 2*pi for the paper's value")
    args = ap.parse_args()

    if args.selftest:
        polarity = synthetic_polarity()
        rgb = [colorize(p, None) for p in polarity]
        write_gif(args.output, rgb, args.out_fps)
        return

    if args.input is None:
        raise SystemExit("give --input path/to/clip.mp4 (or --selftest).")

    frames, in_fps = read_frames(args.input, args.max_frames,
                                 start=args.start_frame, scale=args.scale,
                                 crop=parse_crop(args.crop))
    print(f"read {len(frames)} frames from {args.input} (source fps: {in_fps})")

    # --- the seam: YOUR code. This is the one call into maa/simulate.py. ---
    try:
        from maa.simulate import simulate
    except (ImportError, AttributeError):
        raise SystemExit(
            "maa.simulate.simulate() isn't implemented yet — that part is yours.\n"
            "Write the threshold-crossing loop in maa/simulate.py, then re-run.\n"
            "Until then, `python examples/demo.py --selftest` proves the plumbing."
        )

    events = simulate(frames, fps=in_fps, threshold_c=args.threshold)
    # --- end seam ---

    if args.noise:
        events = apply_noise(events, frames.shape, in_fps, args)

    polarity = to_polarity_frames(events, frames.shape, in_fps)
    rgb = [colorize(p, frames[i + 1]) for i, p in enumerate(polarity)]

    if args.reconstruct:
        lo, hi = log_window(frames)
        # One row per transition k, so all three panels show the same instant:
        # the source frame k+1, the events that produced it, the reconstruction.
        original = [to_log_display(np.log(f + 1e-6), lo, hi) for f in frames[1:]]
        recon = reconstruction_panels(events, frames, in_fps, args)
        rgb = [side_by_side(o, e, r) for o, e, r in zip(original, rgb, recon)]

    write_gif(args.output, rgb, args.out_fps)


if __name__ == "__main__":
    main()
