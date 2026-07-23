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


def read_frames(path: Path, max_frames: int | None):
    """Decode a video into stacked [0,1] grayscale frames (+ its source fps)."""
    reader = imageio.get_reader(str(path))
    fps = reader.get_meta_data().get("fps")
    frames = []
    for i, frame in enumerate(reader):
        if max_frames is not None and i >= max_frames:
            break
        frames.append(to_grayscale(frame))
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
        if len(k) and (k.min() < 0 or k.max() >= T - 1):
            raise SystemExit(
                f"event timestamps bin to transitions {k.min()}..{k.max()}, "
                f"outside 0..{T - 2} — check the t = (k+1)/fps convention."
            )
        maps[k, y.astype(np.intp), x.astype(np.intp)] = np.sign(p)
        return maps
    maps = np.asarray(events)
    if maps.ndim != 3:
        raise SystemExit(
            "demo expects (T-1, H, W) polarity maps or an (x, y, t, p) tuple; "
            f"got shape {maps.shape}. Adjust to_polarity_frames()."
        )
    return np.sign(maps).astype(np.int8)


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
    ap.add_argument("--out-fps", type=float, default=25.0,
                    help="playback fps of the output GIF")
    ap.add_argument("--selftest", action="store_true",
                    help="run the colour+GIF plumbing on a synthetic pattern; "
                         "no clip or simulate() needed")
    args = ap.parse_args()

    if args.selftest:
        polarity = synthetic_polarity()
        rgb = [colorize(p, None) for p in polarity]
        write_gif(args.output, rgb, args.out_fps)
        return

    if args.input is None:
        raise SystemExit("give --input path/to/clip.mp4 (or --selftest).")

    frames, in_fps = read_frames(args.input, args.max_frames)
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

    polarity = to_polarity_frames(events, frames.shape, in_fps)
    rgb = [colorize(p, frames[i + 1]) for i, p in enumerate(polarity)]
    write_gif(args.output, rgb, args.out_fps)


if __name__ == "__main__":
    main()
