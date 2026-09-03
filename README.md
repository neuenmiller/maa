# maa 馬/ม้า

> 馬 (Japanese) / ม้า (Thai): *horse*. Named for Eadweard Muybridge's
> galloping horse (1878), the first photo sequence fast enough to show
> what happens between the frames the eye can see.

Minimal event-camera simulator, converts high-fps video into event
streams. 馬/ม้า: see Muybridge, 1878.

![demo](results/demo.gif)

<sub>Green = brighter (ON), red = darker (OFF). The last 200 m of the 2026
BMW Hong Kong Derby, number 5 in front — 25 fps broadcast footage, cropped
to the turf and downscaled to 384&times;112.</sub>

## Install

```bash
git clone https://github.com/neuenmiller/maa maa
cd maa
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Run

Smoke-test the install:

```bash
python -c "import maa; print(maa.__version__)"
```

Reproduce the headline GIF — drop the race clip into the gitignored `data/`
by hand (`data/fetch.py` is still a stub), then:

```bash
python examples/demo.py --input data/replay-full_20260322_07_eng_2500kbps.mp4 --start-frame 2895 --max-frames 145 --crop 1920:560:0:445 --scale 5 --threshold 0.3
```

Those flags are doing real work. The clip is 25 fps broadcast footage rather
than the 240 fps slow-mo this pipeline wants, so the gap between frames is
large and the events are coarser than a real sensor's. `--crop` drops the
running rail and the sponsor arch, whose panning edges generate more events
than the horses do, and `--threshold 0.3` (up from the 0.2 default) holds
back the turf speckle that the coarse timing turns into noise. Any other
clip works too — `--input data/clip.mp4` on its own is still the short form.

No clip handy? Exercise the colour + GIF plumbing on a synthetic pattern
(write it somewhere throwaway so it doesn't clobber the committed GIF):

```bash
python examples/demo.py --selftest --output /tmp/selftest.gif
```

`simulate` turns the frames into a sparse `(x, y, t, p)` event stream and
the demo paints it green (ON) / red (OFF). Full loop: video in →
`simulate` → events → `noise` → `reconstruct` → frames out, all three
implemented.

Add `--noise` for sensor noise, or `--reconstruct` for a three-panel GIF —
source, events, and the scene rebuilt from those events alone. Integration
alone drifts; `--alpha 6.283` adds the leak that stops it.

## Limitations

- frame-interpolated, not microsecond-accurate; see v2e for a serious simulator
- timing resolution is bounded by input video fps (plus interpolation)
- the noise model is a rough approximation of real sensor behavior

## Roadmap

- [x] **First end-to-end demo GIF in `results/`** — 240 fps clip → log-intensity diffs → threshold → green/red events → GIF. Ugly, no noise, no reconstruction, but visible on day one.
- [x] Implement `simulate` — threshold-crossing events from frames; emit a sparse `(x, y, t, p)` event stream (struct-of-arrays)
- [x] Implement `noise` — background activity, threshold jitter, hot pixels
- [x] Implement `reconstruct` — integrate events back to intensity; leaky variant and output-fps rendering included
- [ ] `experiments/reproduce_v2e` — sanity-check against v2e. Run this **before** adding pixel-model sophistication: the diff against v2e *is* the requirements list — it names which of the candidates below actually move the output.
- [ ] **Pixel-model sophistication** — implement what the v2e diff demands, in NumPy, with tests. Deterministic pixel physics lives in `simulate` (the oracle holds no RNG); anything random stays in `noise`.
- [ ] **v1 C++ kernel** (pybind11) — port the hot loop *once the algorithm is frozen*; validate against the NumPy oracle; benchmark NumPy events/sec → C++ speedup
- [ ] `experiments/e2vid_bench` — reconstruction benchmark
- [ ] `experiments/sim2real` *(v2 stretch)* — does sim-trained transfer to real events?
- [ ] **v2+ — generalize past events** *(more of a direction, not a milestone)* — the pipeline is already sensor-shaped: scene → transduction → noise → reconstruction → evaluation. Pull that into a modality-agnostic core and the event camera becomes sensor #1, not the product.

## Sophistication candidates

Status: proposed → confirmed (the v2e diff says it matters) → landed (in
NumPy, with tests). Placement follows the roadmap rule — deterministic
physics in `simulate`, randomness in `noise`; factories draw a sensor
description once, the caller hands it to whichever stage consumes it.

| Candidate | Lives in | Status |
|---|---|---|
| per-pixel threshold variation (FPN) | factory in `noise`, map consumed by `simulate` | proposed |
| multiple events per over-threshold jump | `simulate` | proposed |
| sub-frame timestamps | `simulate` | proposed |
| shot noise (balanced ON/OFF, rate rises in the dark) | `noise` | proposed |
| intensity-dependent latency / bandwidth | `simulate` | proposed |
| asymmetric ON/OFF thresholds | `simulate` | proposed |
| dead pixels | `noise` | proposed |
