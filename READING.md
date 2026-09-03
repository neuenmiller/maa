# Reading

Papers behind each roadmap item, in the order they become useful. Links go to
arXiv where there is one; the published versions are paywalled and no better.

## reconstruct

**Scheerlinck, Barnes, Mahony — *Continuous-time Intensity Estimation Using
Event Cameras*, ACCV 2018.** <https://arxiv.org/abs/1811.00386>
The main text, and two subsections of it are the whole read. §3 is the
derivation: 3.1 sets the notation, 3.2 builds the complementary filter and
the ODE behind it, which is the same ladder as the TODOs already sitting in
`maa/reconstruct.py`. §4.2 is the practical half, a per-pixel asynchronous
update with a closed form between events, written out as Algorithm 1. Code at
github.com/cedric-scheerlinck/dvs_image_reconstruction — read the math first,
the code after yours works.

**Brandli, Muller, Delbrück — *Real-time, high-speed video decompression using
a frame- and event-based DAVIS sensor*, ISCAS 2014.**
The simplest published direct integration, with periodic frame resets. Short,
and honest about how fast it drifts.

**Gallego et al. — *Event-based Vision: A Survey*, T-PAMI 2020.**
<https://arxiv.org/abs/1904.08405>
Two subsections, ten minutes. §II-D, the event generation model, is the
canonical statement of what `simulate` already does, reset behaviour and all.
§IV-F, image reconstruction, puts direct integration, filtering, variational
and learned methods on one map, so you know what you are choosing not to
build.

Worth holding while reading: does a reconstruction recover the intensity it
lost on a fast edge once the motion stops? what does an all-ON noise floor do
to a buffer whose only move is to sum? log or linear on the way out, and what
is the initial condition when the stream never carries absolute intensity?

## reproduce_v2e

**Hu, Liu, Delbrück — *v2e: From Video Frames to Realistic DVS Events*, CVPRW
2021.** <https://arxiv.org/abs/2006.07722>
The thing being diffed against, so the requirements list comes out of here.
Code at github.com/SensorsINI/v2e. §III is the noise model — leak versus shot,
and why shot noise climbs in the dark.

## sophistication

**Lichtsteiner, Posch, Delbrück — *A 128×128 120 dB 15 µs Latency Asynchronous
Temporal Contrast Vision Sensor*, JSSC 43(2), 2008.**
The original DVS. Where thresholds, comparator mismatch (FPN) and
intensity-dependent latency come from in silicon, rather than as knobs. Two
parts matter: the pixel circuit, and the measured characteristics — the
mismatch numbers set the spread in `make_threshold_map`, latency against
illumination sets the bandwidth row. Skip fabrication and layout.

**NumPy — broadcasting rules.**
<https://numpy.org/doc/stable/user/basics.broadcasting.html>
Not a paper, but the FPN row rests on it: `diff >= threshold_c` compares each
pixel against its own threshold, so a scalar `C` and an `(H, W)` map go
through the same line.

## v1 kernel

Read back in v0, when the event stream needed a shape. Comes back at the port.

**Wikipedia — *AoS and SoA*.** <https://en.wikipedia.org/wiki/AoS_and_SoA>
The vocabulary in two minutes. maa went SoA — four parallel arrays — because
every stage is a single-field sweep and NumPy is a SoA machine anyway.

**Fabian — *Data-Oriented Design*.** <https://www.dataorienteddesign.com/dodbook/>
The long form of the same argument, and optional next to Drepper — same
conclusions, less measurement. Mike Acton's CppCon 2014 talk *Data-Oriented
Design and C++* is the ninety-minute version if the book stalls.

**Drepper — *What Every Programmer Should Know About Memory*, 2007.**
<https://people.freebsd.org/~lstewart/articles/cpumemory.pdf>
Why any of it pays: cache lines, prefetch, the price of a random gather. The
`argsort` permutation in `noise()` is four gathers — this is the paper that
prices them, and the reason the refractory loop is the obvious first thing to
port. 114 pages, but §3 (CPU caches) and §6 (what programmers can do) carry
the whole argument; the DRAM, virtual-memory and NUMA chapters are optional.

## e2vid_bench

**Rebecq, Ranftl, Koltun, Scaramuzza — *High Speed and High Dynamic Range Video
with an Event Camera*, T-PAMI 2019.** <https://arxiv.org/abs/1906.07165>
The learned baseline the bench measures against. Code at
github.com/uzh-rpg/rpg_e2vid. Skim the intro now, read it properly then.

## more

github.com/uzh-rpg/event-based_vision_resources — the curated everything-list.

Skipped on purpose for v0: Munda, Reinbacher, Pock (variational, manifold
regularisation). Good work, wrong altitude until the naive integrator exists.
