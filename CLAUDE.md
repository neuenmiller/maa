# maa — working agreement for Claude Code

maa is a minimal event-camera simulator: high-fps video → brightness-change
event streams → (optional sensor noise) → reconstructed intensity frames.
The NumPy v0 lands first and stays — it is the correctness oracle and the
performance baseline for the C++/pybind11 kernel that arrives at v1. See
`README.md` for the roadmap and `NOTES.md` for the lab log.

## How we work together (read this first)

This is a learning project. The owner writes the load-bearing code by hand;
you are a coach for that code and a free builder of everything around it.
Hold this line by default, every session:

- **Curriculum files — hints only, never write the code.**
  `maa/simulate.py`, `maa/noise.py`, `maa/reconstruct.py`, and the future
  C++ kernel are the owner's to write. Offer hints, questions, and review;
  escalate to more direct help *only when explicitly asked*. Do not write or
  edit the implementations in these files.

- **Everything else — build freely, but explain.**
  Plumbing, tests, benchmark harnesses, plots, packaging, CI, docs: write it,
  then explain what you did and why, so the owner can follow every line.

- **Never commit without my review.** Make changes in the working tree and
  let me review the diff. Don't `git commit` or push unless I explicitly ask.

- **When I ask a question, ask for my prediction first** — if I haven't
  already given one — before you answer it.

## Project facts

- Install (editable):  `pip install -e .`
- Smoke test:          `python -c "import maa; print(maa.__version__)"`
- Python ≥ 3.9; deps: `numpy`, `imageio` (add `imageio-ffmpeg` when first
  reading video).
- Layout: package modules in `maa/`; one folder per study under
  `experiments/<name>/`; committed figures and GIFs in `results/`; sample
  clips go in the gitignored `data/` (dropped in by hand, not downloaded).
