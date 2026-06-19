"""Fetch datasets into data/.

The data/ directory is gitignored — we don't commit sample clips or event
recordings. Run this script to populate it locally instead.

Usage (once implemented):
    python data/fetch.py
"""

# TODO: define dataset registry (name -> URL + checksum).
# TODO: download into data/ with progress + checksum verification.
# TODO: skip files that already exist; --force to re-download.
# TODO: optionally extract archives.

if __name__ == "__main__":
    raise SystemExit("data/fetch.py is a stub — not implemented yet.")
