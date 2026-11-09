"""Fetch stage: download every RSS feed listed in feeds.toml.

Purpose: regular data collection for its own sake — keep the local
archive growing day by day so later method testing, calibration, and
offline reprocessing always have material, regardless of what the
analysis pipeline currently looks at. No LLM, no filtering, no digest.

Reads the `archives` list from feeds.toml (repo root by default) and
runs each entry through `load_or_fetch(archive)` — cache-first, so
already-stored days are served from disk and only missing feeds are
downloaded. All requests stay sequential and throttled by the shared
ArxivRSS throttle; do not run this at the same time as another
ArXiv-fetching process.

Usage:
    uv run python src/fetch_feeds.py                 # all feeds in feeds.toml
    uv run python src/fetch_feeds.py --feeds my.toml # a different list
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from arxiv_rss import load_or_fetch, valid_archive_name

DEFAULT_FEEDS_FILE = "feeds.toml"


def load_archive_list(path: Path) -> list[str]:
    """Read the `archives` list from a feeds TOML file."""
    if not path.exists():
        raise FileNotFoundError(f"feeds file not found: {path}")
    with open(path, "rb") as fh:
        config = tomllib.load(fh)
    archives = config.get("archives")
    if not isinstance(archives, list) or not archives:
        raise ValueError(f"{path}: no non-empty 'archives' list — nothing to fetch")
    invalid = [a for a in archives if not valid_archive_name(a)]
    if invalid:
        raise ValueError(
            f"{path}: invalid archive name(s): {', '.join(invalid)}\n"
            f"expected e.g. astro-ph, astro-ph.CO, cs.LG"
        )
    # de-duplicate while preserving order; a repeat would only hit the cache
    return list(dict.fromkeys(archives))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--feeds",
        default=DEFAULT_FEEDS_FILE,
        help=f"feeds TOML file with the archive list (default: {DEFAULT_FEEDS_FILE})",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    try:
        archives = load_archive_list(Path(args.feeds))
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(exc)
    print(f"fetching {len(archives)} feed(s) from {args.feeds}: {', '.join(archives)}")

    failures: list[str] = []
    for archive in archives:
        try:
            papers, source = load_or_fetch(archive)
        except RuntimeError as exc:
            print(f"{archive}: FAILED — {exc}")
            failures.append(archive)
            continue
        print(f"{archive}: {len(papers)} papers ({source})")

    if failures:
        sys.exit(f"failed feed(s): {', '.join(failures)}")
    print("done")


if __name__ == "__main__":
    main(sys.argv[1:])
