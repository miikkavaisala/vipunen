"""Deliverable stage: the filtered article list and the digest in one .md.

Purpose: a "deliverable" document type — the final collected result file.
It combines two existing pipeline artifacts into a single markdown file
for convenience, with no LLM use at all:

- the filtered article list — the overlap file data/filtered/
  (<archive>-<yyyymmdd>-<filters>-overlap.txt, or
  combined-<archives>-<dayslug>-<filters>-overlap.txt for multi-day /
  multi-archive runs) written by the filter stage;
- the state-of-the-topic digest data/digests/<archive>-<day>-<filters>
  [-long|-clustered][-NN].md written by the digest stage (the latest numbered one).

The result is written to data/deliverables/ under the digest's own file
name, so every digest has exactly one matching deliverable and reruns
never overwrite each other:
    data/deliverables/<archive>-<yyyymmdd>-<filters>[-long|-clustered][-NN].md

main.py writes the deliverable automatically after every digest. This
module's CLI rebuilds one offline from stored files (no LLM, no network):

Usage:
    uv run python src/deliverable.py [archive] [filter ...] [--date D] [--long | --clustered]
    # default: astro-ph, default filters, today
    # e.g. uv run python src/deliverable.py astro-ph star-formation --date 20261019
    # --date also accepts comma lists and ranges: --date 20261021..20261023
"""

from __future__ import annotations

import argparse
import re
import sys
from itertools import permutations
from pathlib import Path

from rich.console import Console

from arxiv_rss import canonical_archive_arg, day_spec_text, days_slug, parse_days
from digest import digest_basename
from filter import DEFAULT_FILTERS, FILTER_CATALOG, combined_label, overlap_filename


def overlap_path_for(
    archive: str,
    filter_names: list[str],
    days: list[str | None] | None = None,
    filtered_dir: str | Path = "data/filtered",
    canonical: bool = True,
) -> Path:
    """Path of the overlap file for one archive (or the combined overlap
    for a multi-day / multi-archive run): <filtered_dir>/<archive>-<day>-
    <filters>-overlap.txt, or combined-<archives>-<dayslug>-<filters>-
    overlap.txt when the run was combined (`archive` is then the '+'-joined
    archive list, as in the combined digest's file name). Falls back to the
    legacy combined-<dayslug>-<filters>-overlap.txt name (no archive list)
    so overlaps written before the archive-aware name stay rebuildable.
    With canonical=True the archive list is sorted into canonical order;
    canonical=False keeps the order exactly as given, to find files written
    before canonical naming. Raises FileNotFoundError when nothing matches."""
    day = days_slug(days) if days else None
    combined = (days is not None and len(days) > 1) or "+" in archive
    label = combined_label(archive.split("+"), sort=canonical) if combined else archive
    path = Path(filtered_dir) / overlap_filename(label, filter_names, day)
    if not path.exists() and combined:
        legacy = Path(filtered_dir) / overlap_filename("combined", filter_names, day)
        if legacy.exists():
            return legacy
    if not path.exists():
        raise FileNotFoundError(
            f"no overlap file {path}\n"
            f"run the filter stage first: uv run python src/filter.py {archive} "
            f"{' '.join(filter_names)}" + (f" --date {day}" if days else "")
        )
    return path


def find_latest_digest(
    archive: str,
    filter_names: list[str],
    days: list[str | None] | None = None,
    long: bool = False,
    clustered: bool = False,
    digests_dir: str | Path = "data/digests",
) -> Path:
    """The newest digest file for (archive, days, filters[, long/clustered]).

    Digest reruns are numbered (-01, -02, ...), never overwritten, so the
    latest is the base name when present, else the highest number.
    Raises FileNotFoundError with a hint when no digest matches."""
    base = digest_basename(archive, filter_names, days, long=long, clustered=clustered)
    d = Path(digests_dir)
    candidates: list[tuple[int, Path]] = []
    plain = d / f"{base}.md"
    if plain.exists():
        candidates.append((0, plain))
    pattern = re.compile(re.escape(base) + r"-(\d+)\.md$")
    if d.is_dir():
        for p in d.iterdir():
            m = pattern.match(p.name)
            if m:
                candidates.append((int(m.group(1)), p))
    if not candidates:
        raise FileNotFoundError(
            f"no digest matching {base}*.md in {d}\n"
            f"run the digest stage first: uv run python src/digest.py {archive} "
            f"{' '.join(filter_names)}"
            + (f" --date {days_slug(days)}" if days else "")
            + (" --long" if long else "")
            + (" --clustered" if clustered else "")
        )
    return max(candidates)[1]


def build_deliverable(
    overlap_path: str | Path,
    digest_path: str | Path,
    title: str | None = None,
    out_dir: str | Path = "data/deliverables",
) -> Path:
    """Combine one overlap file and one digest file into a single .md.

    Pure file concatenation — no LLM. The output goes to
    <out_dir>/<digest file name>, so the deliverable's stem always matches
    the digest it embeds. Returns the written path.
    """
    overlap_path = Path(overlap_path)
    digest_path = Path(digest_path)
    overlap_lines = [
        line for line in overlap_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    digest_text = digest_path.read_text(encoding="utf-8").rstrip("\n")
    title = title or digest_path.stem

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / digest_path.name
    header = (
        f"# Deliverable: {title}\n\n"
        "Single-file convenience combination of the pipeline's two result\n"
        "artifacts — the filtered article list and the state-of-the-topic\n"
        "digest. Assembled from existing files; no LLM involved.\n\n"
        f"- Filtered list ({len(overlap_lines)} papers): `{overlap_path}`\n"
        f"- Digest: `{digest_path}`\n\n"
        f"## Filtered articles ({len(overlap_lines)} papers passing every filter)\n\n"
    )
    text = f"{header}{chr(10).join(overlap_lines)}\n\n## Digest\n\n{digest_text}\n"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archive", nargs="?", default="astro-ph")
    parser.add_argument("filters", nargs="*", help="filter names (default: the general defaults)")
    parser.add_argument(
        "--date",
        help="YYYYMMDD of a stored day, a comma-separated list, or an "
        "inclusive range D1..D2 (default: today)",
    )
    parser.add_argument(
        "--long",
        action="store_true",
        help="use the long digest (feature every surviving paper)",
    )
    parser.add_argument(
        "--clustered",
        action="store_true",
        help="use the clustered digest (every paper, one clause each)",
    )
    args = parser.parse_args()

    if args.long and args.clustered:
        parser.error("--long and --clustered are mutually exclusive")
    chosen = args.filters or [f.name for f in DEFAULT_FILTERS]
    unknown = set(chosen) - set(FILTER_CATALOG)
    if unknown:
        sys.exit(f"unknown filter(s): {', '.join(sorted(unknown))}")

    console = Console(width=max(100, Console().width))
    try:
        days = parse_days(args.date) if args.date else None
        # try the canonical (sorted) archive label first, then every other
        # ordering, so files written before canonical naming (when the
        # command-line order went straight into the file name) are found
        # no matter which order the archive list is typed in
        if "+" in args.archive:
            parts = args.archive.split("+")
            candidates = ["+".join(p) for p in ([tuple(sorted(parts))] + list(permutations(parts)))]
            candidates = list(dict.fromkeys(candidates))
        else:
            candidates = [args.archive]
        candidates = [canonical_archive_arg(args.archive)] + [
            c for c in candidates if c != canonical_archive_arg(args.archive)
        ]
        overlap_path = digest_path = None
        for label in candidates:
            try:
                overlap_path = overlap_path_for(label, chosen, days, canonical=False)
                digest_path = find_latest_digest(
                    label, chosen, days, long=args.long, clustered=args.clustered
                )
                break
            except FileNotFoundError:
                if label is candidates[-1]:
                    raise
        title = (
            f"{args.archive}, {day_spec_text(days) if days else 'today'} — "
            f"filters: {', '.join(chosen)}"
        )
        path = build_deliverable(overlap_path, digest_path, title=title)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(exc)

    n = len([l for l in overlap_path.read_text(encoding="utf-8").splitlines() if l.strip()])
    console.print(
        f"deliverable: {n} filtered papers + digest `{digest_path.name}`\n"
        f"saved: {path}"
    )


if __name__ == "__main__":
    main()
