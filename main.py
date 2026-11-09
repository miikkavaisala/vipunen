"""Run the whole pipeline for one or more ArXiv archives.

Stages:
    1. RSS fetch/parse  — download today's feed if not already stored,
                          else load from disk (no network).
    2. One-liners       — id, title, authors per paper, 100-char cap,
                          written to data/oneliners/ and stdout.
    3. Summaries        — optional: ~2-sentence LLM summaries of a random
                          sample (off by default; enable with -n N). The
                          digest stage has replaced it as the main output.
    4. Filters          — criteria filters (PASS/FAIL) over all papers,
                          verdicts to data/filtered/ (--cascade: run them
                          in sequence over shrinking pools instead — same
                          overlap, fewer LLM calls, pool verdict files).
    5. Digest           — state-of-the-topic document for papers passing
                          every filter, written to data/digests/ and stdout.
                          With several archives and/or several days
                          (--date takes a day, a comma list, or a range
                          D1..D2), one combined digest covers the
                          cross-archive cross-day overlap (de-duplicated
                          by base arxiv_id, version-insensitive) instead
                          of one digest per archive-day.
    6. Deliverable      — data/deliverables/<digest file name>: the overlap
                          list and the digest combined into a single .md
                          (no LLM — existing files joined for convenience).

Usage:
    uv run python main.py                          # astro-ph, sample size 5
    uv run python main.py astro-ph.CO astro-ph.SR  # multiple archives
    uv run python main.py cs.LG cs.AI cs.CL --filters physics  # combined digest
    uv run python main.py -n 20                    # astro-ph, 20 summaries
    uv run python main.py --filters star-formation # one filter, digest on its passers
    uv run python main.py --date 20261019 --filters star-formation  # re-run a stored day offline
    uv run python main.py astro-ph --filters mhd --date 20261021..20261023  # multi-day sweep
    uv run python main.py --filters simulation llm --cascade  # sequential filter pools
    uv run python main.py --long                   # long digest: feature every surviving paper
    uv run python main.py --clustered              # cheaper every-paper digest: one clause per paper, by theme
    uv run python main.py --no-llm                  # skip all LLM stages
    uv run python main.py --llm-url http://localhost:1234/v1 --llm-model your-model  # per-run endpoint
    uv run python main.py --llm-url https://api.example.com/v1 --llm-model m --llm-api-key KEY  # hosted endpoint with key
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent / "src"))

from arxiv_rss import (
    day_spec_text,
    days_slug,
    papers_for_sweep,
    parse_days,
    valid_archive_name,
)
from deliverable import build_deliverable
from digest import combined_digest, digest
from filter import (
    AI_FILTERS,
    ASTRO_FILTERS,
    DEFAULT_FILTERS,
    FILTER_CATALOG,
    MATH_FILTERS,
    SUBJECT_FILTERS,
    combined_label,
    combined_overlap,
    overlap_of,
    run_filters,
    save_overlap,
)
from llm import add_llm_args, apply_llm_args
from oneliner import one_liner, one_liner_text, save_lines
from summarize import (
    metrics_report,
    save_metrics,
    save_summaries,
    summarize_papers,
)

# number of random papers per archive to pass through the LLM summary stage;
# 0 (off) is the default now that the digest stage produces the main output —
# enable with `main.py -n N` when a random-sample overview is wanted
SUMMARY_SAMPLE_SIZE = 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "archives",
        nargs="*",
        help=f"ArXiv archive or category names (default: astro-ph); "
        f"a sample size is given with -n, not as a bare number",
    )
    parser.add_argument(
        "-n",
        "--sample-size",
        type=int,
        default=SUMMARY_SAMPLE_SIZE,
        help=f"papers per archive for the optional random-sample summary stage "
        f"(default: {SUMMARY_SAMPLE_SIZE} = off; the digest is the main output)",
    )
    parser.add_argument(
        "--filters",
        default=",".join(f.name for f in DEFAULT_FILTERS),
        help="comma-separated filter names for the filter stage; the digest "
        "covers papers passing every listed filter "
        f"(defaults: {', '.join(f.name for f in DEFAULT_FILTERS)}; "
        f"named groups — astrophysics: {', '.join(f.name for f in ASTRO_FILTERS)}; "
        f"AI-focused: {', '.join(f.name for f in AI_FILTERS)}; "
        f"subject: {', '.join(f.name for f in SUBJECT_FILTERS)}; "
        f"mathematics: {', '.join(f.name for f in MATH_FILTERS)})",
    )
    parser.add_argument(
        "--date",
        help="YYYYMMDD of a stored day, a comma-separated list, or an "
        "inclusive range D1..D2: run every stage on those days' stored "
        "papers, fully offline on the data side (LLM calls still run). "
        "Several days give per-day outputs plus one combined digest over "
        "the de-duplicated union",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="skip all LLM stages (summaries, filters, digest)",
    )
    parser.add_argument(
        "--long",
        action="store_true",
        help="long digest: feature every paper passing the filters "
        "(default: representative selection)",
    )
    parser.add_argument(
        "--clustered",
        action="store_true",
        help="clustered digest: cover every paper passing the filters with "
        "one clause each, grouped by theme (cheaper than --long)",
    )
    parser.add_argument(
        "--cascade",
        action="store_true",
        help="run the filters in sequence, each judging only the previous "
        "filter's PASS pool (cheapest discriminator first), instead of "
        "every filter over the full list; stores cascade pool verdict "
        "files instead of full-coverage verdicts — the overlap is the "
        "same either way (modulo verdict wobble)",
    )
    add_llm_args(parser)
    args = parser.parse_args(argv)
    if not args.archives:
        args.archives = ["astro-ph"]
    if args.long and args.clustered:
        parser.error("--long and --clustered are mutually exclusive")
    # canonical (sorted) archive order for all combined file names, so the
    # same combination maps to one name regardless of command-line order
    args.archives = sorted(args.archives)
    invalid = [a for a in args.archives if not valid_archive_name(a)]
    if invalid:
        parser.error(
            f"invalid archive name(s): {', '.join(invalid)}\n"
            f"expected e.g. astro-ph, astro-ph.CO, cs.LG — "
            f"use -n for the summary sample size"
        )
    known = set(FILTER_CATALOG)
    args.filter_names = [name.strip() for name in args.filters.split(",") if name.strip()]
    unknown = set(args.filter_names) - known
    if unknown:
        parser.error(
            f"unknown filter name(s): {', '.join(sorted(unknown))}; "
            f"known: {', '.join(sorted(known))}"
        )
    return args


def _one_liner_stage(
    papers: list, archive: str, date: str | None, console: Console
) -> None:
    """Print and save the day's one-liners for one archive."""
    source = f"stored day {date}" if date else "today"
    lines = [one_liner(paper, max_len=None) for paper in papers]
    path = save_lines(lines, archive, date=date)
    console.print(f"[bold]{archive}[/bold]: {len(papers)} papers ({source})")
    for paper in papers:
        console.print(one_liner_text(paper), no_wrap=True, overflow="ignore")
    console.print(f"saved: {path}")


def _summary_stage(
    papers: list, archive: str, sample_size: int, date: str | None, console: Console
) -> None:
    """Optional random-sample LLM summaries of one archive's papers."""
    sample = random.sample(papers, min(sample_size, len(papers)))
    console.print(f"[bold]{archive}[/bold]: summarizing {len(sample)} of {len(papers)} papers")
    entries, metrics = summarize_papers(sample, desc=f"summarize {archive}")
    for _arxiv_id, summary in entries:
        console.print(f"\n{summary}", style="bright_white")
    console.print("\n[bold]--- summary metrics ---[/bold]")
    console.print(metrics_report(metrics))
    path = save_summaries(entries, archive, date=date)
    console.print(f"\nsaved: {path}")
    metrics_path = save_metrics(metrics, archive, date=date)
    console.print(f"saved: {metrics_path}")


def _filter_stage(
    papers: list,
    archive: str,
    filter_defs: list,
    console: Console,
    date: str | None,
    cascade: bool = False,
) -> set[str]:
    """Run the filters over one archive's papers; return the overlap
    (ids that passed every filter)."""
    mode = " (cascade)" if cascade else ""
    console.print(f"\n[bold]{archive}[/bold]: filtering {len(papers)} papers{mode}")
    passed = run_filters(
        papers, filter_defs, archive, console, date=date, cascade=cascade
    )
    for name, ids in passed.items():
        console.print(f"{name}: {len(ids)} passed", style="dim")
    return overlap_of(passed)


def _print_digest(text: str, digest_path: Path, filter_defs: list, console: Console) -> None:
    """Print a digest the same way for single-archive and combined runs."""
    console.print(f"\n[bold]# State of the topic: {', '.join(f.name for f in filter_defs)}[/bold]\n")
    console.print(text)
    console.print(f"\nsaved: {digest_path}")


def _combined_digest_stage(
    archives: list[str],
    days: list[str | None],
    filter_defs: list,
    overlaps: dict[tuple[str, str | None], set[str]],
    titles: dict[str, str],
    console: Console,
    long: bool,
    clustered: bool = False,
    cascade: bool = False,
) -> None:
    """One digest over the cross-archive cross-day overlap — papers from
    any of the lists on any of the days passing every filter, de-duplicated
    by base arxiv_id (version-insensitive)."""
    combined = combined_overlap(list(overlaps.values()))
    console.print(
        f"\n[bold]--- combined overlap ({len(combined)} papers pass all filters "
        f"on any of {len(overlaps)} feeds) ---[/bold]"
    )
    for pid in sorted(combined):
        console.print(f"[{pid}] {titles[pid]}", markup=False, no_wrap=True, overflow="ignore")
    overlap_path = save_overlap(
        combined, titles, combined_label(archives), [f.name for f in filter_defs], date=days_slug(days)
    )
    console.print(f"saved: {overlap_path}")

    if not combined:
        console.print(
            "no papers passed every filter on any feed — combined digest skipped",
            style="dim",
        )
        return
    console.print(f"\n[bold]combined: digesting {len(combined)} papers")
    text, _survivors, digest_path = combined_digest(
        archives,
        [f.name for f in filter_defs],
        days=days,
        long=long,
        clustered=clustered,
        cascade=cascade,
    )
    _print_digest(text, digest_path, filter_defs, console)
    deliverable_path = build_deliverable(
        overlap_path,
        digest_path,
        title=(
            f"{'+'.join(archives)}, {day_spec_text(days)} — "
            f"filters: {', '.join(f.name for f in filter_defs)}"
        ),
    )
    console.print(f"saved: {deliverable_path}")


def run_pipeline(
    archives: list[str],
    sample_size: int = SUMMARY_SAMPLE_SIZE,
    filter_names: list[str] | None = None,
    date: str | None = None,
    long: bool = False,
    clustered: bool = False,
    cascade: bool = False,
) -> None:
    # canonical (sorted) archive order for all combined file names
    archives = sorted(archives)
    days = parse_days(date) if date else [None]
    filter_defs = [FILTER_CATALOG[name] for name in (filter_names or [])]
    multi = len(archives) > 1 or len(days) > 1
    # keep lines intact: one-liners are capped at 100 chars, so never let a
    # narrow console crop or wrap them
    console = Console(width=max(100, Console().width))
    overlaps: dict[tuple[str, str | None], set[str]] = {}
    titles: dict[str, str] = {}
    for archive in archives:
        for day in days:
            papers = papers_for_sweep(
                archive,
                day,
                multi,
                on_missing=lambda a, d: console.print(
                    f"no stored papers for {a} on {d or 'today'} — skipped",
                    style="dim",
                ),
            )
            if papers is None:
                continue  # a multi sweep runs on whatever is stored
            titles.update({p.arxiv_id: p.title for p in papers})

            _one_liner_stage(papers, archive, day, console)
            if sample_size > 0:
                _summary_stage(papers, archive, sample_size, day, console)

            if not filter_defs:
                continue
            overlaps[(archive, day)] = _filter_stage(
                papers, archive, filter_defs, console, day, cascade=cascade
            )

            if multi:
                # per-archive-day digests are replaced by one combined
                # digest below
                continue
            overlap = overlaps[(archive, day)]
            overlap_path = save_overlap(
                overlap, titles, archive, [f.name for f in filter_defs], date=day
            )
            console.print(f"saved: {overlap_path}")
            if not overlap:
                console.print(
                    f"[bold]{archive}[/bold]: no papers passed every filter — digest skipped",
                    style="dim",
                )
                continue
            console.print(f"\n[bold]{archive}[/bold]: digesting {len(overlap)} papers")
            text, _survivors, digest_path = digest(
                archive,
                [f.name for f in filter_defs],
                days=[day],
                long=long,
                clustered=clustered,
                cascade=cascade,
            )
            _print_digest(text, digest_path, filter_defs, console)
            deliverable_path = build_deliverable(
                overlap_path,
                digest_path,
                title=(
                    f"{archive}, {day_spec_text([day])} — "
                    f"filters: {', '.join(f.name for f in filter_defs)}"
                ),
            )
            console.print(f"saved: {deliverable_path}")

    if multi and filter_defs:
        _combined_digest_stage(
            archives, days, filter_defs, overlaps, titles, console, long,
            clustered=clustered, cascade=cascade,
        )


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    apply_llm_args(args)
    sample_size = 0 if args.no_llm else args.sample_size
    filter_names = [] if args.no_llm else args.filter_names
    try:
        run_pipeline(
            args.archives,
            sample_size,
            filter_names,
            date=args.date,
            long=args.long,
            clustered=args.clustered,
            cascade=args.cascade,
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(exc)


if __name__ == "__main__":
    main(sys.argv[1:])
