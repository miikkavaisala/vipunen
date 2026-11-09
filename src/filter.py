"""Filter stage: per-criteria verdicts over a whole archive via LLM calls.

Purpose: the volume-reduction stage of the pipeline. Each filter is one
cheap call per abstract (thinking off): the model must answer with exactly
one line starting with the filter's pass word or fail word plus a short
reason — a format that can be parsed mechanically. Replies without a
parseable verdict rerun the prompt up to MAX_ATTEMPTS times.

Filter criteria are user intent, not code: they live as plain text
files in the repo's filter/ directory — filter/<group>/<name>.txt holds
one filter's criteria wording, editable without touching Python. The
file stem is the filter name (selectable via the CLI or main.py
--filters); groups: general (the default set), ai, astro, subject, and
math. Add a filter by dropping a file in a group directory. All verdicts
use one vocabulary — PASS / FAIL — never mixed. New or reworded filters
must be calibrated: run over a
known list and inspect verdicts and reasons before trusting them.

Keyword pre-filter (the two-tier funnel from doc/IDEA.md): a filter may
carry an optional keyword list in filter/<group>/<name>.keywords — one
phrase per line (# comments, blank lines ignored). Papers whose title +
abstract contain none of the phrases are FAILED instantly, at zero LLM
cost, before the filter call ever runs; the verdict file records them
with the reason "keyword prefilter: no vocabulary hit". Keywords are
near-lossless for obvious irrelevance but bad at topicality, so the list
should stay recall-first: let it over-accept and let the LLM filter
resolve topicality on the shortlist. A filter without a .keywords file
runs over every paper as before. Keyword lists must be calibrated like
criteria wording: check the stored PASS verdicts of the filter for
keyword misses and tune until the loss is acceptable.

Verdicts are written per filter to
data/filtered/<archive>-<yyyymmdd>-<filter>.txt, the overlap (papers passing
every filter) to data/filtered/<archive>-<yyyymmdd>-<filters>-overlap.txt —
the filter set is part of the name, so different filter sets on the same
day never overwrite each other.

Cascade mode (--cascade / run_filters(cascade=True)): when a run combines
filters, run them in sequence instead of all over the full list — each
filter judges only the previous filter's PASS pool, and the overlap is
the last filter's passes (mathematically the same set as the parallel
intersection: a paper some filter fails is in neither). At each step the
cheapest discriminator runs next — the remaining filter with the smallest
judged set (keyword hits within the pool; the whole pool without a
keyword list), a zero-cost prediction of the LLM calls ahead. Keyword
pre-filters compose naturally: each filter's gate applies to its incoming
pool. The switch is an artifact choice, not a correctness tradeoff:
parallel mode (default) keeps full-coverage verdict files (the audit
trail), cascade mode stores only the cascade pools, as
data/filtered/<archive>-<yyyymmdd>-<filter>.cascade.txt so the two modes
never overwrite each other's artifacts. Verdict wobble still makes
borderline flips differ between cascade and parallel runs even when the
sets are mathematically equivalent.

Verdict reuse (the verdict-file analogue of load_or_fetch): a filter's
verdict for a paper is a property of (criteria wording, paper) — not of
the pool the paper arrived in — so stored verdicts are a cache. Before a
filter judges a pool, any verdicts already stored for the same
(archive, day, filter) are loaded (full-coverage file first, then the
cascade pool artifact) and only papers without a stored verdict go to
the LLM; reused and fresh verdicts are merged back in input order. A
repeat query over the same day costs zero LLM calls, and a cascade
stage whose pool was already judged in full coverage reuses those
verdicts instead of duplicating them. The cache is guarded by a
criteria+model fingerprint: every verdict file starts with a
`# filter-fingerprint:` line (a hash of the criteria wording and the
judging model) plus a `# filter-model:` line naming the model, and a
file whose fingerprint no longer matches the current wording and model
is not reused — criteria wording is the quality lever, so a reworded
filter re-judges everything, and switching LLM_MODEL re-judges too
instead of silently reusing another model's verdicts. Header-less
files (written before verdict reuse existed) are also never reused;
they are re-judged once and rewritten with their fingerprint.
Keyword-gate FAILs are not cached — the gate is recomputed live at zero
cost, so editing a keyword list never leaves a stale gate verdict in
the cache.

With several archives, one extra file
holds the combined overlap — papers from any of the lists passing every
filter, de-duplicated by base arxiv id — version
suffix stripped, so cross-day re-announcements count once (papers are cross-listed, so the same
paper can appear in several feeds on the same day):
data/filtered/combined-<archives>-<yyyymmdd>-<filters>-overlap.txt
(archives joined by '+', so different archive combinations on the same
day get different files).

Usage:
    uv run python src/filter.py [archive ...] [filter ...]   # default: astro-ph, all filters
    uv run python src/filter.py astro-ph.CO astro-ph simulation --date 20261019
    uv run python src/filter.py astro-ph mhd --date 20261021..20261023  # multi-day sweep
    uv run python src/filter.py astro-ph simulation llm --cascade  # sequential pools
    # known filter names select filters; the other arguments are archives
    # --date takes one day, a comma-separated list, or an inclusive range;
    # multi-day sweeps also write one combined overlap (de-duplicated by
    # arxiv_id) and skip archive-days with no stored papers
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from tqdm import tqdm

from arxiv_rss import (
    ArxivPaper,
    base_id,
    day_spec_text,
    day_stamp,
    days_slug,
    papers_for_sweep,
    parse_days,
    valid_archive_name,
)
from llm import add_llm_args, apply_llm_args, chat_with_metrics, current_model

# filter calls are the cheapest pipeline calls: answer directly, no reasoning
THINK = False
# if a reply has no parseable verdict, the filter prompt is rerun up to this
# many total attempts before giving up with verdict ERROR
MAX_ATTEMPTS = 3
# cap the reply so a runaway generation stops cleanly (finish_reason
# "length") instead of burning the whole context window and crashing the
# server's inference engine mid-request; a verdict plus a one-line reason
# needs far less than 256 tokens
MAX_TOKENS = 256

# one verdict vocabulary for every filter — never mix terminology
PASS_WORD = "PASS"
FAIL_WORD = "FAIL"


@dataclass
class FilterDef:
    """One named criteria filter. Verdicts are always PASS or FAIL.

    `keywords` is the optional zero-cost pre-filter (see module docstring):
    papers whose title + abstract hit no keyword skip the LLM call and
    FAIL immediately. Empty list = no pre-filter, every paper judged by
    the LLM.
    """

    name: str
    criteria: str
    keywords: list[str] = field(default_factory=list)


# reason recorded for papers the keyword pre-filter discards
KEYWORD_MISS = "keyword prefilter: no vocabulary hit"


# Filter criteria are user intent, not code: they live as editable text
# files in the repo's filter/ directory, one per filter —
# filter/<group>/<name>.txt contains the criteria wording (whitespace is
# normalized on load; the name is the file stem, selectable via the CLI).
# Add a filter by dropping a file in a group directory; edit the wording
# by editing the file. Group directories:
#   general  — the default filters (run when no filter is named)
#   ai       — AI-focused filters for cs.LG / cs.AI / cs.CL pages
#   astro    — astrophysics topic and methodology filters
#   subject  — physics/astronomy papers on method-focused pages
#   math     — mathematics topic filters
FILTERS_DIR = Path(__file__).resolve().parent.parent / "filter"
_NAME_RE = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


def _load_keywords(group: str, name: str) -> list[str]:
    """Load a filter's optional keyword list from
    `filter/<group>/<name>.keywords`: one phrase per line, # comments and
    blank lines ignored. Missing file = empty list = no pre-filter."""
    path = FILTERS_DIR / group / f"{name}.keywords"
    if not path.exists():
        return []
    phrases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        phrase = line.split("#", 1)[0].strip().lower()
        if phrase:
            phrases.append(phrase)
    if not phrases:
        raise RuntimeError(f"empty keyword file: {path}")
    return phrases


def _load_group(group: str) -> list[FilterDef]:
    """Load one group's filters from `filter/<group>/*.txt`.

    The file stem is the filter name; the file content is the criteria
    wording, normalized to single-spaced prose so files can be freely
    line-wrapped. A sibling `<stem>.keywords` file, when present, becomes
    the filter's zero-cost keyword pre-filter.
    """
    gdir = FILTERS_DIR / group
    if not gdir.is_dir():
        raise RuntimeError(f"filter group directory not found: {gdir}")
    fdefs: list[FilterDef] = []
    for fpath in sorted(gdir.glob("*.txt")):
        if not _NAME_RE.fullmatch(fpath.stem):
            raise RuntimeError(
                f"bad filter file name (lowercase words with hyphens): {fpath}"
            )
        criteria = " ".join(fpath.read_text(encoding="utf-8").split())
        if not criteria:
            raise RuntimeError(f"empty filter file: {fpath}")
        fdefs.append(
            FilterDef(
                name=fpath.stem,
                criteria=criteria,
                keywords=_load_keywords(group, fpath.stem),
            )
        )
    if not fdefs:
        raise RuntimeError(f"no filter files in {gdir}")
    return fdefs


DEFAULT_FILTERS = _load_group("general")
AI_FILTERS = _load_group("ai")
ASTRO_FILTERS = _load_group("astro")
SUBJECT_FILTERS = _load_group("subject")
MATH_FILTERS = _load_group("math")

# every known filter, by name
FILTER_CATALOG = {
    f.name: f
    for f in [
        *DEFAULT_FILTERS,
        *AI_FILTERS,
        *ASTRO_FILTERS,
        *SUBJECT_FILTERS,
        *MATH_FILTERS,
    ]
}

PROMPT = """You are a strict filter in an arXiv monitoring pipeline.

Criteria: {criteria}.
If the criteria is fulfilled, the verdict is {pass_word}.
If it is not, the verdict is {fail_word}.
Judge only from the title and abstract below. Do not speculate.

Reply with exactly one line in this format and nothing else:
{pass_word}: <short reason, max 10 words>
or
{fail_word}: <short reason, max 10 words>

Title: {title}
Abstract: {abstract}"""

# every filter speaks the same verdict vocabulary
_VERDICT = re.compile(rf"^\s*({PASS_WORD}|{FAIL_WORD})\b[:\s-]*", re.IGNORECASE)


def filter_paper(paper: ArxivPaper, fdef: FilterDef) -> tuple[str, str, dict]:
    """One LLM call: PASS or FAIL the paper against the filter's criteria.

    If the reply does not start with PASS or FAIL, the prompt is rerun up
    to MAX_ATTEMPTS times. Returns (verdict, reason, metrics). Verdict is
    "PASS", "FAIL", or "ERROR" when every attempt failed the format check.
    """
    prompt = PROMPT.format(
        criteria=fdef.criteria,
        pass_word=PASS_WORD,
        fail_word=FAIL_WORD,
        title=paper.title,
        abstract=paper.abstract,
    )
    t0 = time.monotonic()

    usage: dict = {}
    verdict, reason, attempts, text = "ERROR", "", 0, ""
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        try:
            reply, usage = chat_with_metrics(
                [{"role": "user", "content": prompt}], think=THINK, max_tokens=MAX_TOKENS
            )
        except RuntimeError as exc:
            # transient LLM endpoint error (e.g. server busy) — retry; on
            # the final attempt, record an ERROR verdict like unparseable
            # replies instead of aborting the whole filter run
            if attempts >= MAX_ATTEMPTS:
                reason = " ".join(str(exc).split())[:80]
                break
            continue
        text = reply.strip()
        match = _VERDICT.match(text)
        if match:
            verdict = match.group(1).upper()
            # one line: a multi-line reason would write extra "[id] verdict:"
            # lines into the verdict file, where they would be re-parsed as
            # verdicts on reload (cache poisoning). Whitespace collapse only —
            # no truncation, the stored reason is the audit trail.
            reason = " ".join(text[match.end():].split())
            break
    else:
        # unparseable reply — same one-line normalization for the file
        # (the 80-char cap is pre-existing; this text is junk anyway)
        reason = " ".join(text.split())[:80]

    wall_seconds = time.monotonic() - t0

    details = usage.get("completion_tokens_details") or {}
    metrics = {
        "arxiv_id": paper.arxiv_id,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "reasoning_tokens": details.get("reasoning_tokens", 0),
        "wall_seconds": round(wall_seconds, 2),
        "attempts": attempts,
    }
    return verdict, reason, metrics


def filter_papers(
    papers: list[ArxivPaper], fdef: FilterDef
) -> tuple[list[tuple[str, str, str]], list[dict]]:
    """Filter each paper in `papers` against the filter's criteria.

    Returns (verdicts, metrics): verdicts as (arxiv_id, verdict, reason)
    triples in input order, and per-call metrics dicts.
    """
    verdicts: list[tuple[str, str, str]] = []
    metrics: list[dict] = []
    for paper in tqdm(papers, desc=f"filter: {fdef.name}", unit="paper"):
        verdict, reason, m = filter_paper(paper, fdef)
        verdicts.append((paper.arxiv_id, verdict, reason))
        metrics.append(m)
    return verdicts, metrics


def _verdict_path(
    archive: str, stem: str, out_dir: str | Path, date: str | None
) -> Path:
    """Path of one filter's verdict file: `<out_dir>/<archive>-<date>-<stem>.txt`."""
    return Path(out_dir) / f"{archive}-{day_stamp(date)}-{stem}.txt"


def criteria_fingerprint(fdef: FilterDef) -> str:
    """Short hash of a filter's criteria wording. Stored verdicts are only
    reusable while the wording they were judged under is the wording that
    would be used now, so every verdict file records this fingerprint and
    a mismatch means the file is re-judged."""
    return hashlib.sha1(fdef.criteria.encode("utf-8")).hexdigest()[:12]


def verdict_fingerprint(fdef: FilterDef, model: str) -> str:
    """Short hash of (criteria wording, judging model). A stored verdict is
    a property of both, so a verdict file is reusable only while its
    fingerprint matches the fingerprint the current wording and the current
    model would produce now — a reworded filter or a different LLM
    re-judges everything. Files written before the model was part of the
    fingerprint have a criteria-only hash, so they mismatch once and are
    re-judged and rewritten."""
    return hashlib.sha1(
        f"{criteria_fingerprint(fdef)}\n{model}".encode("utf-8")
    ).hexdigest()[:12]


# first line of every verdict file since verdict reuse exists
FINGERPRINT_RE = re.compile(r"^# filter-fingerprint:\s*(\S+)")
# one stored verdict line: `[<arxiv_id>] PASS|FAIL: <reason>`
VERDICT_LINE_RE = re.compile(r"^\[(\S+)\]\s+(PASS|FAIL):\s*(.*)$")


def load_cached_verdicts(
    archive: str,
    fdef: FilterDef,
    out_dir: str | Path = "data/filtered",
    date: str | None = None,
    model: str | None = None,
) -> dict[str, tuple[str, str]]:
    """Reusable verdicts for (archive, day, filter) from previous runs.

    Sources in order: the full-coverage file, then the cascade pool
    artifact — a paper judged in either is not judged again. Only files
    whose fingerprint line matches the current (criteria wording, model)
    fingerprint are trusted: a reworded filter or a different judging LLM
    re-judges everything, and header-less files (written before verdict
    reuse existed) are re-judged once and rewritten. Keyword-gate FAILs
    are never cached — the gate is recomputed live at zero cost, so
    keyword edits cannot leave a stale gate verdict in the cache.
    Returns arxiv_id -> (verdict, reason) for LLM-judged PASS/FAIL lines
    only (ERROR lines are not cached either; they get a fresh attempt)."""
    cache: dict[str, tuple[str, str]] = {}
    want = verdict_fingerprint(fdef, current_model(model))
    for stem in (fdef.name, f"{fdef.name}.cascade"):
        path = _verdict_path(archive, stem, out_dir, date)
        if not path.exists():
            continue
        trusted = False
        for line in path.read_text(encoding="utf-8").splitlines():
            fingerprint_match = FINGERPRINT_RE.match(line)
            if fingerprint_match:
                trusted = fingerprint_match.group(1) == want
                continue
            if not trusted:
                break  # wording changed since this file was written
            verdict_match = VERDICT_LINE_RE.match(line)
            if verdict_match:
                pid, verdict, reason = verdict_match.groups()
                if reason == KEYWORD_MISS:
                    continue  # re-derived by the live keyword gate
                cache.setdefault(pid, (verdict, reason))
    return cache


def save_verdicts(
    verdicts: list[tuple[str, str, str]],
    archive: str,
    fdef: FilterDef,
    out_dir: str | Path = "data/filtered",
    date: str | None = None,
    cascade: bool = False,
    model: str | None = None,
) -> Path:
    """Write one verdict line per paper to `<out_dir>/<archive>-<date>-<filter>.txt`.

    Cascade mode writes `<filter>.cascade.txt` instead — the cascade pool
    artifact, never overwriting the parallel full-coverage file. The
    header records both the fingerprint (hash of criteria wording +
    judging model) and the model name itself, so a later run can tell
    whether the file is reusable."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = current_model(model)
    stem = f"{fdef.name}.cascade" if cascade else fdef.name
    path = _verdict_path(archive, stem, out_dir, date)
    lines = [
        f"# filter-fingerprint: {verdict_fingerprint(fdef, model)}",
        f"# filter-model: {model}",
    ]
    # the file format is one verdict per line: reasons are collapsed to a
    # single line here (no truncation), so a reply that continues onto a
    # second line in "[id] verdict: ..." form cannot be written as an extra
    # verdict line and re-parsed as one on reload (cache poisoning)
    lines.extend(
        f"[{pid}] {verdict}: {' '.join(reason.split())}"
        for pid, verdict, reason in verdicts
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def overlap_of(passed: dict[str, set[str]]) -> set[str]:
    """Papers that passed every filter — the reliable signal of the set.

    `passed` maps filter name -> the ids that passed it.
    """
    sets = list(passed.values())
    return set.intersection(*sets) if sets else set()


def combined_overlap(overlaps: list[set[str]]) -> set[str]:
    """Union of per-archive-day overlaps — papers from any list passing every
    filter. Union by base arxiv id (version suffix stripped) de-duplicates
    both cross-listed papers, which appear on several feeds the same day,
    and cross-day re-announcements under a different version number; one
    full id (the lowest, deterministically) is kept per paper."""
    union = set().union(*overlaps) if overlaps else set()
    by_base: dict[str, str] = {}
    for pid in sorted(union):
        by_base.setdefault(base_id(pid), pid)
    return set(by_base.values())


def combined_label(archives: list[str], sort: bool = True) -> str:
    """Label for the cross-archive / multi-day overlap file:
    `combined-<archives joined by '+'>`. Used as the `archive` argument of
    `overlap_filename`, so the combined overlap is
    `combined-<archives>-<dayslug>-<filters>-overlap.txt` and different
    archive combinations never overwrite each other's file.

    Archives are sorted by default, so the same combination always maps to
    one file name regardless of command-line order; pass sort=False only
    to look up a label written before canonical (sorted) naming existed.
    """
    parts = sorted(archives) if sort else list(archives)
    return "combined-" + "+".join(parts)


def overlap_filename(
    archive: str, filter_names: list[str], date: str | None = None
) -> str:
    """Overlap file name: `<archive>-<day>-<filters>-overlap.txt`.

    For the cross-archive / multi-day overlap pass `combined_label(archives)`
    as `archive`: `combined-<archives>-<daysspec>-<filters>-overlap.txt`.
    The filter set and (for combined files) the archive list are part of
    the name, so different filter sets or archive combinations on the same
    day never overwrite each other.
    """
    return f"{archive}-{day_stamp(date)}-{'-'.join(filter_names)}-overlap.txt"


def save_overlap(
    ids: set[str] | list[str],
    titles: dict[str, str],
    archive: str,
    filter_names: list[str],
    out_dir: str | Path = "data/filtered",
    date: str | None = None,
) -> Path:
    """Write the overlap list to `<out_dir>/<overlap_filename(...)>`.

    `ids` are paper ids (already de-duplicated by the caller when combined);
    `titles` maps arxiv_id -> title. With archive=combined_label(archives)
    this is the cross-archive overlap of several lists:
    data/filtered/combined-<archives>-<yyyymmdd>-<filters>-overlap.txt.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / overlap_filename(archive, filter_names, date)
    lines = [f"[{pid}] {titles[pid]}" for pid in sorted(ids)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def keyword_hit(paper: ArxivPaper, keywords: list[str]) -> bool:
    """True if title or abstract contains any keyword phrase
    (case-insensitive substring match)."""
    text = f"{paper.title} {paper.abstract}".lower()
    return any(keyword in text for keyword in keywords)


@dataclass
class FilterResult:
    """Outcome of one filter over one archive's papers.

    Engine-only data — terminal presentation is `print_filter_result`'s
    job (the oneliner.py model: pure core, presentation at the edge).
    """

    fdef: FilterDef
    verdicts: list[tuple[str, str, str]]  # (arxiv_id, verdict, reason)
    metrics: list[dict]
    passed: set[str]


def _zero_cost_metrics(arxiv_id: str, **flags: bool) -> dict:
    """Metrics entry for a paper that cost no LLM call — keyword-discarded
    or reused from a stored verdict."""
    return {
        "arxiv_id": arxiv_id,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "wall_seconds": 0.0,
        "attempts": 0,
        **flags,
    }


def run_one_filter(
    papers: list[ArxivPaper],
    fdef: FilterDef,
    archive: str,
    date: str | None = None,
    out_dir: str | Path = "data/filtered",
    cascade: bool = False,
    cache: dict[str, tuple[str, str]] | None = None,
) -> FilterResult:
    """Engine: one filter's verdicts over `papers`, saved to its verdict
    file. No terminal output.

    With a keyword list, papers whose title + abstract hit no keyword are
    FAILED instantly at zero LLM cost; only the shortlist reaches the LLM.
    Papers with a reusable stored verdict (see `load_cached_verdicts`) keep
    it at zero LLM cost; the rest are judged and their verdicts extend the
    cache. Verdicts keep the input order, so the verdict file still covers
    every paper. In cascade mode `papers` is the incoming pool and the
    verdict file is the `.cascade` pool artifact instead of the
    full-coverage file.
    """
    if cache is None:
        cache = load_cached_verdicts(archive, fdef, out_dir=out_dir, date=date)
    if fdef.keywords:
        shortlist = [p for p in papers if keyword_hit(p, fdef.keywords)]
    else:
        shortlist = papers
    # stored verdicts are reused at zero LLM cost; only uncached papers run
    to_judge = [p for p in shortlist if p.arxiv_id not in cache]
    llm_verdicts, llm_metrics = filter_papers(to_judge, fdef)
    judged_by_pid = {pid: (verdict, reason) for pid, verdict, reason in llm_verdicts}
    metric_by_pid = {m["arxiv_id"]: m for m in llm_metrics}

    short_pids = {p.arxiv_id for p in shortlist}
    verdicts, metrics = [], []
    for paper in papers:
        pid = paper.arxiv_id
        if pid not in short_pids:
            # keyword-discarded: instant FAIL at zero LLM cost
            verdicts.append((pid, FAIL_WORD, KEYWORD_MISS))
            metrics.append(_zero_cost_metrics(pid, prefiltered=True))
        elif pid in judged_by_pid:
            verdict, reason = judged_by_pid[pid]
            verdicts.append((pid, verdict, reason))
            metrics.append(metric_by_pid.get(pid) or _zero_cost_metrics(pid))
        else:
            # reused from a previous run's stored verdict
            verdict, reason = cache[pid]
            verdicts.append((pid, verdict, reason))
            metrics.append(_zero_cost_metrics(pid, cached=True))

    passed = {pid for pid, verdict, _reason in verdicts if verdict == PASS_WORD}
    save_verdicts(verdicts, archive, fdef, out_dir=out_dir, date=date, cascade=cascade)
    return FilterResult(fdef=fdef, verdicts=verdicts, metrics=metrics, passed=passed)


def print_filter_result(result: FilterResult, console: Console) -> None:
    """Presentation: one filter's verdict lines and timing summary."""
    for pid, verdict, reason in result.verdicts:
        style = {
            PASS_WORD: "green",
            FAIL_WORD: "dim",
            "ERROR": "bold red",
        }[verdict]
        console.print(
            f"[{pid}] {verdict}: {reason}",
            style=style,
            # old-style arxiv ids contain "/" — rich would eat "[astro-ph/...]"
            # as a markup tag, so never parse markup here
            markup=False,
            no_wrap=True,
            overflow="ignore",
        )

    n = len(result.verdicts)
    n_pass = len(result.passed)
    n_fail = sum(1 for _, v, _ in result.verdicts if v == FAIL_WORD)
    errors = n - n_pass - n_fail
    # keyword-discarded and cache-reused papers cost nothing: report LLM
    # time on LLM calls only
    discarded = sum(1 for m in result.metrics if m.get("prefiltered"))
    reused = sum(1 for m in result.metrics if m.get("cached"))
    llm_seconds = sum(
        m["wall_seconds"]
        for m in result.metrics
        if not m.get("prefiltered") and not m.get("cached")
    )
    n_llm = n - discarded - reused
    parts = []
    if discarded:
        parts.append(f"{discarded} keyword-discarded")
    if reused:
        parts.append(f"{reused} reused from cache")
    parts.append(f"{n_llm} judged by LLM")
    tail = f" — {', '.join(parts)}"
    console.print(
        f"\n[bold]{result.fdef.name}[/bold]: {n_pass} {PASS_WORD}, {n_fail} "
        f"{FAIL_WORD}, {errors} errors of {n}{tail} "
        f"({llm_seconds:.1f}s LLM, {(llm_seconds / n_llm if n_llm else 0):.2f}s/paper)\n"
    )


def judged_pool_size(
    pool: list[ArxivPaper],
    fdef: FilterDef,
    cache: dict[str, tuple[str, str]] | None = None,
) -> int:
    """Zero-cost prediction of a filter's LLM calls over `pool`: the keyword
    gate's survivors (the whole pool without a keyword list), minus papers
    with a reusable stored verdict."""
    if not fdef.keywords:
        candidates = pool
    else:
        candidates = [p for p in pool if keyword_hit(p, fdef.keywords)]
    if cache:
        candidates = [p for p in candidates if p.arxiv_id not in cache]
    return len(candidates)


def run_filters_cascade(
    papers: list[ArxivPaper],
    filters: list[FilterDef],
    archive: str,
    console: Console,
    date: str | None = None,
) -> dict[str, set[str]]:
    """Cascade: run the filters in sequence, each judging only the previous
    filter's PASS pool — the overlap is the last filter's passes, the same
    intersection as the parallel run (a paper any filter fails is in
    neither). At each step the cheapest discriminator runs next: the
    remaining filter with the smallest judged pool (keyword-gate survivors
    within the pool), so the narrow filter never sees papers a wide one
    already failed. Verdict files are the cascade pool artifacts
    (`<filter>.cascade.txt`), not the parallel full-coverage audit trail.
    Returns a mapping filter name -> set of arxiv ids that passed it.
    """
    passed: dict[str, set[str]] = {}
    pool = list(papers)
    remaining = list(filters)
    # caches loaded once: the cascade chooser predicts LLM calls net of
    # reusable stored verdicts, and each stage reuses what it can
    caches = {f.name: load_cached_verdicts(archive, f, date=date) for f in filters}
    llm_calls = 0
    while remaining:
        sizes = [judged_pool_size(pool, f, caches[f.name]) for f in remaining]
        step = sizes.index(min(sizes))
        fdef = remaining.pop(step)
        judged = sizes[step]
        llm_calls += judged
        keyword_survivors = judged_pool_size(pool, fdef)
        reused = keyword_survivors - judged
        discarded = len(pool) - keyword_survivors
        if fdef.keywords and discarded:
            console.print(
                f"{fdef.name}: keyword pre-filter discards {discarded} of "
                f"{len(pool)} pool papers, {keyword_survivors} go to the LLM",
                style="dim",
            )
        if reused:
            console.print(
                f"{fdef.name}: {reused} verdicts reused from cache, "
                f"{judged} judged by LLM",
                style="dim",
            )
        result = run_one_filter(
            pool, fdef, archive, date=date, cascade=True, cache=caches[fdef.name]
        )
        print_filter_result(result, console)
        passed[fdef.name] = result.passed
        pool = [p for p in pool if p.arxiv_id in result.passed]
    parallel_calls = sum(judged_pool_size(papers, f) for f in filters)
    console.print(
        f"cascade: {llm_calls} LLM calls over "
        f"{len(filters)} filters vs {parallel_calls} parallel "
        f"(keyword gates applied per filter) — saved "
        f"{parallel_calls - llm_calls}",
        style="dim",
    )
    return passed


def run_filters(
    papers: list[ArxivPaper],
    filters: list[FilterDef],
    archive: str,
    console: Console,
    date: str | None = None,
    cascade: bool = False,
) -> dict[str, set[str]]:
    """Run every filter over `papers`; print and save each filter's results.

    The pipeline-facing composition of the engine (`run_one_filter`) and
    its presentation (`print_filter_result`). Parallel mode (default):
    every filter judges the full list and stores full-coverage verdict
    files. Cascade mode: filters run in sequence over shrinking pools
    (`run_filters_cascade`). Returns a mapping filter name -> set of
    arxiv ids that passed it.
    """
    if cascade:
        return run_filters_cascade(papers, filters, archive, console, date=date)
    passed: dict[str, set[str]] = {}
    for fdef in filters:
        cache = load_cached_verdicts(archive, fdef, date=date)
        if fdef.keywords:
            survivors = sum(1 for p in papers if keyword_hit(p, fdef.keywords))
            to_llm = judged_pool_size(papers, fdef, cache)
            reused = survivors - to_llm
            console.print(
                f"{fdef.name}: keyword pre-filter discards "
                f"{len(papers) - survivors} of {len(papers)} papers, "
                f"{to_llm} go to the LLM"
                + (f", {reused} reused from cache" if reused else ""),
                style="dim",
            )
        result = run_one_filter(papers, fdef, archive, date=date, cache=cache)
        print_filter_result(result, console)
        passed[fdef.name] = result.passed
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "names",
        nargs="*",
        help="archives and filter names in any order: known filter names "
        "select filters, the other names are archives (default: astro-ph; "
        "default filters: the general defaults)",
    )
    parser.add_argument(
        "--cascade",
        action="store_true",
        help="run the filters in sequence, each judging only the previous "
        "filter's PASS pool (cheapest discriminator first), instead of "
        "every filter over the full list; stores cascade pool verdict "
        "files (<filter>.cascade.txt) instead of full-coverage verdicts — "
        "the overlap is the same either way (modulo verdict wobble)",
    )
    parser.add_argument(
        "--date",
        help="YYYYMMDD of a stored day, a comma-separated list, or an "
        "inclusive range D1..D2 (default: today). Multi-day sweeps write "
        "per-day verdict and overlap files plus one combined overlap",
    )
    add_llm_args(parser)
    args = parser.parse_args()
    apply_llm_args(args)

    chosen = [n for n in args.names if n in FILTER_CATALOG]
    archives = [n for n in args.names if n not in FILTER_CATALOG] or ["astro-ph"]
    invalid = [a for a in archives if not valid_archive_name(a)]
    if invalid:
        sys.exit(
            f"unknown filter or invalid archive name(s): {', '.join(invalid)}; "
            f"known filters: {', '.join(sorted(FILTER_CATALOG))}"
        )
    filters = [FILTER_CATALOG[name] for name in chosen] or DEFAULT_FILTERS
    console = Console(width=max(100, Console().width))

    # per-(archive, day) overlaps; combined below when several combos
    overlaps: dict[tuple[str, str | None], set[str]] = {}
    titles: dict[str, str] = {}
    try:
        days = parse_days(args.date) if args.date else [None]
        multi = len(archives) > 1 or len(days) > 1
        source = day_spec_text(days)
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
                    continue  # a multi-day sweep runs on whatever is stored
                day_tag = day_spec_text([day])
                titles.update({p.arxiv_id: p.title for p in papers})
                console.print(
                    f"[bold]{archive}[/bold], {day_tag}: {len(papers)} papers, "
                    f"filters: {', '.join(f.name for f in filters)} ({source})\n"
                )

                passed = run_filters(
                    papers, filters, archive, console, date=day, cascade=args.cascade
                )
                overlaps[(archive, day)] = overlap_of(passed)
                overlap = overlaps[(archive, day)]

                console.print(
                    f"[bold]--- overlap {archive}, {day_tag} "
                    f"({len(overlap)} papers pass all filters) ---[/bold]"
                )
                for pid in sorted(overlap):
                    console.print(
                        f"[{pid}] {titles[pid]}",
                        markup=False,
                        no_wrap=True,
                        overflow="ignore",
                    )
                for name, ids in passed.items():
                    console.print(f"{name}: {len(ids)} passed", style="dim")
                path = save_overlap(
                    overlap, titles, archive, [f.name for f in filters], date=day
                )
                console.print(f"saved: {path}\n")
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(exc)

    if not multi:
        return

    combined = combined_overlap(list(overlaps.values()))
    console.print(
        f"[bold]--- combined overlap ({len(combined)} papers pass all filters "
        f"on any of {len(overlaps)} feeds) ---[/bold]"
    )
    for pid in sorted(combined):
        console.print(f"[{pid}] {titles[pid]}", markup=False, no_wrap=True, overflow="ignore")
    path = save_overlap(
        combined,
        titles,
        combined_label(archives),
        [f.name for f in filters],
        date=days_slug(days),
    )
    console.print(f"saved: {path}")


if __name__ == "__main__":
    main()
