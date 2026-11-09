"""Digest stage: a "state of the topic today" document for filtered papers.

Purpose: the final pipeline stage so far. Takes the papers that passed the
chosen filters (verdict files written by the filter stage) and makes one
LLM synthesis call to produce the day's state of the topic.

If the passed set is small (total abstract text within a character budget),
abstracts are passed directly to the synthesis call — no per-article
summarization step. Per-article summaries only matter when many abstracts
pass the filter; above the budget they are generated first and the
synthesis runs over the compressed set.

The synthesis prompts frame the input honestly: the survivors are a
small, intentionally filtered sample of a day or two of listings, so
the digest describes what those papers report and must not generalize
into field-level trend claims. Each paper block carries an announcement
tag after its id — [new], [cross-listed], or [re-announcement vN, first
appeared 20YY-MM] — and the synthesis presents re-announcements as
updates to earlier work, never as brand-new results (they stay in the
input; they are only flagged).

Two self-audit mechanisms guard the output (added 2026-10-29 after a
digest quality evaluation):

- Every saved digest ends with a `<!-- digest-audit -->` comment block
  (invisible in rendered markdown): the criteria fingerprints of the
  filters used and the exact survivor ids the synthesis saw. Verdict
  caches get re-judged when filters are reworded, which erases a stored
  digest's traceable survivor set — the embedded block keeps every
  digest auditable against its own inputs forever.
- Each synthesis result passes a zero-LLM-cost citation post-check:
  bracketed arXiv ids that are not among the input papers are stripped
  with a stderr warning (observed once: a corrupted duplicate id
  2610.04904v4 next to the real 2610.04906v4), so no digest carries a
  citation that resolves to nothing. Ids match version-insensitively.

Printed to stdout and written to
data/digests/<archive>-<yyyymmdd>-<filter names>.md.

Usage:
    uv run python src/digest.py [archive] [filter ...] [--topic TOPIC] [--date D] [--long | --clustered] [--cascade]
    # default: astro-ph, all default filters, today
    # e.g. uv run python src/digest.py astro-ph star-formation --date 20261019
    # --long: feature every surviving paper instead of a representative selection
    # --clustered: cheaper every-paper mode — one clause per paper, grouped by
    #   theme (~20 words/paper); the file gets a "-clustered" suffix
    # --cascade: recompute the survivors from the cascade pool verdict files
    #   (<filter>.cascade.txt) — use after `src/filter.py ... --cascade`
    # --date also accepts comma lists and ranges: --date 20261021..20261023
    #   gives one digest over the union of those days (de-duplicated by id)

Cross-archive digests (one digest over the combined overlap of several
lists) are produced by main.py via combined_digest() when several archives
are given.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

from rich.console import Console

from arxiv_rss import (
    ArxivPaper,
    base_id,
    canonical_archive_arg,
    day_spec_text,
    day_stamp,
    days_slug,
    papers_for,
    papers_for_sweep,
    parse_days,
)
from filter import DEFAULT_FILTERS, FILTER_CATALOG, criteria_fingerprint
from llm import add_llm_args, apply_llm_args, chat
from summarize import summarize_papers

# if the total abstract text of survivors fits in this many characters,
# abstracts go directly into the synthesis call; otherwise per-article
# summaries are generated first to compress the input. Sized against the
# hosted model's ~29k-token context: 30k chars of abstracts is roughly
# 8k tokens, leaving ~21k for reasoning plus the digest text (long mode
# budgets ~50 words/paper, capped at 3000 words ~ 4.5k tokens) — enough
# headroom unless reasoning runs away, which _synthesis_call handles.
DIRECT_CHAR_BUDGET = 30_000

# shared framing for every digest prompt: the input is a small,
# intentionally filtered sample of a day or two of listings, and the
# synthesis must not generalize it into field-level claims
SAMPLE_NOTE = """Keep in mind what this input is: a small, intentionally
biased sample — one or two days of arXiv listings, filtered by explicit
monitoring criteria for a specific interest. It is not a survey of the
field. Do not write about what "the field", "researchers", or "the
community" are doing, and avoid sweeping trend claims (e.g. a "broader
shift toward..."). Describe what these specific papers report; any
pattern you mention is an observation about this sample, tied to the
cited papers, not a statement about research at large.
"""

# shared note for every digest prompt about the announcement tag each paper
# block carries after its id: re-announcements (new versions of older
# papers, the majority of a daily cs feed) must be framed as updates to
# earlier work, not as today's new results — they stay in the input
ANNOUNCE_NOTE = """Each paper below is tagged after its arXiv id: [new] means
first announced in this feed run, [cross-listed] means a new paper also
listed on another archive, and [re-announcement vN, first appeared
20YY-MM] means a new version of an older paper. Write about [new] and
[cross-listed] papers as today's submissions. When you cite a
re-announcement, say so in a few words (e.g. "an updated version of
[id] reports ...") — present it as an update to earlier work, never as a
brand-new result.
"""


def _announce_tag(p: ArxivPaper) -> str:
    """Compact announcement tag placed after a paper's id in the digest
    input: " [new]", " [cross-listed]", or " [re-announcement vN, first
    appeared 20YY-MM]" — the YYMM prefix of the base arXiv id dates the
    first version (the RSS pubDate is only the day this entry appeared)."""
    announce_type = (p.announce_type or "new").strip().lower()
    if announce_type == "replace":
        version = p.arxiv_id.rsplit("v", 1)[-1] if "v" in p.arxiv_id else "?"
        match = re.match(r"^(\d{2})(\d{2})\.", base_id(p.arxiv_id))
        first = f"20{match.group(1)}-{match.group(2)}" if match else "earlier"
        return f" [re-announcement v{version}, first appeared {first}]"
    if announce_type == "cross":
        return " [cross-listed]"
    return " [new]"

DIGEST_PROMPT = """You are writing a daily research-monitoring digest.

Monitored topic: "{topic}" ({day_spec}).
Below are {n} arXiv papers from {day_spec} that passed the topic filters, each
with its title and {input_kind}.

Write a "state of the topic today" section of 200-350 words for a
professional audience: what the day's papers collectively say about the
topic, the main themes, methods in use, and notable results. Plain prose,
no bullet lists. Cite papers inline by their arXiv id in brackets where
relevant. Use only the papers given below — do not invent papers, results,
or numbers.

{announce_note}
{sample_note}
Papers:
{papers_block}"""

# long mode: feature every paper in the list, not a representative selection
LONG_DIGEST_PROMPT = """You are writing a daily research-monitoring digest.

Monitored topic: "{topic}" ({day_spec}).
Below are {n} arXiv papers from {day_spec} that passed the topic filters, each
with its title and {input_kind}.

Write a comprehensive "state of the topic today" digest for a professional
audience. Feature EVERY paper in the list: group the papers by theme,
describe what each contributes to that theme, and cite every paper by its
arXiv id in brackets at least once — no paper may be left out. Plain prose,
no bullet lists. Target length: about {word_budget} words. Use only the
papers given below — do not invent papers, results, or numbers.

{announce_note}
{sample_note}
Papers:
{papers_block}"""

# long-mode continuation: chunks after the first must read as the next
# section of the same document, not as a fresh digest
LONG_CONTINUATION_PROMPT = """You are continuing a daily research-monitoring digest.

Monitored topic: "{topic}" ({day_spec}).
Earlier sections of the digest already exist. Below are the next {n}
arXiv papers that passed the topic filters, each with its title and
{input_kind}.

Write the next section of the digest in the same style: group the papers
by theme, describe what each contributes to that theme, and cite every
paper by its arXiv id in brackets at least once — no paper may be left
out. Do NOT re-introduce the topic, repeat the earlier framing, or write
an opening or closing overview of the whole digest: continue the
existing document as its next section. Plain prose, no bullet lists.
Target length: about {word_budget} words. Use only the papers given
below — do not invent papers, results, or numbers.

{announce_note}
{sample_note}
Papers:
{papers_block}"""

# long-mode completion pass: if a synthesis call left papers uncited,
# one small focused call covers only the missing ones and is appended
COMPLETION_PROMPT = """You are completing a daily research-monitoring digest.

Monitored topic: "{topic}" ({day_spec}).
A digest was already written covering the papers, but these {n} papers
from the list were left out. Write additional prose paragraphs in the same
style that cover them, grouped by theme where natural, describing each
paper's contribution in at least one clause — a bare citation without any
description does not count — and citing each paper by its arXiv id in
brackets at least once. Plain prose, no bullet lists. Use only the papers
given below — do not invent papers, results, or numbers.

{announce_note}
{sample_note}
Papers:
{papers_block}"""

# clustered mode: every survivor covered, but cheaply — one clause or short
# sentence per paper inside its theme paragraph (~20 words/paper), instead
# of long mode's ~50 words/paper. It sits between the standard digest
# (representative selection only) and --long (full feature per paper)
CLUSTERED_DIGEST_PROMPT = """You are writing a daily research-monitoring digest.

Monitored topic: "{topic}" ({day_spec}).
Below are {n} arXiv papers from {day_spec} that passed the topic filters, each
with its title and {input_kind}.

Write a "state of the topic today" digest that covers EVERY paper in the
list: group the papers into a few themes, and within each theme's prose
give every paper one clause or short sentence, citing its arXiv id in
brackets — no paper may be left uncited, but no paper gets more than one
sentence. Plain prose, no bullet lists. Target length: about {word_budget}
words. Use only the papers given below — do not invent papers, results, or
numbers.

{announce_note}
{sample_note}
Papers:
{papers_block}"""

# clustered-mode continuation: chunks after the first read as the next
# section of the same document, same clustered style
CLUSTERED_CONTINUATION_PROMPT = """You are continuing a daily research-monitoring digest.

Monitored topic: "{topic}" ({day_spec}).
Earlier sections of the digest already exist. Below are the next {n}
arXiv papers that passed the topic filters, each with its title and
{input_kind}.

Write the next section in the same clustered style: group these papers by
theme, give each paper one clause or short sentence with its arXiv id in
brackets — no paper may be left uncited, no paper gets more than one
sentence. Do NOT re-introduce the topic, repeat the earlier framing, or
write an opening or closing overview of the whole digest: continue the
existing document as its next section. Plain prose, no bullet lists.
Target length: about {word_budget} words. Use only the papers given
below — do not invent papers, results, or numbers.

{announce_note}
{sample_note}
Papers:
{papers_block}"""

# long mode is synthesized in chunks of this many survivors: one call
# cannot feature a large list — the 3000-word budget covers only ~80
# papers, and 50 words/paper for hundreds of papers would overflow the
# model's context. Each chunk gets its own synthesis call plus completion
# check; the parts are concatenated into one digest.
LONG_CHUNK_SIZE = 30

# clustered mode chunks larger than long mode: the per-paper budget is
# ~20 words instead of ~50, so more papers fit in one call's context
CLUSTERED_CHUNK_SIZE = 60

# a bracketed arXiv id as the prompts ask the model to cite papers; an
# optional preceding space is part of the match, so a dropped citation
# takes its preceding space with it ("study [id] claims" -> "study claims",
# "says X [id]." -> "says X.") and leaves no double spaces or space-period
CITED_ID_RE = re.compile(r" ?\[(\d{4}\.\d{4,5}(?:v\d+)?)\]")

# the citation post-check and audit warnings go to stderr so the digest
# text on stdout stays clean prose
_WARN_CONSOLE = Console(stderr=True)


def _strip_unknown_citations(text: str, valid_ids: set[str]) -> str:
    """Zero-LLM-cost citation post-check: remove bracket citations whose
    id is not among the papers given to the synthesis call.

    The model occasionally corrupts an id (observed once, 2026-10-23: the
    real [2610.04906v4] also emitted as a drifted duplicate [2610.04904v4]).
    The claim stays in the text; the defective bracket goes, and a warning
    names the dropped ids. `valid_ids` are base ids, so citations with or
    without the version suffix both count as grounded.
    """
    dropped: list[str] = []

    def _drop(match: re.Match) -> str:
        if base_id(match.group(1)) in valid_ids:
            return match.group(0)
        dropped.append(match.group(1))
        # the match includes the citation's preceding space: dropping the
        # citation takes that space with it and leaves no double spaces
        # or space-period artifacts behind
        return ""

    cleaned = CITED_ID_RE.sub(_drop, text)
    if dropped:
        _WARN_CONSOLE.print(
            f"citation check: dropped {len(dropped)} citation(s) not in "
            f"the digest input: {', '.join(dropped)}"
        )
    return cleaned


def _audit_block(filter_names: list[str], survivors: list[ArxivPaper]) -> str:
    """The self-audit comment appended to every saved digest: the criteria
    fingerprints of the filters used, the exact survivor ids the
    synthesis saw, and the announcement tag of every survivor that was
    not a first announcement in this feed run. HTML comment — invisible
    in rendered markdown — so a digest can always be audited against its
    inputs even after the verdict caches are re-judged by later filter
    rewordings (which otherwise erases the digest's traceable survivor set).
    """
    fingerprints = ", ".join(
        f"{name}:{criteria_fingerprint(FILTER_CATALOG[name])}"
        for name in filter_names
        if name in FILTER_CATALOG
    )
    id_line = " ".join(f"[{p.arxiv_id}]" for p in survivors)
    # wrap long survivor lists instead of one multi-thousand-char line
    wrapped = "\n".join(textwrap.wrap(id_line, width=96)) or "(none)"
    lines = ["<!-- digest-audit"]
    if fingerprints:
        lines.append(f"filters: {fingerprints}")
    lines.append(f"survivors ({len(survivors)}):")
    lines.append(wrapped)
    # zero-cost flag record: which survivors were not first announcements in
    # this feed run (re-announcements / cross-listings), with their tags
    non_new = [
        f"[{p.arxiv_id}]{_announce_tag(p)}"
        for p in survivors
        if (p.announce_type or "new").strip().lower() != "new"
    ]
    if non_new:
        lines.append("announce tags: " + " ".join(non_new))
    lines.append("-->")
    return "\n".join(lines) + "\n"


def read_passed_ids(
    archive: str,
    filter_name: str,
    date: str | None = None,
    verdicts_dir: str | Path = "data/filtered",
    cascade: bool = False,
) -> set[str]:
    """Read a day's verdict file for one filter; return the ids that passed.

    `date` defaults to today (local time); YYYYMMDD selects a stored day.
    Cascade mode reads the cascade pool artifact (`<filter>.cascade.txt`)
    written by `filter.py --cascade` — its PASS ids are the same overlap
    signal, because a pool verdict file's PASS set is exactly the papers
    that survived the filters before it plus this one.
    """
    date = day_stamp(date)
    stem = f"{filter_name}.cascade" if cascade else filter_name
    path = Path(verdicts_dir) / f"{archive}-{date}-{stem}.txt"
    if not path.exists():
        flag = " --cascade" if cascade else ""
        raise FileNotFoundError(
            f"no verdict file for filter {filter_name!r}: {path}\n"
            f"run `uv run python src/filter.py {archive} {filter_name}{flag}` first"
        )
    passed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\[(\S+)\]\s*PASS\b", line)
        if match:
            passed.add(match.group(1))
    return passed


def survivors_for(
    archive: str,
    filter_names: list[str],
    date: str | None = None,
    cascade: bool = False,
) -> list[ArxivPaper]:
    """Papers from the chosen day's list that passed every chosen filter."""
    passed = None
    for name in filter_names:
        ids = read_passed_ids(archive, name, date, cascade=cascade)
        passed = ids if passed is None else passed & ids
    if passed is None or not passed:
        raise ValueError(
            f"no papers passed the chosen filters ({', '.join(filter_names)}) — nothing to digest"
        )
    papers = papers_for(archive, date)
    survivors = [p for p in papers if p.arxiv_id in passed]
    if not survivors:
        raise ValueError("filter ids not found in the papers file — re-run the filter stage")
    return survivors


def combined_survivors(
    archives: list[str],
    filter_names: list[str],
    days: list[str] | None = None,
    cascade: bool = False,
) -> list[ArxivPaper]:
    """Survivors from several lists and/or days: per-(archive, day)
    intersection of the filters' PASS sets, unioned across combos and
    de-duplicated by base arxiv id (version-insensitive).

    Papers cross-listed on several archives (or announced in several feeds
    the same day) appear once; so do re-announcements of the same paper on
    different days under a different version number — union members are
    de-duplicated by base arxiv id, keeping the first-seen version.
    `days` defaults to today.

    Mirror of the filter stage's sweep semantics: with several
    archive-days, combos with no stored papers are skipped (they were
    never filtered); a single combo stays strict. A combo with stored
    papers but no verdict file still errors — the filters were never
    run for it.
    """
    day_list = days if days else [None]
    multi = len(archives) > 1 or len(day_list) > 1
    survivors: dict[str, ArxivPaper] = {}
    for day in day_list:
        for archive in archives:
            papers = papers_for_sweep(archive, day, multi)
            if papers is None:
                continue  # no stored papers: never filtered, skip
            passed = None
            for name in filter_names:
                ids = read_passed_ids(archive, name, day, cascade=cascade)
                passed = ids if passed is None else passed & ids
            if not passed:
                continue
            for paper in papers:
                if paper.arxiv_id in passed:
                    survivors.setdefault(base_id(paper.arxiv_id), paper)
    if not survivors:
        raise ValueError(
            f"no papers passed the chosen filters ({', '.join(filter_names)}) "
            f"on any of {len(archives)} list-day combinations — nothing to digest"
        )
    return list(survivors.values())


def digest_basename(
    archive: str,
    filter_names: list[str],
    days: list[str | None] | None = None,
    long: bool = False,
    clustered: bool = False,
) -> str:
    """Digest file base name (no extension, no rerun number):
    `<archive>-<daysspec>-<filters>[-long|-clustered]`, where <daysspec> is
    the single day, a contiguous range FIRST..LAST, or days joined by '+'
    (None/today = today's stamp). Long-mode digests get a "-long" suffix
    and clustered-mode ones "-clustered", so they never collide with the
    standard digest of the same set.
    """
    day = days_slug(days) if days else day_stamp()
    suffix = "-long" if long else ("-clustered" if clustered else "")
    return f"{archive}-{day}-{'-'.join(filter_names)}{suffix}"


def save_digest(
    text: str,
    archive: str,
    filter_names: list[str],
    topic: str,
    days: list[str] | None = None,
    long: bool = False,
    clustered: bool = False,
    audit: str | None = None,
) -> Path:
    """Write the digest to data/digests/, never overwriting.

    First save: <digest_basename(...)>.md. If that name exists (e.g. a
    rerun of the same day and filter set), a running number is appended:
    <name>-01.md, -02.md, ...

    `audit` (the self-audit comment from _audit_block) is appended after
    the digest text: invisible in rendered markdown, machine-readable
    for later grounding checks.
    """
    out_dir = Path("data/digests")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = digest_basename(archive, filter_names, days, long=long, clustered=clustered)
    path = out_dir / f"{base}.md"
    n = 1
    while path.exists():
        path = out_dir / f"{base}-{n:02d}.md"
        n += 1
    header = f"# State of the topic: {topic}\n\n"
    path.write_text(header + text + "\n" + (audit or ""), encoding="utf-8")
    return path


def digest(
    archive: str,
    filter_names: list[str],
    topic: str | None = None,
    days: list[str] | None = None,
    long: bool = False,
    clustered: bool = False,
    cascade: bool = False,
) -> tuple[str, list[ArxivPaper], Path]:
    """Produce a state-of-the-topic digest for papers passing all chosen
    filters on one archive — one day, or several days unioned and
    de-duplicated — and save it.

    With long=True the digest must feature every surviving paper instead
    of a representative selection; with clustered=True every paper gets
    one clause in a theme group (cheaper than long). long and clustered
    are mutually exclusive. With cascade=True the survivors are
    recomputed from the cascade pool verdict files (`<filter>.cascade.txt`)
    — use it after a filter stage run with `--cascade`.

    The synthesis call runs with reasoning on (think=True): it is a single
    call over the whole survivor set, where deeper reasoning pays for its
    token cost in synthesis quality.

    Returns (digest_text, survivors, digest_path).
    """
    if long and clustered:
        raise ValueError("--long and --clustered are mutually exclusive")
    day_list = days if days else [None]
    if len(day_list) == 1:
        survivors = survivors_for(archive, filter_names, day_list[0], cascade=cascade)
    else:
        survivors = combined_survivors(
            [archive], filter_names, day_list, cascade=cascade
        )
    return _synthesize(
        survivors, archive, filter_names, topic, day_list, long=long, clustered=clustered
    )


def combined_digest(
    archives: list[str],
    filter_names: list[str],
    topic: str | None = None,
    days: list[str] | None = None,
    long: bool = False,
    clustered: bool = False,
    cascade: bool = False,
) -> tuple[str, list[ArxivPaper], Path]:
    """Like `digest`, but spanning several archives and/or days: one digest
    over the combined overlap (papers from any of the lists on any of the
    days passing every filter, de-duplicated by base arxiv id). The digest file
    is named after the joined archive list (in canonical sorted order) and
    the day spec, e.g. data/digests/cs.AI+cs.CL+cs.LG-20261021..20261023-<filters>.md.
    """
    if long and clustered:
        raise ValueError("--long and --clustered are mutually exclusive")
    day_list = days if days else [None]
    survivors = combined_survivors(archives, filter_names, day_list, cascade=cascade)
    label = canonical_archive_arg("+".join(archives))
    return _synthesize(
        survivors, label, filter_names, topic, day_list, long=long, clustered=clustered
    )


# synthesis is one big call: with thinking on and a multi-thousand-word
# long-mode output it can take minutes — the client's 120 s default
# timeout would kill it, so this stage gets its own generous one
SYNTHESIS_TIMEOUT_SECONDS = 600


def _synthesis_call(prompt: str) -> str:
    """One synthesis call, thinking on by design.

    If the model burns the whole token budget on reasoning before any
    content (the endpoint returns finish_reason "length" with empty
    content — a big prompt in a small context window makes this likelier),
    retry once with thinking off: a no-reasoning digest beats a crashed
    pipeline after minutes of filter calls.
    """
    try:
        return chat(
            [{"role": "user", "content": prompt}],
            think=True,
            timeout=SYNTHESIS_TIMEOUT_SECONDS,
        ).strip()
    except RuntimeError as exc:
        if "max_tokens" not in str(exc):
            raise
        return chat(
            [{"role": "user", "content": prompt}],
            think=False,
            timeout=SYNTHESIS_TIMEOUT_SECONDS,
        ).strip()


def _chunk_text(
    chunk: list[ArxivPaper],
    blocks: dict[str, str],
    input_kind: str,
    topic: str,
    day_spec: str,
    prompt_cls: str,
    word_budget: int,
) -> str:
    """Every-paper digest text for a group of survivors (long or clustered
    mode): one synthesis call that must cite every paper, plus a
    completion call for any it drops.

    `prompt_cls` is the first-chunk or continuation prompt of the mode;
    continuations read as the next section of the same document instead
    of a fresh digest.
    """
    prompt = prompt_cls.format(
        topic=topic,
        day_spec=day_spec,
        n=len(chunk),
        input_kind=input_kind,
        word_budget=word_budget,
        announce_note=ANNOUNCE_NOTE,
        sample_note=SAMPLE_NOTE,
        papers_block="\n".join(blocks[p.arxiv_id] for p in chunk),
    )
    text = _synthesis_call(prompt)
    chunk_ids = {base_id(p.arxiv_id) for p in chunk}
    text = _strip_unknown_citations(text, chunk_ids)

    cited = {base_id(c) for c in re.findall(r"\[(\S+)\]", text)}
    missing = [p for p in chunk if base_id(p.arxiv_id) not in cited]
    if missing:
        completion = COMPLETION_PROMPT.format(
            topic=topic,
            day_spec=day_spec,
            n=len(missing),
            announce_note=ANNOUNCE_NOTE,
            sample_note=SAMPLE_NOTE,
            papers_block="\n".join(blocks[p.arxiv_id] for p in missing),
        )
        extra = _synthesis_call(completion)
        extra = _strip_unknown_citations(extra, chunk_ids)
        text = f"{text}\n\n{extra}"
    return text


def _input_blocks(survivors: list[ArxivPaper], label: str) -> tuple[str, dict[str, str]]:
    """(input_kind, blocks) for the synthesis call: abstracts go in directly
    when the total text fits DIRECT_CHAR_BUDGET, otherwise per-article
    summaries compress the input first. Each block starts with the paper's
    id in brackets followed by its announcement tag ([new] / [cross-listed]
    / [re-announcement ...]), so the synthesis can tell new papers from
    version re-announcements."""
    tags = {p.arxiv_id: _announce_tag(p) for p in survivors}
    total_chars = sum(len(p.title) + len(p.abstract) for p in survivors)
    if total_chars <= DIRECT_CHAR_BUDGET:
        return "abstract", {
            p.arxiv_id: f"[{p.arxiv_id}]{tags[p.arxiv_id]} {p.title}\n{p.abstract}"
            for p in survivors
        }
    # too much text for one direct call: compress via per-article summaries
    entries, _metrics = summarize_papers(survivors, desc=f"digest {label}")
    blocks = {}
    for pid, summary in entries:
        if summary.startswith(f"[{pid}]"):
            summary = summary[len(f"[{pid}]") :].lstrip()
        blocks[pid] = f"[{pid}]{tags[pid]} {summary}"
    return "short summary", blocks


def _synthesize(
    survivors: list[ArxivPaper],
    label: str,
    filter_names: list[str],
    topic: str | None,
    days: list[str] | None,
    long: bool = False,
    clustered: bool = False,
) -> tuple[str, list[ArxivPaper], Path]:
    """Run the synthesis call(s) over `survivors` and save the digest."""
    day_list = days if days else [None]
    day_spec = day_spec_text(day_list)
    input_kind, blocks = _input_blocks(survivors, label)

    topic = topic or ", ".join(filter_names)
    if long or clustered:
        # feature-every-paper is a hard requirement of both modes, and one
        # call cannot feature a large list (the word budget covers only
        # ~80 long-mode papers; hundreds would overflow the context) —
        # synthesize in chunks, each checked and completed on its own
        if long:
            first_prompt, cont_prompt = LONG_DIGEST_PROMPT, LONG_CONTINUATION_PROMPT
            chunk_size = LONG_CHUNK_SIZE
            # ~50 words per paper, clamped to a sane range
            budget = lambda n: max(400, min(3000, 50 * n))
        else:
            first_prompt, cont_prompt = (
                CLUSTERED_DIGEST_PROMPT,
                CLUSTERED_CONTINUATION_PROMPT,
            )
            chunk_size = CLUSTERED_CHUNK_SIZE
            # ~20 words per paper, clamped to a sane range
            budget = lambda n: max(400, min(2500, 20 * n))
        parts = []
        for i in range(0, len(survivors), chunk_size):
            chunk = survivors[i : i + chunk_size]
            parts.append(
                _chunk_text(
                    chunk,
                    blocks,
                    input_kind,
                    topic,
                    day_spec,
                    prompt_cls=first_prompt if i == 0 else cont_prompt,
                    word_budget=budget(len(chunk)),
                )
            )
        text = "\n\n".join(parts)
    else:
        prompt = DIGEST_PROMPT.format(
            topic=topic,
            day_spec=day_spec,
            n=len(survivors),
            input_kind=input_kind,
            announce_note=ANNOUNCE_NOTE,
            sample_note=SAMPLE_NOTE,
            papers_block="\n".join(blocks[p.arxiv_id] for p in survivors),
        )
        text = _synthesis_call(prompt)
        text = _strip_unknown_citations(
            text, {base_id(p.arxiv_id) for p in survivors}
        )

    audit = _audit_block(filter_names, survivors)
    path = save_digest(
        text, label, filter_names, topic, day_list, long=long, clustered=clustered, audit=audit
    )
    return text, survivors, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archive", nargs="?", default="astro-ph")
    parser.add_argument("filters", nargs="*", help="filter names (default: all)")
    parser.add_argument("--topic", help="topic label for the digest (default: filter names)")
    parser.add_argument(
        "--date",
        help="YYYYMMDD of a stored day, a comma-separated list, or an "
        "inclusive range D1..D2 (default: today); several days give one "
        "digest over the de-duplicated union",
    )
    parser.add_argument(
        "--long",
        action="store_true",
        help="long digest: feature every surviving paper (default: representative selection)",
    )
    parser.add_argument(
        "--clustered",
        action="store_true",
        help="clustered digest: cover every surviving paper with one clause "
        "each, grouped by theme (cheaper than --long); mutually exclusive with --long",
    )
    parser.add_argument(
        "--cascade",
        action="store_true",
        help="recompute the survivors from the cascade pool verdict files "
        "(<filter>.cascade.txt) instead of the full-coverage verdict "
        "files — use after running the filter stage with --cascade",
    )
    add_llm_args(parser)
    args = parser.parse_args()
    apply_llm_args(args)

    if args.long and args.clustered:
        parser.error("--long and --clustered are mutually exclusive")
    # canonical (sorted) archive order in the digest's file name, so the
    # same combination maps to one name regardless of command-line order
    args.archive = canonical_archive_arg(args.archive)

    chosen = args.filters or [f.name for f in DEFAULT_FILTERS]
    unknown = set(chosen) - set(FILTER_CATALOG)
    if unknown:
        sys.exit(f"unknown filter(s): {', '.join(sorted(unknown))}")

    console = Console(width=max(100, Console().width))
    try:
        days = parse_days(args.date) if args.date else None
        text, survivors, path = digest(
            args.archive,
            chosen,
            args.topic,
            days,
            long=args.long,
            clustered=args.clustered,
            cascade=args.cascade,
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(exc)

    console.print(f"\n[bold]{path.read_text(encoding='utf-8').splitlines()[0]}[/bold]")
    console.print(f"[dim]{len(survivors)} papers[/dim]\n")
    console.print(text)
    console.print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
