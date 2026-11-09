"""Summarization stage: short summaries for a random sample of papers.

Purpose: basic summary mechanism — no filtering yet. Takes a random sample
of N papers from today's stored papers, sends each abstract through the
local LLM, and writes a ~2-sentence summary per paper, easy to read, for a
professional audience. Papers are identified only by their arXiv code.

Performance metrics (token use, wall clock time, output length) are
collected per call, printed, and saved next to the summaries.

Printed to stdout and written to data/summaries/<archive>-<yyyymmdd>.txt.

Usage:
    uv run python src/summarize.py [archive] [N]     # default: astro-ph 5
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from rich.console import Console
from tqdm import tqdm

from arxiv_rss import ArxivPaper, day_stamp, load_or_fetch
from llm import add_llm_args, apply_llm_args, chat_with_metrics

DEFAULT_SAMPLE_SIZE = 5
# summaries are cheap pipeline calls: answer directly, no reasoning block
THINK = False
# cap the reply so a runaway generation stops cleanly (finish_reason
# "length") instead of burning the whole context window and crashing the
# server's inference engine mid-request; the prompt asks for <=60 words
# (~100 tokens), 256 leaves headroom
MAX_TOKENS = 256

PROMPT = """Summarize the following arXiv paper for a professional audience.

Requirements:
- Two sentences, at most 60 words total: one on what the paper does,
  one on the main result or claim.
- Easy to read, plain prose, no bullet points. Do not restate the abstract;
  be much shorter than it.
- Use only information given below. Do not speculate or invent references.

Return the answer as a single paragraph starting with the arXiv identifier
in square brackets, exactly in this format:
[{arxiv_id}] <summary sentences>

Title: {title}
Abstract: {abstract}"""


def id_prefixed(arxiv_id: str, summary: str) -> str:
    """Summary line with the id in brackets, added when missing."""
    if not summary.startswith(f"[{arxiv_id}]"):
        return f"[{arxiv_id}] {summary}"
    return summary


def summarize_paper(paper: ArxivPaper) -> tuple[str, dict]:
    """One LLM call: short summary of the paper's abstract.

    Returns (summary, metrics) where metrics records token use, wall clock
    time, and input/output lengths for performance measurement.
    """
    prompt = PROMPT.format(arxiv_id=paper.arxiv_id, title=paper.title, abstract=paper.abstract)
    t0 = time.monotonic()
    reply, usage = chat_with_metrics(
        [{"role": "user", "content": prompt}], think=THINK, max_tokens=MAX_TOKENS
    )
    wall_seconds = time.monotonic() - t0

    summary = id_prefixed(paper.arxiv_id, reply.strip())

    details = usage.get("completion_tokens_details") or {}
    metrics = {
        "arxiv_id": paper.arxiv_id,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "reasoning_tokens": details.get("reasoning_tokens", 0),
        "wall_seconds": round(wall_seconds, 2),
        "abstract_chars": len(paper.abstract),
        "summary_chars": len(summary),
    }
    return summary, metrics


def summarize_papers(
    sample: list[ArxivPaper], desc: str = "summarize"
) -> tuple[list[tuple[str, str]], list[dict]]:
    """Summarize each paper in `sample`.

    Returns (entries, metrics): entries as (arxiv_id, summary) pairs (each
    summary starts with the id in brackets) and per-call metrics dicts.
    """
    entries: list[tuple[str, str]] = []
    metrics: list[dict] = []
    for paper in tqdm(sample, desc=desc, unit="paper"):
        summary, m = summarize_paper(paper)
        entries.append((paper.arxiv_id, summary))
        metrics.append(m)
    return entries, metrics


def metrics_report(metrics: list[dict]) -> str:
    """Aggregate per-call metrics into a short human-readable report."""
    n = len(metrics)
    if n == 0:
        return "no metrics (no papers summarized)"
    prompt_tokens = sum(m["prompt_tokens"] for m in metrics)
    completion_tokens = sum(m["completion_tokens"] for m in metrics)
    reasoning_tokens = sum(m["reasoning_tokens"] for m in metrics)
    total_seconds = sum(m["wall_seconds"] for m in metrics)
    abstract_chars = sum(m["abstract_chars"] for m in metrics)
    summary_chars = sum(m["summary_chars"] for m in metrics)
    mean_seconds = total_seconds / n
    ratio = (summary_chars / abstract_chars * 100) if abstract_chars else 0.0
    lines = [
        f"papers: {n}",
        f"wall clock: {total_seconds:.1f}s total, {mean_seconds:.1f}s mean per paper",
        f"tokens: {prompt_tokens} prompt, {completion_tokens} completion "
        f"({reasoning_tokens} of them reasoning)",
        f"length: summary {summary_chars} chars from {abstract_chars} chars of abstracts "
        f"({ratio:.0f}% of original)",
    ]
    return "\n".join(lines)


def save_metrics(
    metrics: list[dict],
    archive: str,
    out_dir: str | Path = "data/summaries",
    date: str | None = None,
) -> Path:
    """Write the metrics report to `<out_dir>/<archive>-<yyyymmdd>.metrics.txt`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{archive}-{day_stamp(date)}.metrics.txt"
    per_paper = "\n".join(str(m) for m in metrics)
    path.write_text(per_paper + "\n\n" + metrics_report(metrics) + "\n", encoding="utf-8")
    return path


def save_summaries(
    entries: list[tuple[str, str]],
    archive: str,
    out_dir: str | Path = "data/summaries",
    date: str | None = None,
) -> Path:
    """Write summaries to `<out_dir>/<archive>-<yyyymmdd>.txt`.

    `entries` is a list of (arxiv_id, summary) pairs; the summary may or may
    not already start with the id in brackets. `date` defaults to today.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{archive}-{day_stamp(date)}.txt"
    lines = [id_prefixed(arxiv_id, summary) for arxiv_id, summary in entries]
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archive", nargs="?", default="astro-ph")
    parser.add_argument(
        "n",
        nargs="?",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"sample size (default: {DEFAULT_SAMPLE_SIZE})",
    )
    add_llm_args(parser)
    args = parser.parse_args()
    if args.n < 0:
        parser.error(f"sample size must be >= 0: {args.n}")
    apply_llm_args(args)
    archive = args.archive
    n = args.n

    console = Console(width=max(100, Console().width))
    papers, source = load_or_fetch(archive)
    sample = random.sample(papers, min(n, len(papers)))
    console.print(f"[bold]{archive}[/bold]: summarizing {len(sample)} of {len(papers)} papers ({source})")

    entries, metrics = summarize_papers(sample, desc=f"summarize {archive}")
    for _arxiv_id, summary in entries:
        console.print(f"\n{summary}", style="bright_white")

    console.print("\n[bold]--- summary metrics ---[/bold]")
    console.print(metrics_report(metrics))

    path = save_summaries(entries, archive)
    metrics_path = save_metrics(metrics, archive)
    console.print(f"\nsaved: {path}")
    console.print(f"saved: {metrics_path}")


if __name__ == "__main__":
    main()
