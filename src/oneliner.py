"""One-liner stage: one compact line per paper.

Purpose: give the user a simple, raw, non-LLM overview of the day's papers
before any filtering or analysis happens. No LLM is involved — this is
plain formatting of already-parsed fields.

Input: today's parsed papers (from the RSS stage, loaded from disk if already
downloaded). Output: `<arxiv_id> <title> — <authors>` per line.

Terminal display is capped at MAX_LEN (100) characters with a trailing '...'
when cut. The .txt file is written with unlimited line width — full titles
and complete author lists — so it stays useful for postprocessing.

Written to data/oneliners/<archive>-<yyyymmdd>.txt.

Usage:
    uv run python src/oneliner.py [archive]  # default: astro-ph
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.text import Text

from arxiv_rss import ArxivPaper, day_stamp, load_or_fetch, valid_archive_name

MAX_LEN = 100

# terminal styles for the one-liner parts (file output stays plain text)
STYLE_ID = "bold cyan"
STYLE_TITLE = "bright_white"
STYLE_SEP = "dim"
STYLE_AUTHORS = "italic green"


def one_liner(paper: ArxivPaper, max_len: int | None = MAX_LEN) -> str:
    """Compact '<arxiv_id> <title> — <authors>' line.

    With max_len set, the line is capped at that many characters with a
    trailing '...'. With max_len=None the line has unlimited width — use
    this for file output meant for postprocessing.
    """
    authors = ", ".join(paper.authors)
    line = f"[{paper.arxiv_id}] {paper.title} — {authors}"
    if max_len is not None and len(line) > max_len:
        line = line[: max_len - 3].rstrip() + "..."
    return line


def one_liner_text(paper: ArxivPaper, max_len: int = MAX_LEN) -> Text:
    """Colored version of `one_liner` for terminal display.

    Same 100-char plain-text cap as `one_liner`; styles do not count toward
    the limit.
    """
    prefix = f"[{paper.arxiv_id}] "
    body = f"{paper.title} — {', '.join(paper.authors)}"
    if len(prefix) + len(body) > max_len:
        room = max_len - len(prefix) - 3
        body = body[:room].rstrip() + "..."

    head, sep, tail = body.partition(" — ")
    text = Text(prefix, style=STYLE_ID)
    if sep:
        text.append(head, style=STYLE_TITLE)
        text.append(sep, style=STYLE_SEP)
        text.append(tail, style=STYLE_AUTHORS)
    else:
        text.append(head, style=STYLE_TITLE)
    return text


def save_lines(
    lines: list[str],
    archive: str,
    out_dir: str | Path = "data/oneliners",
    date: str | None = None,
) -> Path:
    """Write one-liners to `<out_dir>/<archive>-<yyyymmdd>.txt`.

    `date` defaults to today; a YYYYMMDD date writes the stored day's file.
    The archive name is validated — it lands in the file name.
    """
    if not valid_archive_name(archive):
        raise ValueError(f"bad archive name: {archive!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{archive}-{day_stamp(date)}.txt"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    # keep lines intact: one-liners are capped at MAX_LEN, so never let a
    # narrow console crop or wrap them
    console = Console(width=max(MAX_LEN, Console().width))
    archive = sys.argv[1] if len(sys.argv) > 1 else "astro-ph"
    papers, source = load_or_fetch(archive)
    lines = [one_liner(paper, max_len=None) for paper in papers]
    path = save_lines(lines, archive)

    console.print(f"[bold]{archive}[/bold]: {len(lines)} one-liners ({source})")
    for paper in papers:
        console.print(one_liner_text(paper), no_wrap=True, overflow="ignore")
    console.print(f"saved: {path}")


if __name__ == "__main__":
    main()
