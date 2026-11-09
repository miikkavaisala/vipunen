"""Fetch and parse ArXiv RSS feeds.

Purpose: non-invasive data fetching. Only download if not already there —
`load_or_fetch(archive)` serves today's papers from disk whenever today's
entry exists, and hits the network only when it does not.

Uses only the standard library. Designed around the project's data-source
principle: RSS is the primary feed (low stress on ArXiv servers), requests
are throttled (5-second pause between feeds), raw snapshots are stored for
later offline processing, and the ArXiv API is never touched here.

Usage:
    from arxiv_rss import ArxivRSS, load_or_fetch

    # downloads only if today's entry is not already on disk:
    papers, source = load_or_fetch("astro-ph")

    # direct fetch when a fresh network copy is explicitly wanted:
    papers = ArxivRSS().fetch("astro-ph", save=True)

    # later, offline, from a stored snapshot:
    papers = ArxivRSS.from_snapshot(Path("data/rss/astro-ph-20261019.xml"))
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

RSS_BASE = "https://export.arxiv.org/rss/"
# ArXiv asks clients to make at most one request per 3 seconds; we are more
# conservative: pause 5 seconds between RSS feeds to stay friendly with the
# ArXiv terms of use. Only applies between requests, never before the first.
REQUEST_DELAY_SECONDS = 5.0

_NAMESPACES = {
    "dc": "http://purl.org/dc/elements/1.1/",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# e.g. astro-ph, hep-ex, cs, astro-ph.CO, cond-mat.mtrl-sci, q-bio.NC
ARCHIVE_NAME_RE = re.compile(r"^[a-z]+(-[a-z]+)*(\.[A-Za-z]+(-[A-Za-z]+)*)?$")


def valid_archive_name(name: str) -> bool:
    """True if `name` looks like an ArXiv archive or category name."""
    return bool(ARCHIVE_NAME_RE.fullmatch(name))


_DAY_RE = re.compile(r"^\d{8}$")


def _check_day(day: str) -> str:
    """Validate one YYYYMMDD day; raise ValueError if malformed."""
    if not _DAY_RE.fullmatch(day):
        raise ValueError(f"bad day {day!r}: expected YYYYMMDD")
    try:
        _dt.datetime.strptime(day, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"bad day {day!r}: {exc}") from exc
    return day


def day_stamp(date: str | None = None) -> str:
    """YYYYMMDD stamp for output file names: `date` if given, else today."""
    return date or time.strftime("%Y%m%d")


def base_id(arxiv_id: str) -> str:
    """ArXiv id without the version suffix: 2610.27888v2 -> 2610.27888.

    The same paper can be announced on several days (or lists) under
    different version numbers; de-duplication across archive-days uses
    the base id so it is counted — and digested — once."""
    return re.sub(r"v\d+$", "", arxiv_id)


def canonical_archive_arg(archive: str) -> str:
    """Canonical '+'-joined archive label for file names: the archives are
    sorted, so the same combination always maps to one name regardless of
    the order given on the command line — 'cs.LG+cs.CL+cs.AI' and
    'cs.LG+cs.AI+cs.CL' both become 'cs.AI+cs.CL+cs.LG'. Single-archive
    arguments pass through unchanged."""
    if "+" in archive:
        return "+".join(sorted(archive.split("+")))
    return archive


def parse_days(spec: str) -> list[str]:
    """Parse a --date spec into a de-duplicated list of YYYYMMDD days.

    Accepted forms, freely mixed, comma-separated: a single day
    (20261023), an inclusive calendar range (20261021..20261023 —
    expands over calendar days, gaps included), or a mix
    (20261019,20261021..20261023). Order is preserved.
    """
    days: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ".." in part:
            first, _, last = part.partition("..")
            _check_day(first)
            _check_day(last)
            start = _dt.datetime.strptime(first, "%Y%m%d")
            end = _dt.datetime.strptime(last, "%Y%m%d")
            if end < start:
                raise ValueError(f"bad range {part!r}: end before start")
            span = (end.date() - start.date()).days
            if span > 366:
                raise ValueError(f"bad range {part!r}: more than a year")
            days.extend(
                (start.date() + _dt.timedelta(days=i)).strftime("%Y%m%d")
                for i in range(span + 1)
            )
        else:
            days.append(_check_day(part))
    if not days:
        raise ValueError(f"empty date spec: {spec!r}")
    seen: set[str] = set()
    return [d for d in days if not (d in seen or seen.add(d))]


def days_slug(days: list[str | None]) -> str:
    """Filename slug for a day list: the single day, a contiguous range
    as FIRST..LAST, otherwise days joined by '+' (None = today)."""
    stamped = [d or time.strftime("%Y%m%d") for d in days]
    if len(stamped) == 1:
        return stamped[0]
    ords = [_dt.datetime.strptime(d, "%Y%m%d").toordinal() for d in stamped]
    if ords == list(range(ords[0], ords[0] + len(ords))):
        return f"{stamped[0]}..{stamped[-1]}"
    return "+".join(stamped)


def day_spec_text(days: list[str | None]) -> str:
    """Human phrase for prompts and console lines: "today", "day D",
    or "days D1..D2" / "days D1+D2"."""
    slug = days_slug(days)
    if len(days) == 1:
        return "today" if days[0] is None else f"day {slug}"
    return f"days {slug}"

_DESC_PREFIX = re.compile(r"^arXiv:\S+\s+Announce Type:\s*([\w-]+)\s*\n?Abstract:\s*")
_COMMENTS_TAIL = re.compile(r"\s*Comments:\s.*$", re.S)


@dataclass
class ArxivPaper:
    """One paper as announced in an ArXiv RSS feed."""

    arxiv_id: str
    url: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    category: str = ""
    announce_type: str = ""
    published: str = ""

    def __str__(self) -> str:
        authors = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors += f" (+{len(self.authors) - 3})"
        return f"[{self.arxiv_id}] ({self.announce_type}) {self.title} — {authors}"


class ArxivRSS:
    """Fetch and parse RSS feeds for any ArXiv archive or category.

    Throttles requests to comply with the ArXiv terms of use and can store
    raw XML snapshots for offline re-processing.
    """

    _last_request_time: float = 0.0  # shared across instances

    def __init__(self, base_url: str = RSS_BASE, raw_dir: str | Path | None = None):
        self.base_url = base_url.rstrip("/") + "/"
        self.raw_dir = Path(raw_dir) if raw_dir is not None else None

    def fetch(self, archive: str, save: bool = False) -> list[ArxivPaper]:
        """Fetch the RSS feed for `archive` (e.g. 'astro-ph', 'astro-ph.CO').

        Returns the parsed papers. With save=True (or a raw_dir set at
        construction), the raw XML is kept as a snapshot for offline use.
        The response is parsed before anything is stored: a non-XML
        response (e.g. an HTML error page) raises RuntimeError, so garbage
        is never stored as the day's snapshot. The archive name is
        validated before anything else — it lands in file names (and the
        URL), and the snapshot would otherwise escape the data directory
        via names like '../weird'.
        """
        if not valid_archive_name(archive):
            raise ValueError(
                f"bad archive name: {archive!r} — expected e.g. astro-ph, "
                "astro-ph.CO, cs.LG"
            )
        url = f"{self.base_url}{archive}"
        xml_text = self._request(url)

        try:
            papers = self.parse(xml_text)
        except ET.ParseError as exc:
            raise RuntimeError(
                f"the response for {archive!r} is not ArXiv RSS XML: {exc}"
            ) from exc

        raw_dir = self.raw_dir
        if save or raw_dir is not None:
            if raw_dir is None:
                raw_dir = Path("data/rss")
            raw_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d")
            snapshot = raw_dir / f"{archive}-{stamp}.xml"
            snapshot.write_text(xml_text, encoding="utf-8")

        return papers

    @staticmethod
    def parse(xml_text: str) -> list[ArxivPaper]:
        """Parse ArXiv RSS XML into ArxivPaper objects. No network involved."""
        root = ET.fromstring(xml_text)
        papers: list[ArxivPaper] = []
        for item in root.iter("item"):
            papers.append(ArxivRSS._parse_item(item))
        return papers

    @staticmethod
    def from_snapshot(path: str | Path) -> list[ArxivPaper]:
        """Re-parse a stored raw snapshot, fully offline."""
        return ArxivRSS.parse(Path(path).read_text(encoding="utf-8"))

    def _request(self, url: str) -> str:
        self._throttle()
        request = urllib.request.Request(
            url, headers={"User-Agent": "vipunen-agent/1.0 (personal research use)"}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"ArXiv rejected the request for {url!r}: HTTP {exc.code} — "
                "is the archive or category name valid?"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(
                f"ArXiv request failed for {url!r}: {reason}"
            ) from exc

    def _throttle(self) -> None:
        elapsed = time.monotonic() - ArxivRSS._last_request_time
        if elapsed < REQUEST_DELAY_SECONDS:
            time.sleep(REQUEST_DELAY_SECONDS - elapsed)
        ArxivRSS._last_request_time = time.monotonic()

    @staticmethod
    def _parse_item(item: ET.Element) -> ArxivPaper:
        def text(tag: str) -> str:
            node = item.findtext(tag, default="", namespaces=_NAMESPACES)
            return node.strip()

        raw_description = item.findtext("description", default="", namespaces=_NAMESPACES)
        abstract, announce_type = ArxivRSS._clean_description(raw_description)

        arxiv_id = ArxivRSS._id_from_guid(
            item.findtext("guid", default="", namespaces=_NAMESPACES)
        )
        if not arxiv_id:
            # fall back to the abs URL, e.g. https://arxiv.org/abs/2610.22406
            link = text("link")
            arxiv_id = link.rsplit("/", 1)[-1] if link else ""

        authors = [
            a.strip()
            for a in text("dc:creator").split(",")
            if a.strip()
        ]

        return ArxivPaper(
            arxiv_id=arxiv_id,
            url=text("link"),
            title=re.sub(r"\s+", " ", text("title")).strip(),
            abstract=abstract,
            authors=authors,
            category=text("category"),
            announce_type=announce_type or text("arxiv:announce_type"),
            published=text("pubDate"),
        )

    @staticmethod
    def _clean_description(raw: str) -> tuple[str, str]:
        """Strip the RSS boilerplate from a <description> element.

        Raw form: 'arXiv:2610.22406v1 Announce Type: new\\nAbstract: ...'
        possibly followed by ' Comments: ...'.
        Returns (abstract, announce_type).
        """
        announce_type = ""
        match = _DESC_PREFIX.match(raw)
        if match:
            announce_type = match.group(1)
            raw = raw[match.end():]
        abstract = _COMMENTS_TAIL.sub("", raw)
        abstract = re.sub(r"\s+", " ", abstract).strip()
        return abstract, announce_type

    @staticmethod
    def _id_from_guid(guid: str) -> str:
        # e.g. 'oai:arXiv.org:2610.22406v1' -> '2610.22406v1'
        return guid.split(":")[-1].strip()


def load_or_fetch(
    archive: str, data_dir: str | Path = "data"
) -> tuple[list[ArxivPaper], str]:
    """Return today's papers for `archive`.

    By default, if today's entry is not yet downloaded, it is downloaded
    (raw RSS snapshot + parsed JSON). If it already exists on disk with
    content, it is loaded from disk and no network request is made.

    If today's stored entry is *empty* (zero papers), it was almost
    certainly fetched before ArXiv posted the daily announcement (~04:00
    UTC), so the feed is re-downloaded and the same-day files are
    overwritten. This costs at most one throttled request per run on a day
    that is genuinely empty.

    Returns (papers, source) where source describes where the data came from.
    """
    stamp = time.strftime("%Y%m%d")
    data_dir = Path(data_dir)
    if not valid_archive_name(archive):
        raise ValueError(
            f"bad archive name: {archive!r} — expected e.g. astro-ph, "
            "astro-ph.CO, cs.LG (it lands in file names)"
        )
    papers_path = data_dir / "papers" / f"{archive}-{stamp}.json"
    if papers_path.exists():
        with open(papers_path, encoding="utf-8") as fh:
            papers = [ArxivPaper(**entry) for entry in json.load(fh)]
        if papers:
            return papers, f"cached: {papers_path}"
        # empty stored day: the feed was fetched before the daily
        # announcement — fall through and re-download

    papers = ArxivRSS(raw_dir=data_dir / "rss").fetch(archive, save=True)
    save_papers(papers, archive, out_dir=data_dir / "papers")
    if papers:
        return papers, f"downloaded: {papers_path}"
    return papers, f"downloaded, still empty (announcement not yet posted): {papers_path}"


def papers_for(archive: str, date: str | None = None, data_dir: str | Path = "data") -> list[ArxivPaper]:
    """Load a day's parsed papers.

    `date` defaults to today via `load_or_fetch` (downloads if missing, or
    re-downloads an empty pre-announcement entry); a YYYYMMDD date reads
    the stored file, fully offline.
    """
    if date is None:
        papers, _source = load_or_fetch(archive, data_dir)
        return papers
    path = Path(data_dir) / "papers" / f"{archive}-{date}.json"
    if not path.exists():
        raise FileNotFoundError(f"no stored papers file for {date}: {path}")
    return [ArxivPaper(**entry) for entry in json.loads(path.read_text(encoding="utf-8"))]


def papers_for_sweep(
    archive: str,
    day: str | None,
    multi: bool,
    on_missing=None,
) -> list[ArxivPaper] | None:
    """`papers_for` with the pipeline sweep's missing-day semantics.

    A single exact combo is strict: FileNotFoundError propagates. In a
    multi combo (several archives and/or days) a missing archive-day is
    skipped — it was never stored, so it was never filtered; `on_missing`
    (if given) is called with (archive, day) for reporting. Returns None
    when the combo was skipped.
    """
    try:
        return papers_for(archive, day)
    except FileNotFoundError:
        if not multi:
            raise
        if on_missing is not None:
            on_missing(archive, day)
        return None


def save_papers(
    papers: list[ArxivPaper], archive: str, out_dir: str | Path = "data/papers"
) -> Path:
    """Write parsed papers to `<out_dir>/<archive>-<yyyymmdd>.json`.

    The archive name is validated — it lands in the file name, and an
    unvalidated one would escape `out_dir` via names like '../weird'.
    """
    if not valid_archive_name(archive):
        raise ValueError(f"bad archive name: {archive!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{archive}-{day_stamp()}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(p) for p in papers], fh, indent=2, ensure_ascii=False)
    return path


if __name__ == "__main__":
    import sys

    archive = sys.argv[1] if len(sys.argv) > 1 else "astro-ph"
    papers, source = load_or_fetch(archive)
    print(f"{archive}: {len(papers)} papers ({source})")
    for paper in papers[:3]:
        print(paper)
        print(f"    abstract: {paper.abstract[:140]}...")
