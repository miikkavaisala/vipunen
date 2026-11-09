# vipunen-agent

Agentic harness that uses small LLMs to monitor ArXiv output systematically:
cheap criteria-based filtering of RSS feeds, one-liner explanations, and a
final digest — designed to run on a single home PC with 16 GB of VRAM.

```
ArXiv RSS feeds (per archive, per day)
        |
        v
[One-liners]      non-LLM raw overview of the day's papers
        |
        v
[Filters]         PASS/FAIL criteria verdicts, one cheap call per paper;
        |         the overlap (papers passing every filter) is the signal
        v
[Digest]          state-of-the-topic synthesis of the overlap,
                  standard or long (feature every paper)
        |
        v
[Deliverable]     one .md with the filtered article list and the digest
                  together (data/deliverables/; no LLM — files combined)
```

## Contents

- [Quick start](#quick-start)
- [Usage](#usage): [pipeline](#run-the-pipeline) · [topic suite](#run-the-topic-suite) · [stored days](#replay-stored-days) · [options](#options)
- [Daily data collection](#daily-data-collection)
- [Filters](#filters): [choosing](#choosing-filters) · [how they work](#how-filters-work) · [output files](#filter-output-files) · [definitions of note](#definitions-of-note)
- [Digests](#digests)
- [LLM endpoint](#llm-endpoint)
- [Data layout](#data-layout)
- [Library use](#library-use)
- [Compliance and acknowledgment](#compliance-and-acknowledgment)

## Quick start

```bash
uv sync                                                       # Python 3.11+ + uv; deps: rich and tqdm only
uv run python src/fetch_feeds.py                              # download today's feeds (cache-first, throttled)
./run_topics.sh --llm-url http://127.0.0.1:1234/v1 --llm-model qwen/qwen3.5-9b   # run the topic suite
```

Results land in `data/deliverables/` — one .md per topic with the filtered
article list and the digest. LLM stages need your OpenAI-compatible server
running (see [LLM endpoint](#llm-endpoint)); without it, pass `--no-llm` to
run the data stages only. `./run_topics.sh --help` lists every option.

## Usage

### Run the pipeline

```bash
uv run python main.py                                     # astro-ph, default filters (simulation, llm)
uv run python main.py cs.LG --filters llm-training        # one AI-focused page
uv run python main.py cs.LG cs.AI cs.CL --filters physics # combined cross-archive digest
```

### Run the topic suite

`./run_topics.sh` runs one pipeline query per calibrated topic — astrophysics
topics, AI topics, physics/astronomy subject filters, the math cutoff topic,
plus broad two-filter cascade queries — and writes one digest per topic:

```bash
./run_topics.sh                # today's data
./run_topics.sh --date 20261028  # replay a stored day, offline on the data side
./run_topics.sh --help         # full option list
```

Extra arguments pass through to every query, so `--date`, `--long`, `--no-llm`
and the `--llm-*` flags apply uniformly. Never run two ArXiv-fetching
processes at once: not this script concurrently with `src/fetch_feeds.py`,
another `main.py` run, or a second copy of itself.

The topic list is the author's own monitoring set, kept as plain `run_query`
lines in `run_topics.sh` — edit them or add your own to run a different set
of topics.

### Replay stored days

`--date` takes a single day (`20261023`), a comma-separated list
(`20261019,20261021`), or an inclusive range (`20261021..20261023`). A
multi-day or multi-archive run sweeps every stored (archive, day), skips
feeds with no stored data, and produces one combined overlap and one digest
over the de-duplicated union; a single missing day stays a hard error. Data
comes from disk — no ArXiv requests.

```bash
uv run python main.py astro-ph --filters mhd --date 20261021..20261023   # multi-day sweep
uv run python src/deliverable.py astro-ph mhd --date 20261021..20261023  # rebuild a deliverable offline
```

### Options

| Flag | Meaning | Default |
|---|---|---|
| `archives...` | ArXiv archives or categories, e.g. `astro-ph`, `cs.LG` (validated before any request) | `astro-ph` |
| `--filters a,b` | comma-separated filter names; the digest covers papers passing every one | `simulation,llm` |
| `--date D` | stored day, comma list, or range `D1..D2` (offline on the data side) | today |
| `--long` | long digest: feature every surviving paper, ~50 words each | off |
| `--cascade` | run the filters in sequence over shrinking pools — same overlap, fewer LLM calls | off |
| `-n N` | also summarize a random sample of N papers per archive (never a bare number) | off (0) |
| `--no-llm` | skip all LLM stages (summaries, filters, digest) | off |
| `--llm-url` / `--llm-model` / `--llm-api-key` | per-run LLM endpoint settings | env vars (see below) |

## Daily data collection

`feeds.toml` lists the ArXiv archives to download every day regardless of what
the analysis currently looks at — data keeps accumulating for later use
(method testing, calibration, offline reprocessing):

```toml
archives = [
    "astro-ph",
    "cs.LG",     # machine learning
    "math.PR",   # probability
    # ... any archive or category works
]
```

```bash
uv run python src/fetch_feeds.py                  # every feed in feeds.toml, cache-first
uv run python src/fetch_feeds.py --feeds my.toml  # a different list
```

Fetches are cache-first — an up-to-date stored feed costs no request — so
just re-run the command after the daily announcement (~04:00 UTC). Requests
are throttled to one per 5 s, per the ArXiv terms of use. Invalid archive
names are rejected before any request; a feed that fails mid-run (rejected,
unreachable, or not RSS) is reported and the run continues with the rest.

## Filters

Criteria filters judge each paper PASS/FAIL against plain-text criteria. The
digest covers the overlap — the papers passing every selected filter, which
is the reliable signal (individual verdicts wobble slightly between runs).

### Choosing filters

Select filters by name with `--filters` (or as positional arguments to
`src/filter.py`):

| Group | Filters | Intended for |
|---|---|---|
| General defaults | `simulation`, `llm` | any archive (run when no `--filters` is given) |
| Astrophysics topics | `star-formation`, `mhd`, `x-ray` | astro-ph pages |
| Astrophysics methodology | `theory`, `observational` | how the work is done; theory-informed interpretation of observational data FAILs `theory` |
| AI-focused | `llm-training`, `ai-agents`, `rag`, `local-ai`, `cuda`, `open-weights`, `ai-for-science`, `ai-for-physical-science` | cs.LG / cs.AI / cs.CL pages |
| Subject | `physics`, `astronomy` | finding physics/astronomy papers on method-focused pages, e.g. `--filters physics,local-ai` |
| Mathematics | `cutoff` | the cutoff phenomenon in stochastic dynamics: mixing times, cutoff proofs/criteria/windows, random walks on groups, card shuffling, Glauber/exclusion dynamics |

```bash
uv run python src/filter.py astro-ph mhd                                # single filter
uv run python src/filter.py cs.LG cs.AI cs.CL physics --date 20261021   # combined cross-archive overlap
```

The built-in list reflects the author's monitoring interests and grows over
time. To monitor something else, add your own filter (next section); if you
have a filter idea that others might want too, open an issue to suggest it.

### How filters work

Criteria are plain text files in the repo's `filter/` directory — one per
filter, editable without touching Python:

```
filter/
├── general/   simulation.txt, llm.txt                    # the default set (run when no filter is named)
├── astro/     star-formation, mhd, x-ray, theory, observational
├── ai/        llm-training, ai-agents, rag, local-ai, cuda, open-weights, ai-for-science, ai-for-physical-science
├── subject/   physics, astronomy
└── math/      cutoff
```

The file stem is the filter name; the file content is the criteria wording
(lines wrap freely). Add a filter by dropping a `.txt` file into a group
directory. Wording is the quality lever — after editing, recalibrate: run the
filter over stored days and adjudicate the verdicts on a sample; every
shipped filter was calibrated on stored archive-days before release.

**Keyword pre-filter.** An optional sibling `filter/<group>/<name>.keywords`
file — one phrase per line, `#` comments, matched case-insensitively against
title + abstract — FAILs papers with no hit at zero LLM cost before the
filter call runs (the verdict file still covers every paper). Lists stay
recall-first: they over-accept and let the LLM resolve topicality. Current
lists discard 63-98% of papers depending on filter and listing; the broad
filters (`simulation`, `llm`, `theory`, `observational`) deliberately carry
none — no safe vocabulary exists for them.

### Filter output files

Under `data/filtered/`:

- Per-filter verdicts: `<archive>-<yyyymmdd>-<filter>.txt`
- Per-archive overlap (papers passing every filter):
  `<archive>-<yyyymmdd>-<filters>-overlap.txt` — the filter set is part of
  the name, so different filter sets on the same day never overwrite each
  other
- Combined cross-archive overlap (papers from any of several lists, de-duplicated
  by base `arxiv_id`, version-insensitive):
  `combined-<archives>-<yyyymmdd>-<filters>-overlap.txt` — archives joined by
  `+`, so different archive combinations on the same day never overwrite each
  other

### Definitions of note

- **`local-ai`** covers the hardware range from on-device/edge and home
  computers up to lab-scale clusters — at most a handful of multi-GPU nodes a
  university or research lab might host. Rented hyperscale cloud and anything
  datacenter-scale does not count.
- **`open-weights`** requires open-weight models named or declared open in the
  abstract (Llama, Qwen, Mistral, Gemma, Phi, DeepSeek, GLM, ...).
  Precision-first: papers that quietly use open models without saying so are
  an accepted loss.
- **`ai-for-science` vs `ai-for-physical-science`**: the former covers all
  natural sciences including biomedical and life sciences; the latter excludes
  them completely (medical instrumentation included) and keeps physics,
  astronomy/space, chemistry, materials, and earth/climate/environment.

<details>
<summary>Run individual stages on their own</summary>

```bash
uv run python src/arxiv_rss.py astro-ph    # fetch + parse + store today's feed (skips download if already stored)
uv run python src/oneliner.py astro-ph     # one-liners: id, title, authors (100-char cap)
uv run python src/summarize.py astro-ph 5  # ~2-sentence LLM summaries of a random sample of N papers
uv run python src/digest.py astro-ph star-formation --date 20261019  # digest of a stored day's filter output
uv run python src/digest.py astro-ph star-formation --long           # long digest: feature every surviving paper
```

</details>

Any ArXiv archive or category works, e.g. `astro-ph`, `astro-ph.CO`, `cs.LG`,
`hep-ex`. If today's entry is already stored with content, commands load it
from disk and make no network request; an empty same-day entry (fetched
before the daily announcement) is re-downloaded automatically.

## Digests

The standard digest is a 200-350 word state-of-the-topic synthesis that cites
representative papers. `--long` (on `main.py` and `src/digest.py`) asks for a
comprehensive digest that features every surviving paper, each cited by its
arXiv id with a word budget of ~50 words per paper; long digests are saved
with a `-long` suffix and never overwrite the standard digest of the same set.

Long-mode completeness is mechanically enforced: any paper a synthesis call
leaves uncited triggers one small follow-up call covering only the missing
papers (each needs at least one descriptive clause — bare citations do not
count), and large survivor sets are synthesized in chunks of 30 papers with
continuation prompts so the document reads as one digest.

Digest synthesis runs with reasoning on; if the model spends its whole token
budget on reasoning before writing any content (empty reply with finish
reason "length"), the call is retried once with reasoning off — a
no-reasoning digest beats a crashed pipeline.

## LLM endpoint

LLM stages talk to an OpenAI-compatible `/v1/chat/completions` endpoint via
`src/llm.py` (LM Studio, llama.cpp server, vLLM, Ollama, ... — local or
hosted). Nothing is built in; configure per run with flags or process-wide
with environment variables (a flag beats the env var):

| Setting | CLI flag | Environment variable |
|---|---|---|
| Endpoint base URL — include the `/v1` prefix | `--llm-url URL` | `LLM_BASE_URL` |
| Model name (required — there is no default) | `--llm-model MODEL` | `LLM_MODEL` |
| API key for hosted endpoints (sent as `Authorization: Bearer KEY`, never printed; local servers need none) | `--llm-api-key KEY` | `LLM_API_KEY` |

```bash
# local server
uv run python main.py --llm-url http://localhost:1234/v1 --llm-model your-model
# hosted endpoint with authentication
uv run python main.py --llm-url https://api.example.com/v1 --llm-model your-model --llm-api-key "$MY_API_KEY"
```

The flags exist on `main.py` and the LLM stage scripts (`src/filter.py`,
`src/summarize.py`, `src/digest.py`), and `run_topics.sh` passes them through
to every topic query. Reasoning is off for high-volume per-item calls
(summaries, filters — measured ~56x faster) and on for digest synthesis. The
model's context window is finite — prompt, reasoning, and output must fit in
it — so the digest stage compresses large survivor sets into per-article
summaries before synthesizing.

## Data layout

Raw RSS snapshots and parsed papers are stored for later offline processing
and are never deleted; nothing under `data/` is committed to git:

```
data/
├── rss/           <archive>-<yyyymmdd>.xml                               raw RSS snapshot
├── papers/        <archive>-<yyyymmdd>.json                              parsed papers (id, title, abstract, authors, ...)
├── oneliners/     <archive>-<yyyymmdd>.txt                               one-liner list
├── summaries/     <archive>-<yyyymmdd>.txt + .metrics.txt                LLM summaries + performance report
├── filtered/      <archive>-<yyyymmdd>-<filter>.txt                      filter verdicts
│                  <archive>-<yyyymmdd>-<filters>-overlap.txt             overlap (papers passing all filters)
│                  combined-<archives>-<yyyymmdd>-<filters>-overlap.txt  cross-archive overlap (deduped by arxiv_id)
├── digests/       <archive>-<yyyymmdd>-<filters>[-long][-NN].md          state-of-the-topic digest (reruns numbered, never overwritten)
│                  <archive>+...-<yyyymmdd>-<filters>[-long][-NN].md     combined cross-archive digest
└── deliverables/  <same file name as the digest>.md                       filtered article list + digest in one .md (no LLM)
```

Every digest also gets a **deliverable** in `data/deliverables/`, under the
digest's own file name: the filtered article list followed by the digest —
a single-file result, assembled from existing files with no LLM call.
Rebuild one offline from stored files:

```bash
uv run python src/deliverable.py astro-ph star-formation --date 20261019  # latest digest of that day/set
uv run python src/deliverable.py astro-ph mhd --date 20261021..20261023 --long
```

To make a `.tar.gz` copy of the RSS snapshots and parsed papers (e.g. for
moving to another machine or archival), run `uv run python backup_data.py`
→ `backups/arxiv-data-<date>.tar.gz` (`backups/` is gitignored too).

## Library use

```python
from arxiv_rss import ArxivRSS, load_or_fetch, papers_for

papers, source = load_or_fetch("astro-ph")   # cached on disk if already downloaded today
for paper in papers:
    print(paper.arxiv_id, paper.title)

# offline, from a stored snapshot:
papers = ArxivRSS.from_snapshot("data/rss/astro-ph-20261019.xml")

# a stored day, fully offline — raises FileNotFoundError when not stored,
# so batch loops can skip missing days and continue:
try:
    papers = papers_for("cs.LG", "20261021")
except FileNotFoundError:
    papers = []
```

Library functions raise standard exceptions (`FileNotFoundError` for missing
data, `ValueError` for nothing-to-digest conditions); the CLI entry points
map them to clean one-line exits.

## Compliance and acknowledgment

Thank you to arXiv for use of its open access interoperability.

vipunen-agent is an independent personal project; it is not affiliated with
or endorsed by ArXiv.

This project uses ArXiv RSS feeds and mirrors the ArXiv terms of use:
throttled requests, low-impact access patterns, and stored snapshots for
offline processing rather than repeated downloads; a per-clause
compliance checklist is maintained in the development repository.

This repository was developed with AI coding agents.
[AI_DISCLOSURE_STATEMENT.md](AI_DISCLOSURE_STATEMENT.md) lists every
coding-agent harness used and the LLM model each ran on; it is updated
whenever the harness or model changes.
