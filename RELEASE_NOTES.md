# Release 1.0

vipunen-agent is an agentic harness that uses small LLMs (up to ~10B
parameters, quantized) to monitor ArXiv RSS feeds on a single home PC
with 16 GB of VRAM: aggressive criteria-based filtering of the daily
announcement flow, and a state-of-the-topic digest of the few papers
that are actually worth attention.

How to install and run it is in [README.md](README.md).

## Core features

- **RSS monitoring, compliance-first.** Feeds are fetched from
  `https://export.arxiv.org/rss/<archive>` only — cache-first,
  throttled to one request per 5 seconds, strictly sequential. Raw RSS
  snapshots and parsed JSON are stored permanently and never
  re-downloaded (`data/` is gitignored).
- **One-liners.** A non-LLM overview of each day's papers (id, title,
  authors) before any filtering — `data/oneliners/`.
- **Criteria filters.** PASS/FAIL judgments of abstracts against
  plain-text criteria (`filter/<group>/<name>.txt` — editable without
  touching Python). 18 calibrated filters ship: general (simulation,
  llm), astrophysics topics and methodology (star-formation, mhd,
  x-ray, theory, observational), AI-focused (llm-training, ai-agents,
  rag, local-ai, cuda, open-weights, ai-for-science,
  ai-for-physical-science), subject (physics, astronomy), and the
  math cutoff filter. Optional recall-first `.keywords` pre-gates
  discard the clearly irrelevant at zero cost before the LLM runs.
- **The overlap is the signal.** Multiple narrow filters outperform one
  complicated filter with small models; the papers passing every
  selected filter are the reliable signal, individual verdicts wobble.
- **Verdict caching.** Verdicts are stored per (archive, day, filter)
  and reused at zero LLM cost while the criteria+model fingerprint
  matches; rewording a filter or switching models re-judges cleanly.
  Keyword-gate FAILs are recomputed live, so keyword edits never
  invalidate stored LLM verdicts.
- **Cascade mode.** `--cascade` judges each filter only over the
  previous filter's PASS pool, cheapest discriminator first — the same
  overlap for noticeably fewer LLM calls.
- **Multi-archive and multi-day runs.** Combined cross-archive,
  cross-day overlaps are de-duplicated by base arxiv id; archive lists
  enter file names in canonical sorted order. `--date` replays stored
  days fully offline on the data side (single day, comma list, or
  inclusive range).
- **Digests.** State-of-the-topic synthesis in three budgets: standard
  (representative selection), `--long` (every survivor at ~50 words,
  completeness mechanically enforced), and `--clustered` (every
  survivor, one clause each, grouped by theme). Every input paper is
  tagged `[new]`, `[cross-listed]`, or `[re-announcement vN, first
  appeared 20YY-MM]`, and re-announcements are presented as updates to
  earlier work. Each digest ends with a zero-LLM-cost audit block
  (criteria fingerprints + exact survivor ids) and passes a citation
  post-check that strips any cited id not in the input set.
- **Deliverables.** After every digest, the overlap list and the digest
  are combined into one .md under `data/deliverables/` — no LLM. Any
  deliverable can be rebuilt offline from stored files
  (`src/deliverable.py`).
- **Topic suite.** `./run_topics.sh` runs one pipeline query per
  calibrated topic; edit the plain `run_query` lines to monitor a
  different set.
- **Any OpenAI-compatible LLM endpoint.** Local (LM Studio,
  llama.cpp, vLLM, Ollama, ...) or hosted, configured per run with
  `--llm-url` / `--llm-model` / `--llm-api-key` or the `LLM_*`
  environment variables; nothing is hardcoded. Reasoning is off for
  the high-volume per-item calls (measured ~56x faster) and on for
  digest synthesis, with a reasoning-off fallback if reasoning alone
  exhausts the context window. Per-item calls cap `max_tokens` so a
  runaway generation stops cleanly instead of exhausting the context,
  and transient server-side failures (HTTP 5xx, connection errors, or
  the inference engine crashing mid-request) are retried a few times
  before the pipeline gives up.

## Development record

- Every filter in this release was calibrated on stored archive-days.
- The filter-vs-keyword benchmark, the overlap-vs-combined retest, and
  the digest self-audit were measured on the development corpus; the
  conclusions are summarized in the public docs.
- A full security and bug review (2026-11-02) is closed: every finding
  was fixed or explicitly accepted as a documented risk.
- This repository was developed with AI coding agents;
  [AI_DISCLOSURE_STATEMENT.md](AI_DISCLOSURE_STATEMENT.md) records
  every harness used and the LLM model each ran on.

## Requirements

Python >= 3.11. Dependencies: `rich` (terminal output) and `tqdm`
(progress bars) only. uv-managed (`uv sync`) or any environment that
can install those two packages.

## Compliance and acknowledgment

Thank you to arXiv for use of its open access interoperability.

vipunen-agent is an independent personal project; it is not affiliated
with or endorsed by ArXiv. Access patterns mirror the ArXiv terms of
use: throttled requests, low-impact access, stored snapshots for
offline processing rather than repeated downloads.
