#!/usr/bin/env bash
# Run one pipeline query per topic: one filter per main.py invocation,
# each producing its own digest of the papers that pass it. Filters
# combine by intersection (a paper must pass every listed filter), so
# topic queries are run separately, never stacked.
#
#     ./run_topics.sh                     # today's data
#     ./run_topics.sh --date 2026-10-24   # replay a stored day, offline
#     ./run_topics.sh --date 2026-10-20..2026-10-26
#     ./run_topics.sh --help              # full, friendly option list
#
# Broad topics run as 2-filter queries with --cascade and the narrower
# filter listed first: the cascade stage judges each filter only over
# the previous filter's PASS pool, so the narrow filter shrinks the
# pool before the broad one runs — same overlap, fewer LLM calls. (The
# cascade picker reorders by cheapest discriminator anyway; listing
# narrow-first settles ties.) Single-topic queries skip --cascade: with
# one filter it saves nothing and would swap the full-coverage verdict
# audit trail for cascade-pool artifacts.
#
# Extra arguments are passed through to every invocation, so --date,
# --long, --no-llm, --llm-url, --llm-model and --llm-api-key apply
# uniformly. The LLM server (src/llm.py endpoint) must be up unless
# --no-llm is passed.
#
# ArXiv compliance (doc/ARXIV_COMPLIANCE.md): the RSS throttle is
# per-process, so each query is a separate throttle. Runs are strictly
# sequential and pause GAP_SECONDS between queries to keep the request
# spacing conservative even when runs are fast (--no-llm) and data is
# missing. Do not run this script concurrently with src/fetch_feeds.py,
# another main.py run, or a second instance of itself — one fetching
# run at a time.
#
# Topic trust status: all filters used here are
# calibrated — star-formation, mhd, x-ray, theory, observational,
# physics, astronomy, rag, llm-training, open-weights, local-ai,
# cutoff, ai-for-science, ai-for-physical-science, llm (re-verified
# 2026-10-29), ai-agents and cuda (tightened and re-calibrated
# 2026-10-29).

set -euo pipefail
cd "$(dirname "$0")"

usage() {
    cat <<'EOF'
run_topics.sh — run the whole topic suite, one pipeline query per topic

Each topic below is a separate `uv run python main.py ...` invocation
that filters one day's ArXiv papers by a calibrated criterion and writes
a digest of the survivors. Filters combine by intersection (a paper
must pass every listed filter), so topics are queried separately, never
stacked. Results land in data/digests/ and data/deliverables/.

Usage:
    ./run_topics.sh [options]

Examples:
    ./run_topics.sh                 # today's data, default LLM setup
    ./run_topics.sh --date 20261028 # replay a stored day (offline data side)
    ./run_topics.sh --date 20261020..20261026
    ./run_topics.sh --llm-url http://127.0.0.1:1234/v1 --llm-model qwen/qwen3.5-9b
    ./run_topics.sh --no-llm        # data stages only, no LLM server needed

Settings (every option is passed through to each main.py query, so it
applies uniformly to all topics):

    --date D            Run on a stored day instead of today: a single day
                        (20261028), a comma list, or an inclusive range
                        (20261020..20261026). Data comes from disk — no
                        ArXiv requests — but LLM stages still run.
    --long              Long digest: feature every surviving paper instead
                        of a representative selection.
    --no-llm            Skip all LLM stages (filters, summaries, digest).
    -n N                Also produce LLM summaries of a random sample of
                        N papers per archive (off by default; the digest
                        is the main output).
    --llm-url URL       Local OpenAI-compatible endpoint base URL, e.g.
                        http://127.0.0.1:1234/v1 — include the /v1 prefix
                        (LM Studio, llama.cpp server, vLLM, Ollama all
                        serve under /v1). Overrides the LLM_BASE_URL
                        environment variable.
    --llm-model MODEL   Model name sent with every request, e.g.
                        qwen/qwen3.5-9b. Overrides the LLM_MODEL
                        environment variable. Required unless LLM_MODEL
                        is set — there is no built-in default.
    --llm-api-key KEY   API key for a hosted endpoint that requires one,
                        sent as 'Authorization: Bearer KEY'. Optional —
                        local servers need no key. Overrides the
                        LLM_API_KEY environment variable.

If --llm-url / --llm-model / --llm-api-key are not given, the environment
variables LLM_BASE_URL / LLM_MODEL / LLM_API_KEY are used instead.

Requirements and cautions:
    - The local LLM server must be up and the requested model loaded,
      unless --no-llm is passed.
    - Stored days accumulate via `uv run python src/fetch_feeds.py`;
      run that, not this script, to collect data. Fetching is RSS-only
      and throttled to one request per 5 s — this script also pauses
      5 s between queries.
    - Never run two fetching runs at once: not this script concurrently
      with src/fetch_feeds.py, another main.py run, or a second copy of
      this script.

Topics covered (all filters calibrated against the stored daily feeds):
    astrophysics on astro-ph: star formation, MHD, X-ray, theory,
        observational methodology, GPU/CUDA methods
    AI/ML on cs.LG cs.AI cs.CL cs.DC cs.AR cs.PF cs.MA: LLM training,
        RAG, open weights, local AI, CUDA/GPU computing, AI agents,
        AI for science, AI for physical science
    physics and astronomy subject matter on AI listings
    mathematics on math.PR math.ST math.CO: the cutoff phenomenon
    broad cross-topic queries (2-filter cascades): open-weight LLMs,
        local LLMs, open-weight/local AI agents, physics+LLM,
        astronomy+LLM, astro theory+simulation
EOF
}

# -h/--help anywhere in the arguments prints the instructions and exits
# before any query runs (instead of being forwarded to main.py).
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            usage
            exit 0
            ;;
    esac
done

# Keep request spacing >= 5 s across process boundaries (the in-process
# throttle in arxiv_rss.py cannot see other processes).
GAP_SECONDS=5

# Arguments given to this script itself are appended to every main.py
# invocation below, after the query's own arguments.
EXTRA_ARGS=("$@")

run_query() {
    local label="$1"
    shift
    echo "=== ${label} ==="
    uv run python main.py "$@" "${EXTRA_ARGS[@]}"
    sleep "${GAP_SECONDS}"
}

# --- astrophysics topics ---

run_query "star formation" \
    astro-ph --filters star-formation

run_query "MHD" \
    astro-ph --filters mhd

run_query "X-ray astronomy" \
    astro-ph --filters x-ray

run_query "theory methodology" \
    astro-ph --filters theory

run_query "observational methodology" \
    astro-ph --filters observational

# Astronomy papers that directly use CUDA or other GPU methods.
run_query "GPU methods in astrophysics" \
    astro-ph --filters cuda

# --- LLM / AI topics ---

run_query "LLM training" \
    cs.LG cs.AI cs.CL --filters llm-training

run_query "RAG" \
    cs.LG cs.AI cs.CL --filters rag

run_query "open weights" \
    cs.LG cs.AI cs.CL --filters open-weights

run_query "local AI" \
    cs.LG cs.AI cs.DC cs.AR --filters local-ai

run_query "CUDA / GPU computing" \
    cs.DC cs.AR cs.PF cs.LG --filters cuda

run_query "AI agents" \
    cs.AI cs.MA cs.CL --filters ai-agents

run_query "AI for science" \
    cs.LG cs.AI --filters ai-for-science

run_query "AI for physical science" \
    cs.LG cs.AI --filters ai-for-physical-science

# --- physics/astronomy subject matter on AI-focused listings ---

run_query "physics on AI pages" \
    cs.LG cs.AI cs.CL --filters physics

run_query "astronomy on AI pages" \
    cs.LG cs.AI cs.CL --filters astronomy

# --- mathematics ---

run_query "cutoff phenomenon" \
    math.PR math.ST math.CO --filters cutoff

# --- broad topics: 2-filter cascade queries, narrow filter first ---
#
# Ordering dependency: singles run before pairs on purpose — the verdict
# cache in filter.py reuses stored verdicts per (archive, day, filter),
# so a pair's first cascade stage costs zero LLM calls when the single
# already judged the same (archive, day, filter) earlier in this run.

# Open-weight language transformers.
run_query "LLMs with open weights" \
    cs.LG cs.CL cs.AI --filters open-weights,llm --cascade

# Language transformers that run locally / on-device.
run_query "local LLMs" \
    cs.LG cs.CL cs.AI --filters local-ai,llm --cascade

run_query "AI agents with open weights" \
    cs.LG cs.CL cs.AI --filters open-weights,ai-agents --cascade

run_query "Local AI agents" \
    cs.LG cs.CL cs.AI --filters local-ai,ai-agents --cascade

run_query "physics + LLM" \
    cs.LG cs.AI cs.CL --filters physics,llm --cascade

run_query "astronomy + LLM" \
    cs.LG cs.AI cs.CL --filters astronomy,llm --cascade

# Theoretical / computational simulation work in astrophysics.
run_query "astro theory + simulation" \
    astro-ph --filters theory,simulation --cascade

echo "All topic queries done."
