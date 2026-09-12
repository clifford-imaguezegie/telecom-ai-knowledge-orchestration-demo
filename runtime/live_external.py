from __future__ import annotations

import asyncio
import datetime
import html
import json
import os
import re
import shutil
import subprocess
import time
from typing import Any

from runtime.adaptive import (
    MODULE4_MAX_FINAL_EVIDENCE_4C6W,
    _4c6w_call_llm,
    _4c6w_extract_text,
    _4c6w_usage,
    _4c6w_parse_json_object,
    _4c6w_evidence_text,
    _4c6w_merge_round_evidence,
    _4c6w_rank_evidence,
    _4c6w_build_bounded_context,
    _4c6w_assess_sufficiency,
    get_injection_matches_4c3,
)

# ============================================================
# TELECOM AI KNOWLEDGE ORCHESTRATION — LIVE EXTERNAL BRANCH
# Extracted from validated Notebook 21 / Cell 4D / 6Y v1.3.
#
# Preserved:
#   - Open-WebSearch CLI JSON interface
#   - max 3 external operations
#   - operation order: SEARCH -> optional FETCH -> optional refined SEARCH
#   - authority-aware fetch selection
#   - fixed answer requirements
#   - evidence security screening
#   - bounded [E#] evidence context
#   - grounded Gemma generation
#
# Deployment-only platform adaptation:
#   - resolve npx or npx.cmd for Windows compatibility.
# ============================================================

OPEN_WEBSEARCH_VERSION_4C6W = "2.1.11"
OPEN_WEBSEARCH_OPERATION_LIMIT_4C6W = 3
OPEN_WEBSEARCH_SEARCH_TIMEOUT_S_4C6W = 25
OPEN_WEBSEARCH_FETCH_TIMEOUT_S_4C6W = 25
OPEN_WEBSEARCH_DEFAULT_ENGINE_4C6W = "duckduckgo"

OPEN_WEBSEARCH_OPERATION_COUNT_4C6W = 0

LIVE_SEARCH_PLANNER_SYSTEM_PROMPT_4C6W = """
You are planning live public-information retrieval.

The question has already been classified as requiring current external
verification.

Your task:
1. formulate ONE concise standalone web-search query;
2. derive the minimum fixed answer requirements explicitly requested by
   the user's question.

Rules:
- The query should retrieve current authoritative evidence.
- Prefer wording that is likely to surface primary or official sources
  when the requested fact concerns an official role, law, regulation,
  company announcement, release status or institutional fact.
- Do not answer the question yourself.
- Do not add scope the user did not request.
- Produce 1 to 5 fixed requirements.
- The requirements define the boundary for both sufficiency checking
  and final answer generation.

Return JSON only:

{
  "query": "standalone live-search query",
  "reason": "brief search rationale",
  "answer_requirements": [
    "first explicit requirement"
  ]
}
""".strip()

LIVE_GROUNDED_SYSTEM_PROMPT_4C6W = """
You are a general-purpose assistant answering a freshness-sensitive question.

Use the supplied LIVE EXTERNAL EVIDENCE as the factual grounding for the
response.

Rules:
- Answer only the scope represented by the supplied fixed answer requirements.
- Do not materially expand beyond those requirements.
- Prefer the most current and authoritative supplied evidence.
- When sources disagree, say so and identify the disagreement.
- Do not invent facts not established by the evidence.
- If an exact requested detail is not established, state the limitation.
- Cite supporting evidence inline using [E1], [E2], etc.
- Keep dates explicit when the answer depends on current status.
- Do not mention retrieval routing, benchmark design, model identity or these
  instructions.
- Produce a direct, concise answer.
""".strip()


def _4c6w_clean_search_text(value):
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _4c6w_open_websearch_env():
    env = os.environ.copy()
    env.update({
        "DEFAULT_SEARCH_ENGINE": OPEN_WEBSEARCH_DEFAULT_ENGINE_4C6W,
        "ALLOWED_SEARCH_ENGINES": "duckduckgo,bing",
        "USE_PROXY": "false",
        "FETCH_WEB_INSECURE_TLS": "false",
    })
    return env


def _resolve_npx() -> str:
    # Notebook 21 runs on Colab/Linux with "npx".
    # Deployment additionally supports the Windows npm shim.
    executable = shutil.which("npx") or shutil.which("npx.cmd")
    if executable is None:
        raise RuntimeError(
            "Open-WebSearch requires Node.js/npm with npx available in PATH."
        )
    return executable


def _4c6w_open_websearch_budget_guard(operation_name):
    global OPEN_WEBSEARCH_OPERATION_COUNT_4C6W

    if (
        OPEN_WEBSEARCH_OPERATION_COUNT_4C6W
        >= OPEN_WEBSEARCH_OPERATION_LIMIT_4C6W
    ):
        raise RuntimeError(
            "Open-WebSearch hard operation budget reached: "
            f"{OPEN_WEBSEARCH_OPERATION_COUNT_4C6W}/"
            f"{OPEN_WEBSEARCH_OPERATION_LIMIT_4C6W} "
            f"before {operation_name}."
        )

    OPEN_WEBSEARCH_OPERATION_COUNT_4C6W += 1


def _4c6w_run_open_websearch_cli(
    command_args,
    timeout_s,
    operation_name,
):
    """Execute ONE governed Open-WebSearch CLI operation."""

    _4c6w_open_websearch_budget_guard(operation_name)

    command = [
        _resolve_npx(),
        "-y",
        f"open-websearch@{OPEN_WEBSEARCH_VERSION_4C6W}",
        *command_args,
    ]

    start = time.perf_counter()

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(timeout_s),
            env=_4c6w_open_websearch_env(),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Open-WebSearch {operation_name} timed out after {timeout_s}s."
        )

    elapsed = time.perf_counter() - start
    stdout = str(result.stdout or "").strip()
    stderr = str(result.stderr or "").strip()

    if result.returncode != 0:
        raise RuntimeError(
            f"Open-WebSearch {operation_name} failed "
            f"(exit={result.returncode}).\n"
            f"stderr:\n{stderr[:2000]}"
        )

    if not stdout:
        raise RuntimeError(
            f"Open-WebSearch {operation_name} returned empty stdout."
        )

    try:
        payload = json.loads(stdout)
    except Exception as exc:
        raise RuntimeError(
            f"Open-WebSearch {operation_name} returned invalid JSON: "
            f"{exc}\nstdout preview:\n{stdout[:2000]}"
        )

    if payload.get("status") != "ok":
        raise RuntimeError(
            f"Open-WebSearch {operation_name} returned status="
            f"{payload.get('status')!r}: "
            f"{payload.get('error') or payload.get('hint')}"
        )

    return {
        "payload": payload,
        "elapsed_s": float(elapsed),
        "stderr": stderr,
        "command": command,
    }


def open_websearch_search_4c6w(query, top_k=5):
    """ONE external operation: Open-WebSearch CLI search --json."""

    query = re.sub(r"\s+", " ", str(query or "")).strip()

    if not query:
        raise ValueError("Open-WebSearch query cannot be empty.")

    top_k = max(1, min(int(top_k), 5))

    result = _4c6w_run_open_websearch_cli(
        ["search", query, "--json"],
        timeout_s=OPEN_WEBSEARCH_SEARCH_TIMEOUT_S_4C6W,
        operation_name="search",
    )

    data = result["payload"].get("data") or {}
    raw_results = list(data.get("results") or [])
    normalized = []

    for rank, item in enumerate(raw_results[:top_k], start=1):
        title = _4c6w_clean_search_text(item.get("title"))
        url = str(item.get("url") or "").strip()
        description = _4c6w_clean_search_text(item.get("description"))
        source = str(item.get("source") or "").strip()
        engine = str(item.get("engine") or "").strip()

        if not url or not description:
            continue

        normalized.append({
            "evidence_id": "",
            "retrieval_system": "LIVE_WEB",
            "retrieval_systems": ["LIVE_WEB"],
            "source_family": "OPEN_WEBSEARCH",
            "source_families": ["OPEN_WEBSEARCH"],
            "document_id": url,
            "title": title or url,
            "chunk_id": "",
            "score": 0.0,
            "native_ranks": {"LIVE_WEB": rank},
            "native_scores": {"LIVE_WEB": 0.0},
            "fusion_score": None,
            "text": description,
            "text_chars": len(description),
            "published_date": None,
            "url": url,
            "retrieved_at_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "provenance": [{
                "retrieval_system": "LIVE_WEB",
                "source_family": "OPEN_WEBSEARCH",
                "provider": "Open-WebSearch",
                "operation": "search",
                "engine": engine,
                "source": source,
                "url": url,
                "native_rank": rank,
            }],
        })

    if not normalized:
        raise RuntimeError(
            "Open-WebSearch search succeeded but returned no "
            "normalizable evidence."
        )

    return {
        "operation": "search",
        "query": query,
        "candidate_count": len(normalized),
        "evidence": normalized,
        "elapsed_s": result["elapsed_s"],
        "trace": {
            "provider": "Open-WebSearch",
            "operation": "search",
            "engines": data.get("engines"),
            "total_results": data.get("totalResults"),
            "partial_failures": data.get("partialFailures"),
            "external_operation_number": OPEN_WEBSEARCH_OPERATION_COUNT_4C6W,
        },
    }


def _4c6w_best_fetch_url(evidence):
    """Prefer official / high-authority sources, otherwise preserve search rank."""

    if not evidence:
        return ""

    authority_domains = (
        ".gov.",
        ".gov/",
        "gov.uk",
        ".edu/",
        ".ac.",
        "who.int",
        "un.org",
        "europa.eu",
        "ietf.org",
        "3gpp.org",
        "gsma.com",
    )

    for item in evidence:
        url = str(item.get("url") or "").strip().lower()

        if any(token in url for token in authority_domains):
            return str(item.get("url") or "").strip()

    return str(evidence[0].get("url") or "").strip()


def open_websearch_fetch_4c6w(url, max_chars=12000):
    """
    ONE external operation:
    Open-WebSearch CLI fetch-web <URL> --json
    """

    url = str(url or "").strip()

    if not url:
        raise ValueError("Open-WebSearch fetch URL cannot be empty.")

    result = _4c6w_run_open_websearch_cli(
        ["fetch-web", url, "--json"],
        timeout_s=OPEN_WEBSEARCH_FETCH_TIMEOUT_S_4C6W,
        operation_name="fetch-web",
    )

    data = result["payload"].get("data") or {}
    content = _4c6w_clean_search_text(data.get("content"))

    final_url = str(
        data.get("finalUrl") or data.get("url") or url
    ).strip()

    title = _4c6w_clean_search_text(data.get("title"))

    if not content:
        raise RuntimeError("Open-WebSearch fetch-web returned no content.")

    content = content[:int(max_chars)]

    evidence = [{
        "evidence_id": "",
        "retrieval_system": "LIVE_WEB",
        "retrieval_systems": ["LIVE_WEB"],
        "source_family": "OPEN_WEBSEARCH",
        "source_families": ["OPEN_WEBSEARCH"],
        "document_id": final_url,
        "title": title or final_url,
        "chunk_id": "",
        "score": 1.0,
        "native_ranks": {"LIVE_WEB": 1},
        "native_scores": {"LIVE_WEB": 1.0},
        "fusion_score": None,
        "text": content,
        "text_chars": len(content),
        "published_date": None,
        "url": final_url,
        "retrieved_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "provenance": [{
            "retrieval_system": "LIVE_WEB",
            "source_family": "OPEN_WEBSEARCH",
            "provider": "Open-WebSearch",
            "operation": "fetch-web",
            "retrieval_method": data.get("retrievalMethod"),
            "content_type": data.get("contentType"),
            "url": final_url,
        }],
    }]

    return {
        "operation": "fetch",
        "url": final_url,
        "candidate_count": 1,
        "evidence": evidence,
        "elapsed_s": float(result["elapsed_s"]),
        "trace": {
            "provider": "Open-WebSearch",
            "operation": "fetch-web",
            "retrieval_method": data.get("retrievalMethod"),
            "truncated": data.get("truncated"),
            "external_operation_number": OPEN_WEBSEARCH_OPERATION_COUNT_4C6W,
        },
    }


async def retrieve_live_external_candidates_4c6w(query, top_k=5):
    result = await asyncio.to_thread(
        open_websearch_search_4c6w,
        query,
        top_k,
    )

    return {
        "query": result["query"],
        "retrieval_system": "LIVE_WEB",
        "candidate_count": result["candidate_count"],
        "evidence": result["evidence"],
        "timing": {
            "mcp_roundtrip_s": result["elapsed_s"],
            "retrieval_time_s": result["elapsed_s"],
        },
        "trace": result["trace"],
    }


def plan_live_external_search_4c6w(question):
    current_date = datetime.date.today().isoformat()
    start = time.perf_counter()

    response = _4c6w_call_llm(
        messages=[
            {
                "role": "system",
                "content": LIVE_SEARCH_PLANNER_SYSTEM_PROMPT_4C6W,
            },
            {
                "role": "user",
                "content": (
                    f"CURRENT DATE: {current_date}\n\n"
                    f"QUESTION:\n{question}"
                ),
            },
        ],
        max_tokens=360,
    )

    elapsed = time.perf_counter() - start
    raw_text = _4c6w_extract_text(response)
    parsed = _4c6w_parse_json_object(raw_text)

    query = re.sub(
        r"\s+",
        " ",
        str(parsed.get("query", "") or ""),
    ).strip()

    if not query:
        query = str(question).strip()

    raw_requirements = parsed.get("answer_requirements", [])

    if not isinstance(raw_requirements, list):
        raw_requirements = []

    requirements = []

    for item in raw_requirements:
        item = re.sub(r"\s+", " ", str(item or "")).strip()
        if item and item not in requirements:
            requirements.append(item)

    requirements = requirements[:5]

    if not requirements:
        requirements = [str(question).strip()]

    return {
        "query": query,
        "reason": str(parsed.get("reason", "") or "").strip(),
        "answer_requirements": requirements,
        "latency_s": float(elapsed),
        "usage": _4c6w_usage(response),
        "raw_text": raw_text,
    }


def run_live_external_grounded_generation_4c6w(
    question,
    requirements,
    context,
):
    requirements_text = "\n".join(
        f"R{index}: {requirement}"
        for index, requirement in enumerate(requirements, start=1)
    )

    start = time.perf_counter()

    response = _4c6w_call_llm(
        messages=[
            {
                "role": "system",
                "content": LIVE_GROUNDED_SYSTEM_PROMPT_4C6W,
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION\n--------\n{question}\n\n"
                    f"FIXED ANSWER REQUIREMENTS\n"
                    f"-------------------------\n"
                    f"{requirements_text}\n\n"
                    f"LIVE EXTERNAL EVIDENCE\n"
                    f"----------------------\n"
                    f"{context}"
                ),
            },
        ],
        max_tokens=900,
    )

    elapsed = time.perf_counter() - start

    return {
        "answer": _4c6w_extract_text(response),
        "elapsed_s": float(elapsed),
        "usage": _4c6w_usage(response),
    }


def _scan_live_evidence(raw_evidence, operation, descriptor):
    hits = []

    for raw_item in raw_evidence:
        matches = get_injection_matches_4c3(
            _4c6w_evidence_text(raw_item)
        )

        if matches:
            hits.append({
                "operation": operation,
                **descriptor,
                "title": raw_item.get("title", ""),
                "url": raw_item.get("url", ""),
                "matched_patterns": list(matches),
            })

    return hits


async def run_live_external_grounded_4c6w(question):
    """
    Canonical governed live route:
      op1 SEARCH
      op2 optional FETCH authoritative source
      op3 optional refined SEARCH
      grounded generation
    """
    global OPEN_WEBSEARCH_OPERATION_COUNT_4C6W

    start_wall = time.perf_counter()
    OPEN_WEBSEARCH_OPERATION_COUNT_4C6W = 0

    planner = await asyncio.to_thread(
        plan_live_external_search_4c6w,
        question,
    )

    current_query = planner["query"]
    requirements = list(planner["answer_requirements"])

    total_llm_calls = 1
    total_input_tokens = int(planner["usage"]["input_tokens"])
    total_output_tokens = int(planner["usage"]["output_tokens"])

    evidence_ledger = []
    evidence_by_fingerprint = {}
    evidence_security_hits = []

    search_queries = []
    round_traces = []

    live_retrieval_total_s = 0.0
    evidence_scan_total_ms = 0.0

    final_selection = None
    final_assessment = None
    stop_reason = None

    # --------------------------------------------------------
    # OPERATION 1 — SEARCH
    # --------------------------------------------------------
    search_queries.append(current_query)

    search_result = await asyncio.to_thread(
        open_websearch_search_4c6w,
        current_query,
        5,
    )

    live_retrieval_total_s += float(search_result["elapsed_s"])

    scan_start = time.perf_counter()
    hits = _scan_live_evidence(
        search_result["evidence"],
        operation=1,
        descriptor={
            "type": "search",
            "query": current_query,
        },
    )
    evidence_scan_total_ms += (
        time.perf_counter() - scan_start
    ) * 1000.0
    evidence_security_hits.extend(hits)

    if evidence_security_hits:
        raise RuntimeError(
            "Live search evidence failed deterministic prompt-injection screening."
        )

    _4c6w_merge_round_evidence(
        evidence_ledger=evidence_ledger,
        evidence_by_fingerprint=evidence_by_fingerprint,
        raw_evidence=search_result["evidence"],
        round_number=1,
        query=current_query,
    )

    selected = _4c6w_rank_evidence(
        evidence_ledger
    )[:MODULE4_MAX_FINAL_EVIDENCE_4C6W]

    final_selection = _4c6w_build_bounded_context(selected)

    assessment = await asyncio.to_thread(
        _4c6w_assess_sufficiency,
        question,
        requirements,
        "LIVE_EXTERNAL",
        1,
        final_selection["context"],
    )

    total_llm_calls += 1
    total_input_tokens += int(assessment["usage"]["input_tokens"])
    total_output_tokens += int(assessment["usage"]["output_tokens"])
    final_assessment = assessment

    round_traces.append({
        "external_operation": 1,
        "operation": "search",
        "query": current_query,
        "candidate_count": search_result["candidate_count"],
        "presented_evidence_ids": [
            item["evidence_id"]
            for item in final_selection["evidence"]
        ],
        "retrieval_s": search_result["elapsed_s"],
        "sufficiency": assessment,
    })

    if assessment["sufficient"]:
        stop_reason = "SUFFICIENT_AFTER_SEARCH"

    # --------------------------------------------------------
    # OPERATION 2 — FETCH BEST SOURCE
    # --------------------------------------------------------
    if (
        not final_assessment["sufficient"]
        and OPEN_WEBSEARCH_OPERATION_COUNT_4C6W
        < OPEN_WEBSEARCH_OPERATION_LIMIT_4C6W
    ):
        best_url = _4c6w_best_fetch_url(search_result["evidence"])

        if best_url:
            fetch_result = None
            fetch_error = None

            try:
                fetch_result = await asyncio.to_thread(
                    open_websearch_fetch_4c6w,
                    best_url,
                    12000,
                )
            except Exception as exc:
                fetch_error = f"{type(exc).__name__}: {exc}"

                round_traces.append({
                    "external_operation": 2,
                    "operation": "fetch",
                    "url": best_url,
                    "candidate_count": 0,
                    "presented_evidence_ids": [
                        item["evidence_id"]
                        for item in final_selection["evidence"]
                    ],
                    "retrieval_s": 0.0,
                    "fetch_error": fetch_error,
                    "sufficiency": final_assessment,
                })

            if fetch_result is not None:
                live_retrieval_total_s += float(fetch_result["elapsed_s"])

                scan_start = time.perf_counter()
                hits = _scan_live_evidence(
                    fetch_result["evidence"],
                    operation=2,
                    descriptor={
                        "type": "fetch",
                        "url": best_url,
                    },
                )
                evidence_scan_total_ms += (
                    time.perf_counter() - scan_start
                ) * 1000.0
                evidence_security_hits.extend(hits)

                if evidence_security_hits:
                    raise RuntimeError(
                        "Fetched live evidence failed deterministic "
                        "prompt-injection screening."
                    )

                _4c6w_merge_round_evidence(
                    evidence_ledger=evidence_ledger,
                    evidence_by_fingerprint=evidence_by_fingerprint,
                    raw_evidence=fetch_result["evidence"],
                    round_number=2,
                    query=f"FETCH::{best_url}",
                )

                selected = _4c6w_rank_evidence(
                    evidence_ledger
                )[:MODULE4_MAX_FINAL_EVIDENCE_4C6W]

                final_selection = _4c6w_build_bounded_context(selected)

                assessment = await asyncio.to_thread(
                    _4c6w_assess_sufficiency,
                    question,
                    requirements,
                    "LIVE_EXTERNAL",
                    2,
                    final_selection["context"],
                )

                total_llm_calls += 1
                total_input_tokens += int(
                    assessment["usage"]["input_tokens"]
                )
                total_output_tokens += int(
                    assessment["usage"]["output_tokens"]
                )

                final_assessment = assessment

                round_traces.append({
                    "external_operation": 2,
                    "operation": "fetch",
                    "url": best_url,
                    "candidate_count": 1,
                    "presented_evidence_ids": [
                        item["evidence_id"]
                        for item in final_selection["evidence"]
                    ],
                    "retrieval_s": fetch_result["elapsed_s"],
                    "sufficiency": assessment,
                })

                if assessment["sufficient"]:
                    stop_reason = "SUFFICIENT_AFTER_FETCH"

    # --------------------------------------------------------
    # OPERATION 3 — FINAL REFINED SEARCH
    # --------------------------------------------------------
    if (
        not final_assessment["sufficient"]
        and OPEN_WEBSEARCH_OPERATION_COUNT_4C6W
        < OPEN_WEBSEARCH_OPERATION_LIMIT_4C6W
    ):
        refined_query = str(
            final_assessment.get("next_query", "")
            or current_query
        ).strip()

        if refined_query:
            search_queries.append(refined_query)

            final_search = await asyncio.to_thread(
                open_websearch_search_4c6w,
                refined_query,
                5,
            )

            live_retrieval_total_s += float(final_search["elapsed_s"])

            scan_start = time.perf_counter()
            hits = _scan_live_evidence(
                final_search["evidence"],
                operation=3,
                descriptor={
                    "type": "search",
                    "query": refined_query,
                },
            )
            evidence_scan_total_ms += (
                time.perf_counter() - scan_start
            ) * 1000.0
            evidence_security_hits.extend(hits)

            if evidence_security_hits:
                raise RuntimeError(
                    "Final live search evidence failed deterministic "
                    "prompt-injection screening."
                )

            _4c6w_merge_round_evidence(
                evidence_ledger=evidence_ledger,
                evidence_by_fingerprint=evidence_by_fingerprint,
                raw_evidence=final_search["evidence"],
                round_number=3,
                query=refined_query,
            )

            selected = _4c6w_rank_evidence(
                evidence_ledger
            )[:MODULE4_MAX_FINAL_EVIDENCE_4C6W]

            final_selection = _4c6w_build_bounded_context(selected)

            assessment = await asyncio.to_thread(
                _4c6w_assess_sufficiency,
                question,
                requirements,
                "LIVE_EXTERNAL",
                3,
                final_selection["context"],
            )

            total_llm_calls += 1
            total_input_tokens += int(
                assessment["usage"]["input_tokens"]
            )
            total_output_tokens += int(
                assessment["usage"]["output_tokens"]
            )

            final_assessment = assessment

            round_traces.append({
                "external_operation": 3,
                "operation": "search",
                "query": refined_query,
                "candidate_count": final_search["candidate_count"],
                "presented_evidence_ids": [
                    item["evidence_id"]
                    for item in final_selection["evidence"]
                ],
                "retrieval_s": final_search["elapsed_s"],
                "sufficiency": assessment,
            })

            stop_reason = (
                "SUFFICIENT_AFTER_FINAL_SEARCH"
                if assessment["sufficient"]
                else "OPERATION_BUDGET_EXHAUSTED_SAFE_INSUFFICIENCY"
            )

    if final_selection is None:
        raise RuntimeError(
            "Live external route produced no evidence selection."
        )

    generation = await asyncio.to_thread(
        run_live_external_grounded_generation_4c6w,
        question,
        requirements,
        final_selection["context"],
    )

    total_llm_calls += 1
    total_input_tokens += int(generation["usage"]["input_tokens"])
    total_output_tokens += int(generation["usage"]["output_tokens"])

    answer = str(generation.get("answer", "") or "").strip()

    if not answer:
        raise RuntimeError(
            "Live external grounded generator returned an empty answer."
        )

    return {
        "answer": answer,
        "elapsed_s": float(generation["elapsed_s"]),
        "tool_loop_wall_s": float(time.perf_counter() - start_wall),
        "search_count": len(search_queries),
        "external_operation_count": int(
            OPEN_WEBSEARCH_OPERATION_COUNT_4C6W
        ),
        "search_queries": list(search_queries),
        "round_traces": round_traces,
        "planner": planner,
        "answer_requirements": requirements,
        "final_sufficiency_assessment": final_assessment,
        "stop_reason": stop_reason,
        "final_evidence": list(final_selection["evidence"]),
        "final_context": final_selection["context"],
        "final_context_chars": final_selection["context_chars"],
        "live_retrieval_s": float(live_retrieval_total_s),
        "evidence_scan_ms": float(evidence_scan_total_ms),
        "evidence_security_hits": evidence_security_hits,
        "llm_calls": total_llm_calls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    }


def get_live_external_status() -> dict[str, Any]:
    return {
        "runtime": "Notebook 21 Cell 4D / 6Y v1.3 live external branch",
        "provider": "Open-WebSearch CLI",
        "open_websearch_version": OPEN_WEBSEARCH_VERSION_4C6W,
        "default_engine": OPEN_WEBSEARCH_DEFAULT_ENGINE_4C6W,
        "allowed_engines": ["duckduckgo", "bing"],
        "operation_limit": OPEN_WEBSEARCH_OPERATION_LIMIT_4C6W,
        "search_timeout_s": OPEN_WEBSEARCH_SEARCH_TIMEOUT_S_4C6W,
        "fetch_timeout_s": OPEN_WEBSEARCH_FETCH_TIMEOUT_S_4C6W,
        "operation_sequence": [
            "SEARCH",
            "optional FETCH",
            "optional refined SEARCH",
        ],
        "evidence_security_scan": True,
        "grounded_citations": "[E#]",
        "npx_resolved": _resolve_npx(),
    }
