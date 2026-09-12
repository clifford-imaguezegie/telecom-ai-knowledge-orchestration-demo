from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from runtime.config import (
    DEFAULT_TIMEZONE,
    MAX_WEB_OPERATIONS,
)


# ============================================================
# TELECOM AI KNOWLEDGE ORCHESTRATION — LIVE EXTERNAL GROUNDING
# ============================================================

DATE_TIME_KEYWORDS = {
    "date",
    "today",
    "time",
    "current time",
    "current date",
    "what day is it",
    "what is todays date",
    "what is today's date",
    "what time is it",
}

DEFAULT_SEARCH_ENGINE = os.getenv(
    "DEFAULT_SEARCH_ENGINE",
    "duckduckgo",
)

AUTHORITATIVE_DOMAIN_HINTS = (
    ".gov.",
    ".gov/",
    "gov.uk",
    ".edu",
    ".ac.",
    "who.int",
    "itu.int",
    "3gpp.org",
    "etsi.org",
    "gsma.com",
    "o-ran.org",
)

MAX_FETCH_CHARS = 6000


# ============================================================
# LOCAL DATE / TIME PATH
# ============================================================

def is_local_datetime_question(
    question: str,
) -> bool:
    """
    Detect questions that should be answered from the
    deterministic local runtime clock rather than web search.
    """

    text = str(
        question or ""
    ).strip().lower()

    return any(
        keyword in text
        for keyword in DATE_TIME_KEYWORDS
    )


def get_local_datetime(
    timezone_name: str = DEFAULT_TIMEZONE,
) -> dict[str, Any]:
    """
    Return deterministic local date/time information.
    """

    start = time.perf_counter()

    try:
        timezone = ZoneInfo(
            timezone_name
        )

    except Exception as exc:
        raise RuntimeError(
            f"Invalid timezone: {timezone_name}"
        ) from exc

    now = datetime.now(
        timezone
    )

    elapsed_s = (
        time.perf_counter()
        -
        start
    )

    return {
        "timezone": timezone_name,
        "iso_datetime": now.isoformat(),
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "human_readable": now.strftime(
            "%A, %d %B %Y at %H:%M:%S %Z"
        ),
        "elapsed_s": elapsed_s,
    }


def answer_local_datetime(
    question: str,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> dict[str, Any]:
    """
    Produce a user-facing answer for local date/time questions.
    """

    info = get_local_datetime(
        timezone_name=timezone_name
    )

    text = str(
        question or ""
    ).lower()

    if "time" in text and "date" not in text:
        answer = (
            f"The current time in {timezone_name} is "
            f"{info['time']} on {info['day_of_week']}."
        )

    elif (
        "date" in text
        or
        "today" in text
        or
        "what day" in text
    ):
        answer = (
            f"Today's date in {timezone_name} is "
            f"{info['date']} ({info['day_of_week']})."
        )

    else:
        answer = (
            f"The current local date and time in "
            f"{timezone_name} is "
            f"{info['human_readable']}."
        )

    return {
        "answer": answer,
        "source_type": "LOCAL_RUNTIME",
        "web_search_executed": False,
        "timezone": timezone_name,
        "datetime": info,
        "elapsed_s": info["elapsed_s"],
    }


# ============================================================
# OPEN-WEBSEARCH CLI HELPERS
# ============================================================

def _resolve_npx_executable() -> str:
    """
    Resolve npx across Windows and Linux.
    """

    executable = (
        shutil.which("npx")
        or
        shutil.which("npx.cmd")
    )

    if not executable:
        raise RuntimeError(
            "npx executable was not found on PATH. "
            "Install Node.js/npm before using live web grounding."
        )

    return executable


def _extract_json_from_cli_output(
    stdout: str,
) -> dict[str, Any]:
    """
    Extract the JSON payload from Open-WebSearch CLI output.
    """

    text = str(
        stdout or ""
    )

    start_index = text.find("{")

    if start_index < 0:
        raise RuntimeError(
            "Open-WebSearch returned no JSON payload."
        )

    json_text = text[
        start_index:
    ].strip()

    try:
        return json.loads(
            json_text
        )

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Could not parse Open-WebSearch JSON output."
        ) from exc


# ============================================================
# SEARCH
# ============================================================

def search_web(
    query: str,
    engine: str = DEFAULT_SEARCH_ENGINE,
    max_results: int = 5,
) -> dict[str, Any]:
    """
    Execute one Open-WebSearch query.
    """

    start = time.perf_counter()

    npx_executable = _resolve_npx_executable()

    env = os.environ.copy()
    env["DEFAULT_SEARCH_ENGINE"] = engine

    command = [
        npx_executable,
        "-y",
        "open-websearch@2.1.11",
        "search",
        query,
        "--json",
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=45,
            check=False,
        )

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Open-WebSearch search timed out."
        ) from exc

    if completed.returncode != 0:
        raise RuntimeError(
            "Open-WebSearch search failed with return code "
            f"{completed.returncode}: "
            f"{completed.stderr.strip()}"
        )

    payload = _extract_json_from_cli_output(
        completed.stdout
    )

    if payload.get("status") != "ok":
        raise RuntimeError(
            f"Open-WebSearch returned error status: {payload}"
        )

    data = payload.get(
        "data",
        {},
    )

    raw_results = (
        data.get(
            "results",
            [],
        )
        or
        []
    )

    results = []

    for item in raw_results[
        :max_results
    ]:
        results.append(
            {
                "title": str(
                    item.get(
                        "title",
                        "",
                    )
                ).strip(),

                "url": str(
                    item.get(
                        "url",
                        "",
                    )
                ).strip(),

                "description": str(
                    item.get(
                        "description",
                        "",
                    )
                ).strip(),

                "source": str(
                    item.get(
                        "source",
                        "",
                    )
                ).strip(),

                "engine": str(
                    item.get(
                        "engine",
                        engine,
                    )
                ).strip(),
            }
        )

    return {
        "query": query,
        "engine": engine,
        "results": results,
        "result_count": len(results),
        "partial_failures": data.get(
            "partialFailures",
            [],
        ),
        "elapsed_s": (
            time.perf_counter()
            -
            start
        ),
        "web_operations": 1,
    }


def search_web_with_fallback(
    query: str,
    max_results: int = 5,
) -> dict[str, Any]:
    """
    Search using a bounded fallback policy.
    """

    engines = [
        "duckduckgo",
        "startpage",
        "bing",
    ]

    attempts = []

    total_start = time.perf_counter()

    for engine in engines[
        :MAX_WEB_OPERATIONS
    ]:

        try:
            result = search_web(
                query=query,
                engine=engine,
                max_results=max_results,
            )

            attempts.append(
                {
                    "engine": engine,
                    "result_count":
                        result["result_count"],
                    "elapsed_s":
                        result["elapsed_s"],
                }
            )

            if result["result_count"] > 0:
                result["attempts"] = attempts
                result["total_web_operations"] = len(
                    attempts
                )
                result["total_elapsed_s"] = (
                    time.perf_counter()
                    -
                    total_start
                )

                return result

        except Exception as exc:
            attempts.append(
                {
                    "engine": engine,
                    "error": str(exc),
                }
            )

    return {
        "query": query,
        "engine": None,
        "results": [],
        "result_count": 0,
        "partial_failures": [],
        "elapsed_s": (
            time.perf_counter()
            -
            total_start
        ),
        "web_operations": len(
            attempts
        ),
        "total_web_operations": len(
            attempts
        ),
        "attempts": attempts,
    }


# ============================================================
# FETCH
# ============================================================

def fetch_web_page(
    url: str,
    max_chars: int = MAX_FETCH_CHARS,
) -> dict[str, Any]:
    """
    Fetch one web page through Open-WebSearch.
    """

    start = time.perf_counter()

    npx_executable = _resolve_npx_executable()

    command = [
        npx_executable,
        "-y",
        "open-websearch@2.1.11",
        "fetch-web",
        url,
        "--json",
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45,
            check=False,
        )

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Open-WebSearch fetch timed out: {url}"
        ) from exc

    if completed.returncode != 0:
        raise RuntimeError(
            "Open-WebSearch fetch failed with return code "
            f"{completed.returncode}: "
            f"{completed.stderr.strip()}"
        )

    payload = _extract_json_from_cli_output(
        completed.stdout
    )

    if payload.get("status") != "ok":
        raise RuntimeError(
            f"Open-WebSearch fetch returned error: {payload}"
        )

    data = payload.get(
        "data",
        {},
    )

    content = str(
        data.get(
            "content",
            "",
        )
    ).strip()

    return {
        "url": str(
            data.get(
                "finalUrl",
                url,
            )
        ).strip(),
        "title": str(
            data.get(
                "title",
                "",
            )
        ).strip(),
        "content": content[
            :max_chars
        ],
        "content_chars": min(
            len(content),
            max_chars,
        ),
        "truncated": (
            len(content)
            >
            max_chars
        ),
        "retrieval_method": str(
            data.get(
                "retrievalMethod",
                "",
            )
        ).strip(),
        "elapsed_s": (
            time.perf_counter()
            -
            start
        ),
        "web_operations": 1,
    }


# ============================================================
# AUTHORITY RANKING
# ============================================================

def _authority_score(
    result: dict[str, Any],
) -> int:
    """
    Prefer authoritative public/standards sources.
    """

    url = str(
        result.get(
            "url",
            "",
        )
    ).lower()

    score = 0

    for hint in AUTHORITATIVE_DOMAIN_HINTS:
        if hint in url:
            score += 10

    if "wikipedia.org" in url:
        score += 2

    if result.get("description"):
        score += 1

    return score


def select_fetch_candidates(
    search_results: list[dict[str, Any]],
    max_fetches: int = 2,
) -> list[dict[str, Any]]:
    """
    Select the most authoritative pages for full fetch.
    """

    ranked = sorted(
        search_results,
        key=_authority_score,
        reverse=True,
    )

    selected = []

    seen_urls = set()

    for item in ranked:

        url = str(
            item.get(
                "url",
                "",
            )
        ).strip()

        if not url:
            continue

        if url in seen_urls:
            continue

        selected.append(
            item
        )

        seen_urls.add(
            url
        )

        if len(selected) >= max_fetches:
            break

    return selected


# ============================================================
# LIVE WEB EVIDENCE PACKAGE
# ============================================================

def build_live_web_evidence(
    query: str,
    max_results: int = 5,
) -> dict[str, Any]:
    """
    Search the live web, prefer authoritative sources,
    and fetch pages while respecting MAX_WEB_OPERATIONS.

    Operation budget:
        Search = 1
        Fetch 1 = 1
        Fetch 2 = 1
    """

    total_start = time.perf_counter()

    search_result = search_web_with_fallback(
        query=query,
        max_results=max_results,
    )

    operations_used = int(
        search_result.get(
            "total_web_operations",
            0,
        )
    )

    # --------------------------------------------------------
    # Normalized zero-result branch
    # --------------------------------------------------------

    if search_result["result_count"] == 0:
        return {
            "query": query,
            "search_engine":
                search_result.get(
                    "engine"
                ),
            "search":
                search_result,
            "evidence": [],
            "evidence_count": 0,
            "total_web_operations":
                operations_used,
            "elapsed_s": (
                time.perf_counter()
                -
                total_start
            ),
        }

    remaining_budget = max(
        0,
        MAX_WEB_OPERATIONS
        -
        operations_used,
    )

    candidates = select_fetch_candidates(
        search_result["results"],
        max_fetches=remaining_budget,
    )

    evidence = []

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        try:
            fetched = fetch_web_page(
                candidate["url"]
            )

            operations_used += 1

            evidence.append(
                {
                    "evidence_id":
                        f"W{index}",
                    "title":
                        fetched["title"]
                        or
                        candidate["title"],
                    "url":
                        fetched["url"],
                    "source":
                        candidate.get(
                            "source",
                            "",
                        ),
                    "content":
                        fetched["content"],
                    "retrieval_method":
                        fetched[
                            "retrieval_method"
                        ],
                }
            )

        except Exception as exc:
            evidence.append(
                {
                    "evidence_id":
                        f"W{index}",
                    "title":
                        candidate.get(
                            "title",
                            "",
                        ),
                    "url":
                        candidate.get(
                            "url",
                            "",
                        ),
                    "source":
                        candidate.get(
                            "source",
                            "",
                        ),
                    "content":
                        candidate.get(
                            "description",
                            "",
                        ),
                    "retrieval_method":
                        "SEARCH_SNIPPET_FALLBACK",
                    "fetch_error":
                        str(exc),
                }
            )

    return {
        "query": query,
        "search_engine":
            search_result["engine"],
        "search":
            search_result,
        "evidence":
            evidence,
        "evidence_count":
            len(evidence),
        "total_web_operations":
            operations_used,
        "elapsed_s": (
            time.perf_counter()
            -
            total_start
        ),
    }


# ============================================================
# CONTEXT FORMATTING
# ============================================================

def format_live_web_context(
    evidence: list[dict[str, Any]],
) -> str:
    """
    Format web evidence for grounded Gemma synthesis.
    """

    blocks = []

    for item in evidence:

        evidence_id = item[
            "evidence_id"
        ]

        title = item.get(
            "title",
            "",
        )

        url = item.get(
            "url",
            "",
        )

        content = item.get(
            "content",
            "",
        )

        blocks.append(
            f"[{evidence_id}]\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content:\n{content}"
        )

    return "\n\n".join(
        blocks
    )