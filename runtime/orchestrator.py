from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime as DateTime
from zoneinfo import ZoneInfo
from typing import Any

from openai import OpenAI

from runtime.config import (
    GENERATOR_MODEL,
    ROUTER_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MAX_OUTPUT_TOKENS,
    DEFAULT_TIMEZONE,
)
from runtime.adaptive import run_connected_corpus_grounded
from runtime.live_external import run_live_external_grounded_4c6w


# ============================================================
# TELECOM AI KNOWLEDGE ORCHESTRATION — TOP-LEVEL RUNTIME
# Phase 1 integration from validated Notebook 21 / Cell 4D / 6Y v1.3.
#
# This module validates:
#   deterministic security
#   Gemma-only baseline
#   Granite three-way knowledge-scope routing
#   connected-corpus dispatch
#   stable general-knowledge fallback
#   deterministic local date/time handling
#
# The Open-WebSearch live-external branch is added in the next deployment
# step after this control-plane integration passes.
# ============================================================

TELECOM_GROUNDED_MODE = "TELECOM_GROUNDED"
LIVE_EXTERNAL_GROUNDED_MODE = "LIVE_EXTERNAL_GROUNDED"
GENERAL_KNOWLEDGE_FALLBACK_MODE = "GENERAL_KNOWLEDGE_FALLBACK"

VALID_RESPONSE_MODES = {
    TELECOM_GROUNDED_MODE,
    LIVE_EXTERNAL_GROUNDED_MODE,
    GENERAL_KNOWLEDGE_FALLBACK_MODE,
}

INJECTION_PATTERNS_4C3 = (
    r"\bignore (?:all |any |the )?(?:previous|prior|earlier) instructions?\b",
    r"\bdisregard (?:all |any |the )?(?:previous|prior|earlier) instructions?\b",
    r"\boverride (?:the )?(?:system|developer|previous|prior) instructions?\b",
    r"\breveal (?:the )?(?:system|developer) prompt\b",
    r"\bshow (?:me )?(?:the )?(?:system|developer) prompt\b",
    r"\bprint (?:the )?(?:system|developer) prompt\b",
    r"\byou are now\b",
    r"\bact as\b.*\binstead\b",
    r"\bforget (?:all |the )?(?:previous|prior|earlier) instructions?\b",
)

EVIDENCE_MANIPULATION_PATTERNS_4C3 = (
    r"\bignore (?:the )?(?:retrieved )?evidence\b",
    r"\bignore (?:the )?(?:provided )?context\b",
    r"\bdo not cite\b",
    r"\bfabricate (?:a |the )?citation\b",
    r"\binvent (?:a |the )?citation\b",
    r"\bpretend (?:the )?evidence\b",
    r"\bclaim (?:that )?(?:the )?evidence says\b",
    r"\bchange (?:the )?evidence\b",
    r"\boverride (?:the )?evidence\b",
)

KNOWLEDGE_ROUTER_SYSTEM_PROMPT = """
You are the knowledge-scope router for an AI system.

Classify the user's sanitized question into exactly ONE response mode.

MODE 1 — TELECOM_GROUNDED
Use when the question is materially covered by the connected controlled
technical corpus and should be answered from that corpus.

The connected corpus materially covers:
- telecommunications and mobile networks;
- 3GPP, 5G, 5G SA, NR, RAN, Core, OSS/BSS;
- telecom standards, interfaces, procedures and architecture;
- IETF networking and Internet protocols;
- Kubernetes, cloud-native infrastructure and platform engineering;
- O-RAN, ETSI, GSMA, TM Forum, CAMARA and related technical material;
- technical research and industry material represented in the corpus.

The question does NOT need to explicitly contain telecom terminology.
If the connected technical corpus is the best authoritative source,
use TELECOM_GROUNDED.

MODE 2 — LIVE_EXTERNAL_GROUNDED
Use when a reliable answer materially depends on public information that
may have changed over time, may have changed after model training, or
should be verified against current external sources.

Examples include, but are not limited to:
- current office holders, leaders, executives or organizational roles;
- latest or recent news and events;
- current prices, market values or exchange rates;
- current schedules, availability, standings or results;
- current laws, regulations, policies or official guidance;
- latest software, product, standards or release status;
- recent company announcements, statistics or economic indicators;
- any real-world status whose answer can become stale.

IMPORTANT:
- This is NOT keyword matching.
- A question may require LIVE_EXTERNAL_GROUNDED even without words such as
  "current", "latest", "today", "now" or "recent".
- For example, "Who is the prime minister of the United Kingdom?" normally
  asks for the person holding that office now and should be live-verified.
- A telecom-related question may also require LIVE_EXTERNAL_GROUNDED when
  the requested claim is explicitly about the latest/current state and the
  controlled corpus cannot establish that current state.

MODE 3 — GENERAL_KNOWLEDGE_FALLBACK
Use when the question is outside the connected technical corpus AND its
answer is stable general knowledge that does not reasonably require live
external verification.

Examples:
- explain photosynthesis;
- explain Newton's second law;
- what causes volcanic eruptions;
- stable historical or conceptual general-knowledge questions.

Decision priority:
1. If current/fresh external verification is materially required,
   choose LIVE_EXTERNAL_GROUNDED.
2. Otherwise, if the connected controlled technical corpus is appropriate,
   choose TELECOM_GROUNDED.
3. Otherwise choose GENERAL_KNOWLEDGE_FALLBACK.

Return JSON only:

{
  "response_mode": "TELECOM_GROUNDED | LIVE_EXTERNAL_GROUNDED | GENERAL_KNOWLEDGE_FALLBACK",
  "reason": "brief semantic routing reason"
}
""".strip()

_openrouter: OpenAI | None = None


def _client() -> OpenAI:
    global _openrouter
    if _openrouter is None:
        key = str(OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY", "")).strip()
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")
        _openrouter = OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)
    return _openrouter


def _extract_text(response) -> str:
    choices = getattr(response, "choices", []) or []
    if not choices:
        raise RuntimeError("OpenRouter returned no completion choices.")
    content = getattr(getattr(choices[0], "message", None), "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
            else:
                text = getattr(item, "text", "")
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _usage(response) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    raw_text = str(raw_text or "").strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.I)
        raw_text = re.sub(r"\s*```$", "", raw_text).strip()

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(raw_text[start:end + 1])
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Router response did not contain a valid JSON object.")


def preprocess_user_prompt(prompt: str) -> dict[str, Any]:
    start = time.perf_counter()
    raw_prompt = str(prompt or "").strip()

    injection_matches = [
        pattern for pattern in INJECTION_PATTERNS_4C3
        if re.search(pattern, raw_prompt, flags=re.I | re.S)
    ]
    evidence_matches = [
        pattern for pattern in EVIDENCE_MANIPULATION_PATTERNS_4C3
        if re.search(pattern, raw_prompt, flags=re.I | re.S)
    ]

    sanitized_prompt = raw_prompt
    for pattern in tuple(injection_matches) + tuple(evidence_matches):
        sanitized_prompt = re.sub(
            pattern, " ", sanitized_prompt, flags=re.I | re.S
        )

    sanitized_prompt = re.sub(r"\s+", " ", sanitized_prompt).strip()

    return {
        "raw_prompt": raw_prompt,
        "sanitized_prompt": sanitized_prompt,
        "injection_detected": bool(injection_matches),
        "evidence_manipulation_detected": bool(evidence_matches),
        "matched_patterns": injection_matches + evidence_matches,
        "elapsed_s": float(time.perf_counter() - start),
    }


def generate_gemma_only_baseline(question: str) -> dict[str, Any]:
    start = time.perf_counter()
    response = _client().chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question directly and accurately "
                    "using only your pretrained general knowledge. "
                    "Do not claim access to retrieved evidence, live web "
                    "information, RAG, MCP, or external tools."
                ),
            },
            {"role": "user", "content": str(question)},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    elapsed = time.perf_counter() - start
    usage = _usage(response)

    return {
        "answer": _extract_text(response),
        "elapsed_s": elapsed,
        **usage,
        "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
    }


def generate_general_fallback(question: str) -> dict[str, Any]:
    start = time.perf_counter()
    response = _client().chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using stable general "
                    "knowledge only. Do not claim live or retrieved "
                    "knowledge. Begin the answer exactly with:\n\n"
                    "⚠ Outside Telecom Knowledge Domain: This response "
                    "uses the model's general knowledge and is not grounded "
                    "in the connected Telecom RAG/MCP sources.\n\n"
                    "Then provide the useful answer."
                ),
            },
            {"role": "user", "content": str(question)},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    elapsed = time.perf_counter() - start
    answer = _extract_text(response)

    required_prefix = (
        "⚠ Outside Telecom Knowledge Domain: This response uses "
        "the model's general knowledge and is not grounded in the "
        "connected Telecom RAG/MCP sources."
    )

    if required_prefix not in answer:
        answer = required_prefix + "\n\n" + answer

    usage = _usage(response)

    return {
        "answer": answer,
        "elapsed_s": elapsed,
        **usage,
        "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
    }


def classify_knowledge_scope(question: str) -> dict[str, Any]:
    start = time.perf_counter()

    response = _client().chat.completions.create(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": KNOWLEDGE_ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": str(question).strip()},
        ],
        max_tokens=160,
        extra_body={"reasoning": {"effort": "none"}},
    )

    elapsed = time.perf_counter() - start
    raw_text = _extract_text(response)
    parsed = _parse_json_object(raw_text)

    mode = str(parsed.get("response_mode", "") or "").strip().upper()
    reason = str(parsed.get("reason", "") or "").strip()

    if mode not in VALID_RESPONSE_MODES:
        raise ValueError(f"Granite returned unsupported response mode: {mode!r}")

    return {
        "response_mode": mode,
        "reason": reason,
        "elapsed_s": elapsed,
        "raw_text": raw_text,
    }


def classify_local_datetime_request(question: str) -> dict[str, Any]:
    q = re.sub(r"\s+", " ", str(question or "").strip().lower())

    date_patterns = (
        r"\bwhat(?:'s| is) today'?s date\b",
        r"\bwhat(?:'s| is) todays date\b",
        r"\bwhat(?:'s| is) the date today\b",
        r"\bwhat(?:'s| is) the current date\b",
        r"\btoday'?s date\b",
        r"\btodays date\b",
        r"\bcurrent date\b",
        r"\bwhat day is it today\b",
    )

    time_patterns = (
        r"\bwhat time is it\b",
        r"\bwhat(?:'s| is) the current time\b",
        r"\bwhat time is it now\b",
        r"\bcurrent time\b",
        r"\btime now\b",
    )

    if any(re.search(pattern, q) for pattern in date_patterns):
        return {"matched": True, "kind": "DATE", "timezone": DEFAULT_TIMEZONE}

    if any(re.search(pattern, q) for pattern in time_patterns):
        return {"matched": True, "kind": "TIME", "timezone": DEFAULT_TIMEZONE}

    return {"matched": False, "kind": None, "timezone": None}


def run_local_datetime_tool(question: str) -> dict[str, Any] | None:
    classification = classify_local_datetime_request(question)
    if not classification["matched"]:
        return None

    tz_name = classification["timezone"]
    now = DateTime.now(ZoneInfo(tz_name))

    if classification["kind"] == "DATE":
        answer = f"Today's date is {now.strftime('%d %B %Y')}."
    else:
        answer = (
            f"The current time in London is {now.strftime('%H:%M:%S %Z')} "
            f"on {now.strftime('%d %B %Y')}."
        )

    return {
        "answer": answer,
        "kind": classification["kind"],
        "timezone": tz_name,
        "timestamp_iso": now.isoformat(),
        "runtime_tool": "LOCAL_DATE_TIME",
    }


async def run_runtime_phase1(question: str) -> dict[str, Any]:
    """
    Canonical top-level Notebook 21 control-plane integration.

    TELECOM_GROUNDED:
        baseline + adaptive connected-corpus answer.

    GENERAL_KNOWLEDGE_FALLBACK:
        baseline + stable general fallback.

    LIVE_EXTERNAL_GROUNDED:
        deterministic date/time is executed locally.
        Other freshness-sensitive questions use the validated
        Open-WebSearch grounded branch.
    """
    total_start = time.perf_counter()

    security = preprocess_user_prompt(question)
    sanitized = security["sanitized_prompt"]

    if not sanitized:
        raise ValueError("Question is empty after deterministic security sanitization.")

    # Notebook 21 runs baseline Gemma and Granite router concurrently.
    baseline_task = asyncio.to_thread(generate_gemma_only_baseline, sanitized)
    router_task = asyncio.to_thread(classify_knowledge_scope, sanitized)

    baseline, routing = await asyncio.gather(baseline_task, router_task)

    mode = routing["response_mode"]

    if mode == TELECOM_GROUNDED_MODE:
        flexible = await run_connected_corpus_grounded(sanitized)
        answer = flexible["answer"]
        execution = {
            "connected_corpus": True,
            "live_external": False,
            "general_fallback": False,
            "local_runtime": False,
        }

    elif mode == GENERAL_KNOWLEDGE_FALLBACK_MODE:
        flexible = await asyncio.to_thread(generate_general_fallback, sanitized)
        answer = flexible["answer"]
        execution = {
            "connected_corpus": False,
            "live_external": False,
            "general_fallback": True,
            "local_runtime": False,
        }

    else:
        local_result = run_local_datetime_tool(sanitized)

        if local_result is not None:
            flexible = {
                "answer": local_result["answer"],
                "local_runtime": local_result,
            }
            answer = local_result["answer"]
            execution = {
                "connected_corpus": False,
                "live_external": True,
                "general_fallback": False,
                "local_runtime": True,
            }
        else:
            flexible = await run_live_external_grounded_4c6w(sanitized)
            answer = flexible["answer"]
            execution = {
                "connected_corpus": False,
                "live_external": True,
                "general_fallback": False,
                "local_runtime": False,
            }

    return {
        "raw_question": security["raw_prompt"],
        "sanitized_question": sanitized,
        "security": security,
        "routing": routing,
        "gemma4_only": baseline,
        "flexible": flexible,
        "answer": answer,
        "execution": execution,
        "total_wall_s": float(time.perf_counter() - total_start),
    }


def get_orchestrator_status() -> dict[str, Any]:
    return {
        "runtime": "Notebook 21 Cell 4D / 6Y v1.3 top-level control plane",
        "generator_model": GENERATOR_MODEL,
        "router_model": ROUTER_MODEL,
        "valid_modes": sorted(VALID_RESPONSE_MODES),
        "security": "deterministic pre-LLM sanitization",
        "baseline": "Gemma-only, retrieval-independent",
        "connected_corpus": "adaptive Notebook 21 path",
        "general_fallback": "stable general knowledge + domain caveat",
        "local_datetime": "deterministic Python runtime",
        "live_web": "validated Open-WebSearch grounded branch",
    }
