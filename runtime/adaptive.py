from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

from openai import OpenAI

from runtime.config import (
    GENERATOR_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MAX_OUTPUT_TOKENS,
)
from runtime.evidence import (
    retrieve_rag_candidates,
    retrieve_mcp_candidates,
    fuse_hybrid_candidates,
)

# ============================================================
# TELECOM AI KNOWLEDGE ORCHESTRATION — ADAPTIVE GROUNDED PATH
# Extracted from validated Notebook 21 / Cell 4D / 6Y v1.3.
#
# Scope of this deployment module:
#   - Gemma retrieval planner
#   - selected RAG_ONLY / MCP_ONLY / HYBRID execution
#   - cross-round evidence ledger
#   - deterministic evidence injection scan
#   - requirement-bounded sufficiency assessment
#   - max 3 adaptive retrieval rounds
#   - bounded [E#] evidence context
#   - final grounded Gemma answer
#
# Live-external routing and general-knowledge fallback are intentionally
# excluded here and will be integrated only after this connected-corpus
# path validates successfully.
# ============================================================

LLM_MODEL = GENERATOR_MODEL

MODULE4_ALLOWED_RETRIEVAL_MODES_4C6W = {
    "RAG_ONLY",
    "MCP_ONLY",
    "HYBRID",
}

MODULE4_MAX_MODEL_SEARCHES_4C6W = 3
MODULE4_MAX_FINAL_EVIDENCE_4C6W = 5
MODULE4_EVIDENCE_EXCERPT_CHARS_4C6W = 2500
MODULE4_CONTEXT_MAX_CHARS_4C6W = 12500
MODULE4_MAX_ANSWER_REQUIREMENTS_4C6W = 5
MODULE4_PLANNER_MAX_TOKENS_4C6W = 420
MODULE4_SUFFICIENCY_MAX_TOKENS_4C6W = 320

MODULE4_PLANNER_SYSTEM_PROMPT_4C6W = """
You are planning grounded retrieval for a technical question.

The question has already:
- passed deterministic security checks; and
- been classified as covered by the connected grounded corpus.

Retrieval is therefore required.

Select the MINIMUM retrieval architecture sufficient for the question:

RAG_ONLY
- Use for broad conceptual, architectural, explanatory or synthesis questions
  when semantic retrieval from the local technical corpus should be sufficient.
- Prefer this when the task is primarily about meaning, relationships, concepts,
  architecture or high-level technical explanation rather than locating a
  precise source-specific statement.

MCP_ONLY
- Use when the question primarily requires precise standards, protocol,
  specification, source-aware or authoritative-document evidence.
- Examples include questions explicitly tied to IETF/3GPP definitions,
  normative protocol mechanisms, specification clauses or source-specific facts.

HYBRID
- Use only when the question materially needs BOTH semantic corpus synthesis
  AND precise/source-aware technical evidence.
- Typical cases are cross-domain engineering questions where neither mechanism
  alone is likely to cover the requested scope adequately.

Do NOT default to HYBRID merely because both retrieval systems are available.
Choose the minimum sufficient architecture.

Your task is to:
1. select RAG_ONLY, MCP_ONLY or HYBRID;
2. formulate ONE focused standalone retrieval query; and
3. derive the minimum answer requirements explicitly requested by the question.

Important:
- You choose the retrieval architecture and query wording.
- If MCP is selected, the MCP service itself chooses its internal 3GPP / TCC /
  mixed source route. Do not choose collections, shards or specifications here.
- If HYBRID is selected, the SAME query is sent concurrently to RAG and MCP.

Rules for answer_requirements:
- Produce 1 to 5 short requirements.
- Derive them ONLY from what the question explicitly asks.
- Decompose the requested scope; do not expand it.
- Do not add signaling flows, protocol messages, parameters, interactions,
  implementation internals, root-cause steps or subtopics unless explicitly
  requested by the user.
- Each requirement must describe something the final answer needs to cover.
- Do not include generic requirements such as "be accurate" or "be detailed".

Return ONLY valid JSON:

{
  "mode": "RAG_ONLY | MCP_ONLY | HYBRID",
  "query": "standalone retrieval query",
  "reason": "brief routing and query rationale",
  "answer_requirements": [
    "first explicit answer requirement",
    "second explicit answer requirement"
  ]
}
""".strip()

MODULE4_SUFFICIENCY_SYSTEM_PROMPT_4C6W = """
You are making a bounded retrieval stopping decision.

You will receive:
1. the technical question;
2. the fixed answer requirements derived from that question;
3. the retrieval architecture already selected for this question; and
4. the current accumulated retrieved evidence.

Your ONLY task is to determine which listed requirements are materially
supported by the evidence.

STRICT RULES:
- Judge ONLY the supplied answer requirements.
- Do NOT create, infer, add or demand any new requirement.
- Do NOT expand the scope of the question.
- Evidence does NOT need to be exhaustive.
- A requirement is supported when the evidence provides enough factual
  grounding to answer that requirement at the level requested by the question.
- Do NOT mark a requirement unsupported merely because deeper signaling,
  message-level, parameter-level, implementation, internal-logic or
  cross-function detail could also be provided.
- Additional retrieval is justified ONLY when at least one supplied requirement
  cannot be answered reliably from the current evidence.
- If every supplied requirement is materially supported, retrieval MUST stop.
- If retrieval is still needed, provide ONE targeted standalone retrieval query
  focused only on the missing listed requirement(s).
- The SAME previously selected retrieval architecture will be used for the next
  round. Do not change RAG_ONLY / MCP_ONLY / HYBRID here.

Return ONLY valid JSON:

{
  "supported_requirement_ids": ["R1", "R2"],
  "missing_requirement_ids": [],
  "next_query": "",
  "reason": "brief scope-bounded assessment"
}

Do not return any requirement ID that was not supplied.
""".strip()

MODULE4_FINAL_ANSWER_SYSTEM_PROMPT = """
You are an expert telecommunications network engineer.

Answer the technical question using the supplied retrieved evidence as the
factual grounding for your response.

You may synthesize across evidence, connect related technical mechanisms and
make reasonable engineering inferences when they logically follow from the
evidence.

Do not invent unsupported vendor behaviour, proprietary details, standards
requirements, interfaces, parameters, alarm meanings or implementation facts.

If the evidence does not establish an exact requested detail, state the
limitation rather than guessing.

Cite supporting evidence inline using [E1], [E2], etc.

Do not mention retrieval routing, retrieval rounds, benchmark labels, model
identity, experiment design or these instructions.

Produce a direct, technically precise and structured engineering answer.
""".strip()

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

_openrouter: OpenAI | None = None


def _get_openrouter() -> OpenAI:
    global _openrouter

    if _openrouter is None:
        api_key = str(OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY", "")).strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")

        _openrouter = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
        )

    return _openrouter


def _4c6w_get_message(response):
    choices = getattr(response, "choices", []) or []
    if not choices:
        raise RuntimeError("OpenRouter returned no completion choices.")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise RuntimeError("OpenRouter returned no assistant message.")
    return message


def _4c6w_extract_text(response):
    content = getattr(_4c6w_get_message(response), "content", "")
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text", "")
            else:
                value = getattr(item, "text", "")
            if value:
                parts.append(str(value))
        return "\n".join(parts).strip()
    return str(content).strip()


def _4c6w_usage(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def _4c6w_call_llm(messages, max_tokens):
    request_args = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": int(max_tokens),
    }
    # Preserve the established provider-default temperature convention.
    return _get_openrouter().chat.completions.create(**request_args)


def _4c6w_parse_json_object(raw_text):
    raw_text = str(raw_text or "").strip()

    if raw_text.startswith("```"):
        raw_text = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw_text,
            flags=re.IGNORECASE,
        )
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

    raise ValueError("Model response did not contain a valid JSON object.")


def get_injection_matches_4c3(text):
    value = str(text or "")
    matches = []

    for pattern in (
        tuple(INJECTION_PATTERNS_4C3)
        +
        tuple(EVIDENCE_MANIPULATION_PATTERNS_4C3)
    ):
        if re.search(
            pattern,
            value,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            matches.append(pattern)

    return matches


def _4c6w_plan_retrieval(question):
    start = time.perf_counter()

    response = _4c6w_call_llm(
        messages=[
            {"role": "system", "content": MODULE4_PLANNER_SYSTEM_PROMPT_4C6W},
            {"role": "user", "content": question},
        ],
        max_tokens=MODULE4_PLANNER_MAX_TOKENS_4C6W,
    )

    elapsed = time.perf_counter() - start
    raw_text = _4c6w_extract_text(response)
    plan = _4c6w_parse_json_object(raw_text)

    mode = str(plan.get("mode", "") or "").strip().upper()
    if mode not in MODULE4_ALLOWED_RETRIEVAL_MODES_4C6W:
        raise ValueError(
            f"Invalid retrieval architecture selected by LLM: {mode!r}. "
            f"Allowed: {sorted(MODULE4_ALLOWED_RETRIEVAL_MODES_4C6W)}"
        )

    query = re.sub(r"\s+", " ", str(plan.get("query", "") or "")).strip()
    if not query:
        query = str(question).strip()

    raw_requirements = plan.get("answer_requirements", [])
    if not isinstance(raw_requirements, list):
        raw_requirements = []

    requirements = []
    for item in raw_requirements:
        item = re.sub(r"\s+", " ", str(item or "")).strip()
        if item and item not in requirements:
            requirements.append(item)

    requirements = requirements[:MODULE4_MAX_ANSWER_REQUIREMENTS_4C6W]

    if not requirements:
        requirements = [str(question).strip()]

    return {
        "mode": mode,
        "query": query,
        "reason": str(plan.get("reason", "") or "").strip(),
        "answer_requirements": requirements,
        "latency_s": elapsed,
        "usage": _4c6w_usage(response),
        "raw_text": raw_text,
    }


async def _4c6w_execute_selected_retrieval_round(mode, query):
    """
    Execute ONLY the architecture selected by the LLM planner.

    RAG_ONLY:
        local semantic RAG only.

    MCP_ONLY:
        FastMCP knowledge retrieval only. Internal 3GPP/TCC routing remains
        owned by the MCP service.

    HYBRID:
        same query -> RAG + MCP concurrently -> existing Hybrid fusion.
    """

    if mode not in MODULE4_ALLOWED_RETRIEVAL_MODES_4C6W:
        raise ValueError(f"Unsupported retrieval mode: {mode}")

    round_start = time.perf_counter()

    if mode == "RAG_ONLY":
        rag = await asyncio.to_thread(
            retrieve_rag_candidates,
            query,
        )

        rag_s = float(
            rag.get("timing", {}).get("retrieval_time_s", 0.0) or 0.0
        )

        return {
            "mode": mode,
            "evidence": list(rag.get("evidence", []) or []),
            "rag_candidates": int(rag.get("candidate_count", 0) or 0),
            "mcp_candidates": 0,
            "rag_calls": 1,
            "mcp_calls": 0,
            "rag_retrieval_s": rag_s,
            "mcp_retrieval_s": 0.0,
            "selected_retrieval_s": float(time.perf_counter() - round_start),
            "rag": rag,
            "mcp": None,
            "hybrid": None,
        }

    if mode == "MCP_ONLY":
        mcp_result = await retrieve_mcp_candidates(query)

        mcp_s = float(
            mcp_result.get("timing", {}).get("retrieval_time_s", 0.0) or 0.0
        )

        return {
            "mode": mode,
            "evidence": list(mcp_result.get("evidence", []) or []),
            "rag_candidates": 0,
            "mcp_candidates": int(mcp_result.get("candidate_count", 0) or 0),
            "rag_calls": 0,
            "mcp_calls": 1,
            "rag_retrieval_s": 0.0,
            "mcp_retrieval_s": mcp_s,
            "selected_retrieval_s": float(time.perf_counter() - round_start),
            "rag": None,
            "mcp": mcp_result,
            "hybrid": None,
        }

    # HYBRID — preserve original Module 4 concurrency model.
    rag_task = asyncio.to_thread(
        retrieve_rag_candidates,
        query,
    )
    mcp_task = retrieve_mcp_candidates(query)

    rag, mcp_result = await asyncio.gather(
        rag_task,
        mcp_task,
    )

    parallel_wall_s = float(time.perf_counter() - round_start)

    hybrid = fuse_hybrid_candidates(
        rag_candidates=rag.get("evidence", []),
        mcp_candidates=mcp_result.get("evidence", []),
    )

    fusion_s = float(
        hybrid.get("timing", {}).get("fusion_only_s", 0.0) or 0.0
    )

    rag_s = float(
        rag.get("timing", {}).get("retrieval_time_s", 0.0) or 0.0
    )
    mcp_s = float(
        mcp_result.get("timing", {}).get("retrieval_time_s", 0.0) or 0.0
    )

    return {
        "mode": mode,
        "evidence": list(hybrid.get("evidence", []) or []),
        "rag_candidates": int(rag.get("candidate_count", 0) or 0),
        "mcp_candidates": int(mcp_result.get("candidate_count", 0) or 0),
        "rag_calls": 1,
        "mcp_calls": 1,
        "rag_retrieval_s": rag_s,
        "mcp_retrieval_s": mcp_s,
        "selected_retrieval_s": parallel_wall_s + fusion_s,
        "parallel_retrieval_wall_s": parallel_wall_s,
        "fusion_only_s": fusion_s,
        "rag": rag,
        "mcp": mcp_result,
        "hybrid": hybrid,
    }


def _4c6w_evidence_text(item):
    return str(item.get("text") or item.get("evidence") or "").strip()


def _4c6w_evidence_fingerprint(item):
    text = _4c6w_evidence_text(item)
    return "|".join([
        str(item.get("source_family", "") or "").strip().lower(),
        str(item.get("title", "") or "").strip().lower(),
        str(item.get("document_id", item.get("identifier", "")) or "").strip().lower(),
        re.sub(r"\s+", " ", text.lower())[:1200],
    ])


def _4c6w_normalize_systems(item):
    systems = (
        item.get("retrieval_systems")
        or item.get("retrieval_system")
        or item.get("retriever")
        or []
    )
    if isinstance(systems, str):
        systems = [systems]

    normalized = []
    for value in systems:
        value = str(value or "").strip().upper()
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _4c6w_merge_round_evidence(
    evidence_ledger,
    evidence_by_fingerprint,
    raw_evidence,
    round_number,
    query,
):
    """Merge one round into a stable cross-round evidence ledger."""

    round_items = []

    for fallback_rank, raw_item in enumerate(raw_evidence, start=1):
        item = dict(raw_item)

        source_rank = int(
            item.get("presentation_rank")
            or item.get("rank")
            or fallback_rank
        )

        item["text"] = _4c6w_evidence_text(item)[
            :MODULE4_EVIDENCE_EXCERPT_CHARS_4C6W
        ]

        systems = _4c6w_normalize_systems(item)
        fingerprint = _4c6w_evidence_fingerprint(item)
        rrf_increment = 1.0 / (60.0 + float(source_rank))

        if fingerprint in evidence_by_fingerprint:
            ledger_item = evidence_by_fingerprint[fingerprint]

            for system in systems:
                if system not in ledger_item["retrieval_systems"]:
                    ledger_item["retrieval_systems"].append(system)

            ledger_item["adaptive_rrf_score"] += rrf_increment
            ledger_item["occurrence_count"] += 1
            ledger_item["last_seen_round"] = int(round_number)
            ledger_item["round_numbers"].append(int(round_number))
            ledger_item["retrieval_queries"].append(str(query))

        else:
            ledger_item = dict(item)
            ledger_item["source_presentation_rank"] = source_rank
            ledger_item["presentation_rank"] = len(evidence_ledger) + 1
            ledger_item["evidence_id"] = f"E{ledger_item['presentation_rank']}"
            ledger_item["retrieval_systems"] = systems
            ledger_item["adaptive_rrf_score"] = rrf_increment
            ledger_item["occurrence_count"] = 1
            ledger_item["first_seen_round"] = int(round_number)
            ledger_item["last_seen_round"] = int(round_number)
            ledger_item["round_numbers"] = [int(round_number)]
            ledger_item["retrieval_queries"] = [str(query)]

            evidence_ledger.append(ledger_item)
            evidence_by_fingerprint[fingerprint] = ledger_item

        round_items.append(ledger_item)

    return round_items


def _4c6w_rank_evidence(evidence_ledger):
    return sorted(
        evidence_ledger,
        key=lambda item: (
            -float(item.get("adaptive_rrf_score", 0.0) or 0.0),
            -int(item.get("occurrence_count", 1) or 1),
            int(item.get("source_presentation_rank", 999999) or 999999),
            int(item.get("presentation_rank", 999999) or 999999),
        ),
    )


def _4c6w_select_evidence(evidence_ledger, mode):
    """
    RAG_ONLY / MCP_ONLY:
        top cumulative evidence by adaptive RRF.

    HYBRID:
        preserve the Module 4 representation principle by reserving up to two
        RAG-backed and two MCP-backed items, then fill by adaptive RRF.
    """

    ranked = _4c6w_rank_evidence(evidence_ledger)
    if not ranked:
        return []

    if mode in {"RAG_ONLY", "MCP_ONLY"}:
        return ranked[:MODULE4_MAX_FINAL_EVIDENCE_4C6W]

    available_rag = sum(
        "RAG" in item.get("retrieval_systems", [])
        for item in ranked
    )
    available_mcp = sum(
        "MCP" in item.get("retrieval_systems", [])
        for item in ranked
    )

    required_rag = min(2, available_rag)
    required_mcp = min(2, available_mcp)

    selected = []

    def selected_has(system):
        return sum(
            system in item.get("retrieval_systems", [])
            for item in selected
        )

    for item in ranked:
        if selected_has("RAG") >= required_rag:
            break
        if "RAG" in item.get("retrieval_systems", []) and item not in selected:
            selected.append(item)

    for item in ranked:
        if selected_has("MCP") >= required_mcp:
            break
        if "MCP" in item.get("retrieval_systems", []) and item not in selected:
            selected.append(item)

    for item in ranked:
        if len(selected) >= MODULE4_MAX_FINAL_EVIDENCE_4C6W:
            break
        if item not in selected:
            selected.append(item)

    return selected[:MODULE4_MAX_FINAL_EVIDENCE_4C6W]


def _4c6w_build_bounded_context(selected_evidence):
    bounded_items = []
    blocks = []
    chars_used = 0

    for item in selected_evidence:
        rank = int(item["presentation_rank"])
        systems = item.get("retrieval_systems", []) or []
        retrieval_label = "/".join(str(value) for value in systems if value)
        if not retrieval_label:
            retrieval_label = "GROUNDED"

        header = (
            f"[E{rank}] "
            f"Retrieval={retrieval_label} | "
            f"Source={item.get('source_family', '')} | "
            f"Title={item.get('title', '')} | "
            f"Document={item.get('document_id', item.get('identifier', ''))}"
        )

        separator_chars = 2 if blocks else 0
        remaining = (
            MODULE4_CONTEXT_MAX_CHARS_4C6W
            - chars_used
            - separator_chars
            - len(header)
            - 1
        )

        if remaining <= 0:
            break

        evidence_text = _4c6w_evidence_text(item)[:remaining]
        if not evidence_text:
            continue

        bounded_item = dict(item)
        bounded_item["text"] = evidence_text
        block = header + "\n" + evidence_text

        if blocks:
            chars_used += 2

        blocks.append(block)
        chars_used += len(block)
        bounded_items.append(bounded_item)

    return {
        "evidence": bounded_items,
        "context": "\n\n".join(blocks),
        "context_chars": chars_used,
    }


def _4c6w_assess_sufficiency(
    question,
    requirements,
    mode,
    round_number,
    context,
):
    requirement_map = {
        f"R{index}": requirement
        for index, requirement in enumerate(requirements, start=1)
    }

    requirements_text = "\n".join(
        f"{requirement_id}: {requirement}"
        for requirement_id, requirement in requirement_map.items()
    )

    start = time.perf_counter()

    response = _4c6w_call_llm(
        messages=[
            {"role": "system", "content": MODULE4_SUFFICIENCY_SYSTEM_PROMPT_4C6W},
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
                    f"FIXED ANSWER REQUIREMENTS:\n{requirements_text}\n\n"
                    f"RETRIEVAL MODE:\n{mode}\n\n"
                    f"ROUND:\n{round_number}\n\n"
                    f"EVIDENCE:\n{context}"
                ),
            },
        ],
        max_tokens=MODULE4_SUFFICIENCY_MAX_TOKENS_4C6W,
    )

    elapsed = time.perf_counter() - start
    raw_text = _4c6w_extract_text(response)
    result = _4c6w_parse_json_object(raw_text)

    valid_ids = list(requirement_map.keys())

    supported = []
    for value in result.get("supported_requirement_ids", []) or []:
        value = str(value).strip().upper()
        if value in valid_ids and value not in supported:
            supported.append(value)

    missing = []
    for value in result.get("missing_requirement_ids", []) or []:
        value = str(value).strip().upper()
        if value in valid_ids and value not in missing:
            missing.append(value)

    # Structural reconciliation only; no semantic re-judging in Python.
    for requirement_id in valid_ids:
        if requirement_id not in supported and requirement_id not in missing:
            missing.append(requirement_id)

    supported = [
        requirement_id
        for requirement_id in supported
        if requirement_id not in missing
    ]

    sufficient = len(missing) == 0

    next_query = re.sub(
        r"\s+",
        " ",
        str(result.get("next_query", "") or ""),
    ).strip()

    missing_requirements = [
        requirement_map[requirement_id]
        for requirement_id in missing
    ]

    # Same safe Module 4 fallback: reuse only fixed requirements; invent no scope.
    if not sufficient and not next_query:
        focus = "; ".join(missing_requirements)
        next_query = (
            f"{question} Focus specifically on: {focus}"
        ).strip()

    if sufficient:
        next_query = ""

    return {
        "sufficient": sufficient,
        "supported_requirement_ids": supported,
        "missing_requirement_ids": missing,
        "missing_requirements": missing_requirements,
        "missing_evidence": "; ".join(missing_requirements),
        "next_query": next_query,
        "reason": str(result.get("reason", "") or "").strip(),
        "latency_s": elapsed,
        "usage": _4c6w_usage(response),
        "raw_text": raw_text,
    }


def run_4c_telecom_grounded(question, context):
    user_prompt = (
        "QUESTION\n"
        "--------\n"
        f"{str(question).strip()}\n\n"
        "TECHNICAL EVIDENCE\n"
        "------------------\n"
        f"{str(context).strip()}"
    )

    start = time.perf_counter()
    response = _4c6w_call_llm(
        messages=[
            {
                "role": "system",
                "content": MODULE4_FINAL_ANSWER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    elapsed = time.perf_counter() - start

    usage = _4c6w_usage(response)

    return {
        "answer": _4c6w_extract_text(response),
        "elapsed_s": float(elapsed),
        "prompt_tokens": usage["input_tokens"],
        "completion_tokens": usage["output_tokens"],
        "total_tokens": usage["input_tokens"] + usage["output_tokens"],
    }


async def run_module4_aligned_routed_grounded_4c6w(question):
    start_wall = time.perf_counter()

    total_input_tokens = 0
    total_output_tokens = 0
    total_llm_calls = 0
    total_llm_latency = 0.0

    rag_candidate_total = 0
    mcp_candidate_total = 0
    rag_call_total = 0
    mcp_call_total = 0

    rag_retrieval_total = 0.0
    mcp_retrieval_total = 0.0
    selected_retrieval_total = 0.0
    evidence_scan_total_ms = 0.0

    evidence_security_hits = []
    evidence_ledger = []
    evidence_by_fingerprint = {}

    search_queries = []
    round_traces = []

    # LLM selects architecture, query and fixed requirements.
    planner = await asyncio.to_thread(
        _4c6w_plan_retrieval,
        question,
    )

    total_llm_calls += 1
    total_llm_latency += float(planner["latency_s"])
    total_input_tokens += int(planner["usage"]["input_tokens"])
    total_output_tokens += int(planner["usage"]["output_tokens"])

    selected_mode = planner["mode"]
    current_query = planner["query"]
    requirements = list(planner["answer_requirements"])

    final_selection = None
    final_assessment = None
    stop_reason = None

    for round_number in range(1, MODULE4_MAX_MODEL_SEARCHES_4C6W + 1):
        search_queries.append(current_query)

        round_result = await _4c6w_execute_selected_retrieval_round(
            selected_mode,
            current_query,
        )

        rag_candidates = int(round_result["rag_candidates"])
        mcp_candidates = int(round_result["mcp_candidates"])

        rag_candidate_total += rag_candidates
        mcp_candidate_total += mcp_candidates
        rag_call_total += int(round_result["rag_calls"])
        mcp_call_total += int(round_result["mcp_calls"])

        rag_retrieval_s = float(round_result["rag_retrieval_s"])
        mcp_retrieval_s = float(round_result["mcp_retrieval_s"])
        selected_retrieval_s = float(round_result["selected_retrieval_s"])

        rag_retrieval_total += rag_retrieval_s
        mcp_retrieval_total += mcp_retrieval_s
        selected_retrieval_total += selected_retrieval_s

        raw_evidence = list(round_result.get("evidence", []) or [])

        # Deterministic evidence security boundary.
        scan_start = time.perf_counter()
        round_security_hits = []

        for raw_item in raw_evidence:
            evidence_text = _4c6w_evidence_text(raw_item)
            matches = get_injection_matches_4c3(evidence_text)

            if matches:
                hit = {
                    "round": round_number,
                    "query": current_query,
                    "title": raw_item.get("title", ""),
                    "matched_patterns": list(matches),
                }
                round_security_hits.append(hit)
                evidence_security_hits.append(hit)

        scan_ms = (time.perf_counter() - scan_start) * 1000.0
        evidence_scan_total_ms += scan_ms

        if round_security_hits:
            raise RuntimeError(
                "Retrieved evidence failed deterministic prompt-injection "
                "screening. Grounded generation has been blocked."
            )

        round_evidence = _4c6w_merge_round_evidence(
            evidence_ledger=evidence_ledger,
            evidence_by_fingerprint=evidence_by_fingerprint,
            raw_evidence=raw_evidence,
            round_number=round_number,
            query=current_query,
        )

        selected = _4c6w_select_evidence(
            evidence_ledger,
            selected_mode,
        )

        final_selection = _4c6w_build_bounded_context(selected)

        assessment = await asyncio.to_thread(
            _4c6w_assess_sufficiency,
            question,
            requirements,
            selected_mode,
            round_number,
            final_selection["context"],
        )

        total_llm_calls += 1
        total_llm_latency += float(assessment["latency_s"])
        total_input_tokens += int(assessment["usage"]["input_tokens"])
        total_output_tokens += int(assessment["usage"]["output_tokens"])
        final_assessment = assessment

        round_traces.append({
            "search_number": round_number,
            "mode": selected_mode,
            "query": current_query,
            "rag_candidates": rag_candidates,
            "mcp_candidates": mcp_candidates,
            "rag_calls": int(round_result["rag_calls"]),
            "mcp_calls": int(round_result["mcp_calls"]),
            "round_evidence_ids": [item["evidence_id"] for item in round_evidence],
            "presented_evidence_ids": [
                item["evidence_id"] for item in final_selection["evidence"]
            ],
            "cumulative_unique_evidence": len(evidence_ledger),
            "rag_retrieval_s": rag_retrieval_s,
            "mcp_retrieval_s": mcp_retrieval_s,
            "selected_retrieval_s": selected_retrieval_s,
            "evidence_scan_ms": scan_ms,
            "sufficiency": assessment,
        })

        if assessment["sufficient"]:
            stop_reason = "SUFFICIENT_EVIDENCE"
            break

        if round_number >= MODULE4_MAX_MODEL_SEARCHES_4C6W:
            stop_reason = "MAX_RETRIEVAL_ROUNDS"
            break

        current_query = assessment["next_query"]
        if not current_query:
            raise RuntimeError(
                "Evidence was insufficient but no refined retrieval query was available."
            )

    if final_selection is None:
        raise RuntimeError(
            "Grounded adaptive retrieval completed without an evidence selection."
        )

    generation = await asyncio.to_thread(
        run_4c_telecom_grounded,
        question,
        final_selection["context"],
    )

    final_answer = str(generation.get("answer", "") or "").strip()
    final_generation_s = float(generation.get("elapsed_s", 0.0) or 0.0)

    total_llm_calls += 1
    total_llm_latency += final_generation_s

    if not final_answer:
        raise RuntimeError("Grounded generator returned an empty answer.")

    return {
        "answer": final_answer,
        "elapsed_s": final_generation_s,
        "tool_loop_wall_s": float(time.perf_counter() - start_wall),
        "llm_calls": total_llm_calls,
        "tool_requests": 0,
        "search_count": len(search_queries),
        "search_queries": list(search_queries),
        "round_traces": round_traces,
        "selected_mode": selected_mode,
        "planner": planner,
        "answer_requirements": requirements,
        "final_sufficiency_assessment": final_assessment,
        "stop_reason": stop_reason,
        "all_evidence": evidence_ledger,
        "final_evidence": list(final_selection["evidence"]),
        "final_context": final_selection["context"],
        "final_context_chars": final_selection["context_chars"],
        "rag_candidate_total": rag_candidate_total,
        "mcp_candidate_total": mcp_candidate_total,
        "rag_call_total": rag_call_total,
        "mcp_call_total": mcp_call_total,
        "rag_retrieval_s": rag_retrieval_total,
        "mcp_retrieval_s": mcp_retrieval_total,
        # Backwards-compatible field consumed by the surrounding runtime.
        # It now means total wall time of the LLM-selected retrieval architecture.
        "hybrid_retrieval_s": selected_retrieval_total,
        "selected_retrieval_s": selected_retrieval_total,
        "evidence_scan_ms": evidence_scan_total_ms,
        "evidence_security_hits": evidence_security_hits,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_llm_latency_s": total_llm_latency,
        "completion_mode": (
            "normal"
            if stop_reason == "SUFFICIENT_EVIDENCE"
            else "final_after_search_cap"
        ),
    }


async def run_connected_corpus_grounded(question: str) -> dict[str, Any]:
    """
    Deployment-friendly alias for the canonical Notebook 21 connected-corpus path.
    """
    question = re.sub(r"\s+", " ", str(question or "")).strip()
    if not question:
        raise ValueError("Question must not be empty.")

    return await run_module4_aligned_routed_grounded_4c6w(question)


def get_adaptive_runtime_status() -> dict[str, Any]:
    return {
        "runtime": "Notebook 21 Cell 4D / 6Y v1.3 connected-corpus adaptive path",
        "generator_model": LLM_MODEL,
        "allowed_modes": sorted(MODULE4_ALLOWED_RETRIEVAL_MODES_4C6W),
        "max_retrieval_rounds": MODULE4_MAX_MODEL_SEARCHES_4C6W,
        "max_final_evidence": MODULE4_MAX_FINAL_EVIDENCE_4C6W,
        "max_context_chars": MODULE4_CONTEXT_MAX_CHARS_4C6W,
        "max_answer_requirements": MODULE4_MAX_ANSWER_REQUIREMENTS_4C6W,
        "planner_max_tokens": MODULE4_PLANNER_MAX_TOKENS_4C6W,
        "sufficiency_max_tokens": MODULE4_SUFFICIENCY_MAX_TOKENS_4C6W,
        "evidence_security_scan": True,
        "grounded_citations": "[E#]",
    }
