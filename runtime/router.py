from __future__ import annotations

import json
import time
from typing import Any

from runtime.models import generate_granite


TELECOM_GROUNDED = "TELECOM_GROUNDED"
LIVE_EXTERNAL_GROUNDED = "LIVE_EXTERNAL_GROUNDED"
GENERAL_KNOWLEDGE_FALLBACK = "GENERAL_KNOWLEDGE_FALLBACK"

VALID_RESPONSE_MODES = {
    TELECOM_GROUNDED,
    LIVE_EXTERNAL_GROUNDED,
    GENERAL_KNOWLEDGE_FALLBACK,
}


ROUTER_SYSTEM_PROMPT = """
You are a knowledge-scope router for a telecom AI system.

Your task is to decide where the authoritative knowledge required to answer
the user's question should come from.

Choose exactly one response mode:

1. TELECOM_GROUNDED
   Use when the connected controlled technical corpus is the appropriate
   authoritative source. This includes telecom standards, RAN, 5G Core,
   OSS, cloud-native telecom, Kubernetes for telecom, Open RAN,
   telecom architecture, telecom protocols, technical implementation,
   troubleshooting, engineering mechanisms and related controlled
   technical knowledge.

2. LIVE_EXTERNAL_GROUNDED
   Use when the answer depends on changing, current, recent or externally
   evolving public information. Examples include current people, current
   office holders, recent events, latest versions, live public facts,
   current dates, schedules or information whose truth may have changed.

3. GENERAL_KNOWLEDGE_FALLBACK
   Use when the question is stable general knowledge and does not require
   the connected technical corpus or current external information.

Important:
- Route based on where authoritative knowledge should come from.
- Do not classify based only on keywords.
- Do not answer the user's question.
- Return JSON only.
- Do not include markdown.

Required JSON format:

{
  "response_mode": "TELECOM_GROUNDED | LIVE_EXTERNAL_GROUNDED | GENERAL_KNOWLEDGE_FALLBACK",
  "reason": "brief explanation"
}
""".strip()


def _parse_router_json(
    text: str,
) -> dict[str, Any]:
    """
    Parse and validate Granite router JSON.
    """

    raw = str(text or "").strip()

    # Remove accidental fenced output if provider/model adds it.
    if raw.startswith("```"):
        raw = raw.strip("`").strip()

        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Router returned invalid JSON: {raw}"
        ) from exc

    response_mode = str(
        payload.get(
            "response_mode",
            "",
        )
    ).strip()

    reason = str(
        payload.get(
            "reason",
            "",
        )
    ).strip()

    if response_mode not in VALID_RESPONSE_MODES:
        raise RuntimeError(
            f"Router returned invalid response mode: {response_mode}"
        )

    if not reason:
        raise RuntimeError(
            "Router returned an empty reason."
        )

    return {
        "response_mode": response_mode,
        "reason": reason,
    }


def route_knowledge_scope(
    question: str,
) -> dict[str, Any]:
    """
    Route sanitized user intent to one of the three
    knowledge scopes.
    """

    start = time.perf_counter()

    result = generate_granite(
        messages=[
            {
                "role": "system",
                "content": ROUTER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        max_tokens=160,
    )

    parsed = _parse_router_json(
        result["text"]
    )

    elapsed_s = (
        time.perf_counter()
        -
        start
    )

    return {
        "response_mode":
            parsed["response_mode"],
        "reason":
            parsed["reason"],
        "elapsed_s":
            elapsed_s,
        "model_elapsed_s":
            result["elapsed_s"],
        "model":
            result["model"],
        "raw_output":
            result["text"],
    }