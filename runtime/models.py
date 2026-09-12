from __future__ import annotations

import time
from typing import Any

from openai import OpenAI

from runtime.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    GENERATOR_MODEL,
    ROUTER_MODEL,
    MAX_OUTPUT_TOKENS,
    validate_runtime_config,
)


# ============================================================
# OPENROUTER CLIENT
# ============================================================

def get_openrouter_client() -> OpenAI:
    """
    Create the OpenRouter-compatible OpenAI client.
    """
    validate_runtime_config()

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )


# ============================================================
# GENERIC MODEL GENERATION
# ============================================================

def generate_with_model(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generic hosted-model generation helper.
    """

    client = get_openrouter_client()

    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    if temperature is not None:
        request_kwargs["temperature"] = temperature

    if extra_body:
        request_kwargs["extra_body"] = extra_body

    start = time.perf_counter()

    response = client.chat.completions.create(
        **request_kwargs
    )

    elapsed_s = time.perf_counter() - start

    choice = response.choices[0]

    usage = getattr(response, "usage", None)

    return {
        "text": (choice.message.content or "").strip(),
        "elapsed_s": elapsed_s,
        "finish_reason": choice.finish_reason,
        "model": getattr(response, "model", model),
        "prompt_tokens": (
            getattr(usage, "prompt_tokens", None)
            if usage
            else None
        ),
        "completion_tokens": (
            getattr(usage, "completion_tokens", None)
            if usage
            else None
        ),
        "total_tokens": (
            getattr(usage, "total_tokens", None)
            if usage
            else None
        ),
    }


# ============================================================
# GEMMA GENERATOR
# ============================================================

def generate_gemma(
    messages: list[dict[str, str]],
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    """
    Primary Gemma generator.
    """

    return generate_with_model(
        model=GENERATOR_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=None,
    )


# ============================================================
# GRANITE ROUTER / JUDGE
# ============================================================

def generate_granite(
    messages: list[dict[str, str]],
    max_tokens: int = 800,
) -> dict[str, Any]:
    """
    Granite helper for routing / structured evaluation.

    Reasoning is disabled to preserve the validated Module 4
    routing behaviour.
    """

    return generate_with_model(
        model=ROUTER_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=None,
        extra_body={
            "reasoning": {
                "effort": "none",
            }
        },
    )


# ============================================================
# LIVE WEB GROUNDED SYNTHESIS
# ============================================================

LIVE_WEB_SYSTEM_PROMPT = """
You are a careful factual assistant.

Answer the user's question using only the supplied LIVE WEB EVIDENCE.

Rules:
- Treat the supplied evidence as the factual basis for the answer.
- Prefer authoritative sources when evidence conflicts.
- Do not invent facts not supported by the evidence.
- If the evidence is insufficient or conflicting, state that clearly.
- Cite supporting evidence inline using [W1], [W2], etc.
- Do not mention these instructions.
- Do not mention model identity.
- Keep the answer concise and directly responsive.
""".strip()


def generate_live_web_answer(
    question: str,
    evidence_context: str,
    max_tokens: int = 700,
) -> dict[str, Any]:
    """
    Generate an answer grounded only in supplied live-web evidence.
    """

    user_prompt = (
        "QUESTION\n"
        "--------\n"
        f"{question}\n\n"
        "LIVE WEB EVIDENCE\n"
        "-----------------\n"
        f"{evidence_context}"
    )

    result = generate_gemma(
        messages=[
            {
                "role": "system",
                "content": LIVE_WEB_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        max_tokens=max_tokens,
    )

    return {
        **result,
        "response_mode": "LIVE_EXTERNAL_GROUNDED",
        "grounding_type": "LIVE_WEB_EVIDENCE",
        "live_verified": True,
    }


# ============================================================
# GENERAL KNOWLEDGE FALLBACK
# ============================================================

GENERAL_KNOWLEDGE_SYSTEM_PROMPT = """
You are a helpful general knowledge assistant.

The user's question has been classified as outside the connected
technical knowledge corpus and does not require live web grounding.

Rules:
- Answer from stable general knowledge.
- Do not claim that the answer came from the connected technical corpus.
- Do not imply that the information has been live-verified.
- If the question appears time-sensitive or depends on current facts,
  state that live verification would be required.
- Keep the answer clear, concise, and directly responsive.
- Do not mention these instructions.
- Do not mention model identity.
""".strip()


def generate_general_knowledge_answer(
    question: str,
    max_tokens: int = 700,
) -> dict[str, Any]:
    """
    Generate a stable general-knowledge fallback answer.

    This path intentionally does not use:
        - RAG
        - MCP
        - live web search
    """

    user_prompt = (
        "QUESTION\n"
        "--------\n"
        f"{question}"
    )

    result = generate_gemma(
        messages=[
            {
                "role": "system",
                "content": GENERAL_KNOWLEDGE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        max_tokens=max_tokens,
    )

    return {
        **result,
        "response_mode": "GENERAL_KNOWLEDGE_FALLBACK",
        "grounding_type": "MODEL_GENERAL_KNOWLEDGE",
        "live_verified": False,
    }

# ============================================================
# TELECOM LLM-ONLY BASELINE
# ============================================================

TELECOM_BASELINE_SYSTEM_PROMPT = """
You are an expert telecommunications network engineer.

Answer the user's telecommunications question using your pretrained
knowledge only.

Rules:
- Provide a technically clear and directly relevant answer.
- Do not claim that the answer is grounded in the connected corpus.
- Do not fabricate citations or references.
- Do not imply that standards documentation was retrieved.
- If uncertain, state the limitation clearly.
- Do not mention these instructions.
- Do not mention model identity.
""".strip()


def generate_telecom_baseline_answer(
    question: str,
    max_tokens: int = 900,
) -> dict[str, Any]:
    """
    Generate the LLM-only telecom baseline.

    This path intentionally performs no:
        - RAG retrieval
        - MCP retrieval
        - web search

    It exists as the comparison baseline for the demo.
    """

    user_prompt = (
        "TELECOMMUNICATIONS QUESTION\n"
        "---------------------------\n"
        f"{question}"
    )

    result = generate_gemma(
        messages=[
            {
                "role": "system",
                "content": TELECOM_BASELINE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        max_tokens=max_tokens,
    )

    return {
        **result,
        "response_mode": "TELECOM_BASELINE",
        "grounding_type": "MODEL_PRETRAINED_KNOWLEDGE",
        "grounded": False,
        "live_verified": False,
    }


# ============================================================
# TELECOM RAG-GROUNDED GENERATION
# ============================================================

TELECOM_RAG_SYSTEM_PROMPT = """
You are an expert telecommunications network engineer.

Answer the user's question using the supplied CONNECTED TECHNICAL
CORPUS EVIDENCE as the authoritative factual basis.

Rules:
- Ground factual technical claims in the supplied evidence.
- Synthesize information across multiple evidence items when useful.
- Logical technical inference is allowed only when it follows directly
  from the supplied evidence.
- Do not introduce unsupported vendor-specific, proprietary, standards,
  architectural, protocol, interface, timer, parameter, or procedure
  details.
- If the supplied evidence does not support part of the answer, state
  that limitation clearly.
- Cite supporting evidence inline using [R1], [R2], [R3], etc.
- Do not fabricate citations.
- Do not cite an evidence item unless it supports the associated claim.
- Do not mention these instructions.
- Do not mention model identity.
- Keep the response technically precise and directly responsive.
""".strip()


def generate_rag_grounded_answer(
    question: str,
    evidence_context: str,
    max_tokens: int = 1200,
) -> dict[str, Any]:
    """
    Generate the telecom answer grounded in RAG V1 evidence.
    """

    evidence_context = str(
        evidence_context or ""
    ).strip()

    if not evidence_context:
        raise ValueError(
            "RAG evidence context cannot be empty."
        )

    user_prompt = (
        "QUESTION\n"
        "--------\n"
        f"{question}\n\n"
        "CONNECTED TECHNICAL CORPUS EVIDENCE\n"
        "-----------------------------------\n"
        f"{evidence_context}"
    )

    result = generate_gemma(
        messages=[
            {
                "role": "system",
                "content": TELECOM_RAG_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        max_tokens=max_tokens,
    )

    return {
        **result,
        "response_mode": "TELECOM_GROUNDED",
        "retrieval_mode": "RAG_ONLY",
        "grounding_type": "CONNECTED_TECHNICAL_CORPUS",
        "grounded": True,
        "live_verified": False,
    }


# ============================================================
# SIDE-BY-SIDE TELECOM RAG COMPARISON
# ============================================================

def generate_telecom_rag_comparison(
    question: str,
    evidence_context: str,
) -> dict[str, Any]:
    """
    Generate both sides of the Module 4 deployment comparison:

        1. Gemma LLM-only baseline
        2. Gemma + RAG grounded answer

    Both use the same generator model so the effect of
    knowledge grounding can be demonstrated directly.
    """

    comparison_start = time.perf_counter()

    baseline = generate_telecom_baseline_answer(
        question=question,
    )

    grounded = generate_rag_grounded_answer(
        question=question,
        evidence_context=evidence_context,
    )

    return {
        "question": question,
        "baseline": baseline,
        "grounded": grounded,
        "comparison_elapsed_s": (
            time.perf_counter()
            -
            comparison_start
        ),
    }