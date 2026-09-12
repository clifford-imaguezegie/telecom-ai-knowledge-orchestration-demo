from __future__ import annotations

import asyncio
import time
from typing import Any

from runtime.orchestrator import (
    TELECOM_GROUNDED_MODE,
    LIVE_EXTERNAL_GROUNDED_MODE,
    GENERAL_KNOWLEDGE_FALLBACK_MODE,
    preprocess_user_prompt,
    generate_gemma_only_baseline,
    classify_knowledge_scope,
    generate_general_fallback,
    run_local_datetime_tool,
)
from runtime.adaptive import run_connected_corpus_grounded
from runtime.live_external import run_live_external_grounded_4c6w
from runtime.augmented import (
    run_augmented_reasoning,
    run_augmented_non_telecom,
)


# ============================================================
# THREE-PATH EXPERIMENT CONTROLLER
#
# Experimental conditions:
#
# 1. Gemma LLM-Only
#    Objective:
#       Maximum native model reasoning
#
# 2. Strict Knowledge Grounding
#    Objective:
#       Maximum evidence discipline
#
#    Telecom:
#       Adaptive RAG / MCP / Hybrid
#
#    Current/live non-telecom:
#       Web or local runtime depending on question
#
#    Stable general non-telecom:
#       Gemma pretrained knowledge fallback
#
# 3. Augmented Reasoning
#    Objective:
#       Maximum useful synthesis across available evidence
#       and pretrained model knowledge
#
#    Telecom:
#       Forced RAG + MCP + Web + Model Knowledge
#
#    Non-telecom:
#       Forced Web + Model Knowledge
# ============================================================


def _path_metadata(
    *,
    name: str,
    objective: str,
    mode: str,
    sources: list[str],
    policy: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "objective": objective,
        "mode": mode,
        "sources": sources,
        "policy": policy,
    }


def _llm_only_path_metadata() -> dict[str, Any]:
    return _path_metadata(
        name="Gemma LLM-Only",
        objective="Maximum native model reasoning",
        mode="LLM_ONLY",
        sources=[
            "Gemma pretrained knowledge",
        ],
        policy="No retrieval or external grounding",
    )


def _strict_path_metadata(
    response_mode: str,
) -> dict[str, Any]:
    if response_mode == TELECOM_GROUNDED_MODE:
        return _path_metadata(
            name="Strict Knowledge Grounding",
            objective="Maximum evidence discipline",
            mode="TELECOM_STRICT_GROUNDED",
            sources=[
                "Adaptive RAG",
                "MCP",
                "Hybrid when selected",
            ],
            policy=(
                "Strict evidence-grounded Telecom answer"
            ),
        )

    if response_mode == LIVE_EXTERNAL_GROUNDED_MODE:
        return _path_metadata(
            name="Strict Knowledge Grounding",
            objective="Maximum evidence discipline",
            mode="LIVE_EXTERNAL_GROUNDED",
            sources=[
                "Web Search or Local Runtime",
            ],
            policy=(
                "Use externally grounded current information"
            ),
        )

    return _path_metadata(
        name="Strict Knowledge Grounding",
        objective="Maximum evidence discipline",
        mode="GENERAL_KNOWLEDGE_FALLBACK",
        sources=[
            "Gemma pretrained knowledge",
        ],
        policy=(
            "Stable general-knowledge fallback; no Telecom RAG/MCP"
        ),
    )


def _augmented_path_metadata(
    response_mode: str,
) -> dict[str, Any]:
    if response_mode == TELECOM_GROUNDED_MODE:
        return _path_metadata(
            name="Augmented Reasoning",
            objective=(
                "Maximum useful synthesis across evidence "
                "and model knowledge"
            ),
            mode="TELECOM_AUGMENTED",
            sources=[
                "RAG",
                "MCP",
                "Web Search",
                "Gemma pretrained knowledge",
            ],
            policy=(
                "Forced multi-source evidence gathering + "
                "one Gemma synthesis call"
            ),
        )

    return _path_metadata(
        name="Augmented Reasoning",
        objective=(
            "Maximum useful synthesis across evidence "
            "and model knowledge"
        ),
        mode="NON_TELECOM_AUGMENTED",
        sources=[
            "Web Search",
            "Gemma pretrained knowledge",
        ],
        policy=(
            "Forced web augmentation + one Gemma synthesis call"
        ),
    )


async def _run_strict_path(
    response_mode: str,
    sanitized: str,
) -> dict[str, Any]:
    if response_mode == TELECOM_GROUNDED_MODE:
        return await (
            run_connected_corpus_grounded(
                sanitized
            )
        )

    if response_mode == LIVE_EXTERNAL_GROUNDED_MODE:
        local_result = run_local_datetime_tool(
            sanitized
        )

        if local_result is not None:
            return {
                "answer": local_result[
                    "answer"
                ],
                "local_runtime":
                    local_result,
                "final_evidence": [],
                "timing": {
                    "total_wall_s":
                        float(
                            local_result.get(
                                "elapsed_s",
                                0.0,
                            )
                            or 0.0
                        )
                },
                "orchestration_path": [
                    {
                        "stage":
                            "Local Date/Time Runtime",
                        "status":
                            "completed",
                        "elapsed_s":
                            float(
                                local_result.get(
                                    "elapsed_s",
                                    0.0,
                                )
                                or 0.0
                            ),
                    }
                ],
            }

        return await (
            run_live_external_grounded_4c6w(
                sanitized
            )
        )

    started = time.perf_counter()

    result = await asyncio.to_thread(
        generate_general_fallback,
        sanitized,
    )

    elapsed = (
        time.perf_counter()
        -
        started
    )

    if isinstance(result, dict):
        answer = result.get(
            "answer",
            "",
        )
    else:
        answer = str(
            result
            or ""
        )

    return {
        "answer": answer,
        "final_evidence": [],
        "timing": {
            "total_wall_s": float(
                elapsed
            ),
        },
        "orchestration_path": [
            {
                "stage":
                    "Gemma General-Knowledge Fallback",
                "status":
                    "completed",
                "elapsed_s":
                    float(
                        elapsed
                    ),
            }
        ],
    }


async def _run_augmented_path(
    response_mode: str,
    sanitized: str,
) -> dict[str, Any]:
    if response_mode == TELECOM_GROUNDED_MODE:
        return await run_augmented_reasoning(
            sanitized
        )

    return await run_augmented_non_telecom(
        sanitized
    )


async def run_three_path_experiment(
    question: str,
) -> dict[str, Any]:
    security = preprocess_user_prompt(
        question
    )

    sanitized = str(
        security.get(
            "sanitized_prompt",
            "",
        )
        or ""
    ).strip()

    if not sanitized:
        raise ValueError(
            "Question is empty after security sanitization."
        )

    overall_started = time.perf_counter()

    # Baseline and route start concurrently, preserving the
    # current validated behavior.
    baseline_started = time.perf_counter()

    baseline_task = asyncio.create_task(
        asyncio.to_thread(
            generate_gemma_only_baseline,
            sanitized,
        )
    )

    router_started = time.perf_counter()

    router_task = asyncio.create_task(
        asyncio.to_thread(
            classify_knowledge_scope,
            sanitized,
        )
    )

    routing = await router_task

    router_s = (
        time.perf_counter()
        -
        router_started
    )

    response_mode = routing[
        "response_mode"
    ]

    strict_started = time.perf_counter()
    strict_task = asyncio.create_task(
        _run_strict_path(
            response_mode,
            sanitized,
        )
    )

    augmented_started = time.perf_counter()
    augmented_task = asyncio.create_task(
        _run_augmented_path(
            response_mode,
            sanitized,
        )
    )

    baseline_result = await baseline_task

    baseline_s = (
        time.perf_counter()
        -
        baseline_started
    )

    strict_result, augmented_result = (
        await asyncio.gather(
            strict_task,
            augmented_task,
        )
    )

    strict_s = (
        time.perf_counter()
        -
        strict_started
    )

    augmented_s = (
        time.perf_counter()
        -
        augmented_started
    )

    total_wall_s = (
        time.perf_counter()
        -
        overall_started
    )

    return {
        "question": question,
        "sanitized_question": sanitized,
        "security": security,
        "routing": routing,
        "query_type": response_mode,
        "paths": {
            "llm_only": {
                "metadata":
                    _llm_only_path_metadata(),
                "result":
                    baseline_result,
                "timing": {
                    "total_wall_s":
                        float(
                            baseline_s
                        )
                },
                "orchestration_path": [
                    {
                        "stage":
                            "Gemma LLM-Only Inference",
                        "status":
                            "completed",
                        "elapsed_s":
                            float(
                                baseline_s
                            ),
                    }
                ],
            },
            "strict": {
                "metadata":
                    _strict_path_metadata(
                        response_mode
                    ),
                "result":
                    strict_result,
                "timing": {
                    "total_wall_s":
                        float(
                            strict_s
                        )
                },
                "orchestration_path":
                    strict_result.get(
                        "orchestration_path",
                        [],
                    ),
            },
            "augmented": {
                "metadata":
                    _augmented_path_metadata(
                        response_mode
                    ),
                "result":
                    augmented_result,
                "timing": {
                    "total_wall_s":
                        float(
                            augmented_s
                        )
                },
                "orchestration_path":
                    augmented_result.get(
                        "orchestration_path",
                        [],
                    ),
            },
        },
        "timing": {
            "router_s":
                float(
                    router_s
                ),
            "baseline_s":
                float(
                    baseline_s
                ),
            "strict_s":
                float(
                    strict_s
                ),
            "augmented_s":
                float(
                    augmented_s
                ),
            "total_wall_s":
                float(
                    total_wall_s
                ),
        },
    }


def get_three_path_policy() -> dict[str, Any]:
    return {
        "telecom": {
            "llm_only": [
                "Gemma pretrained knowledge",
            ],
            "strict": [
                "Adaptive RAG / MCP / Hybrid",
                "Strict evidence grounding",
            ],
            "augmented": [
                "Forced RAG",
                "Forced MCP",
                "Forced Web Search",
                "Gemma pretrained knowledge",
                "Knowledge synthesis",
            ],
        },
        "current_live_non_telecom": {
            "llm_only": [
                "Gemma pretrained knowledge",
            ],
            "strict": [
                "Gemma pretrained knowledge or Web Search depending on question",
            ],
            "augmented": [
                "Forced Web Search",
                "Gemma pretrained knowledge",
                "Knowledge synthesis",
            ],
        },
        "stable_general_non_telecom": {
            "llm_only": [
                "Gemma pretrained knowledge",
            ],
            "strict": [
                "Gemma pretrained knowledge or Web Search depending on question",
            ],
            "augmented": [
                "Forced Web Search",
                "Gemma pretrained knowledge",
                "Knowledge synthesis",
            ],
        },
    }
