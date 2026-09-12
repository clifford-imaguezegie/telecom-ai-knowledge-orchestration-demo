from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from runtime.evaluation import (
    JUDGE_DISPLAY_NAME,
    JUDGE_MODEL,
    JUDGE_PROVIDER,
    JUDGE_REASONING_EFFORT,
    GEMMA_ONLY_LABEL,
    GEMMA_RAG_MCP_LABEL,
    RISK_DEFINITION,
    build_valid_evidence_ids,
    call_granite_judge,
    extract_refined_claims,
    hallucination_risk_label,
    relevance_score_label,
    run_granite_single_answer_judge,
    validate_and_summarize_judgment,
)


# ============================================================
# THREE-WAY COMPARATIVE EVALUATION
#
# Systems:
# A = Gemma LLM-Only
# B = Strict Knowledge Grounding
# C = Augmented Reasoning
#
# Metrics:
# 1. Evidence-relative hallucination risk
# 2. Answer relevance
# 3. Answer completeness
#
# Hallucination-risk evidence basis:
# - A and B: Strict RAG/MCP evidence bundle
# - C: Augmented RAG/MCP/Web evidence bundle
#
# Relevance / Completeness:
# - All three judged against the same user question
# - No reward for citations or grounding
# ============================================================

THREE_WAY_QUALITY_MAX_TOKENS = 3600
THREE_WAY_QUALITY_ATTEMPTS = 2

THREE_WAY_QUALITY_PROMPT = """
You are an independent evaluator comparing THREE answers to the same user question.

SYSTEM A = Gemma LLM-Only
SYSTEM B = Strict Knowledge Grounding
SYSTEM C = Augmented Reasoning

Evaluate two DISTINCT qualities:

1. ANSWER RELEVANCE
   How directly and usefully does the answer address what the user actually asked?

2. ANSWER COMPLETENESS
   Which material answer requirements implied by the user's request are actually
   covered by each answer?

Do NOT judge factual correctness here.
Do NOT judge hallucination risk here.
Do NOT reward citations, evidence labels, or grounding.
Do NOT penalize an answer merely because it is ungrounded.
Do NOT reward verbosity or length by itself.

FIRST — DEFINE REQUIRED ANSWER ELEMENTS:
Infer ONLY the material answer elements required by the user's wording and scope.

Scope calibration rules:
- Infer required answer elements from the USER QUESTION ONLY.
- Do not use the content of Answer A, B, or C to decide what the user required.
- Respect the explicit domain boundary and task type in the question.
- A required element must be either:
  (a) explicitly requested by the user, or
  (b) materially necessary to fulfill the request.
- Do not automatically promote adjacent technical domains, background theory,
  standards references, architecture references, implementation stages, or
  contextual information into required elements merely because they are useful,
  technically related, or mentioned in one of the candidate answers.
- If the user asks for TOOLS, required elements should primarily represent
  materially distinct tool categories, tool-selection needs, or requested use
  cases. Standards/specifications/reference documents are NOT tools and must not
  become required elements unless the user explicitly asks for standards,
  compliance references, or design guidance based on standards.
- If the user asks specifically about 5G Radio Network / RAN design, keep the
  requirements inside the radio-network boundary. Core, Transport, Cloud, OSS,
  BSS, orchestration, and other adjacent domains are not required unless explicitly
  requested or strictly necessary to answer the question.
- Related adjacent-domain material may improve usefulness when concise, but it must
  not increase completeness merely because it expands beyond the requested scope.
- Optional recommendations, best-practice advice, or lifecycle guidance should not
  become completeness requirements unless the user asks for recommendations,
  comparisons, steps, workflow, or decision guidance.
- Use no more than 5 required answer elements.
- Assign requirement IDs exactly as R1, R2, R3, ... in order.

RELEVANCE:
Score EACH dimension independently from 0-100:
- Directness
- Coverage of user intent
- Specificity
- Usefulness
- Focus

Python will calculate final relevance deterministically using:
- Directness 20%
- Coverage 25%
- Specificity 20%
- Usefulness 25%
- Focus 10%

Do NOT calculate or optimize the final weighted relevance score yourself.

OUTPUT COMPACTNESS:
- Keep question_intent to one concise sentence.
- Use no more than 5 required answer elements.
- Keep each relevance_rationale and completeness_rationale to one concise sentence.
- Keep comparison_summary to one concise sentence.
- Do not explain scoring outside the required JSON fields.
- Do not repeat answer text or requirement descriptions unnecessarily.
- Return only the required JSON object.

COMPLETENESS:
For each answer, return ONLY the requirement IDs that are materially covered.
A requirement is covered only when the answer meaningfully addresses it.
Do not mark an element covered based on a passing mention.

Python will compute completeness deterministically:
covered requirements / total required requirements * 100.

Return JSON only:

{
  "question_intent": "One-sentence description of user intent.",
  "required_answer_elements": [
    {"id": "R1", "element": "Required element 1"},
    {"id": "R2", "element": "Required element 2"}
  ],
  "answer_a": {
    "directness": 0,
    "coverage": 0,
    "specificity": 0,
    "usefulness": 0,
    "focus": 0,
    "covered_requirement_ids": ["R1"],
    "relevance_rationale": "Concise reason for the relevance judgment.",
    "completeness_rationale": "Concise reason for the completeness judgment."
  },
  "answer_b": {
    "directness": 0,
    "coverage": 0,
    "specificity": 0,
    "usefulness": 0,
    "focus": 0,
    "covered_requirement_ids": ["R1"],
    "relevance_rationale": "Concise reason for the relevance judgment.",
    "completeness_rationale": "Concise reason for the completeness judgment."
  },
  "answer_c": {
    "directness": 0,
    "coverage": 0,
    "specificity": 0,
    "usefulness": 0,
    "focus": 0,
    "covered_requirement_ids": ["R1"],
    "relevance_rationale": "Concise reason for the relevance judgment.",
    "completeness_rationale": "Concise reason for the completeness judgment."
  },
  "comparison_summary": "Optional concise comparison of the three answers."
}

Do NOT return relevance_score.
Do NOT return completeness_score.
Do NOT return preferred_relevance.
Do NOT return preferred_completeness.
Python derives all four deterministically.
""".strip()


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    text = str(raw_text or "").strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    payload = json.loads(text)

    if not isinstance(payload, dict):
        raise RuntimeError("Judge output must be a JSON object.")

    return payload


def _validate_three_way_quality(payload: dict[str, Any]) -> dict[str, Any]:
    intent = str(
        payload.get(
            "question_intent",
            "",
        )
    ).strip()

    if not intent:
        raise RuntimeError(
            "Missing question_intent."
        )

    required_raw = payload.get(
        "required_answer_elements",
        [],
    )

    if not isinstance(
        required_raw,
        list,
    ):
        raise RuntimeError(
            "required_answer_elements must be a list."
        )

    if not required_raw:
        raise RuntimeError(
            "At least one required answer element is required."
        )

    if len(required_raw) > 5:
        raise RuntimeError(
            "required_answer_elements must contain no more than 5 items."
        )

    required_elements = []
    seen_ids = set()

    for index, item in enumerate(
        required_raw,
        start=1,
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise RuntimeError(
                "Each required_answer_elements item must be an object "
                "containing id and element."
            )

        requirement_id = str(
            item.get(
                "id",
                "",
            )
        ).strip().upper()

        element = str(
            item.get(
                "element",
                "",
            )
        ).strip()

        expected_id = f"R{index}"

        if requirement_id != expected_id:
            raise RuntimeError(
                "Requirement IDs must be sequential "
                f"R1..Rn. Expected {expected_id}, got "
                f"{requirement_id or '<missing>'}."
            )

        if not element:
            raise RuntimeError(
                f"{requirement_id}.element is missing."
            )

        if requirement_id in seen_ids:
            raise RuntimeError(
                f"Duplicate requirement ID: {requirement_id}."
            )

        seen_ids.add(
            requirement_id
        )

        required_elements.append(
            {
                "id":
                    requirement_id,
                "element":
                    element,
            }
        )

    payload[
        "required_answer_elements"
    ] = required_elements

    requirement_lookup = {
        item["id"]:
            item["element"]
        for item
        in required_elements
    }

    valid_requirement_ids = set(
        requirement_lookup.keys()
    )

    relevance_weights = {
        "directness":
            0.20,
        "coverage":
            0.25,
        "specificity":
            0.20,
        "usefulness":
            0.25,
        "focus":
            0.10,
    }

    for key in (
        "answer_a",
        "answer_b",
        "answer_c",
    ):
        group = payload.get(
            key
        )

        if not isinstance(
            group,
            dict,
        ):
            raise RuntimeError(
                f"Missing {key}."
            )

        weighted_relevance = 0.0

        for (
            dimension,
            weight,
        ) in relevance_weights.items():
            value = group.get(
                dimension
            )

            # Hosted JSON models occasionally serialize a valid
            # numeric score as a string. Accept numeric strings
            # deterministically; reject everything else.
            if isinstance(
                value,
                str,
            ):
                value = value.strip()

                if re.fullmatch(
                    r"\d+(?:\.\d+)?",
                    value,
                ):
                    value = float(
                        value
                    )

            if (
                isinstance(
                    value,
                    bool,
                )
                or
                not isinstance(
                    value,
                    (
                        int,
                        float,
                    ),
                )
            ):
                raise RuntimeError(
                    f"{key}.{dimension} must be numeric."
                )

            value = int(
                round(
                    float(
                        value
                    )
                )
            )

            if not 0 <= value <= 100:
                raise RuntimeError(
                    f"{key}.{dimension} must be between 0 and 100."
                )

            group[
                dimension
            ] = value

            weighted_relevance += (
                value
                *
                weight
            )

        group[
            "relevance_score"
        ] = int(
            round(
                weighted_relevance
            )
        )

        covered_raw = group.get(
            "covered_requirement_ids",
            [],
        )

        if not isinstance(
            covered_raw,
            list,
        ):
            raise RuntimeError(
                f"{key}.covered_requirement_ids must be a list."
            )

        covered_ids = []

        for requirement_id in covered_raw:
            requirement_id = str(
                requirement_id
            ).strip().upper()

            if not requirement_id:
                continue

            if requirement_id not in valid_requirement_ids:
                raise RuntimeError(
                    f"{key} returned unknown requirement ID: "
                    f"{requirement_id}."
                )

            if requirement_id not in covered_ids:
                covered_ids.append(
                    requirement_id
                )

        group[
            "covered_requirement_ids"
        ] = covered_ids

        missing_ids = [
            requirement_id
            for requirement_id
            in requirement_lookup
            if requirement_id
            not in
            set(
                covered_ids
            )
        ]

        group[
            "missing_requirement_ids"
        ] = missing_ids

        group[
            "covered_elements"
        ] = [
            requirement_lookup[
                requirement_id
            ]
            for requirement_id
            in covered_ids
        ]

        group[
            "missing_elements"
        ] = [
            requirement_lookup[
                requirement_id
            ]
            for requirement_id
            in missing_ids
        ]

        group[
            "completeness_score"
        ] = int(
            round(
                (
                    len(
                        covered_ids
                    )
                    /
                    len(
                        required_elements
                    )
                )
                *
                100.0
            )
        )

        relevance_rationale = str(
            group.get(
                "relevance_rationale",
                "",
            )
        ).strip()

        if not relevance_rationale:
            raise RuntimeError(
                f"{key}.relevance_rationale is missing."
            )

        completeness_rationale = str(
            group.get(
                "completeness_rationale",
                "",
            )
        ).strip()

        if not completeness_rationale:
            raise RuntimeError(
                f"{key}.completeness_rationale is missing."
            )

        group["relevance_rationale"] = relevance_rationale
        group["completeness_rationale"] = completeness_rationale

        # Backward-compatible alias for older UI/test code.
        group["rationale"] = relevance_rationale

    def _preferred_by(
        metric_key: str,
    ) -> str:
        values = {
            "A":
                int(
                    payload[
                        "answer_a"
                    ][
                        metric_key
                    ]
                ),
            "B":
                int(
                    payload[
                        "answer_b"
                    ][
                        metric_key
                    ]
                ),
            "C":
                int(
                    payload[
                        "answer_c"
                    ][
                        metric_key
                    ]
                ),
        }

        best_score = max(
            values.values()
        )

        winners = [
            label
            for (
                label,
                score,
            )
            in values.items()
            if score
            ==
            best_score
        ]

        if len(
            winners
        ) == 1:
            return winners[
                0
            ]

        return "TIE"

    payload[
        "preferred_relevance"
    ] = _preferred_by(
        "relevance_score"
    )

    payload[
        "preferred_completeness"
    ] = _preferred_by(
        "completeness_score"
    )

    comparison_summary = str(
        payload.get(
            "comparison_summary",
            "",
        )
    ).strip()

    if not comparison_summary:
        comparison_summary = (
            "Relevance preference: "
            f"{payload['preferred_relevance']}. "
            "Completeness preference: "
            f"{payload['preferred_completeness']}."
        )

    payload[
        "comparison_summary"
    ] = comparison_summary

    return payload

def run_three_way_quality_judge(
    question: str,
    llm_only_answer: str,
    strict_answer: str,
    augmented_answer: str,
) -> dict[str, Any]:
    user_prompt = (
        "USER QUESTION\n"
        "-------------\n"
        f"{question}\n\n"
        "ANSWER A — GEMMA LLM-ONLY\n"
        "-------------------------\n"
        f"{llm_only_answer}\n\n"
        "ANSWER B — STRICT KNOWLEDGE GROUNDING\n"
        "-------------------------------------\n"
        f"{strict_answer}\n\n"
        "ANSWER C — AUGMENTED REASONING\n"
        "------------------------------\n"
        f"{augmented_answer}"
    )

    base_messages = [
        {
            "role":
                "system",
            "content":
                THREE_WAY_QUALITY_PROMPT,
        },
        {
            "role":
                "user",
            "content":
                user_prompt,
        },
    ]

    started = time.perf_counter()

    last_error = None
    last_raw_text = ""
    last_finish_reason = ""

    for attempt in range(
        1,
        THREE_WAY_QUALITY_ATTEMPTS + 1,
    ):
        attempt_messages = list(
            base_messages
        )

        if attempt > 1:
            attempt_messages.append(
                {
                    "role":
                        "user",
                    "content":
                        (
                            "Your previous response did not satisfy the "
                            "required three-way quality JSON contract. "
                            "Return ONLY one complete JSON object. Include "
                            "question_intent, required_answer_elements, "
                            "answer_a, answer_b, answer_c, and optionally "
                            "comparison_summary. required_answer_elements "
                            "must contain 1 to 5 objects with sequential IDs "
                            "R1, R2, ... and an element string. Each answer "
                            "must include directness, coverage, specificity, "
                            "usefulness, focus, covered_requirement_ids, relevance_rationale, "
                            "and completeness_rationale. All five relevance dimensions "
                            "must be numeric 0-100. covered_requirement_ids "
                            "must contain only the supplied R# IDs that the "
                            "answer materially covers. Do NOT return "
                            "relevance_score, completeness_score, preferred_"
                            "relevance, or preferred_completeness; Python "
                            "derives them. Infer required elements from the "
                            "USER QUESTION ONLY; candidate-answer content must "
                            "not create requirements. Only explicit or materially "
                            "necessary elements may be required. Do not promote "
                            "adjacent domains, standards/reference documents, "
                            "background theory, or optional advice into required "
                            "elements unless the question asks for them or they "
                            "are strictly necessary. For a tools question, "
                            "standards/specifications are not tools. For a "
                            "RAN-specific question, do not make Core, Transport, "
                            "Cloud, OSS, BSS, or orchestration required unless "
                            "explicitly requested or strictly necessary. "
                            "Do not return markdown, commentary, "
                            "code fences, an empty object, or a partial object."
                        ),
                }
            )

        result = call_granite_judge(
            messages=
                attempt_messages,
            max_tokens=
                THREE_WAY_QUALITY_MAX_TOKENS,
        )

        raw_text = str(
            result.get(
                "text",
                "",
            )
            or
            ""
        ).strip()

        finish_reason = str(
            result.get(
                "finish_reason",
                "",
            )
            or
            ""
        ).strip().lower()

        last_raw_text = raw_text
        last_finish_reason = finish_reason

        try:
            payload = _parse_json_object(
                raw_text
            )

            payload = _validate_three_way_quality(
                payload
            )

            if attempt > 1:
                print()
                print(
                    "✓ Granite three-way quality judge "
                    f"recovered on attempt {attempt}/"
                    f"{THREE_WAY_QUALITY_ATTEMPTS}."
                )

            return {
                "payload":
                    payload,
                "result":
                    result,
                "elapsed_s":
                    (
                        time.perf_counter()
                        -
                        started
                    ),
                "attempts":
                    attempt,
                "retried":
                    attempt > 1,
            }

        except Exception as exc:
            last_error = exc

            print()
            print(
                "RAW GRANITE THREE-WAY QUALITY OUTPUT"
            )
            print(
                "-" * 116
            )
            print(
                f"Attempt           : "
                f"{attempt}/"
                f"{THREE_WAY_QUALITY_ATTEMPTS}"
            )
            print(
                f"Finish Reason     : "
                f"{finish_reason or 'UNKNOWN'}"
            )
            print(
                f"Max Output Tokens : "
                f"{THREE_WAY_QUALITY_MAX_TOKENS}"
            )
            print(
                f"Validation Error  : "
                f"{type(exc).__name__}: {exc}"
            )
            print()
            print(
                raw_text
                if raw_text
                else
                "<EMPTY RESPONSE>"
            )

            if attempt < THREE_WAY_QUALITY_ATTEMPTS:
                print()
                print(
                    "⚠ Retrying the three-way quality judge "
                    "once with a corrective JSON-only instruction..."
                )

    raise RuntimeError(
        "Could not parse three-way relevance/completeness judge output "
        f"after {THREE_WAY_QUALITY_ATTEMPTS} attempts. "
        f"Last finish reason: "
        f"{last_finish_reason or 'unknown'}. "
        f"Last returned characters: "
        f"{len(last_raw_text):,}."
    ) from last_error

def _evaluate_single_risk(
    *,
    question: str,
    answer: str,
    prefix: str,
    evidence_context: str,
    evidence_items: list[dict[str, Any]],
    system_label: str,
) -> dict[str, Any]:
    """
    Evaluate one answer independently for evidence-relative
    hallucination risk using the dedicated single-answer
    batched Granite risk judge.
    """

    extraction = extract_refined_claims(
        answer,
        prefix,
    )

    claims = extraction[
        "claims"
    ]

    valid_evidence_ids = build_valid_evidence_ids(
        evidence_items
    )

    execution = run_granite_single_answer_judge(
        question=
            question,
        evidence_context=
            evidence_context,
        claims=
            claims,
        system_label=
            system_label,
    )

    assessment = validate_and_summarize_judgment(
        original_claims=
            claims,
        judge_group_payload=
            execution[
                "payload"
            ],
        valid_evidence_ids=
            valid_evidence_ids,
    )

    return {
        "claim_extraction":
            extraction,
        "assessment":
            assessment,
        "risk_pct":
            float(
                assessment[
                    "estimated_hallucination_risk_pct"
                ]
            ),
        "judge":
            execution,
    }

def _score_label(score: float) -> str:
    return relevance_score_label(score)


def _build_hallucination_reasoning(
    assessment: dict[str, Any],
    *,
    max_claim_examples: int = 5,
) -> dict[str, Any]:
    """
    Build UI-friendly reasoning from existing claim-level Granite judgments.
    No additional judge call is made.
    """
    claims = assessment.get("claims", []) or []

    counts = {
        "supported": int(assessment.get("supported", 0) or 0),
        "partially_supported": int(
            assessment.get("partially_supported", 0) or 0
        ),
        "unsupported": int(assessment.get("unsupported", 0) or 0),
        "contradicted": int(assessment.get("contradicted", 0) or 0),
    }

    priority = {
        "contradicted": 0,
        "unsupported": 1,
        "partially_supported": 2,
        "supported": 3,
    }

    notable = sorted(
        [
            item
            for item in claims
            if str(item.get("status", "")).strip()
            in {
                "contradicted",
                "unsupported",
                "partially_supported",
            }
        ],
        key=lambda item: (
            priority.get(
                str(item.get("status", "")).strip(),
                99,
            ),
            str(item.get("claim_id", "")),
        ),
    )[:max_claim_examples]

    notable_claims = [
        {
            "claim_id": str(item.get("claim_id", "")).strip(),
            "claim": str(item.get("claim", "")).strip(),
            "status": str(item.get("status", "")).strip(),
            "evidence": list(item.get("evidence", []) or []),
            "reason": str(item.get("reason", "")).strip(),
        }
        for item in notable
    ]

    return {
        "summary": (
            f"{counts['supported']} supported, "
            f"{counts['partially_supported']} partially supported, "
            f"{counts['unsupported']} unsupported, "
            f"{counts['contradicted']} contradicted."
        ),
        "counts": counts,
        "notable_claims": notable_claims,
        "all_claims": claims,
    }


def evaluate_three_way_telecom(
    *,
    question: str,
    llm_only_answer: str,
    strict_answer: str,
    augmented_answer: str,
    strict_evidence_context: str,
    strict_evidence_items: list[dict[str, Any]],
    augmented_evidence_context: str,
    augmented_evidence_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Three-way Telecom evaluation.

    Hallucination-risk basis:
      - LLM-only is assessed independently against Strict connected-corpus evidence.
      - Strict is assessed independently against Strict connected-corpus evidence.
      - Augmented is assessed independently against its broader RAG/MCP/Web evidence.

    Quality basis:
      - All three answers are compared together against the same user question
        for relevance and completeness.

    Execution:
      - Risk A, Risk B, Risk C, and the three-way quality judge run concurrently.
    """

    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=4) as executor:
        llm_risk_future = executor.submit(
            _evaluate_single_risk,
            question=question,
            answer=llm_only_answer,
            prefix="A",
            evidence_context=strict_evidence_context,
            evidence_items=strict_evidence_items,
            system_label="Gemma LLM-Only",
        )

        strict_risk_future = executor.submit(
            _evaluate_single_risk,
            question=question,
            answer=strict_answer,
            prefix="B",
            evidence_context=strict_evidence_context,
            evidence_items=strict_evidence_items,
            system_label="Strict Knowledge Grounding",
        )

        augmented_risk_future = executor.submit(
            _evaluate_single_risk,
            question=question,
            answer=augmented_answer,
            prefix="C",
            evidence_context=augmented_evidence_context,
            evidence_items=augmented_evidence_items,
            system_label="Augmented Reasoning",
        )

        quality_future = executor.submit(
            run_three_way_quality_judge,
            question,
            llm_only_answer,
            strict_answer,
            augmented_answer,
        )

        llm_risk_result = llm_risk_future.result()
        strict_risk_result = strict_risk_future.result()
        augmented_risk_result = augmented_risk_future.result()
        quality = quality_future.result()

    wall_s = time.perf_counter() - started

    q = quality["payload"]

    llm_risk = llm_risk_result["risk_pct"]
    strict_risk = strict_risk_result["risk_pct"]
    augmented_risk = augmented_risk_result["risk_pct"]

    systems = {
        "llm_only": {
            "label": "Gemma LLM-Only",
            "hallucination_risk_pct": llm_risk,
            "hallucination_risk_label": hallucination_risk_label(llm_risk),
            "risk_claim_extraction": llm_risk_result["claim_extraction"],
            "risk_assessment": llm_risk_result["assessment"],
            "relevance_score": q["answer_a"]["relevance_score"],
            "relevance_label": _score_label(
                q["answer_a"]["relevance_score"]
            ),
            "relevance_breakdown": {
                "directness": q["answer_a"]["directness"],
                "coverage": q["answer_a"]["coverage"],
                "specificity": q["answer_a"]["specificity"],
                "usefulness": q["answer_a"]["usefulness"],
                "focus": q["answer_a"]["focus"],
            },
            "completeness_score": q["answer_a"]["completeness_score"],
            "completeness_label": _score_label(
                q["answer_a"]["completeness_score"]
            ),
            "covered_elements": q["answer_a"]["covered_elements"],
            "missing_elements": q["answer_a"]["missing_elements"],
            "covered_requirement_ids": q["answer_a"]["covered_requirement_ids"],
            "missing_requirement_ids": q["answer_a"]["missing_requirement_ids"],
            "relevance_rationale": q["answer_a"]["relevance_rationale"],
            "completeness_rationale": q["answer_a"]["completeness_rationale"],
            "quality_rationale": q["answer_a"]["relevance_rationale"],
            "hallucination_reasoning": _build_hallucination_reasoning(
                llm_risk_result["assessment"]
            ),
        },
        "strict": {
            "label": "Strict Knowledge Grounding",
            "hallucination_risk_pct": strict_risk,
            "hallucination_risk_label": hallucination_risk_label(strict_risk),
            "risk_claim_extraction": strict_risk_result["claim_extraction"],
            "risk_assessment": strict_risk_result["assessment"],
            "relevance_score": q["answer_b"]["relevance_score"],
            "relevance_label": _score_label(
                q["answer_b"]["relevance_score"]
            ),
            "relevance_breakdown": {
                "directness": q["answer_b"]["directness"],
                "coverage": q["answer_b"]["coverage"],
                "specificity": q["answer_b"]["specificity"],
                "usefulness": q["answer_b"]["usefulness"],
                "focus": q["answer_b"]["focus"],
            },
            "completeness_score": q["answer_b"]["completeness_score"],
            "completeness_label": _score_label(
                q["answer_b"]["completeness_score"]
            ),
            "covered_elements": q["answer_b"]["covered_elements"],
            "missing_elements": q["answer_b"]["missing_elements"],
            "covered_requirement_ids": q["answer_b"]["covered_requirement_ids"],
            "missing_requirement_ids": q["answer_b"]["missing_requirement_ids"],
            "relevance_rationale": q["answer_b"]["relevance_rationale"],
            "completeness_rationale": q["answer_b"]["completeness_rationale"],
            "quality_rationale": q["answer_b"]["relevance_rationale"],
            "hallucination_reasoning": _build_hallucination_reasoning(
                strict_risk_result["assessment"]
            ),
        },
        "augmented": {
            "label": "Augmented Reasoning",
            "hallucination_risk_pct": augmented_risk,
            "hallucination_risk_label": hallucination_risk_label(
                augmented_risk
            ),
            "risk_claim_extraction": augmented_risk_result["claim_extraction"],
            "risk_assessment": augmented_risk_result["assessment"],
            "relevance_score": q["answer_c"]["relevance_score"],
            "relevance_label": _score_label(
                q["answer_c"]["relevance_score"]
            ),
            "relevance_breakdown": {
                "directness": q["answer_c"]["directness"],
                "coverage": q["answer_c"]["coverage"],
                "specificity": q["answer_c"]["specificity"],
                "usefulness": q["answer_c"]["usefulness"],
                "focus": q["answer_c"]["focus"],
            },
            "completeness_score": q["answer_c"]["completeness_score"],
            "completeness_label": _score_label(
                q["answer_c"]["completeness_score"]
            ),
            "covered_elements": q["answer_c"]["covered_elements"],
            "missing_elements": q["answer_c"]["missing_elements"],
            "covered_requirement_ids": q["answer_c"]["covered_requirement_ids"],
            "missing_requirement_ids": q["answer_c"]["missing_requirement_ids"],
            "relevance_rationale": q["answer_c"]["relevance_rationale"],
            "completeness_rationale": q["answer_c"]["completeness_rationale"],
            "quality_rationale": q["answer_c"]["relevance_rationale"],
            "hallucination_reasoning": _build_hallucination_reasoning(
                augmented_risk_result["assessment"]
            ),
        },
    }

    return {
        "applicable": True,
        "question_intent": q["question_intent"],
        "required_answer_elements": [
            item["element"]
            for item in q["required_answer_elements"]
        ],
        "required_answer_element_details": q["required_answer_elements"],
        "preferred_relevance": q["preferred_relevance"],
        "preferred_completeness": q["preferred_completeness"],
        "comparison_summary": q["comparison_summary"],
        "systems": systems,
        "differences": {
            "strict_vs_llm_risk_reduction_pp":
                llm_risk - strict_risk,
            "augmented_vs_llm_risk_reduction_pp":
                llm_risk - augmented_risk,
            "augmented_vs_strict_relevance_pp":
                systems["augmented"]["relevance_score"]
                - systems["strict"]["relevance_score"],
            "augmented_vs_strict_completeness_pp":
                systems["augmented"]["completeness_score"]
                - systems["strict"]["completeness_score"],
            "augmented_vs_llm_relevance_pp":
                systems["augmented"]["relevance_score"]
                - systems["llm_only"]["relevance_score"],
            "augmented_vs_llm_completeness_pp":
                systems["augmented"]["completeness_score"]
                - systems["llm_only"]["completeness_score"],
        },
        "definitions": {
            "hallucination_risk": RISK_DEFINITION,
            "relevance": (
                "How directly and usefully the answer addresses the user's "
                "actual question."
            ),
            "completeness": (
                "How comprehensively the answer covers the material answer "
                "elements implied by the user's request."
            ),
        },
        "judge": {
            "provider": JUDGE_PROVIDER,
            "model": JUDGE_MODEL,
            "display_name": JUDGE_DISPLAY_NAME,
            "reasoning_effort": JUDGE_REASONING_EFFORT,
            "wall_s": wall_s,

            # New individual-risk timings.
            "llm_risk_judge_s":
                llm_risk_result["judge"]["elapsed_s"],
            "strict_risk_judge_s":
                strict_risk_result["judge"]["elapsed_s"],
            "augmented_risk_judge_s":
                augmented_risk_result["judge"]["elapsed_s"],

            # Quality judge timing.
            "quality_judge_s": quality["elapsed_s"],
            "quality_attempts": quality["attempts"],
            "quality_retried": quality["retried"],

            # Execution diagnostics.
            "parallel_judge_operations": 4,
            "risk_architecture": "Three independent single-answer risk judges",
            "risk_execution_mode": "INDIVIDUAL_CONCURRENT",
            "quality_execution_mode": "THREE_WAY_COMPARATIVE",
            "llm_risk_batches":
                llm_risk_result["judge"].get("batch_count"),
            "strict_risk_batches":
                strict_risk_result["judge"].get("batch_count"),
            "augmented_risk_batches":
                augmented_risk_result["judge"].get("batch_count"),
        },
    }


def get_three_way_evaluation_status() -> dict[str, Any]:
    return {
        "systems": [
            "Gemma LLM-Only",
            "Strict Knowledge Grounding",
            "Augmented Reasoning",
        ],
        "metrics": [
            "Evidence-relative Hallucination Risk",
            "Answer Relevance",
            "Answer Completeness",
        ],
        "quality_rubric": {
            "relevance_dimension_scale": "Each dimension independently scored 0-100",
            "relevance_weights": {
                "directness": 0.20,
                "coverage": 0.25,
                "specificity": 0.20,
                "usefulness": 0.25,
                "focus": 0.10,
            },
            "relevance_score": "Python-weighted deterministic 0-100 score",
            "relevance_rationale": "Granite concise question-relative explanation",
            "completeness_rationale": "Granite concise requirement-coverage explanation",
            "hallucination_reasoning": "Claim-level Granite evidence-support reasons; no extra judge call",
            "preferred_relevance": "Derived deterministically in Python",
            "preferred_completeness": "Derived deterministically in Python",
            "completeness": (
                "Deterministic Python score = covered required elements / "
                "total required elements * 100"
            ),
        },
        "risk_evidence_basis": {
            "llm_only": "Strict connected-corpus evidence",
            "strict": "Strict connected-corpus evidence",
            "augmented": "Augmented RAG + MCP + Web evidence",
        },
        "judge_model": JUDGE_MODEL,
        "parallel_judge_operations": 4,
        "risk_architecture": "Three independent single-answer risk judges",
    }
