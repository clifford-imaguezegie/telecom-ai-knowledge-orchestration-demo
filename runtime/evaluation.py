from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from openai import OpenAI

from runtime.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    ROUTER_MODEL,
)

# ============================================================
# TELECOM AI KNOWLEDGE ORCHESTRATION — COMPARATIVE EVALUATION
# Frozen Notebook 21 / Cell 4D comparative-judge path.
#
# Connected-corpus only.
# Live external and general knowledge comparisons are N/A.
# ============================================================

JUDGE_PROVIDER = "OpenRouter"
JUDGE_MODEL = ROUTER_MODEL
JUDGE_DISPLAY_NAME = "Granite 4.2 8B"
JUDGE_REASONING_EFFORT = "low"
JUDGE_MAX_TOKENS = 300

_judge_client: OpenAI | None = None


def _get_judge_client() -> OpenAI:
    global _judge_client

    if _judge_client is None:
        api_key = str(
            OPENROUTER_API_KEY
            or os.getenv("OPENROUTER_API_KEY", "")
        ).strip()

        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is unavailable."
            )

        _judge_client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )

    return _judge_client


def call_granite_judge(
    messages,
    max_tokens=JUDGE_MAX_TOKENS,
):

    start = (
        time.perf_counter()
    )

    response = (
        _get_judge_client().chat.completions.create(

            model=
                JUDGE_MODEL,

            messages=
                messages,

            max_tokens=
                max_tokens,

            temperature=
                0.0,

            top_p=
                1.0,

            response_format={
                "type":
                    "json_object"
            },

            extra_body={

                "reasoning": {

                    "effort":
                        JUDGE_REASONING_EFFORT,

                    "exclude":
                        True,
                }
            },
        )
    )

    elapsed = (
        time.perf_counter()
        -
        start
    )

    message = (
        response.choices[0].message
    )

    content = str(
        message.content
        or
        ""
    ).strip()

    usage = getattr(
        response,
        "usage",
        None,
    )

    prompt_tokens = int(
        getattr(
            usage,
            "prompt_tokens",
            0,
        )
        or
        0
    )

    completion_tokens = int(
        getattr(
            usage,
            "completion_tokens",
            0,
        )
        or
        0
    )

    total_tokens = int(
        getattr(
            usage,
            "total_tokens",
            0,
        )
        or
        0
    )

    return {

        "text":
            content,

        "elapsed_s":
            float(
                elapsed
            ),

        "prompt_tokens":
            prompt_tokens,

        "completion_tokens":
            completion_tokens,

        "total_tokens":
            total_tokens,

        "finish_reason":
            response.choices[
                0
            ].finish_reason,

        "requested_model":
            JUDGE_MODEL,

        "response_model":
            getattr(
                response,
                "model",
                "",
            ),
    }

GEMMA_ONLY_LABEL = (
    "Gemma 4 Only"
)


GEMMA_RAG_MCP_LABEL = (
    "Gemma 4 + RAG + MCP"
)


RISK_DISPLAY_LABEL = (
    "Estimated Hallucination Risk"
)


RISK_DEFINITION = (
    "Estimated Hallucination Risk reflects how strongly the "
    "answer is supported by the retrieved technical evidence. "
    "It does not prove that unsupported claims are false."
)



# 4B.6 — ESTIMATED HALLUCINATION RISK WEIGHTS
# ============================================================

HALLUCINATION_RISK_WEIGHTS = {

    "supported":
        0.00,

    "partially_supported":
        0.35,

    "unsupported":
        0.75,

    "contradicted":
        1.00,
}


VALID_JUDGE_STATUSES = set(
    HALLUCINATION_RISK_WEIGHTS.keys()
)


# ============================================================
# 4B.7 — CLAIM TEXT NORMALIZATION
# ============================================================

def remove_evidence_citations(
    text,
):

    return re.sub(
        r"\[(?:\s*E\d+\s*,?)+\]",
        "",
        str(text or ""),
    )


def clean_claim_text(
    text,
):

    text = str(
        text or ""
    )


    text = remove_evidence_citations(
        text
    )


    text = re.sub(
        r"\*\*",
        "",
        text,
    )


    text = re.sub(
        r"`",
        "",
        text,
    )


    text = text.replace(
        "$",
        "",
    )


    text = re.sub(
        r"\s+",
        " ",
        text,
    )


    text = re.sub(
        r"\s+([.,;:!?])",
        r"\1",
        text,
    )


    return text.strip(
        " \t\r\n-*|"
    )


# ============================================================
# 4B.8 — ABBREVIATION-AWARE SENTENCE HANDLING
# ============================================================

CLAIM_ABBREVIATIONS = [

    "vs.",
    "e.g.",
    "i.e.",
    "etc.",
    "Fig.",
    "No.",
]


ABBREVIATION_DOT_TOKEN = (
    "<ABBR_DOT>"
)


def protect_abbreviations(
    text,
):

    protected = str(
        text or ""
    )


    for abbreviation in CLAIM_ABBREVIATIONS:

        protected = re.sub(

            re.escape(
                abbreviation
            ),

            lambda match:
                match.group(
                    0
                ).replace(
                    ".",
                    ABBREVIATION_DOT_TOKEN,
                ),

            protected,

            flags=re.IGNORECASE,
        )


    return protected


def restore_abbreviations(
    text,
):

    return str(
        text or ""
    ).replace(
        ABBREVIATION_DOT_TOKEN,
        ".",
    )


# ============================================================
# 4B.9 — STRUCTURAL STATEMENT FILTER
# ============================================================

STRUCTURAL_PATTERNS = [

    r"^the primary responsibilities.*(?:include|are|can be)",
    r"^its primary responsibilities.*(?:include|are|can be)",
    r"^the specific responsibilities.*(?:include|are|can be)",
    r"^the responsibilities.*(?:include|are|can be)",
    r"^the .* responsibilities .* categorized",

    r"^specific tasks include",
    r"^the following .* include",

    r"^in summary\b",
    r"^summary\b",

    r"^feature\s+[-—]\s+.*responsibility",

    r"^it is important to distinguish\b",
    r"^it is critical to distinguish\b",

    r"^to ensure .*clarity\b",
    r"^to avoid .*confusion\b",

    r"^the .* include:$",
    r"^the .* includes:$",

    r"^these include:$",

    r"^while .* plays .* role:$",
]


def is_structural_statement(
    text,
):

    text = clean_claim_text(
        text
    )


    if not text:

        return True


    lowered = text.lower()


    for pattern in STRUCTURAL_PATTERNS:

        if re.search(
            pattern,
            lowered,
        ):

            return True


    if text.endswith(
        ":"
    ):

        return True


    return False


# ============================================================
# 4B.10 — MARKDOWN TABLE DETECTION
# ============================================================

def is_markdown_table_separator(
    line,
):

    line = str(
        line
    ).strip()


    if "|" not in line:

        return False


    cells = [

        cell.strip()

        for cell
        in line.strip("|").split("|")
    ]


    if not cells:

        return False


    return all(

        re.fullmatch(
            r":?-{3,}:?",
            cell,
        )
        is not None

        for cell
        in cells
    )


def is_markdown_table_row(
    line,
):

    line = str(
        line
    ).strip()


    return (
        line.startswith("|")
        and
        line.endswith("|")
    )


# ============================================================
# 4B.11 — NON-PROPOSITIONAL LIST FRAGMENT FILTER
# ============================================================

PREDICATE_PATTERN = re.compile(
    r"\b("
    r"is|are|was|were|be|been|being|"
    r"has|have|had|"
    r"does|do|did|"
    r"can|could|may|might|must|shall|should|will|would|"
    r"acts|allows|applies|assists|"
    r"connects|contains|controls|coordinates|"
    r"enables|ensures|establishes|"
    r"facilitates|forwards|"
    r"handles|includes|interacts|"
    r"maintains|manages|maps|"
    r"offers|oversees|performs|processes|provides|"
    r"receives|represents|requires|responsible|routes|"
    r"selects|serves|supports|"
    r"terminates|tracks|triggers|uses|works"
    r")\b",
    flags=re.IGNORECASE,
)


def is_non_propositional_list_fragment(
    text,
):

    text = clean_claim_text(
        text
    )


    if not text:

        return True


    if ":" in text:

        return False


    if PREDICATE_PATTERN.search(
        text
    ):

        return False


    if re.search(
        r"[.!?]$",
        text,
    ):

        return False


    return True


# ============================================================
# 4B.12 — SENTENCE SPLITTER
# ============================================================

def split_into_sentence_units(
    text,
):

    text = clean_claim_text(
        text
    )


    if not text:

        return []


    protected = protect_abbreviations(
        text
    )


    parts = re.split(

        r"(?<=[.!?])\s+(?=[A-Z0-9])",

        protected,
    )


    cleaned_parts = []


    for part in parts:

        part = restore_abbreviations(
            part
        )


        part = clean_claim_text(
            part
        )


        if len(
            part
        ) < 20:

            continue


        if is_structural_statement(
            part
        ):

            continue


        cleaned_parts.append(
            part
        )


    return cleaned_parts


# ============================================================
# 4B.13 — REFINED FIXED CLAIM EXTRACTOR
# ============================================================

def extract_refined_claims(
    answer,
    prefix,
):

    answer = str(
        answer or ""
    )


    extracted = []


    skipped = {

        "headings":
            0,

        "tables":
            0,

        "structural":
            0,

        "fragments":
            0,

        "short":
            0,

        "duplicates":
            0,
    }


    for raw_line in answer.splitlines():

        line = raw_line.strip()


        if not line:

            continue


        if re.match(
            r"^#{1,6}\s+",
            line,
        ):

            skipped[
                "headings"
            ] += 1

            continue


        if (
            is_markdown_table_separator(
                line
            )
            or
            is_markdown_table_row(
                line
            )
        ):

            skipped[
                "tables"
            ] += 1

            continue


        bullet_match = re.match(

            r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$",

            line,
        )


        if bullet_match:

            bullet_text = clean_claim_text(

                bullet_match.group(
                    1
                )
            )


            if is_structural_statement(
                bullet_text
            ):

                skipped[
                    "structural"
                ] += 1

                continue


            if is_non_propositional_list_fragment(
                bullet_text
            ):

                skipped[
                    "fragments"
                ] += 1

                continue


            sentence_units = (

                split_into_sentence_units(
                    bullet_text
                )
            )


            if not sentence_units:

                skipped[
                    "short"
                ] += 1


            extracted.extend(
                sentence_units
            )

            continue


        cleaned_line = clean_claim_text(
            line
        )


        if is_structural_statement(
            cleaned_line
        ):

            skipped[
                "structural"
            ] += 1

            continue


        sentence_units = (

            split_into_sentence_units(
                cleaned_line
            )
        )


        if not sentence_units:

            skipped[
                "short"
            ] += 1


        extracted.extend(
            sentence_units
        )


    # ========================================================
    # EXACT NORMALIZED DEDUPLICATION
    # ========================================================

    unique_claims = []

    seen = set()


    for claim in extracted:

        canonical = re.sub(

            r"[^a-z0-9]+",

            " ",

            claim.lower(),

        ).strip()


        if not canonical:

            continue


        if canonical in seen:

            skipped[
                "duplicates"
            ] += 1

            continue


        seen.add(
            canonical
        )


        unique_claims.append(
            claim
        )


    if not unique_claims:

        raise RuntimeError(
            f"No refined claims extracted for {prefix}."
        )


    numbered_claims = []


    for index, claim in enumerate(
        unique_claims,
        start=1,
    ):

        numbered_claims.append(

            {
                "claim_id":
                    f"{prefix}{index}",

                "claim":
                    claim,
            }
        )


    return {

        "claims":
            numbered_claims,

        "skipped":
            skipped,

        "final_claim_count":
            len(
                numbered_claims
            ),
    }


def format_claims_for_judge(
    claims,
):

    return "\n".join(

        f"{item['claim_id']}: {item['claim']}"

        for item
        in claims
    )


# ============================================================
# 4B.14 — GRANITE COMPARATIVE JUDGE PROMPT
# ============================================================
#
# ONLY CHANGE FROM THE LAST WORKING VERSION:
#
# Before final verdict, Granite checks whether another supplied
# evidence item supports the claim more directly.
#
# ============================================================

MODULE4_JUDGE_SYSTEM_PROMPT = """
You are an independent evaluator of technical answers.

Evaluate every supplied claim using ONLY the supplied evidence.

Classify each claim as exactly one of:

supported:
The evidence supports the claim.

partially_supported:
The evidence supports part of the claim, but does not fully establish it.

unsupported:
The evidence does not establish the claim.

contradicted:
The evidence conflicts with the claim.

Rules:
- Do not use external knowledge.
- Judge both answer groups using the same evidence and the same standard.
- Do not treat general topic similarity as sufficient evidence.
- A reasonable inference from the supplied evidence may be accepted.
- If the evidence supports only part of a claim, use partially_supported.
- If the evidence is missing, use unsupported rather than contradicted.
- Before assigning a verdict, check whether another supplied evidence item
  supports the claim more directly than the first evidence item considered.
- Evaluate every supplied claim exactly once.
- Copy every claim_id exactly and unchanged.
- Evidence must contain only relevant E# identifiers from the supplied evidence.
- Give a short reason for each judgment.
- Do not calculate totals, scores, percentages or risk.
- Return JSON only.

Return exactly this structure:

{
  "gemma4_only": {
    "claims": [
      {
        "claim_id": "A1",
        "status": "supported",
        "evidence": ["E1"],
        "reason": "Short evidence-based reason."
      }
    ]
  },
  "gemma4_rag_mcp": {
    "claims": [
      {
        "claim_id": "B1",
        "status": "supported",
        "evidence": ["E1"],
        "reason": "Short evidence-based reason."
      }
    ]
  }
}
""".strip()


# ============================================================
# 4B.15 — VALID EVIDENCE IDS
# ============================================================

def build_valid_evidence_ids(
    evidence_items,
):

    return {

        f"E{item['presentation_rank']}"

        for item
        in evidence_items
    }


# ============================================================
# 4B.16 — PARSE SINGLE COMPARATIVE GRANITE OUTPUT
# ============================================================

def parse_granite_comparative_output(
    raw_text,
):

    raw_text = str(
        raw_text or ""
    ).strip()


    if raw_text.startswith(
        "```"
    ):

        raw_text = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw_text,
            flags=re.IGNORECASE,
        )


        raw_text = re.sub(
            r"\s*```$",
            "",
            raw_text,
        )


    payload = json.loads(
        raw_text
    )


    if not isinstance(
        payload,
        dict,
    ):

        raise RuntimeError(
            "Granite comparative output must be a JSON object."
        )


    gemma_only_group = payload.get(
        "gemma4_only"
    )


    rag_mcp_group = payload.get(
        "gemma4_rag_mcp"
    )


    if gemma_only_group is None:

        for key, value in payload.items():

            canonical = re.sub(
                r"[^a-z0-9]+",
                "",
                str(key).lower(),
            )


            if canonical in {
                "gemma4only",
                "standalone",
                "answera",
            }:

                gemma_only_group = value

                break


    if rag_mcp_group is None:

        for key, value in payload.items():

            canonical = re.sub(
                r"[^a-z0-9]+",
                "",
                str(key).lower(),
            )


            if canonical in {
                "gemma4ragmcp",
                "grounded",
                "answerb",
            }:

                rag_mcp_group = value

                break


    if gemma_only_group is None:

        raise RuntimeError(
            "Granite output missing Gemma 4 Only group."
        )


    if rag_mcp_group is None:

        raise RuntimeError(
            "Granite output missing Gemma 4 + RAG + MCP group."
        )


    def extract_claim_list(
        group,
        group_name,
    ):

        if isinstance(
            group,
            list,
        ):

            return group


        if isinstance(
            group,
            dict,
        ):

            if isinstance(
                group.get(
                    "claims"
                ),
                list,
            ):

                return group[
                    "claims"
                ]


            if isinstance(
                group.get(
                    "evaluations"
                ),
                list,
            ):

                return group[
                    "evaluations"
                ]


        raise RuntimeError(
            f"Granite group {group_name} does not contain a claim list."
        )


    return {

        "gemma4_only": {

            "claims":
                extract_claim_list(
                    gemma_only_group,
                    "gemma4_only",
                )
        },

        "gemma4_rag_mcp": {

            "claims":
                extract_claim_list(
                    rag_mcp_group,
                    "gemma4_rag_mcp",
                )
        },
    }


# ============================================================
# 4B.17 — STRUCTURAL VALIDATION + RISK CALCULATION
# ============================================================

def validate_and_summarize_judgment(
    original_claims,
    judge_group_payload,
    valid_evidence_ids,
):

    evaluations = judge_group_payload.get(
        "claims",
        [],
    )


    if not isinstance(
        evaluations,
        list,
    ):

        raise RuntimeError(
            "Granite claims must be a list."
        )


    expected_ids = [

        item[
            "claim_id"
        ]

        for item
        in original_claims
    ]


    expected_set = set(
        expected_ids
    )


    # --------------------------------------------------------
    # Hosted-model claim-ID resilience
    # --------------------------------------------------------
    #
    # Required original claim IDs remain strict:
    #   - every expected ID must be present;
    #   - expected IDs must not be duplicated;
    #   - each expected ID is validated exactly once.
    #
    # A hosted judge may occasionally append an invented extra
    # claim ID (for example B20 when the fixed claim list ends at
    # B19). Such extras are not part of the supplied evaluation
    # contract and are therefore discarded with an explicit
    # warning rather than failing an otherwise complete judgment.
    # --------------------------------------------------------

    unexpected_items = []

    filtered_evaluations = []

    for item in evaluations:

        if not isinstance(
            item,
            dict,
        ):
            continue

        claim_id = str(
            item.get(
                "claim_id",
                "",
            )
        ).strip()

        if claim_id in expected_set:
            filtered_evaluations.append(
                item
            )
        else:
            unexpected_items.append(
                item
            )


    unexpected_ids = sorted(
        {
            str(
                item.get(
                    "claim_id",
                    "",
                )
            ).strip()

            for item
            in unexpected_items

            if str(
                item.get(
                    "claim_id",
                    "",
                )
            ).strip()
        }
    )


    if unexpected_ids:

        print()
        print(
            "⚠ GRANITE JUDGE WARNING — "
            "discarding unexpected claim ID(s): "
            +
            ", ".join(
                unexpected_ids
            )
        )


    evaluations = filtered_evaluations


    returned_ids = [

        str(
            item.get(
                "claim_id",
                "",
            )
        ).strip()

        for item
        in evaluations
    ]


    returned_set = set(
        returned_ids
    )


    if len(
        returned_ids
    ) != len(
        returned_set
    ):

        raise RuntimeError(
            "Granite returned duplicate expected claim IDs."
        )


    missing_ids = sorted(

        expected_set
        -
        returned_set
    )


    if missing_ids:

        raise RuntimeError(

            "Granite omitted required claim IDs: "
            +
            ", ".join(
                missing_ids
            )
        )


    if len(
        evaluations
    ) != len(
        original_claims
    ):

        raise RuntimeError(
            "Granite output is not 1:1 with supplied required claims "
            "after removing unexpected claim IDs."
        )


    original_lookup = {

        item[
            "claim_id"
        ]:
            item[
                "claim"
            ]

        for item
        in original_claims
    }


    evaluation_lookup = {

        str(
            item[
                "claim_id"
            ]
        ).strip():
            item

        for item
        in evaluations
    }


    normalized = []


    counts = {

        "supported":
            0,

        "partially_supported":
            0,

        "unsupported":
            0,

        "contradicted":
            0,
    }


    invalid_evidence_references = (
        0
    )


    unsupported_with_evidence = (
        0
    )


    empty_reason_count = (
        0
    )


    for claim_id in expected_ids:

        evaluation = (

            evaluation_lookup[
                claim_id
            ]
        )


        status = str(

            evaluation.get(
                "status",
                "",
            )

        ).strip().lower()


        if status not in VALID_JUDGE_STATUSES:

            raise RuntimeError(

                f"Invalid Granite status for "
                f"{claim_id}: {status!r}"
            )


        evidence = evaluation.get(
            "evidence",
            [],
        )


        if evidence is None:

            evidence = []


        if isinstance(
            evidence,
            str,
        ):

            evidence = [
                evidence
            ]


        if not isinstance(
            evidence,
            list,
        ):

            evidence = []


        cleaned_evidence = []


        for evidence_id in evidence:

            evidence_id = str(
                evidence_id
            ).strip()


            if not evidence_id:

                continue


            if (
                evidence_id
                not in
                valid_evidence_ids
            ):

                invalid_evidence_references += 1

                continue


            if (
                evidence_id
                not in
                cleaned_evidence
            ):

                cleaned_evidence.append(
                    evidence_id
                )


        reason = str(

            evaluation.get(
                "reason",
                "",
            )

        ).strip()


        if not reason:

            empty_reason_count += 1


        if (
            status
            ==
            "unsupported"
            and
            cleaned_evidence
        ):

            unsupported_with_evidence += 1


        normalized.append(

            {
                "claim_id":
                    claim_id,

                "claim":
                    original_lookup[
                        claim_id
                    ],

                "status":
                    status,

                "evidence":
                    cleaned_evidence,

                "reason":
                    reason,
            }
        )


        counts[
            status
        ] += 1


    total_claims = len(
        normalized
    )


    weighted_risk = sum(

        counts[
            status
        ]
        *
        HALLUCINATION_RISK_WEIGHTS[
            status
        ]

        for status
        in HALLUCINATION_RISK_WEIGHTS
    )


    estimated_risk_pct = (

        weighted_risk
        /
        total_claims
        *
        100.0
    )


    return {

        "claims":
            normalized,

        "total_claims":
            total_claims,

        "supported":
            counts[
                "supported"
            ],

        "partially_supported":
            counts[
                "partially_supported"
            ],

        "unsupported":
            counts[
                "unsupported"
            ],

        "contradicted":
            counts[
                "contradicted"
            ],

        "invalid_evidence_references":
            invalid_evidence_references,

        "unsupported_with_evidence":
            unsupported_with_evidence,

        "empty_reason_count":
            empty_reason_count,

        "estimated_hallucination_risk_pct":
            float(
                min(
                    100.0,
                    max(
                        0.0,
                        estimated_risk_pct,
                    )
                )
            ),
    }


def hallucination_risk_label(
    risk_pct,
):

    if risk_pct <= 20:

        return "LOW"


    if risk_pct <= 40:

        return "MODERATE"


    if risk_pct <= 60:

        return "ELEVATED"


    if risk_pct <= 80:

        return "HIGH"


    return "VERY HIGH"

# ============================================================
# 4B.19 — BATCHED GRANITE COMPARATIVE JUDGE HELPER
# ============================================================
#
# PURPOSE
# -------
# Preserve the existing evidence-relative hallucination-risk methodology
# while making hosted Granite structured output more reliable.
#
# Only the execution strategy changes:
#   fixed claims -> deterministic batches -> Granite -> merge ->
#   existing validation + existing risk calculation.
#
# The following remain unchanged:
# - claim extraction
# - evidence supplied to the judge
# - Granite model and rubric
# - supported / partially_supported / unsupported / contradicted labels
# - evidence-ID validation
# - hallucination-risk weights
# - final risk calculation
# ============================================================

GRANITE_CLAIM_BATCH_SIZE = 8
GRANITE_BATCH_MAX_ATTEMPTS = 2
GRANITE_BATCH_BASE_MAX_TOKENS = 1200
GRANITE_BATCH_TOKENS_PER_CLAIM = 140
GRANITE_BATCH_MAX_TOKENS = 3200


def _chunk_claims(
    claims,
    batch_size=GRANITE_CLAIM_BATCH_SIZE,
):
    """Split a fixed claim list into deterministic ordered batches."""

    claims = list(claims or [])

    if batch_size <= 0:
        raise ValueError(
            "Granite claim batch size must be greater than zero."
        )

    return [
        claims[start:start + batch_size]
        for start in range(0, len(claims), batch_size)
    ]


def _expected_claim_ids(claims):
    """Return fixed claim IDs in original order."""

    return [
        str(item["claim_id"]).strip()
        for item in claims
    ]


def _validate_batch_claim_ids(
    *,
    payload,
    expected_a,
    expected_b,
):
    """
    Validate one batch or the final merged payload.

    Full status/evidence/reason validation remains in
    validate_and_summarize_judgment().
    """

    expected_groups = {
        "gemma4_only": list(expected_a),
        "gemma4_rag_mcp": list(expected_b),
    }

    for group_name, expected_ids in expected_groups.items():
        group = payload.get(group_name)

        if not isinstance(group, dict):
            raise RuntimeError(
                f"Granite batch missing group: {group_name}"
            )

        claims = group.get("claims", [])

        if not isinstance(claims, list):
            raise RuntimeError(
                f"Granite batch group {group_name} does not contain a claims list."
            )

        returned_ids = [
            str(item.get("claim_id", "")).strip()
            for item in claims
            if isinstance(item, dict)
        ]

        if len(returned_ids) != len(set(returned_ids)):
            raise RuntimeError(
                f"Granite batch returned duplicate claim IDs for {group_name}."
            )

        expected_set = set(expected_ids)
        returned_set = set(returned_ids)

        missing_ids = sorted(expected_set - returned_set)
        unexpected_ids = sorted(returned_set - expected_set)

        if missing_ids:
            raise RuntimeError(
                f"Granite batch omitted required claim IDs for {group_name}: "
                + ", ".join(missing_ids)
            )

        if unexpected_ids:
            raise RuntimeError(
                f"Granite batch invented unexpected claim IDs for {group_name}: "
                + ", ".join(unexpected_ids)
            )

        if len(returned_ids) != len(expected_ids):
            raise RuntimeError(
                f"Granite batch claim count mismatch for {group_name}: "
                f"expected {len(expected_ids)}, returned {len(returned_ids)}."
            )


def _build_batch_prompt(
    *,
    question,
    evidence_context,
    claims_a,
    claims_b,
):
    """Build the existing comparative-judge request for one bounded batch."""

    ids_a = _expected_claim_ids(claims_a)
    ids_b = _expected_claim_ids(claims_b)

    formatted_a = (
        format_claims_for_judge(claims_a)
        if claims_a
        else "(NO CLAIMS IN THIS BATCH)"
    )

    formatted_b = (
        format_claims_for_judge(claims_b)
        if claims_b
        else "(NO CLAIMS IN THIS BATCH)"
    )

    required_a = ", ".join(ids_a) if ids_a else "(NONE)"
    required_b = ", ".join(ids_b) if ids_b else "(NONE)"

    return (
        f"QUESTION\n"
        f"--------\n"
        f"{question}\n\n"
        f"TECHNICAL EVIDENCE\n"
        f"------------------\n"
        f"{evidence_context}\n\n"
        f"ANSWER A — {GEMMA_ONLY_LABEL}\n"
        f"FIXED CLAIMS — CURRENT BATCH ONLY\n"
        f"---------------------------------\n"
        f"{formatted_a}\n\n"
        f"ANSWER B — {GEMMA_RAG_MCP_LABEL}\n"
        f"FIXED CLAIMS — CURRENT BATCH ONLY\n"
        f"---------------------------------\n"
        f"{formatted_b}\n\n"
        f"REQUIRED ANSWER A CLAIM IDS — CURRENT BATCH\n"
        f"-------------------------------------------\n"
        f"{required_a}\n\n"
        f"REQUIRED ANSWER B CLAIM IDS — CURRENT BATCH\n"
        f"-------------------------------------------\n"
        f"{required_b}\n\n"
        f"IMPORTANT BATCH RULES\n"
        f"---------------------\n"
        f"- Evaluate ONLY the claims listed in this batch.\n"
        f"- Do not add claims from outside this batch.\n"
        f"- Include every required claim ID exactly once.\n"
        f"- If one answer has no claims in this batch, return an empty claims list for that answer group.\n"
        f"- Return both required top-level JSON groups."
    )


def _run_granite_claim_batch(
    *,
    question,
    evidence_context,
    claims_a,
    claims_b,
    batch_number,
    total_batches,
):
    """Execute and structurally validate one bounded Granite batch."""

    expected_a = _expected_claim_ids(claims_a)
    expected_b = _expected_claim_ids(claims_b)

    total_claims = len(claims_a) + len(claims_b)

    judge_max_tokens = min(
        GRANITE_BATCH_MAX_TOKENS,
        max(
            GRANITE_BATCH_BASE_MAX_TOKENS,
            (GRANITE_BATCH_TOKENS_PER_CLAIM * total_claims) + 500,
        ),
    )

    user_prompt = _build_batch_prompt(
        question=question,
        evidence_context=evidence_context,
        claims_a=claims_a,
        claims_b=claims_b,
    )

    base_messages = [
        {
            "role": "system",
            "content": MODULE4_JUDGE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    batch_started = time.perf_counter()
    last_exc = None
    last_raw_text = ""
    last_finish_reason = ""

    for attempt in range(1, GRANITE_BATCH_MAX_ATTEMPTS + 1):
        messages = list(base_messages)

        if attempt > 1:
            repair_instruction = (
                "Your previous response did not satisfy the required batch JSON "
                "contract. Return ONLY one complete JSON object matching the original "
                "schema. Include BOTH top-level groups 'gemma4_only' and "
                "'gemma4_rag_mcp'. Include EVERY required claim ID from THIS BATCH "
                "exactly once and no additional claim IDs. For each claim return "
                "claim_id, status, evidence, and reason. If a group has no claims in "
                "this batch, return that group's claims as an empty list. Do not return "
                "markdown, commentary, code fences, an empty object, or a partial object."
            )

            messages.append(
                {
                    "role": "user",
                    "content": repair_instruction,
                }
            )

        result = call_granite_judge(
            messages=messages,
            max_tokens=judge_max_tokens,
        )

        raw_text = str(result.get("text", "") or "").strip()
        finish_reason = str(
            result.get("finish_reason", "") or ""
        ).strip().lower()

        last_raw_text = raw_text
        last_finish_reason = finish_reason

        try:
            payload = parse_granite_comparative_output(raw_text)

            _validate_batch_claim_ids(
                payload=payload,
                expected_a=expected_a,
                expected_b=expected_b,
            )

            elapsed_s = time.perf_counter() - batch_started

            if attempt > 1:
                print(
                    f"✓ Granite batch {batch_number}/{total_batches} recovered "
                    f"on attempt {attempt}/{GRANITE_BATCH_MAX_ATTEMPTS}."
                )

            return {
                "payload": payload,
                "result": result,
                "raw_text": raw_text,
                "elapsed_s": float(elapsed_s),
                "max_tokens": int(judge_max_tokens),
                "claim_count": int(total_claims),
                "attempts": int(attempt),
                "retried": bool(attempt > 1),
                "batch_number": int(batch_number),
                "finish_reason": finish_reason,
            }

        except Exception as exc:
            last_exc = exc

            print()
            print("GRANITE BATCH OUTPUT WARNING")
            print("-" * 116)
            print(
                f"Batch              : {batch_number}/{total_batches}"
            )
            print(
                f"Attempt            : {attempt}/{GRANITE_BATCH_MAX_ATTEMPTS}"
            )
            print(
                f"Claims in Batch    : {total_claims}"
            )
            print(
                f"Finish Reason      : {finish_reason or 'UNKNOWN'}"
            )
            print(
                f"Max Output Tokens  : {judge_max_tokens}"
            )
            print(
                f"Returned Characters: {len(raw_text):,}"
            )
            print(
                f"Validation Error   : {type(exc).__name__}: {exc}"
            )

            if attempt < GRANITE_BATCH_MAX_ATTEMPTS:
                print()
                print(
                    "⚠ Retrying this batch once with a corrective JSON-only instruction..."
                )

    raise RuntimeError(
        f"Granite comparative judge batch {batch_number}/{total_batches} failed "
        f"after {GRANITE_BATCH_MAX_ATTEMPTS} attempts. Last finish reason: "
        f"{last_finish_reason or 'unknown'}. Last returned characters: "
        f"{len(last_raw_text):,}."
    ) from last_exc


def run_granite_comparative_judge(
    question,
    evidence_context,
    gemma_only_claims,
    rag_mcp_claims,
):
    """
    Evaluate two fixed claim groups against one evidence bundle using bounded
    Granite batches while preserving the existing public return contract.
    """

    gemma_only_claims = list(gemma_only_claims or [])
    rag_mcp_claims = list(rag_mcp_claims or [])

    if not gemma_only_claims and not rag_mcp_claims:
        raise RuntimeError(
            "No claims supplied to Granite comparative judge."
        )

    batches_a = _chunk_claims(gemma_only_claims)
    batches_b = _chunk_claims(rag_mcp_claims)

    total_batches = max(len(batches_a), len(batches_b))

    while len(batches_a) < total_batches:
        batches_a.append([])

    while len(batches_b) < total_batches:
        batches_b.append([])

    print()
    print("GRANITE COMPARATIVE JUDGE — BATCHED EXECUTION")
    print("-" * 116)
    print(
        f"Answer A Claims     : {len(gemma_only_claims)}"
    )
    print(
        f"Answer B Claims     : {len(rag_mcp_claims)}"
    )
    print(
        f"Batch Size / Answer : {GRANITE_CLAIM_BATCH_SIZE}"
    )
    print(
        f"Total Batches       : {total_batches}"
    )

    judge_started = time.perf_counter()

    merged_a = []
    merged_b = []
    batch_metadata = []
    last_result = None
    raw_outputs = []
    total_attempts = 0
    any_retry = False
    total_requested_tokens = 0

    for batch_index in range(total_batches):
        claims_a = batches_a[batch_index]
        claims_b = batches_b[batch_index]

        print()
        print(
            f"Running Granite batch {batch_index + 1}/{total_batches} "
            f"(A={len(claims_a)}, B={len(claims_b)})..."
        )

        batch_result = _run_granite_claim_batch(
            question=question,
            evidence_context=evidence_context,
            claims_a=claims_a,
            claims_b=claims_b,
            batch_number=batch_index + 1,
            total_batches=total_batches,
        )

        payload = batch_result["payload"]

        merged_a.extend(
            payload["gemma4_only"]["claims"]
        )
        merged_b.extend(
            payload["gemma4_rag_mcp"]["claims"]
        )

        last_result = batch_result.get("result")
        raw_outputs.append(batch_result.get("raw_text", ""))
        total_attempts += int(batch_result.get("attempts", 1))
        any_retry = any_retry or bool(
            batch_result.get("retried", False)
        )
        total_requested_tokens += int(
            batch_result.get("max_tokens", 0)
        )

        batch_metadata.append(
            {
                "batch_number": batch_result.get("batch_number"),
                "claim_count": batch_result.get("claim_count"),
                "a_claim_count": len(claims_a),
                "b_claim_count": len(claims_b),
                "elapsed_s": batch_result.get("elapsed_s"),
                "attempts": batch_result.get("attempts"),
                "retried": batch_result.get("retried"),
                "finish_reason": batch_result.get("finish_reason"),
                "max_tokens": batch_result.get("max_tokens"),
            }
        )

        print(
            f"✓ Batch {batch_index + 1}/{total_batches} complete "
            f"in {batch_result['elapsed_s']:.2f}s."
        )

    judge_elapsed_s = time.perf_counter() - judge_started

    merged_payload = {
        "gemma4_only": {
            "claims": merged_a,
        },
        "gemma4_rag_mcp": {
            "claims": merged_b,
        },
    }

    _validate_batch_claim_ids(
        payload=merged_payload,
        expected_a=_expected_claim_ids(gemma_only_claims),
        expected_b=_expected_claim_ids(rag_mcp_claims),
    )

    print()
    print("✓ Granite batched comparative judge complete.")
    print(
        f"  Answer A returned : {len(merged_a)}/{len(gemma_only_claims)} claims"
    )
    print(
        f"  Answer B returned : {len(merged_b)}/{len(rag_mcp_claims)} claims"
    )
    print(
        f"  Total batches     : {total_batches}"
    )
    print(
        f"  Total judge time  : {judge_elapsed_s:.2f}s"
    )

    return {
        "result": last_result,
        "payload": merged_payload,
        "raw_text": "\n\n".join(raw_outputs),
        "elapsed_s": float(judge_elapsed_s),
        "max_tokens": int(total_requested_tokens),
        "total_claims": int(
            len(gemma_only_claims) + len(rag_mcp_claims)
        ),
        "attempts": int(total_attempts),
        "retried": bool(any_retry),
        "batch_size": int(GRANITE_CLAIM_BATCH_SIZE),
        "batch_count": int(total_batches),
        "batches": batch_metadata,
    }


# ============================================================

# ============================================================
# 4B.20 — SINGLE-ANSWER GRANITE RISK JUDGE
# ============================================================
#
# PURPOSE
# -------
# Evaluate ONE answer against ONE evidence bundle.
#
# This is the canonical deployment path for three-system risk
# evaluation. It removes the architectural workaround of
# sending an empty second comparison group.
#
# Preserved methodology:
# - same fixed claim extraction upstream
# - same evidence-relative status labels
# - same evidence-ID validation downstream
# - same risk weights / risk calculation downstream
# - same bounded batching / retry behavior
# ============================================================

SINGLE_RISK_JUDGE_SYSTEM_PROMPT = """
You are an independent evaluator of technical claims.

Evaluate every supplied claim using ONLY the supplied evidence.

Classify each claim as exactly one of:

supported:
The evidence supports the claim.

partially_supported:
The evidence supports part of the claim, but does not fully establish it.

unsupported:
The evidence does not establish the claim.

contradicted:
The evidence conflicts with the claim.

Rules:
- Do not use external knowledge.
- Do not treat general topic similarity as sufficient evidence.
- A reasonable inference from the supplied evidence may be accepted.
- If the evidence supports only part of a claim, use partially_supported.
- If the evidence is missing, use unsupported rather than contradicted.
- Before assigning a verdict, check whether another supplied evidence item
  supports the claim more directly than the first evidence item considered.
- Evaluate every supplied claim exactly once.
- Copy every claim_id exactly and unchanged.
- Evidence must contain only relevant E# identifiers from the supplied evidence.
- Give a short reason for each judgment.
- Do not calculate totals, scores, percentages or risk.
- Return JSON only.

Return exactly this structure:

{
  "claims": [
    {
      "claim_id": "A1",
      "status": "supported",
      "evidence": ["E1"],
      "reason": "Short evidence-based reason."
    }
  ]
}
""".strip()


def _parse_single_risk_output(raw_text):
    raw_text = str(raw_text or "").strip()

    if raw_text.startswith("```"):
        raw_text = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw_text,
            flags=re.IGNORECASE,
        )
        raw_text = re.sub(
            r"\s*```$",
            "",
            raw_text,
        )

    payload = json.loads(raw_text)

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Granite single-answer risk output must be a JSON object."
        )

    claims = payload.get("claims")

    if not isinstance(claims, list):
        raise RuntimeError(
            "Granite single-answer risk output must contain a claims list."
        )

    return {
        "claims": claims,
    }


def _validate_single_risk_claim_ids(
    *,
    payload,
    expected_ids,
):
    claims = payload.get("claims", [])

    returned_ids = [
        str(item.get("claim_id", "")).strip()
        for item in claims
        if isinstance(item, dict)
    ]

    if len(returned_ids) != len(set(returned_ids)):
        raise RuntimeError(
            "Granite single-answer batch returned duplicate claim IDs."
        )

    expected_set = set(expected_ids)
    returned_set = set(returned_ids)

    missing_ids = sorted(
        expected_set - returned_set
    )

    unexpected_ids = sorted(
        returned_set - expected_set
    )

    if missing_ids:
        raise RuntimeError(
            "Granite single-answer batch omitted required claim IDs: "
            + ", ".join(missing_ids)
        )

    if unexpected_ids:
        raise RuntimeError(
            "Granite single-answer batch invented unexpected claim IDs: "
            + ", ".join(unexpected_ids)
        )

    if len(returned_ids) != len(expected_ids):
        raise RuntimeError(
            "Granite single-answer batch claim count mismatch: "
            f"expected {len(expected_ids)}, returned {len(returned_ids)}."
        )


def _build_single_risk_batch_prompt(
    *,
    question,
    evidence_context,
    claims,
):
    ids = _expected_claim_ids(claims)

    return (
        f"QUESTION\n"
        f"--------\n"
        f"{question}\n\n"
        f"TECHNICAL EVIDENCE\n"
        f"------------------\n"
        f"{evidence_context}\n\n"
        f"FIXED CLAIMS — CURRENT BATCH ONLY\n"
        f"---------------------------------\n"
        f"{format_claims_for_judge(claims)}\n\n"
        f"REQUIRED CLAIM IDS — CURRENT BATCH\n"
        f"----------------------------------\n"
        f"{', '.join(ids)}\n\n"
        f"IMPORTANT BATCH RULES\n"
        f"---------------------\n"
        f"- Evaluate ONLY the claims listed in this batch.\n"
        f"- Include every required claim ID exactly once.\n"
        f"- Do not add any claim IDs.\n"
        f"- Return one JSON object containing only the claims list."
    )


def _run_single_risk_batch(
    *,
    question,
    evidence_context,
    claims,
    batch_number,
    total_batches,
    system_label="Answer",
):
    expected_ids = _expected_claim_ids(claims)

    judge_max_tokens = min(
        GRANITE_BATCH_MAX_TOKENS,
        max(
            GRANITE_BATCH_BASE_MAX_TOKENS,
            (
                GRANITE_BATCH_TOKENS_PER_CLAIM
                *
                len(claims)
            )
            + 500,
        ),
    )

    base_messages = [
        {
            "role": "system",
            "content": SINGLE_RISK_JUDGE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": _build_single_risk_batch_prompt(
                question=question,
                evidence_context=evidence_context,
                claims=claims,
            ),
        },
    ]

    started = time.perf_counter()
    last_exc = None
    last_raw_text = ""
    last_finish_reason = ""

    for attempt in range(
        1,
        GRANITE_BATCH_MAX_ATTEMPTS + 1,
    ):
        messages = list(base_messages)

        if attempt > 1:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response did not satisfy the required "
                        "single-answer JSON contract. Return ONLY one complete "
                        "JSON object with a top-level 'claims' list. Include "
                        "EVERY required claim ID from THIS BATCH exactly once "
                        "and no additional claim IDs. Each claim must contain "
                        "claim_id, status, evidence, and reason. Do not return "
                        "markdown, commentary, code fences, an empty object, "
                        "or a partial object."
                    ),
                }
            )

        result = call_granite_judge(
            messages=messages,
            max_tokens=judge_max_tokens,
        )

        raw_text = str(
            result.get("text", "")
            or ""
        ).strip()

        finish_reason = str(
            result.get("finish_reason", "")
            or ""
        ).strip().lower()

        last_raw_text = raw_text
        last_finish_reason = finish_reason

        try:
            payload = _parse_single_risk_output(
                raw_text
            )

            _validate_single_risk_claim_ids(
                payload=payload,
                expected_ids=expected_ids,
            )

            elapsed_s = (
                time.perf_counter()
                -
                started
            )

            if attempt > 1:
                print(
                    f"✓ {system_label} batch "
                    f"{batch_number}/{total_batches} recovered "
                    f"on attempt {attempt}/"
                    f"{GRANITE_BATCH_MAX_ATTEMPTS}."
                )

            return {
                "payload": payload,
                "result": result,
                "raw_text": raw_text,
                "elapsed_s": float(elapsed_s),
                "max_tokens": int(judge_max_tokens),
                "claim_count": int(len(claims)),
                "attempts": int(attempt),
                "retried": bool(attempt > 1),
                "batch_number": int(batch_number),
                "finish_reason": finish_reason,
            }

        except Exception as exc:
            last_exc = exc

            print()
            print("GRANITE SINGLE-ANSWER BATCH WARNING")
            print("-" * 116)
            print(
                f"System             : {system_label}"
            )
            print(
                f"Batch              : "
                f"{batch_number}/{total_batches}"
            )
            print(
                f"Attempt            : "
                f"{attempt}/{GRANITE_BATCH_MAX_ATTEMPTS}"
            )
            print(
                f"Claims in Batch    : {len(claims)}"
            )
            print(
                f"Finish Reason      : "
                f"{finish_reason or 'UNKNOWN'}"
            )
            print(
                f"Max Output Tokens  : "
                f"{judge_max_tokens}"
            )
            print(
                f"Returned Characters: "
                f"{len(raw_text):,}"
            )
            print(
                f"Validation Error   : "
                f"{type(exc).__name__}: {exc}"
            )

            if attempt < GRANITE_BATCH_MAX_ATTEMPTS:
                print()
                print(
                    f"⚠ Retrying {system_label} batch "
                    f"{batch_number}/{total_batches} once "
                    "with a corrective JSON-only instruction..."
                )

    raise RuntimeError(
        f"Granite single-answer risk batch for {system_label} "
        f"{batch_number}/{total_batches} failed after "
        f"{GRANITE_BATCH_MAX_ATTEMPTS} attempts. "
        f"Last finish reason: "
        f"{last_finish_reason or 'unknown'}. "
        f"Last returned characters: "
        f"{len(last_raw_text):,}."
    ) from last_exc


def run_granite_single_answer_judge(
    *,
    question,
    evidence_context,
    claims,
    system_label="Answer",
):
    """
    Evaluate one fixed claim list against one evidence bundle.

    Public payload:
        {
            "claims": [...]
        }

    Downstream risk scoring remains handled by
    validate_and_summarize_judgment().
    """

    claims = list(claims or [])

    if not claims:
        raise RuntimeError(
            "No claims supplied to Granite single-answer risk judge."
        )

    batches = _chunk_claims(
        claims
    )

    total_batches = len(batches)

    print()
    print(
        "GRANITE SINGLE-ANSWER RISK JUDGE "
        "— BATCHED EXECUTION"
    )
    print("-" * 116)
    print(
        f"System              : {system_label}"
    )
    print(
        f"Claims              : {len(claims)}"
    )
    print(
        f"Batch Size          : "
        f"{GRANITE_CLAIM_BATCH_SIZE}"
    )
    print(
        f"Total Batches       : {total_batches}"
    )

    judge_started = time.perf_counter()

    merged_claims = []
    batch_metadata = []
    raw_outputs = []
    last_result = None
    total_attempts = 0
    any_retry = False
    total_requested_tokens = 0

    for batch_index, claim_batch in enumerate(
        batches,
        start=1,
    ):
        print()
        print(
            f"Running {system_label} risk batch "
            f"{batch_index}/{total_batches} "
            f"({len(claim_batch)} claims)..."
        )

        batch_result = _run_single_risk_batch(
            question=question,
            evidence_context=evidence_context,
            claims=claim_batch,
            batch_number=batch_index,
            total_batches=total_batches,
            system_label=system_label,
        )

        merged_claims.extend(
            batch_result[
                "payload"
            ][
                "claims"
            ]
        )

        last_result = batch_result.get(
            "result"
        )

        raw_outputs.append(
            batch_result.get(
                "raw_text",
                "",
            )
        )

        total_attempts += int(
            batch_result.get(
                "attempts",
                1,
            )
        )

        any_retry = (
            any_retry
            or
            bool(
                batch_result.get(
                    "retried",
                    False,
                )
            )
        )

        total_requested_tokens += int(
            batch_result.get(
                "max_tokens",
                0,
            )
        )

        batch_metadata.append(
            {
                "batch_number":
                    batch_result.get(
                        "batch_number"
                    ),
                "claim_count":
                    batch_result.get(
                        "claim_count"
                    ),
                "elapsed_s":
                    batch_result.get(
                        "elapsed_s"
                    ),
                "attempts":
                    batch_result.get(
                        "attempts"
                    ),
                "retried":
                    batch_result.get(
                        "retried"
                    ),
                "finish_reason":
                    batch_result.get(
                        "finish_reason"
                    ),
                "max_tokens":
                    batch_result.get(
                        "max_tokens"
                    ),
            }
        )

        print(
            f"✓ {system_label} batch "
            f"{batch_index}/{total_batches} complete "
            f"in {batch_result['elapsed_s']:.2f}s."
        )

    elapsed_s = (
        time.perf_counter()
        -
        judge_started
    )

    merged_payload = {
        "claims":
            merged_claims,
    }

    _validate_single_risk_claim_ids(
        payload=merged_payload,
        expected_ids=_expected_claim_ids(
            claims
        ),
    )

    print()
    print(
        f"✓ Granite single-answer risk judge "
        f"complete — {system_label}."
    )
    print(
        f"  Returned claims : "
        f"{len(merged_claims)}/{len(claims)}"
    )
    print(
        f"  Total batches   : {total_batches}"
    )
    print(
        f"  Total judge time: {elapsed_s:.2f}s"
    )

    return {
        "result":
            last_result,
        "payload":
            merged_payload,
        "raw_text":
            "\n\n".join(
                raw_outputs
            ),
        "elapsed_s":
            float(
                elapsed_s
            ),
        "max_tokens":
            int(
                total_requested_tokens
            ),
        "total_claims":
            int(
                len(claims)
            ),
        "attempts":
            int(
                total_attempts
            ),
        "retried":
            bool(
                any_retry
            ),
        "batch_size":
            int(
                GRANITE_CLAIM_BATCH_SIZE
            ),
        "batch_count":
            int(
                total_batches
            ),
        "batches":
            batch_metadata,
        "system_label":
            str(
                system_label
            ),
    }


# ============================================================
# 4C.6Y.A — MODULE-4-ALIGNED LLM-SELECTED RETRIEVAL ORCHESTRATION
# ============================================================
#
# PURPOSE
# -------
# Restore the complete proven Module 4 natural-routing pattern:
#
#     Grounded Question
#          ↓
#     SAME LLM — Retrieval Planner
#          ├─ selects MINIMUM retrieval architecture
#          │      RAG_ONLY / MCP_ONLY / HYBRID
#          ├─ formulates initial retrieval query
#          └─ derives fixed answer requirements
#          ↓
#     Python executes ONLY the selected architecture
#          ↓
#     deterministic evidence-security scan
#          ↓
#     SAME LLM — Requirement-Bounded Sufficiency Assessor
#          ├─ sufficient → stop
#          └─ insufficient → formulate ONE refined next_query
#                                  ↓
#                    SAME retrieval architecture, next round
#          ↓
#     Final bounded evidence
#          ↓
#     EXISTING frozen grounded-answer generator
#          ↓
#     EXISTING frozen Granite comparative judge
#
# RESPONSIBILITY SPLIT
# --------------------
# LLM:
#   - selects RAG_ONLY / MCP_ONLY / HYBRID
#   - formulates retrieval query
#   - derives explicit answer requirements
#   - assesses evidence sufficiency
#   - formulates refined query when a requirement remains unsupported
#
# Python:
#   - validates the LLM-selected architecture
#   - executes only that architecture
#   - keeps source-family / shard / specification routing internal to MCP
#   - enforces maximum 3 retrieval rounds
#   - scans evidence for prompt injection
#   - accumulates / deduplicates evidence across rounds
#   - enforces maximum 5 final evidence items / context budget
#
# Native OpenAI-style tool_calls are NOT used.
# No retrieval route is hard-coded.
# ============================================================





# ============================================================
# DEPLOYMENT EXTENSION — ANSWER RELEVANCE JUDGE
# ============================================================
#
# This is intentionally separate from the frozen Notebook 21
# hallucination-risk judge.
#
# Hallucination risk:
#   measures evidence support.
#
# Answer relevance:
#   measures how directly and adequately the answer addresses
#   the user's actual question.
#
# Relevance does NOT judge factual correctness or evidence support.
# ============================================================

RELEVANCE_JUDGE_MAX_TOKENS = 900
RELEVANCE_JUDGE_ATTEMPTS = 2

RELEVANCE_JUDGE_SYSTEM_PROMPT = """
You are an independent evaluator comparing two answers to the same user question.

Evaluate ANSWER A and ANSWER B for ANSWER RELEVANCE only.

Answer relevance means how directly and adequately the answer addresses what
the user actually asked.

Evaluate each answer using these dimensions:
- Directness: does it answer the requested question rather than drifting?
- Coverage: does it address the important parts of the request?
- Specificity: does it provide useful detail appropriate to the question?
- Usefulness: would the answer help the user act, decide, or understand?
- Focus: does it avoid unnecessary or off-topic material?

Important rules:
- Do NOT judge factual correctness.
- Do NOT judge hallucination or evidence support.
- Do NOT reward an answer merely because it contains citations.
- Do NOT penalize an answer merely because it is ungrounded.
- Evaluate both answers against the USER QUESTION using the same standard.
- Scores must be integers from 0 to 100.
- Give a concise rationale for each score.
- Return JSON only.

Return exactly this structure:

{
  "answer_a": {
    "score": 0,
    "rationale": "Concise relevance rationale."
  },
  "answer_b": {
    "score": 0,
    "rationale": "Concise relevance rationale."
  },
  "comparison_summary": "Concise comparison of which answer is more relevant and why."
}
""".strip()


def parse_relevance_judge_output(raw_text: str) -> dict[str, Any]:
    raw_text = str(raw_text or "").strip()

    if raw_text.startswith("```"):
        raw_text = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw_text,
            flags=re.IGNORECASE,
        )
        raw_text = re.sub(
            r"\s*```$",
            "",
            raw_text,
        )

    payload = json.loads(raw_text)

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Granite relevance output must be a JSON object."
        )

    for key in ("answer_a", "answer_b"):
        group = payload.get(key)

        if not isinstance(group, dict):
            raise RuntimeError(
                f"Granite relevance output missing {key}."
            )

        score = group.get("score")

        if isinstance(score, bool) or not isinstance(
            score,
            (int, float),
        ):
            raise RuntimeError(
                f"Granite relevance score for {key} must be numeric."
            )

        score = int(round(float(score)))

        if score < 0 or score > 100:
            raise RuntimeError(
                f"Granite relevance score for {key} must be 0-100."
            )

        rationale = str(
            group.get("rationale", "")
        ).strip()

        if not rationale:
            raise RuntimeError(
                f"Granite relevance rationale missing for {key}."
            )

        group["score"] = score
        group["rationale"] = rationale

    payload["comparison_summary"] = str(
        payload.get("comparison_summary", "")
    ).strip()

    return payload


def run_granite_relevance_judge(
    question: str,
    baseline_answer: str,
    grounded_answer: str,
) -> dict[str, Any]:
    user_prompt = (
        f"USER QUESTION\n"
        f"-------------\n"
        f"{question}\n\n"
        f"ANSWER A — {GEMMA_ONLY_LABEL}\n"
        f"----------------------------\n"
        f"{baseline_answer}\n\n"
        f"ANSWER B — {GEMMA_RAG_MCP_LABEL}\n"
        f"---------------------------------\n"
        f"{grounded_answer}"
    )

    messages = [
        {
            "role": "system",
            "content": RELEVANCE_JUDGE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    started = time.perf_counter()
    last_exc = None
    last_finish_reason = ""

    for attempt in range(
        1,
        RELEVANCE_JUDGE_ATTEMPTS + 1,
    ):
        result = call_granite_judge(
            messages=messages,
            max_tokens=RELEVANCE_JUDGE_MAX_TOKENS,
        )

        raw_text = str(
            result.get("text", "")
        ).strip()

        last_finish_reason = str(
            result.get("finish_reason", "")
        ).strip()

        try:
            payload = parse_relevance_judge_output(
                raw_text
            )

            return {
                "payload": payload,
                "result": result,
                "elapsed_s": (
                    time.perf_counter()
                    -
                    started
                ),
                "attempts": attempt,
                "retried": attempt > 1,
            }

        except Exception as exc:
            last_exc = exc

            if attempt < RELEVANCE_JUDGE_ATTEMPTS:
                continue

    raise RuntimeError(
        "Could not parse Granite relevance output after "
        f"{RELEVANCE_JUDGE_ATTEMPTS} attempts. "
        f"Last finish reason: {last_finish_reason or 'unknown'}."
    ) from last_exc


def relevance_score_label(score: float) -> str:
    score = float(score)

    if score >= 90:
        return "EXCELLENT"

    if score >= 80:
        return "HIGH"

    if score >= 60:
        return "MODERATE"

    if score >= 40:
        return "LOW"

    return "VERY LOW"


def evaluate_connected_comparison(
    question: str,
    baseline_answer: str,
    grounded_answer: str,
    evidence_context: str,
    evidence_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Deployment wrapper around the frozen Notebook 21 comparative judge
    plus a separate deployment relevance judge.

    Hallucination risk is evidence-relative:
    unsupported does NOT mean a claim is factually false; it means the
    supplied connected-corpus evidence did not establish the claim.

    Answer relevance is question-relative:
    it measures how directly and adequately each answer addresses the
    user's request. It does not judge factual correctness.
    """

    baseline_extraction = extract_refined_claims(
        baseline_answer,
        "A",
    )

    grounded_extraction = extract_refined_claims(
        grounded_answer,
        "B",
    )

    baseline_claims = baseline_extraction["claims"]
    grounded_claims = grounded_extraction["claims"]

    valid_evidence_ids = build_valid_evidence_ids(
        evidence_items
    )

    evaluation_started = time.perf_counter()

    # Run both independent Granite evaluations concurrently so the
    # relevance metric does not simply add its full latency to the
    # hallucination-risk judge.
    with ThreadPoolExecutor(max_workers=2) as executor:
        risk_future = executor.submit(
            run_granite_comparative_judge,
            question,
            evidence_context,
            baseline_claims,
            grounded_claims,
        )

        relevance_future = executor.submit(
            run_granite_relevance_judge,
            question,
            baseline_answer,
            grounded_answer,
        )

        execution = risk_future.result()
        relevance_execution = relevance_future.result()

    evaluation_wall_s = (
        time.perf_counter()
        -
        evaluation_started
    )

    baseline_assessment = validate_and_summarize_judgment(
        original_claims=baseline_claims,
        judge_group_payload=execution["payload"]["gemma4_only"],
        valid_evidence_ids=valid_evidence_ids,
    )

    grounded_assessment = validate_and_summarize_judgment(
        original_claims=grounded_claims,
        judge_group_payload=execution["payload"]["gemma4_rag_mcp"],
        valid_evidence_ids=valid_evidence_ids,
    )

    baseline_risk = float(
        baseline_assessment["estimated_hallucination_risk_pct"]
    )

    grounded_risk = float(
        grounded_assessment["estimated_hallucination_risk_pct"]
    )

    relevance_payload = relevance_execution["payload"]

    baseline_relevance = int(
        relevance_payload["answer_a"]["score"]
    )

    grounded_relevance = int(
        relevance_payload["answer_b"]["score"]
    )

    return {
        "applicable": True,
        "definition": RISK_DEFINITION,
        "relevance_definition": (
            "Answer Relevance measures how directly and adequately the "
            "response addresses the user's question. It is separate from "
            "factual correctness and evidence support."
        ),
        "baseline": {
            "label": GEMMA_ONLY_LABEL,
            "claim_extraction": baseline_extraction,
            "assessment": baseline_assessment,
            "risk_pct": baseline_risk,
            "risk_label": hallucination_risk_label(
                baseline_risk
            ),
            "relevance_score": baseline_relevance,
            "relevance_label": relevance_score_label(
                baseline_relevance
            ),
            "relevance_rationale": relevance_payload[
                "answer_a"
            ]["rationale"],
        },
        "grounded": {
            "label": GEMMA_RAG_MCP_LABEL,
            "claim_extraction": grounded_extraction,
            "assessment": grounded_assessment,
            "risk_pct": grounded_risk,
            "risk_label": hallucination_risk_label(
                grounded_risk
            ),
            "relevance_score": grounded_relevance,
            "relevance_label": relevance_score_label(
                grounded_relevance
            ),
            "relevance_rationale": relevance_payload[
                "answer_b"
            ]["rationale"],
        },
        "risk_difference_pct": (
            baseline_risk
            -
            grounded_risk
        ),
        "relevance_difference_points": (
            grounded_relevance
            -
            baseline_relevance
        ),
        "relevance_comparison_summary": relevance_payload.get(
            "comparison_summary",
            "",
        ),
        "judge": {
            "provider": JUDGE_PROVIDER,
            "model": JUDGE_MODEL,
            "display_name": JUDGE_DISPLAY_NAME,
            "reasoning_effort": JUDGE_REASONING_EFFORT,
            "elapsed_s": evaluation_wall_s,
            "risk_elapsed_s": execution["elapsed_s"],
            "relevance_elapsed_s": relevance_execution["elapsed_s"],
            "total_claims": execution["total_claims"],
            "max_tokens": execution["max_tokens"],
            "attempts": execution["attempts"],
            "retried": execution["retried"],
            "finish_reason": execution["result"].get(
                "finish_reason",
                "",
            ),
            "prompt_tokens": execution["result"].get(
                "prompt_tokens",
                0,
            ),
            "completion_tokens": execution["result"].get(
                "completion_tokens",
                0,
            ),
            "total_tokens": execution["result"].get(
                "total_tokens",
                0,
            ),
            "relevance_attempts": relevance_execution[
                "attempts"
            ],
            "relevance_retried": relevance_execution[
                "retried"
            ],
            "relevance_finish_reason": relevance_execution[
                "result"
            ].get(
                "finish_reason",
                "",
            ),
        },
    }


def evaluation_not_applicable(reason: str) -> dict[str, Any]:
    return {
        "applicable": False,
        "reason": str(reason),
        "baseline": None,
        "grounded": None,
        "risk_difference_pct": None,
        "relevance_difference_points": None,
        "relevance_comparison_summary": None,
        "definition": RISK_DEFINITION,
        "relevance_definition": (
            "Answer Relevance measures how directly and adequately the "
            "response addresses the user's question."
        ),
    }


def get_evaluation_status() -> dict[str, Any]:
    return {
        "runtime": "Notebook 21 frozen comparative evaluation",
        "scope": "CONNECTED_CORPUS_ONLY",
        "judge_model": JUDGE_MODEL,
        "judge_reasoning_effort": JUDGE_REASONING_EFFORT,
        "judge_retry_limit": 2,
        "risk_weights": dict(HALLUCINATION_RISK_WEIGHTS),
        "risk_bands": {
            "LOW": "<=20%",
            "MODERATE": ">20% and <=40%",
            "ELEVATED": ">40% and <=60%",
            "HIGH": ">60% and <=80%",
            "VERY HIGH": ">80%",
        },
        "risk_definition": RISK_DEFINITION,
        "relevance_evaluation": {
            "enabled": True,
            "scope": "CONNECTED_CORPUS_ONLY",
            "score_range": "0-100",
            "dimensions": [
                "directness",
                "coverage",
                "specificity",
                "usefulness",
                "focus",
            ],
            "correctness_is_not_part_of_relevance": True,
            "runs_in_parallel_with_risk_judge": True,
        },
    }
