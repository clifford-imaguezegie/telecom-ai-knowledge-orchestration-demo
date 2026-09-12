from __future__ import annotations

import re
import time
from typing import Any


# ============================================================
# TELECOM AI KNOWLEDGE ORCHESTRATION — SECURITY LAYER
# ============================================================

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?previous\s+instructions",
    r"override\s+(the\s+)?system\s+prompt",
    r"override\s+(the\s+)?developer\s+prompt",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"show\s+(the\s+)?system\s+prompt",
    r"print\s+(the\s+)?system\s+prompt",
    r"reveal\s+(the\s+)?developer\s+prompt",
    r"show\s+(the\s+)?developer\s+prompt",
    r"print\s+(the\s+)?developer\s+prompt",
    r"you\s+are\s+now",
    r"act\s+as\s+.*\s+instead",
    r"forget\s+(all\s+)?previous\s+instructions",
]

EVIDENCE_MANIPULATION_PATTERNS = [
    r"ignore\s+(the\s+)?evidence",
    r"ignore\s+(the\s+)?context",
    r"do\s+not\s+cite",
    r"fabricate\s+(a\s+)?citation",
    r"invent\s+(a\s+)?citation",
    r"pretend\s+(the\s+)?evidence",
    r"claim\s+(the\s+)?evidence\s+says",
    r"change\s+(the\s+)?evidence",
    r"override\s+(the\s+)?evidence",
]


def _find_matches(
    text: str,
    patterns: list[str],
) -> list[str]:
    """
    Return matched security patterns.
    """

    matches: list[str] = []

    for pattern in patterns:
        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            matches.append(pattern)

    return matches


def get_injection_matches(
    text: str,
) -> list[str]:
    """
    Detect prompt-injection and evidence-manipulation patterns.
    """

    all_patterns = (
        INJECTION_PATTERNS
        +
        EVIDENCE_MANIPULATION_PATTERNS
    )

    return _find_matches(
        text,
        all_patterns,
    )


def scan_for_injection(
    text: str,
) -> bool:
    """
    Return True when prompt-injection patterns are detected.
    """

    return bool(
        _find_matches(
            text,
            INJECTION_PATTERNS,
        )
    )


def scan_for_evidence_manipulation(
    text: str,
) -> bool:
    """
    Return True when evidence-manipulation patterns are detected.
    """

    return bool(
        _find_matches(
            text,
            EVIDENCE_MANIPULATION_PATTERNS,
        )
    )


def preprocess_user_prompt(
    prompt: str,
) -> dict[str, Any]:
    """
    Create trusted sanitized intent before downstream
    LLM or knowledge-access activity.
    """

    start = time.perf_counter()

    raw_prompt = str(
        prompt or ""
    ).strip()

    injection_detected = scan_for_injection(
        raw_prompt
    )

    evidence_manipulation_detected = (
        scan_for_evidence_manipulation(
            raw_prompt
        )
    )

    matched_patterns = get_injection_matches(
        raw_prompt
    )

    sanitized_prompt = raw_prompt

    for pattern in matched_patterns:
        sanitized_prompt = re.sub(
            pattern,
            " ",
            sanitized_prompt,
            flags=re.IGNORECASE,
        )

    # Normalize whitespace after removing matched patterns.
    sanitized_prompt = re.sub(
        r"\s+",
        " ",
        sanitized_prompt,
    ).strip()

    # Remove orphaned conjunctions that can remain after
    # malicious instruction fragments are stripped.
    sanitized_prompt = re.sub(
        r"^(and|or|but)\b[\s,.;:-]*",
        "",
        sanitized_prompt,
        flags=re.IGNORECASE,
    )

    # Remove whitespace immediately before punctuation.
    sanitized_prompt = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        sanitized_prompt,
    )

    # Remove leading punctuation left after sanitization.
    sanitized_prompt = re.sub(
        r"^[\s,.;:-]+",
        "",
        sanitized_prompt,
    )

    sanitized_prompt = sanitized_prompt.strip()

    elapsed_s = (
        time.perf_counter()
        -
        start
    )

    return {
        "raw_prompt": raw_prompt,
        "sanitized_prompt": sanitized_prompt,
        "injection_detected": injection_detected,
        "evidence_manipulation_detected":
            evidence_manipulation_detected,
        "matched_patterns": matched_patterns,
        "elapsed_s": elapsed_s,
    }


def validate_evidence_text(
    text: str,
) -> dict[str, Any]:
    """
    Scan retrieved evidence before it reaches the generator.
    """

    injection_detected = scan_for_injection(
        text
    )

    evidence_manipulation_detected = (
        scan_for_evidence_manipulation(
            text
        )
    )

    return {
        "safe": not (
            injection_detected
            or
            evidence_manipulation_detected
        ),
        "injection_detected":
            injection_detected,
        "evidence_manipulation_detected":
            evidence_manipulation_detected,
        "matched_patterns":
            get_injection_matches(text),
    }