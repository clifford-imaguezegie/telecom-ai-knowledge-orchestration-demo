from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import streamlit as st

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
from runtime.evaluation_threeway import (
    evaluate_three_way_telecom,
    run_three_way_quality_judge,
)


# ============================================================
# PRESENTATION CONSTANTS
# ============================================================

NON_TELECOM_WARNING = (
    "⚠ Outside Telecom Knowledge Domain: this response is not grounded "
    "in the connected Telecom RAG/MCP knowledge sources."
)

KNOWLEDGE_BASE_DISCLAIMER = (
    "⚠ Knowledge-base limitation: this response is grounded in the connected "
    "Telecom knowledge sources. Its coverage and completeness are therefore "
    "limited by the information available in those sources."
)


LLM_ONLY_WARNING = (
    "⚠ Pretrained-knowledge response: this answer is generated from the model's "
    "pretrained knowledge without external grounding. It may contain outdated, "
    "unsupported, or hallucinated information."
)

AUGMENTED_REASONING_WARNING = (
    "⚠ Augmented-reasoning response: this answer combines retrieved evidence "
    "with the model's pretrained knowledge and synthesis. Although evidence is "
    "used to improve grounding and coverage, some statements may extend beyond "
    "the retrieved sources and may still be unsupported or hallucinated."
)

SYSTEM_KEYS = ("llm_only", "strict", "augmented")

SYSTEM_LABELS = {
    "llm_only": "Gemma LLM-Only",
    "strict": "Strict Knowledge Grounding",
    "augmented": "Augmented Reasoning",
}

SYSTEM_OBJECTIVES = {
    "llm_only": "Maximum native model reasoning",
    "strict": "Maximum evidence discipline",
    "augmented": "Maximum useful synthesis",
}


# ============================================================
# SMALL HELPERS
# ============================================================

def _answer_from_result(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("answer", "") or "")
    return str(result or "")


def _format_seconds(value: float | None, pending: str = "Pending…") -> str:
    if value is None:
        return pending
    return f"{float(value):.1f}s"


def _risk_badge(label: str | None) -> str:
    return str(label or "N/A").upper()


def _score_label(score: int | float) -> str:
    value = float(score)
    if value >= 90:
        return "EXCELLENT"
    if value >= 80:
        return "STRONG"
    if value >= 70:
        return "MODERATE"
    if value >= 60:
        return "LIMITED"
    return "WEAK"


def _render_answer_card(
    slot,
    *,
    title: str,
    objective: str,
    answer: str | None,
    mode: str | None = None,
    sources: list[str] | None = None,
    policy: str | None = None,
    warning: str | None = None,
):
    with slot.container():
        st.subheader(title)
        st.caption(f"Objective: {objective}")

        if mode:
            st.markdown(f"**Mode:** `{mode}`")

        if sources:
            st.markdown("**Sources used:** " + " · ".join(sources))

        if policy:
            st.caption(f"Policy: {policy}")

        if warning:
            st.warning(warning)

        if answer:
            st.markdown(answer)
        else:
            st.info("Waiting for response…")


def _render_path(
    slot,
    *,
    title: str,
    objective: str,
    mode: str | None,
    sources: list[str] | None,
    stages: list[dict[str, Any]] | None,
    pending_message: str | None = None,
):
    with slot.container():
        st.markdown(f"### {title}")
        st.caption(objective)

        if mode:
            st.markdown(f"**Mode:** `{mode}`")

        if sources:
            st.markdown("**Sources:** " + " · ".join(sources))

        if not stages:
            st.info(pending_message or "Waiting for orchestration trace…")
            return

        for idx, stage in enumerate(stages, start=1):
            name = str(stage.get("stage", f"Stage {idx}"))
            status = str(stage.get("status", "completed"))
            elapsed = stage.get("elapsed_s")

            if elapsed is None:
                st.write(f"{idx}. **{name}** — {status}")
            else:
                st.write(
                    f"{idx}. **{name}** — {status} · "
                    f"{float(elapsed):.2f}s"
                )


def _compact_value(value: Any, max_chars: int = 180) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _clean_heading_candidate(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^[#>\-\*\s]+", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_storage_like_title(value: str) -> bool:
    text = str(value or "").strip()
    lower = text.lower()

    if not text:
        return True

    generic = {
        "raw",
        "raw.md",
        "document",
        "source",
        "untitled source",
        "01-genodebs",
        "01-genodebs.md",
    }

    if lower in generic:
        return True

    # Path-ish / artifact-ish names.
    if "/" in text or "\\" in text:
        return True

    if re.fullmatch(r"\d{6,8}", text):
        return True

    if re.fullmatch(r"(ts|tr|es|en|gr|gs|spec)[_\-.]?[a-z0-9_.-]+", lower):
        return True

    if lower.endswith((".md", ".txt", ".json", ".parquet")):
        return True

    return False


def _extract_document_heading(item: dict[str, Any]) -> str:
    """
    Recover a human-readable document/paper/spec title from the retrieved
    evidence text when the normalized title is only a storage filename.
    """
    text = str(item.get("text", "") or "").strip()

    if not text:
        return ""

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # 1. Prefer Markdown headings.
    for line in lines[:40]:
        if re.match(r"^#{1,6}\s+\S", line):
            candidate = _clean_heading_candidate(line)

            if (
                8 <= len(candidate) <= 220
                and not _is_storage_like_title(candidate)
            ):
                return candidate

    # 2. Standards documents often start with a formal title line without #.
    title_signals = (
        "technical specification",
        "technical report",
        "specification",
        "architecture",
        "requirements",
        "radio access",
        "radio network",
        "network",
        "protocol",
        "procedures",
        "management",
        "study on",
        "system",
        "security",
        "interface",
        "open radio access",
        "o-ran",
        "oran",
        "3gpp",
        "etsi",
        "ietf",
        "rfc ",
        "itu-t",
        "gsma",
    )

    for line in lines[:30]:
        candidate = _clean_heading_candidate(line)

        if not (8 <= len(candidate) <= 220):
            continue

        lower = candidate.lower()

        if any(signal in lower for signal in title_signals):
            # Avoid table rows / boilerplate.
            if candidate.count("|") <= 2:
                return candidate

    # 3. Last-resort first prose-like line.
    for line in lines[:20]:
        candidate = _clean_heading_candidate(line)

        if (
            12 <= len(candidate) <= 180
            and candidate.count("|") <= 1
            and not candidate.startswith(("http://", "https://"))
            and not _is_storage_like_title(candidate)
        ):
            return candidate

    return ""


def _infer_display_source(item: dict[str, Any]) -> str:
    """
    Infer a human-readable source label from retrieval provenance.

    IMPORTANT:
    - Retrieval provenance is authoritative.
    - Do not classify a RAG/MCP item as Web merely because its evidence
      text contains a URL.
    - Do not classify a Web item as 3GPP/ETSI merely because the page
      text mentions a standards body.
    - This affects presentation only; retrieval/ranking are unchanged.
    """

    retrieval_system = str(
        item.get("retrieval_system", "") or ""
    ).strip().upper()

    retrieval_systems = {
        str(value).strip().upper()
        for value in (item.get("retrieval_systems", []) or [])
        if str(value).strip()
    }

    provenance = item.get("provenance", []) or []

    # --------------------------------------------------------
    # 1. Retrieval system is the strongest source signal.
    # --------------------------------------------------------

    if (
        retrieval_system == "LIVE_WEB"
        or "LIVE_WEB" in retrieval_systems
    ):
        return "Web"

    for prov in provenance:
        if not isinstance(prov, dict):
            continue

        prov_system = str(
            prov.get("retrieval_system", "") or ""
        ).strip().upper()

        provider = str(
            prov.get("provider", "") or ""
        ).strip().lower()

        if (
            prov_system == "LIVE_WEB"
            or "open-websearch" in provider
            or "open_websearch" in provider
        ):
            return "Web"


    # --------------------------------------------------------
    # 2. Prefer explicit source-family metadata.
    # --------------------------------------------------------

    explicit_families = []

    source_family = str(
        item.get("source_family", "") or ""
    ).strip()

    if source_family:
        explicit_families.append(source_family)

    for prov in provenance:
        if not isinstance(prov, dict):
            continue

        value = str(
            prov.get("source_family", "") or ""
        ).strip()

        if value:
            explicit_families.append(value)

    for family in explicit_families:

        normalized = family.upper()

        if normalized == "TCC":
            return "GSMA TCC"

        if normalized in {
            "3GPP",
            "ETSI",
            "O-RAN",
            "IETF",
            "ITU-T",
            "GSMA",
            "CAMARA",
        }:
            return normalized

        if family.lower() in {
            "open_source",
            "rag_corpus",
        }:
            # Keep evaluating metadata paths below so an open-source
            # standards document can still be labelled accurately.
            continue

        if family.lower() not in {
            "unknown",
            "mcp",
            "rag",
        }:
            return family


    # --------------------------------------------------------
    # 3. Infer from metadata/provenance ONLY.
    #    Deliberately exclude evidence text and raw URLs.
    # --------------------------------------------------------

    metadata_parts = [
        str(item.get("document_id", "") or ""),
        str(item.get("title", "") or ""),
    ]

    for prov in provenance:
        if not isinstance(prov, dict):
            continue

        for key in (
            "collection",
            "identifier",
            "source_path",
            "file_path",
            "source",
        ):
            value = prov.get(key)

            if value:
                metadata_parts.append(
                    str(value)
                )

    blob = " ".join(
        metadata_parts
    ).lower()


    source_rules = [
        (
            (
                "3gpp",
                "3gpp ts",
                "3gpp tr",
                "3gpp_rel",
                "/23.",
                "/24.",
                "/28.",
                "/29.",
                "/36.",
                "/37.",
                "/38.",
            ),
            "3GPP",
        ),
        (
            (
                "o-ran",
                "o_ran",
                "oran.",
                "oran ",
                "wg4.ts",
                "wg1.ts",
                "wg2.ts",
                "wg3.ts",
            ),
            "O-RAN",
        ),
        (
            (
                "ietf",
                "rfc ",
                "rfc-",
                "/rfc",
            ),
            "IETF",
        ),
        (
            (
                "etsi",
                "/etsi/",
                "tr_",
                "ts_",
                "es_",
                "etsi tr",
                "etsi ts",
            ),
            "ETSI",
        ),
        (
            (
                "itu-t",
                "itu t",
                "itu recommendation",
            ),
            "ITU-T",
        ),
        (
            (
                "telco common corpus",
                "wikipedia-telecom",
                "wikidata-telecom",
                "ieee-access",
                "openalex",
                "uspto",
                "epo",
                "3gpp-tsg",
            ),
            "GSMA TCC",
        ),
        (
            (
                "camara",
            ),
            "CAMARA",
        ),
        (
            (
                "tm forum",
                "tmforum",
            ),
            "TM Forum",
        ),
    ]


    for signals, label in source_rules:

        if any(
            signal in blob
            for signal in signals
        ):
            return label


    # --------------------------------------------------------
    # 4. Retrieval-system-aware generic fallback.
    # --------------------------------------------------------

    if (
        retrieval_system == "MCP"
        or "MCP" in retrieval_systems
    ):
        return "MCP Telecom Knowledge"

    return "Open-source Telecom Corpus"


def _source_display_name(item: dict[str, Any]) -> str:
    """
    Display the actual document/paper/spec title when possible, not the
    local storage filename.
    """
    title = str(item.get("title", "") or "").strip()

    # Web/MCP titles are often already human-readable.
    if title and not _is_storage_like_title(title):
        return title

    # RAG chunks often carry storage filenames; recover the first useful
    # heading from the actual evidence text.
    heading = _extract_document_heading(item)
    if heading:
        return heading

    # MCP may provide section or identifier metadata.
    provenance = item.get("provenance", []) or []

    for prov in provenance:
        if not isinstance(prov, dict):
            continue

        section_heading = _clean_heading_candidate(
            prov.get("section_heading", "")
        )
        identifier = str(
            prov.get("identifier", "") or ""
        ).strip()

        if section_heading and not _is_storage_like_title(section_heading):
            if identifier:
                return f"{identifier} — {section_heading}"
            return section_heading

        if identifier and not _is_storage_like_title(identifier):
            return identifier

    # Final fallback: show a concise document/spec identifier rather than raw.md.
    document_id = str(
        item.get("document_id", "") or ""
    ).strip()

    if document_id:
        normalized = document_id.replace("\\", "/").rstrip("/")
        bits = [bit for bit in normalized.split("/") if bit]

        for candidate in reversed(bits):
            if candidate.lower() not in {"raw", "raw.md", "marked"}:
                return candidate

    return "Telecom technical source"



def _clean_reference_identifier(value: str) -> str:
    """
    Convert machine-style identifiers into a concise checkable reference.
    Examples:
      STD:3GPP:TS_37.141 -> 3GPP TS 37.141
      ts_38104v190000p   -> TS 38.104
    """
    value = str(value or "").strip()

    if not value:
        return ""

    cleaned = (
        value
        .replace("\\:", ":")
        .replace("\\_", "_")
        .replace("STD:3GPP:TS_", "3GPP TS ")
        .replace("STD:3GPP:TR_", "3GPP TR ")
        .replace("STD:ETSI:TS_", "ETSI TS ")
        .replace("STD:ETSI:TR_", "ETSI TR ")
    )

    # Common ETSI/3GPP compact storage identifier.
    match = re.search(
        r"\b(?:ts|tr)[_-]?(\d{2})(\d{3})",
        cleaned,
        flags=re.IGNORECASE,
    )

    if match:
        prefix = "TS" if cleaned.lower().startswith("ts") else "TR"
        cleaned = f"{prefix} {match.group(1)}.{match.group(2)}"

    return cleaned.strip()


def _source_reference_topic(item: dict[str, Any]) -> str:
    """
    Build a human-readable, checkable reference label from all available
    provenance rather than trusting one title field.
    """

    provenance = item.get("provenance", []) or []

    identifiers = []
    sections = []

    for prov in provenance:
        if not isinstance(prov, dict):
            continue

        identifier = _clean_reference_identifier(
            prov.get("identifier", "")
        )

        if identifier and identifier not in identifiers:
            identifiers.append(identifier)

        section = _clean_heading_candidate(
            prov.get("section_heading", "")
        )

        if (
            section
            and not _is_storage_like_title(section)
            and section not in sections
        ):
            sections.append(section)

    title = str(
        item.get("title", "") or ""
    ).strip()

    if _is_storage_like_title(title):
        title = ""

    document_id = _clean_reference_identifier(
        item.get("document_id", "")
    )

    recovered_heading = _extract_document_heading(item)

    candidates = []

    for value in identifiers:
        if value and value not in candidates:
            candidates.append(value)

    if title and title not in candidates:
        candidates.append(title)

    for value in sections:
        if value and value not in candidates:
            candidates.append(value)

    if (
        recovered_heading
        and not _is_storage_like_title(recovered_heading)
        and recovered_heading not in candidates
    ):
        candidates.append(recovered_heading)

    if document_id and document_id not in candidates:
        candidates.append(document_id)

    # Keep the display concise but useful.
    return " · ".join(candidates[:3]) or "Reference topic unavailable"


def _render_evidence_group(
    container,
    title: str,
    evidence: list[dict[str, Any]],
):
    with container:
        st.markdown(f"### {title}")

        if not evidence:
            st.info("No external evidence items were returned.")
            return

        for position, item in enumerate(evidence, start=1):
            evidence_id = str(item.get("evidence_id", "") or "").strip()
            source_family = _infer_display_source(item)
            retrieval_systems = [
                str(x)
                for x in (item.get("retrieval_systems", []) or [])
            ]
            document_id = str(
                item.get("document_id", "") or ""
            ).strip()
            chunk_id = str(
                item.get("chunk_id", "") or ""
            ).strip()
            url = str(item.get("url", "") or "").strip()
            text = str(item.get("text", "") or "").strip()

            native_ranks = item.get("native_ranks", {}) or {}
            native_scores = item.get("native_scores", {}) or {}
            fusion_score = item.get("fusion_score")
            provenance = item.get("provenance", []) or []

            display_name = _source_display_name(item)
            reference_topic = _source_reference_topic(item)

            retrieval_label = (
                "/".join(retrieval_systems)
                if retrieval_systems
                else "UNKNOWN"
            )

            header_parts = [
                f"{position}. {display_name}",
                f"Source: {source_family}",
                f"Ref: {reference_topic}",
            ]

            with st.expander(" · ".join(header_parts)):
                # Core provenance
                st.markdown("**Source provenance**")

                provenance_rows = {
                    "Evidence ID": evidence_id or "N/A",
                    "Source": source_family or "N/A",
                    "Reference / topic": reference_topic or "N/A",
                    "Retrieval system": retrieval_label,
                    "Document ID": document_id or "N/A",
                    "Chunk ID": chunk_id or "N/A",
                }

                st.write(provenance_rows)

                # Retrieval ranking / scoring
                score_rows = {}

                if native_ranks:
                    score_rows["Native rank(s)"] = native_ranks

                if native_scores:
                    score_rows["Native score(s)"] = native_scores

                if fusion_score is not None:
                    try:
                        score_rows["Fusion score"] = round(
                            float(fusion_score),
                            6,
                        )
                    except Exception:
                        score_rows["Fusion score"] = fusion_score

                if score_rows:
                    st.markdown("**Retrieval details**")
                    st.write(score_rows)

                # Detailed provenance records
                if provenance:
                    st.markdown("**Underlying source metadata**")

                    for prov_index, prov in enumerate(
                        provenance,
                        start=1,
                    ):
                        if not isinstance(prov, dict):
                            continue

                        cleaned = {}

                        preferred_keys = (
                            "retrieval_system",
                            "source_family",
                            "provider",
                            "collection",
                            "identifier",
                            "release",
                            "section_heading",
                            "source_path",
                            "file_path",
                            "source_shard",
                            "page",
                            "engine",
                            "source",
                            "operation",
                            "native_rank",
                            "native_score",
                            "url",
                        )

                        for key in preferred_keys:
                            value = prov.get(key)
                            if value not in (None, "", [], {}):
                                cleaned[key] = _compact_value(value)

                        if cleaned:
                            if len(provenance) > 1:
                                st.caption(
                                    f"Provenance record {prov_index}"
                                )
                            st.write(cleaned)

                if url:
                    st.markdown(f"[Open source]({url})")

                if text:
                    st.markdown("**Evidence excerpt**")
                    st.write(text[:2200])


# ============================================================
# METRIC RENDERING
# ============================================================

def _render_pending_metrics(
    risk_slot,
    relevance_slot,
    completeness_slot,
):
    with risk_slot.container():
        st.subheader("Evidence-Relative Hallucination Risk")
        st.caption(
            "Measures how strongly each answer is supported by the evidence "
            "available to its evaluation path. Unsupported does not automatically "
            "mean factually false."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Gemma LLM-Only", "Pending…")
        c2.metric("Strict Knowledge Grounding", "Pending…")
        c3.metric("Augmented Reasoning", "Pending…")

    with relevance_slot.container():
        st.subheader("Answer Relevance")
        st.caption(
            "Measures directness, intent coverage, specificity, usefulness, "
            "and focus. It is separate from factual correctness and grounding."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Gemma LLM-Only", "Pending…")
        c2.metric("Strict Knowledge Grounding", "Pending…")
        c3.metric("Augmented Reasoning", "Pending…")

    with completeness_slot.container():
        st.subheader("Answer Completeness")
        st.caption(
            "Measures how comprehensively each answer covers the material answer "
            "elements implied by the user's request."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Gemma LLM-Only", "Pending…")
        c2.metric("Strict Knowledge Grounding", "Pending…")
        c3.metric("Augmented Reasoning", "Pending…")


def _render_three_way_metrics(
    *,
    risk_slot,
    relevance_slot,
    completeness_slot,
    evaluation: dict[str, Any],
    telecom_applicable: bool,
):
    # st.empty() placeholders may retain a previously rendered multi-element
    # container unless they are explicitly cleared first.
    risk_slot.empty()
    relevance_slot.empty()
    completeness_slot.empty()

    systems = evaluation["systems"]

    with risk_slot.container():
        st.subheader("Evidence-Relative Hallucination Risk")
        st.caption(
            "Measures evidence support, not factual truth. A claim can be correct "
            "yet still score as unsupported if it is absent from the evidence bundle."
        )

        if not telecom_applicable:
            st.info(
                "Hallucination-risk comparison is not applicable to non-telecom queries."
            )
        else:
            c1, c2, c3 = st.columns(3)

            for col, key in zip((c1, c2, c3), SYSTEM_KEYS):
                system = systems[key]
                col.metric(
                    SYSTEM_LABELS[key],
                    f"{system['hallucination_risk_pct']:.2f}%",
                    _risk_badge(system["hallucination_risk_label"]),
                )

            with st.expander("Claim-support assessment"):
                for key in SYSTEM_KEYS:
                    system = systems[key]
                    extraction = system.get("risk_claim_extraction", {})
                    assessment = system.get("risk_assessment", {})

                    st.markdown(f"**{SYSTEM_LABELS[key]}**")
                    st.write({
                        "claims": extraction.get("final_claim_count"),
                        "supported": assessment.get("supported"),
                        "partially_supported": assessment.get("partially_supported"),
                        "unsupported": assessment.get("unsupported"),
                        "contradicted": assessment.get("contradicted"),
                    })

    with relevance_slot.container():
        st.subheader("Answer Relevance")
        st.caption(
            "Scored from directness (20), coverage (25), specificity (20), "
            "usefulness (25), and focus (10)."
        )

        c1, c2, c3 = st.columns(3)

        for col, key in zip((c1, c2, c3), SYSTEM_KEYS):
            system = systems[key]
            col.metric(
                SYSTEM_LABELS[key],
                f"{system['relevance_score']}%",
                system["relevance_label"],
            )

        with st.expander("Relevance assessment"):
            st.markdown(
                f"**Question intent:** {evaluation.get('question_intent', '')}"
            )

            required = evaluation.get("required_answer_elements", [])
            if required:
                st.markdown("**Required answer elements**")
                for element in required:
                    st.write(f"- {element}")

            for key in SYSTEM_KEYS:
                system = systems[key]
                st.markdown(f"**{SYSTEM_LABELS[key]}**")
                st.write(system.get("relevance_breakdown", {}))
                st.write(system.get("quality_rationale", ""))

            st.markdown(
                f"**Preferred relevance:** "
                f"{evaluation.get('preferred_relevance', 'N/A')}"
            )

    with completeness_slot.container():
        st.subheader("Answer Completeness")
        st.caption(
            "Measures coverage of the material answer elements inferred from the question."
        )

        c1, c2, c3 = st.columns(3)

        for col, key in zip((c1, c2, c3), SYSTEM_KEYS):
            system = systems[key]
            col.metric(
                SYSTEM_LABELS[key],
                f"{system['completeness_score']}%",
                system["completeness_label"],
            )

        with st.expander("Completeness assessment"):
            for key in SYSTEM_KEYS:
                system = systems[key]
                st.markdown(f"**{SYSTEM_LABELS[key]}**")

                missing = system.get("missing_elements", [])
                if missing:
                    st.write("Important missing elements:")
                    for item in missing:
                        st.write(f"- {item}")
                else:
                    st.write("No material missing elements identified.")

            st.markdown(
                f"**Preferred completeness:** "
                f"{evaluation.get('preferred_completeness', 'N/A')}"
            )

            summary = str(evaluation.get("comparison_summary", "") or "")
            if summary:
                st.markdown("**Comparison summary**")
                st.write(summary)


def _render_latency(
    slot,
    timings: dict[str, float | None],
    judge_applicable: bool | None,
):
    slot.empty()
    with slot.container():
        st.subheader("Latency")
        st.caption(
            "Each system is timed independently. Judge latency is shown separately."
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "LLM-Only",
            _format_seconds(timings.get("llm_only_s"), "Running…"),
        )
        c2.metric(
            "Strict",
            _format_seconds(timings.get("strict_s"), "Running…"),
        )
        c3.metric(
            "Augmented",
            _format_seconds(timings.get("augmented_s"), "Running…"),
        )

        if judge_applicable is False:
            judge_text = "N/A"
        else:
            judge_text = _format_seconds(
                timings.get("judge_s"),
                "Pending…",
            )

        c4.metric("Judge", judge_text)

        router_s = timings.get("router_s")
        total_s = timings.get("total_s")

        detail = []
        if router_s is not None:
            detail.append(f"Router {router_s:.2f}s")
        if total_s is not None:
            detail.append(f"End-to-end {total_s:.2f}s")

        if detail:
            st.caption(" · ".join(detail))


# ============================================================
# PATH EXECUTION
# ============================================================

async def _run_strict_path(
    response_mode: str,
    sanitized: str,
) -> dict[str, Any]:
    if response_mode == TELECOM_GROUNDED_MODE:
        return await run_connected_corpus_grounded(sanitized)

    if response_mode == LIVE_EXTERNAL_GROUNDED_MODE:
        local_result = run_local_datetime_tool(sanitized)

        if local_result is not None:
            elapsed = float(local_result.get("elapsed_s", 0.0) or 0.0)
            return {
                "answer": local_result["answer"],
                "local_runtime": local_result,
                "final_evidence": [],
                "final_context": "",
                "timing": {"total_wall_s": elapsed},
                "orchestration_path": [
                    {
                        "stage": "Local Date/Time Runtime",
                        "status": "completed",
                        "elapsed_s": elapsed,
                    }
                ],
            }

        return await run_live_external_grounded_4c6w(sanitized)

    started = time.perf_counter()
    result = await asyncio.to_thread(
        generate_general_fallback,
        sanitized,
    )
    elapsed = time.perf_counter() - started

    return {
        "answer": _answer_from_result(result),
        "final_evidence": [],
        "final_context": "",
        "timing": {"total_wall_s": float(elapsed)},
        "orchestration_path": [
            {
                "stage": "Gemma General-Knowledge Fallback",
                "status": "completed",
                "elapsed_s": float(elapsed),
            }
        ],
    }


async def _run_augmented_path(
    response_mode: str,
    sanitized: str,
) -> dict[str, Any]:
    if response_mode == TELECOM_GROUNDED_MODE:
        return await run_augmented_reasoning(sanitized)

    return await run_augmented_non_telecom(sanitized)


def _strict_metadata(response_mode: str) -> dict[str, Any]:
    if response_mode == TELECOM_GROUNDED_MODE:
        return {
            "mode": "TELECOM_STRICT_GROUNDED",
            "sources": ["Adaptive RAG", "MCP", "Hybrid when selected"],
            "policy": "Strict evidence-grounded Telecom answer",
        }

    if response_mode == LIVE_EXTERNAL_GROUNDED_MODE:
        return {
            "mode": "LIVE_EXTERNAL_GROUNDED",
            "sources": ["Web Search or Local Runtime"],
            "policy": "Externally grounded current-information response",
        }

    return {
        "mode": "GENERAL_KNOWLEDGE_FALLBACK",
        "sources": ["Gemma pretrained knowledge"],
        "policy": "Stable general-knowledge fallback; no Telecom RAG/MCP",
    }


def _augmented_metadata(response_mode: str) -> dict[str, Any]:
    if response_mode == TELECOM_GROUNDED_MODE:
        return {
            "mode": "TELECOM_AUGMENTED",
            "sources": [
                "RAG",
                "MCP",
                "Web Search",
                "Gemma pretrained knowledge",
            ],
            "policy": (
                "Forced multi-source evidence gathering + "
                "one Gemma synthesis call"
            ),
        }

    return {
        "mode": "NON_TELECOM_AUGMENTED",
        "sources": ["Web Search", "Gemma pretrained knowledge"],
        "policy": "Forced web augmentation + one Gemma synthesis call",
    }


def _strict_trace_from_result(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build a truthful presentation trace from fields already measured by
    the validated Strict runtime. No synthetic timings are introduced.
    """
    if not isinstance(result, dict):
        return []

    existing = result.get("orchestration_path", []) or []
    if existing:
        return existing

    stages: list[dict[str, Any]] = []

    rag_calls = int(result.get("rag_call_total", 0) or 0)
    mcp_calls = int(result.get("mcp_call_total", 0) or 0)

    rag_s = result.get("rag_retrieval_s")
    mcp_s = result.get("mcp_retrieval_s")

    if rag_calls > 0:
        stage = {
            "stage": f"RAG Retrieval ({rag_calls} call{'s' if rag_calls != 1 else ''})",
            "status": "completed",
        }
        if rag_s is not None:
            stage["elapsed_s"] = float(rag_s)
        stages.append(stage)

    if mcp_calls > 0:
        stage = {
            "stage": f"MCP Retrieval ({mcp_calls} call{'s' if mcp_calls != 1 else ''})",
            "status": "completed",
        }
        if mcp_s is not None:
            stage["elapsed_s"] = float(mcp_s)
        stages.append(stage)

    # The adaptive runtime records every sufficiency judgment by round.
    round_traces = result.get("round_traces", []) or []
    sufficiency_s = 0.0
    sufficiency_found = False

    for round_trace in round_traces:
        if not isinstance(round_trace, dict):
            continue

        assessment = round_trace.get("sufficiency", {}) or {}
        latency = assessment.get("latency_s")

        if latency is not None:
            try:
                sufficiency_s += float(latency)
                sufficiency_found = True
            except Exception:
                pass

    if sufficiency_found:
        stages.append({
            "stage": "Evidence Sufficiency Assessment",
            "status": "completed",
            "elapsed_s": sufficiency_s,
        })
    elif round_traces:
        stages.append({
            "stage": "Evidence Sufficiency Assessment",
            "status": "completed",
        })

    generation_s = result.get("elapsed_s")

    if generation_s is not None:
        stages.append({
            "stage": "Strict Gemma Generation",
            "status": "completed",
            "elapsed_s": float(generation_s),
        })
    elif result.get("answer"):
        stages.append({
            "stage": "Strict Gemma Generation",
            "status": "completed",
        })

    return stages


def _render_strict_runtime_summary(
    slot,
    result: dict[str, Any],
):
    """
    Render Strict-specific runtime facts that are returned by the validated
    adaptive runtime but are not naturally represented as sequential stages.
    """
    if not isinstance(result, dict):
        return

    with slot.container():
        st.caption(
            "Strict runtime summary · "
            f"Mode: {result.get('selected_mode', 'N/A')} · "
            f"Searches: {result.get('search_count', 'N/A')} · "
            f"Stop: {result.get('stop_reason', 'N/A')} · "
            f"Evidence: {len(result.get('final_evidence', []) or [])}"
        )


# ============================================================
# NON-TELECOM QUALITY-ONLY EVALUATION
# ============================================================

def _build_quality_only_evaluation(
    quality_execution: dict[str, Any],
) -> dict[str, Any]:
    q = quality_execution["payload"]

    systems = {}

    mapping = {
        "llm_only": "answer_a",
        "strict": "answer_b",
        "augmented": "answer_c",
    }

    for key, answer_key in mapping.items():
        group = q[answer_key]
        systems[key] = {
            "label": SYSTEM_LABELS[key],
            "hallucination_risk_pct": None,
            "hallucination_risk_label": "N/A",
            "risk_claim_extraction": {},
            "risk_assessment": {},
            "relevance_score": group["relevance_score"],
            "relevance_label": _score_label(group["relevance_score"]),
            "relevance_breakdown": {
                "directness": group["directness"],
                "coverage": group["coverage"],
                "specificity": group["specificity"],
                "usefulness": group["usefulness"],
                "focus": group["focus"],
            },
            "completeness_score": group["completeness_score"],
            "completeness_label": _score_label(group["completeness_score"]),
            "missing_elements": group["missing_elements"],
            "quality_rationale": group["rationale"],
        }

    return {
        "applicable": False,
        "question_intent": q["question_intent"],
        "required_answer_elements": q["required_answer_elements"],
        "preferred_relevance": q["preferred_relevance"],
        "preferred_completeness": q["preferred_completeness"],
        "comparison_summary": q["comparison_summary"],
        "systems": systems,
        "judge": {
            "wall_s": quality_execution["elapsed_s"],
            "quality_judge_s": quality_execution["elapsed_s"],
            "quality_attempts": quality_execution["attempts"],
            "quality_retried": quality_execution["retried"],
        },
    }


# ============================================================
# PROGRESSIVE THREE-PATH EXECUTION
# ============================================================

async def run_progressive_three_path(
    *,
    question: str,
    risk_slot,
    relevance_slot,
    completeness_slot,
    latency_slot,
    path_slots: dict[str, Any],
    answer_slots: dict[str, Any],
    timer_slots: dict[str, Any],
    evidence_slot,
    trace_slot,
):
    security = preprocess_user_prompt(question)
    sanitized = str(
        security.get("sanitized_prompt", "") or ""
    ).strip()

    if not sanitized:
        raise ValueError("Question is empty after security sanitization.")

    if (
        security.get("injection_detected")
        or security.get("evidence_manipulation_detected")
    ):
        st.warning(
            "Potential prompt-injection language was detected and removed "
            "before routing."
        )

    overall_started = time.perf_counter()

    timings: dict[str, float | None] = {
        "router_s": None,
        "llm_only_s": None,
        "strict_s": None,
        "augmented_s": None,
        "judge_s": None,
        "total_s": None,
    }

    _render_pending_metrics(
        risk_slot,
        relevance_slot,
        completeness_slot,
    )
    _render_latency(
        latency_slot,
        timings,
        judge_applicable=None,
    )

    # Initial cards
    _render_answer_card(
        answer_slots["llm_only"],
        title=SYSTEM_LABELS["llm_only"],
        objective=SYSTEM_OBJECTIVES["llm_only"],
        answer=None,
        mode="LLM_ONLY",
        sources=["Gemma pretrained knowledge"],
        policy="No retrieval or external grounding",
        warning=LLM_ONLY_WARNING,
    )
    _render_answer_card(
        answer_slots["strict"],
        title=SYSTEM_LABELS["strict"],
        objective=SYSTEM_OBJECTIVES["strict"],
        answer=None,
        policy="Route and grounding policy are being determined.",
    )
    _render_answer_card(
        answer_slots["augmented"],
        title=SYSTEM_LABELS["augmented"],
        objective=SYSTEM_OBJECTIVES["augmented"],
        answer=None,
        policy="Route and augmentation policy are being determined.",
        warning=AUGMENTED_REASONING_WARNING,
    )

    _render_path(
        path_slots["llm_only"],
        title=SYSTEM_LABELS["llm_only"],
        objective=SYSTEM_OBJECTIVES["llm_only"],
        mode="LLM_ONLY",
        sources=["Gemma pretrained knowledge"],
        stages=None,
        pending_message="Gemma inference has started.",
    )
    _render_path(
        path_slots["strict"],
        title=SYSTEM_LABELS["strict"],
        objective=SYSTEM_OBJECTIVES["strict"],
        mode=None,
        sources=None,
        stages=None,
        pending_message="Waiting for Granite routing decision.",
    )
    _render_path(
        path_slots["augmented"],
        title=SYSTEM_LABELS["augmented"],
        objective=SYSTEM_OBJECTIVES["augmented"],
        mode=None,
        sources=None,
        stages=None,
        pending_message="Waiting for Granite routing decision.",
    )

    # Baseline and router begin together.
    baseline_started = time.perf_counter()
    router_started = time.perf_counter()

    baseline_task = asyncio.create_task(
        asyncio.to_thread(
            generate_gemma_only_baseline,
            sanitized,
        )
    )
    router_task = asyncio.create_task(
        asyncio.to_thread(
            classify_knowledge_scope,
            sanitized,
        )
    )

    routing = None
    response_mode = None
    strict_meta = None
    augmented_meta = None

    strict_task = None
    augmented_task = None
    strict_started = None
    augmented_started = None

    baseline_result = None
    strict_result = None
    augmented_result = None

    while (
        baseline_result is None
        or routing is None
        or strict_result is None
        or augmented_result is None
    ):
        now = time.perf_counter()

        # LLM-only path
        if baseline_result is None:
            if baseline_task.done():
                baseline_result = baseline_task.result()
                timings["llm_only_s"] = (
                    time.perf_counter() - baseline_started
                )

                timer_slots["llm_only"].success(
                    f"Completed in {timings['llm_only_s']:.1f}s"
                )

                _render_answer_card(
                    answer_slots["llm_only"],
                    title=SYSTEM_LABELS["llm_only"],
                    objective=SYSTEM_OBJECTIVES["llm_only"],
                    answer=_answer_from_result(baseline_result),
                    mode="LLM_ONLY",
                    sources=["Gemma pretrained knowledge"],
                    policy="No retrieval or external grounding",
                    warning=LLM_ONLY_WARNING,
                )

                _render_path(
                    path_slots["llm_only"],
                    title=SYSTEM_LABELS["llm_only"],
                    objective=SYSTEM_OBJECTIVES["llm_only"],
                    mode="LLM_ONLY",
                    sources=["Gemma pretrained knowledge"],
                    stages=[
                        {
                            "stage": "Gemma LLM-Only Inference",
                            "status": "completed",
                            "elapsed_s": timings["llm_only_s"],
                        }
                    ],
                )

                _render_latency(
                    latency_slot,
                    timings,
                    judge_applicable=None,
                )
            else:
                timer_slots["llm_only"].info(
                    f"Gemma inference running · "
                    f"{now - baseline_started:.1f}s"
                )

        # Router
        if routing is None:
            if router_task.done():
                routing = router_task.result()
                timings["router_s"] = (
                    time.perf_counter() - router_started
                )

                response_mode = routing["response_mode"]
                strict_meta = _strict_metadata(response_mode)
                augmented_meta = _augmented_metadata(response_mode)

                strict_started = time.perf_counter()
                augmented_started = time.perf_counter()

                strict_task = asyncio.create_task(
                    _run_strict_path(
                        response_mode,
                        sanitized,
                    )
                )
                augmented_task = asyncio.create_task(
                    _run_augmented_path(
                        response_mode,
                        sanitized,
                    )
                )

                _render_answer_card(
                    answer_slots["strict"],
                    title=SYSTEM_LABELS["strict"],
                    objective=SYSTEM_OBJECTIVES["strict"],
                    answer=None,
                    mode=strict_meta["mode"],
                    sources=strict_meta["sources"],
                    policy=strict_meta["policy"],
                )

                _render_answer_card(
                    answer_slots["augmented"],
                    title=SYSTEM_LABELS["augmented"],
                    objective=SYSTEM_OBJECTIVES["augmented"],
                    answer=None,
                    mode=augmented_meta["mode"],
                    sources=augmented_meta["sources"],
                    policy=augmented_meta["policy"],
                    warning=AUGMENTED_REASONING_WARNING,
                )

                _render_path(
                    path_slots["strict"],
                    title=SYSTEM_LABELS["strict"],
                    objective=SYSTEM_OBJECTIVES["strict"],
                    mode=strict_meta["mode"],
                    sources=strict_meta["sources"],
                    stages=None,
                    pending_message=(
                        "Routing complete. Strict processing has started."
                    ),
                )

                _render_path(
                    path_slots["augmented"],
                    title=SYSTEM_LABELS["augmented"],
                    objective=SYSTEM_OBJECTIVES["augmented"],
                    mode=augmented_meta["mode"],
                    sources=augmented_meta["sources"],
                    stages=None,
                    pending_message=(
                        "Routing complete. Augmented processing has started."
                    ),
                )

                # Refresh baseline warning if route resolved after baseline completed.
                if baseline_result is not None and response_mode != TELECOM_GROUNDED_MODE:
                    _render_answer_card(
                        answer_slots["llm_only"],
                        title=SYSTEM_LABELS["llm_only"],
                        objective=SYSTEM_OBJECTIVES["llm_only"],
                        answer=_answer_from_result(baseline_result),
                        mode="LLM_ONLY",
                        sources=["Gemma pretrained knowledge"],
                        policy="No retrieval or external grounding",
                        warning=LLM_ONLY_WARNING,
                    )
            else:
                timer_slots["strict"].info(
                    f"Routing question · {now - router_started:.1f}s"
                )
                timer_slots["augmented"].info(
                    f"Routing question · {now - router_started:.1f}s"
                )

        # Strict path
        if (
            routing is not None
            and strict_result is None
            and strict_task is not None
        ):
            if strict_task.done():
                strict_result = strict_task.result()
                timings["strict_s"] = (
                    time.perf_counter() - float(strict_started)
                )

                timer_slots["strict"].success(
                    f"Completed in {timings['strict_s']:.1f}s"
                )

                _render_answer_card(
                    answer_slots["strict"],
                    title=SYSTEM_LABELS["strict"],
                    objective=SYSTEM_OBJECTIVES["strict"],
                    answer=_answer_from_result(strict_result),
                    mode=strict_meta["mode"],
                    sources=strict_meta["sources"],
                    policy=strict_meta["policy"],
                    warning=(
                        KNOWLEDGE_BASE_DISCLAIMER
                        if response_mode == TELECOM_GROUNDED_MODE
                        else NON_TELECOM_WARNING
                        if response_mode == GENERAL_KNOWLEDGE_FALLBACK_MODE
                        else None
                    ),
                )

                _render_path(
                    path_slots["strict"],
                    title=SYSTEM_LABELS["strict"],
                    objective=SYSTEM_OBJECTIVES["strict"],
                    mode=strict_meta["mode"],
                    sources=strict_meta["sources"],
                    stages=_strict_trace_from_result(
                        strict_result
                    ),
                )

                _render_strict_runtime_summary(
                    path_slots["strict"],
                    strict_result,
                )

                _render_latency(
                    latency_slot,
                    timings,
                    judge_applicable=None,
                )
            else:
                timer_slots["strict"].info(
                    f"Strict processing · "
                    f"{now - float(strict_started):.1f}s"
                )

        # Augmented path
        if (
            routing is not None
            and augmented_result is None
            and augmented_task is not None
        ):
            if augmented_task.done():
                augmented_result = augmented_task.result()
                timings["augmented_s"] = (
                    time.perf_counter() - float(augmented_started)
                )

                timer_slots["augmented"].success(
                    f"Completed in {timings['augmented_s']:.1f}s"
                )

                _render_answer_card(
                    answer_slots["augmented"],
                    title=SYSTEM_LABELS["augmented"],
                    objective=SYSTEM_OBJECTIVES["augmented"],
                    answer=_answer_from_result(augmented_result),
                    mode=augmented_meta["mode"],
                    sources=augmented_meta["sources"],
                    policy=augmented_meta["policy"],
                    warning=AUGMENTED_REASONING_WARNING,
                )

                _render_path(
                    path_slots["augmented"],
                    title=SYSTEM_LABELS["augmented"],
                    objective=SYSTEM_OBJECTIVES["augmented"],
                    mode=augmented_meta["mode"],
                    sources=augmented_meta["sources"],
                    stages=augmented_result.get(
                        "orchestration_path",
                        [],
                    ),
                )

                _render_latency(
                    latency_slot,
                    timings,
                    judge_applicable=None,
                )
            else:
                timer_slots["augmented"].info(
                    f"Augmented processing · "
                    f"{now - float(augmented_started):.1f}s"
                )

        await asyncio.sleep(0.5)

    # ========================================================
    # SOURCES / REFERENCES
    # ========================================================
    with evidence_slot.container():
        st.subheader("Sources / References")

        if response_mode == TELECOM_GROUNDED_MODE:
            left, right = st.columns(2, gap="large")

            _render_evidence_group(
                left,
                "Strict Knowledge Grounding",
                strict_result.get("final_evidence", []),
            )
            _render_evidence_group(
                right,
                "Augmented Reasoning",
                augmented_result.get("final_evidence", []),
            )
        else:
            _render_evidence_group(
                st.container(),
                "Augmented Web Evidence",
                augmented_result.get("final_evidence", []),
            )

    # ========================================================
    # JUDGE PHASE
    # ========================================================
    telecom_applicable = (
        response_mode == TELECOM_GROUNDED_MODE
    )

    judge_started = time.perf_counter()

    if telecom_applicable:
        evaluation_task = asyncio.create_task(
            asyncio.to_thread(
                evaluate_three_way_telecom,
                question=sanitized,
                llm_only_answer=_answer_from_result(
                    baseline_result
                ),
                strict_answer=_answer_from_result(
                    strict_result
                ),
                augmented_answer=_answer_from_result(
                    augmented_result
                ),
                strict_evidence_context=str(
                    strict_result.get(
                        "final_context",
                        "",
                    )
                    or ""
                ),
                strict_evidence_items=list(
                    strict_result.get(
                        "final_evidence",
                        [],
                    )
                    or []
                ),
                augmented_evidence_context=str(
                    augmented_result.get(
                        "final_context",
                        "",
                    )
                    or ""
                ),
                augmented_evidence_items=list(
                    augmented_result.get(
                        "final_evidence",
                        [],
                    )
                    or []
                ),
            )
        )
    else:
        evaluation_task = asyncio.create_task(
            asyncio.to_thread(
                run_three_way_quality_judge,
                sanitized,
                _answer_from_result(baseline_result),
                _answer_from_result(strict_result),
                _answer_from_result(augmented_result),
            )
        )

    while not evaluation_task.done():
        elapsed = time.perf_counter() - judge_started

        with risk_slot.container():
            st.subheader("Evidence-Relative Hallucination Risk")
            if telecom_applicable:
                st.info(
                    f"Granite claim-support evaluation running · "
                    f"{elapsed:.0f}s"
                )
            else:
                st.info(
                    "Hallucination-risk comparison is not applicable "
                    "to non-telecom queries."
                )

        with relevance_slot.container():
            st.subheader("Answer Relevance")
            st.info(
                f"Granite three-way quality evaluation running · "
                f"{elapsed:.0f}s"
            )

        with completeness_slot.container():
            st.subheader("Answer Completeness")
            st.info(
                f"Granite three-way quality evaluation running · "
                f"{elapsed:.0f}s"
            )

        _render_latency(
            latency_slot,
            timings,
            judge_applicable=telecom_applicable,
        )

        await asyncio.sleep(1.0)

    judge_error = None
    raw_evaluation = None

    try:
        raw_evaluation = await evaluation_task
    except Exception as exc:
        judge_error = exc

    timings["judge_s"] = (
        time.perf_counter() - judge_started
    )
    timings["total_s"] = (
        time.perf_counter() - overall_started
    )

    if judge_error is None:
        if telecom_applicable:
            evaluation = raw_evaluation
        else:
            evaluation = _build_quality_only_evaluation(
                raw_evaluation
            )

        _render_three_way_metrics(
            risk_slot=risk_slot,
            relevance_slot=relevance_slot,
            completeness_slot=completeness_slot,
            evaluation=evaluation,
            telecom_applicable=telecom_applicable,
        )
    else:
        error_text = str(judge_error or "").strip()

        risk_slot.empty()
        relevance_slot.empty()
        completeness_slot.empty()

        with risk_slot.container():
            st.subheader("Evidence-Relative Hallucination Risk")
            if telecom_applicable:
                c1, c2, c3 = st.columns(3)
                c1.metric("Gemma LLM-Only", "Unavailable")
                c2.metric("Strict Knowledge Grounding", "Unavailable")
                c3.metric("Augmented Reasoning", "Unavailable")
                st.warning(
                    "Judge evaluation unavailable for this run. "
                    "Granite did not return a valid comparative judgment "
                    "after its configured retries. The three generated "
                    "answers are still valid experiment outputs."
                )
            else:
                st.info(
                    "Hallucination-risk comparison is not applicable "
                    "to non-telecom queries."
                )

        with relevance_slot.container():
            st.subheader("Answer Relevance")
            c1, c2, c3 = st.columns(3)
            c1.metric("Gemma LLM-Only", "Unavailable")
            c2.metric("Strict Knowledge Grounding", "Unavailable")
            c3.metric("Augmented Reasoning", "Unavailable")
            st.warning(
                "Judge evaluation unavailable for this run. "
                "Relevance was not scored because the Granite evaluation "
                "response could not be parsed reliably."
            )

        with completeness_slot.container():
            st.subheader("Answer Completeness")
            c1, c2, c3 = st.columns(3)
            c1.metric("Gemma LLM-Only", "Unavailable")
            c2.metric("Strict Knowledge Grounding", "Unavailable")
            c3.metric("Augmented Reasoning", "Unavailable")
            st.warning(
                "Judge evaluation unavailable for this run. "
                "Completeness was not scored because the Granite evaluation "
                "response could not be parsed reliably."
            )

        st.session_state["last_judge_error"] = error_text

    _render_latency(
        latency_slot,
        timings,
        judge_applicable=telecom_applicable,
    )

    # ========================================================
    # TECHNICAL TRACE
    # ========================================================
    with trace_slot.container():
        with st.expander("Technical Experiment Trace"):
            st.write({
                "knowledge_scope": response_mode,
                "routing_reason": routing.get("reason", ""),
                "router_latency_s": timings["router_s"],
                "llm_only_latency_s": timings["llm_only_s"],
                "strict_latency_s": timings["strict_s"],
                "augmented_latency_s": timings["augmented_s"],
                "judge_latency_s": timings["judge_s"],
                "judge_status": (
                    "FAILED"
                    if judge_error is not None
                    else "COMPLETED"
                ),
                "judge_error": (
                    str(judge_error)
                    if judge_error is not None
                    else None
                ),
                "total_elapsed_s": timings["total_s"],
            })

            if telecom_applicable:
                st.write({
                    "strict_selected_mode":
                        strict_result.get("selected_mode"),
                    "strict_search_count":
                        strict_result.get("search_count"),
                    "strict_stop_reason":
                        strict_result.get("stop_reason"),
                    "strict_evidence_count":
                        len(strict_result.get("final_evidence", [])),
                    "augmented_evidence_count":
                        len(augmented_result.get("final_evidence", [])),
                    "augmented_selection_policy":
                        augmented_result.get("selection_policy"),
                })

            st.write({
                "preferred_relevance":
                    evaluation.get("preferred_relevance"),
                "preferred_completeness":
                    evaluation.get("preferred_completeness"),
                "comparison_summary":
                    evaluation.get("comparison_summary"),
            })

    return {
        "security": security,
        "routing": routing,
        "baseline": baseline_result,
        "strict": strict_result,
        "augmented": augmented_result,
        "evaluation": evaluation,
        "timings": timings,
    }


# ============================================================
# STREAMLIT APP
# ============================================================

def render_app():
    st.set_page_config(
        page_title="Telecom AI Knowledge Orchestration Runtime",
        page_icon="📡",
        layout="wide",
    )


    # Compact dashboard presentation:
    # preserve the existing information hierarchy while reducing vertical
    # whitespace so metrics and responses are visible together more easily.
    st.markdown(
        """
        <style>
        /* Reduce main page vertical padding */
        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 1.5rem;
        }

        /* Tighten spacing around headings */
        h1 {
            margin-top: 0.15rem !important;
            margin-bottom: 0.35rem !important;
        }

        h2, h3 {
            margin-top: 0.55rem !important;
            margin-bottom: 0.30rem !important;
        }

        /* Compact metric cards */
        div[data-testid="stMetric"] {
            padding: 0.35rem 0.55rem !important;
            min-height: 72px !important;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.78rem !important;
            line-height: 1.05rem !important;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.55rem !important;
            line-height: 1.75rem !important;
        }

        div[data-testid="stMetricDelta"] {
            font-size: 0.72rem !important;
            line-height: 0.95rem !important;
        }

        /* Reduce vertical gaps between Streamlit elements */
        div[data-testid="stVerticalBlock"] {
            gap: 0.55rem !important;
        }

        /* Keep alert boxes useful but less tall */
        div[data-testid="stAlert"] {
            padding-top: 0.45rem !important;
            padding-bottom: 0.45rem !important;
        }

        /* Slightly compact captions */
        .stCaption {
            margin-top: -0.10rem !important;
            margin-bottom: 0.15rem !important;
        }

        /* Reduce divider footprint */
        hr {
            margin-top: 0.55rem !important;
            margin-bottom: 0.55rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Telecom AI Knowledge Orchestration — Demo")

    st.markdown(
        "**A comparative AI engineering demonstrator evaluating three ways of "
        "using a general-purpose LLM: native model reasoning, strict telecom "
        "knowledge grounding, and augmented multi-source reasoning.**"
    )

    st.caption(
        "The experiment explores the trade-offs between answer relevance, "
        "completeness, evidence-relative hallucination risk, latency, and "
        "knowledge-base coverage."
    )
    st.caption("Built by Clifford Imaguezegie")

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------
    with st.sidebar:
        st.header("Demo Infrastructure")

        st.table({
            "Component": [
                "Generator LLM",
                "System 1",
                "System 2",
                "System 3",
                "Router / Evaluator",
                "LLM Gateway",
                "Embedding Model",
                "Vector Search",
                "RAG Corpus Scale",
                "MCP Framework",
                "MCP Technical Sources",
                "TCC Retrieval",
                "3GPP Retrieval",
                "Live Web Grounding",
                "Web Engines",
                "Strict Retrieval",
                "Augmented Retrieval",
                "Hybrid Fusion",
                "Security",
                "Demo UI",
            ],
            "Technology": [
                "Google Gemma 4 26B A4B IT",
                "Gemma LLM-Only",
                "Strict Knowledge Grounding",
                "Augmented Reasoning",
                "IBM Granite 4.2 8B",
                "OpenRouter",
                "BAAI/bge-m3",
                "FAISS IndexFlatIP",
                "~1.51M vectors · 1024 dimensions",
                "FastMCP",
                "GSMA Telco Common Corpus + GSMA/3GPP",
                "DuckDB HTTPFS",
                "Direct remote document retrieval",
                "Open-WebSearch CLI",
                "DuckDuckGo / Bing",
                "Adaptive RAG / MCP / Hybrid",
                "Forced RAG + MCP + Web",
                "RRF + deduplication",
                "Deterministic pre-LLM checks",
                "Streamlit",
            ],
        })

        with st.expander("Three-System Architecture"):
            st.code(
                """USER QUERY
   │
   ├──────────────────────────────┐
   │                              │
   ↓                              ↓
GEMMA LLM-ONLY              SECURITY + ROUTER
   │                              │
   ↓                    ┌─────────┴─────────┐
ANSWER                  ↓                   ↓
                    STRICT              AUGMENTED
                    GROUNDING            REASONING
                       │                   │
                 Adaptive RAG /      RAG + MCP + Web
                 MCP / Hybrid        + Model Knowledge
                       │                   │
                       ↓                   ↓
                    ANSWER              SYNTHESIS
                                           │
                                           ↓
                                        ANSWER

             ───── GRANITE EVALUATION ─────
          Hallucination Risk · Relevance · Completeness
""",
                language="text",
            )

        with st.expander("Experimental Objectives"):
            st.markdown(
                "**Gemma LLM-Only**  \n"
                "Maximum native model reasoning."
            )
            st.markdown(
                "**Strict Knowledge Grounding**  \n"
                "Maximum evidence discipline."
            )
            st.markdown(
                "**Augmented Reasoning**  \n"
                "Maximum useful synthesis across evidence and model knowledge."
            )

    st.info(
        "Grounded telecom responses are constrained by the coverage of the "
        "connected knowledge base. Missing operational, optimization, tuning, "
        "troubleshooting, or vendor-specific material may limit answer completeness."
    )

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    st.markdown("**Recommended reference questions**")

    reference_questions = [
        "About Xn Interface",
        "What is the role of the AMF in a 5G Standalone Network",
        "Function of the SMO in ORAN Architecture",
    ]

    ref_cols = st.columns(3)

    for ref_index, (ref_col, ref_question) in enumerate(
        zip(ref_cols, reference_questions),
        start=1,
    ):
        with ref_col:
            if st.button(
                f"{ref_index}. {ref_question}",
                key=f"reference_question_{ref_index}",
                use_container_width=True,
            ):
                st.session_state["question_input"] = ref_question
                st.session_state["auto_submit_reference"] = True

    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""

    with st.form("question_form"):
        question = st.text_area(
            "Ask a question",
            key="question_input",
            height=110,
            placeholder=(
                "Ask a telecom, live-current, or general-knowledge question…"
            ),
        )

        submitted = st.form_submit_button(
            "Run Three-System Experiment",
            type="primary",
            use_container_width=True,
        )

    if st.session_state.pop(
        "auto_submit_reference",
        False,
    ):
        question = st.session_state.get(
            "question_input",
            "",
        )
        submitted = True

    if not submitted:
        return

    # --------------------------------------------------------
    # 1. EVALUATION METRICS
    # --------------------------------------------------------
    risk_slot = st.empty()
    relevance_slot = st.empty()
    completeness_slot = st.empty()

    # --------------------------------------------------------
    # 2. LATENCY
    # --------------------------------------------------------
    latency_slot = st.empty()

    st.divider()

    # --------------------------------------------------------
    # 3. ORCHESTRATION PATHS
    # --------------------------------------------------------
    st.subheader("Orchestration Paths")

    path_col_1, path_col_2, path_col_3 = st.columns(
        3,
        gap="large",
    )

    with path_col_1:
        path_slot_llm = st.empty()

    with path_col_2:
        path_slot_strict = st.empty()

    with path_col_3:
        path_slot_augmented = st.empty()

    path_slots = {
        "llm_only": path_slot_llm,
        "strict": path_slot_strict,
        "augmented": path_slot_augmented,
    }

    st.divider()

    # --------------------------------------------------------
    # 4. RESPONSES — PROGRESSIVE
    # --------------------------------------------------------
    st.subheader("Responses")

    answer_col_1, answer_col_2, answer_col_3 = st.columns(
        3,
        gap="large",
    )

    with answer_col_1:
        timer_llm = st.empty()
        answer_llm = st.empty()

    with answer_col_2:
        timer_strict = st.empty()
        answer_strict = st.empty()

    with answer_col_3:
        timer_augmented = st.empty()
        answer_augmented = st.empty()

    timer_slots = {
        "llm_only": timer_llm,
        "strict": timer_strict,
        "augmented": timer_augmented,
    }

    answer_slots = {
        "llm_only": answer_llm,
        "strict": answer_strict,
        "augmented": answer_augmented,
    }

    st.divider()

    # --------------------------------------------------------
    # 5. SOURCES / REFERENCES
    # --------------------------------------------------------
    evidence_slot = st.empty()

    # --------------------------------------------------------
    # 6. TECHNICAL TRACE
    # --------------------------------------------------------
    trace_slot = st.empty()

    try:
        asyncio.run(
            run_progressive_three_path(
                question=question,
                risk_slot=risk_slot,
                relevance_slot=relevance_slot,
                completeness_slot=completeness_slot,
                latency_slot=latency_slot,
                path_slots=path_slots,
                answer_slots=answer_slots,
                timer_slots=timer_slots,
                evidence_slot=evidence_slot,
                trace_slot=trace_slot,
            )
        )
    except Exception as exc:
        st.exception(exc)
