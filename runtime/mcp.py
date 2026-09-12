from __future__ import annotations

import asyncio
import gc
import os
import re
import time
import urllib.request
from collections import defaultdict
from typing import Any

import duckdb
import pandas as pd
import psutil
from fastmcp import FastMCP
from huggingface_hub import HfApi

# ============================================================
# TELECOM AI KNOWLEDGE ORCHESTRATION — MCP VERSION A
# Extracted from validated Notebook 21 Cells 2A, 2C and 2D.
# Notebook-only validation dashboards/smoke tests are excluded.
# Retrieval rules, scoring, routing and evidence contract are preserved.
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()


ARCHITECTURE_VERSION = (
    "Version A"
)

RETRIEVAL_ARCHITECTURE = (
    "Remote raw-corpus retrieval"
)

REMOTE_CORPORA = {

    "tcc": {

        "repo_id":
            "GSMA/Telco-Common-Corpus",

        "repo_type":
            "dataset",

        "source_family":
            "TCC",

        "file_pattern":
            "data/*.parquet",

        "format":
            "parquet",
    },

    "3gpp": {

        "repo_id":
            "GSMA/3GPP",

        "repo_type":
            "dataset",

        "source_family":
            "3GPP",

        "file_pattern":
            "marked/**/raw.md",

        "format":
            "markdown",
    },

}

MAX_REMOTE_SHARDS = (
    5
)

PARALLEL_SHARD_CONCURRENCY = (
    5
)

TOP_K_RESULTS = (
    5
)

RETRIEVAL_SCORING = (
    "lexical + phrase + proximity"
)

MAX_MCP_SEARCHES = (
    3
)

MAX_RETRIEVED_SOURCES = (
    5
)

MCP_EXCERPT_CHARS = (
    2500
)

DUCKDB_THREADS = (
    os.cpu_count()
    or
    1
)

DUCKDB_DATABASE = (
    ":memory:"
)

VERSION_A_REFERENCE_TCC_FILES = (
    100
)

VERSION_A_REFERENCE_3GPP_FILES = (
    15_052
)

PER_SHARD_LIMIT = 10

TCC_SEARCH_TIMEOUT_SECONDS = 120

EARLY_TEXT_CHARS = 3000

PROXIMITY_WINDOW = 200

GPP_FETCH_TIMEOUT_SECONDS = 60

GPP_WINDOW_CHARS = 5000

GPP_WINDOW_OVERLAP = 1000

GPP_WINDOWS_PER_SPEC = 2

MAX_GPP_SPEC_CANDIDATES = 3

LOW_INFORMATION_TERMS = {
    "5g",
    "nr",
    "network",
    "procedure",
    "function",
    "role",
    "system",
    "handling",
    "management",
}

QUERY_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to",
    "for", "in", "on", "with", "by", "from",
    "as", "at", "what", "which", "how", "why",
    "when", "where", "is", "are", "was", "were",
    "be", "been", "being", "do", "does", "did",
    "explain", "describe", "identify", "including",
    "according", "primary",
}

GPP_SOURCE_SIGNALS = {
    "3gpp",
    "3gpp ts",
    "3gpp tr",
    "5g standalone",
    "5g sa",
    "5gs",
    "5g core",
    "ng-ran",
    "amf",
    "smf",
    "upf",
    "ausf",
    "udm",
    "nssf",
    "pcf",
    "pdu session",
    "registration",
    "mobility management",
    "s-nssai",
    "network slicing",
    "network slice",
    "5qi",
    "qos flow",
    "ngap",
    "xnap",
    "xn interface",
    "rrc",
    "radio link failure",
    "rlf",
    "n1 interface",
    "n2 interface",
    "n3 interface",
    "n4 interface",
    "pfcp",
}

TCC_IETF_SIGNALS = {
    "ietf",
    "rfc",
    "quic",
    "http",
    "http3",
    "http/3",
    "tls",
    "tcp",
    "udp",
    "dns",
}

TCC_RESEARCH_SIGNALS = {
    "research",
    "paper",
    "study",
    "ieee",
    "openalex",
}

TCC_PATENT_SIGNALS = {
    "patent",
    "invention",
    "uspto",
    "epo",
}

TCC_KNOWLEDGE_SIGNALS = {
    "wikipedia",
    "wikidata",
    "general telecom",
    "general telecommunications",
}

GPP_SPEC_RULES = [

    {
        "name":
            "5GS architecture",

        "signals": {
            "amf",
            "smf",
            "upf",
            "nssf",
            "s-nssai",
            "network slice",
            "network slicing",
            "5qi",
            "qos flow",
            "pdu session",
            "5g core",
            "5g standalone",
            "5g sa",
        },

        "specs": [
            "23.501",
        ],
    },

    {
        "name":
            "5GS procedures",

        "signals": {
            "registration",
            "mobility management",
            "pdu session",
            "handover",
            "inter-gnb handover",
            "service request",
            "session release",
            "pdu session release",
        },

        "specs": [
            "23.502",
        ],
    },

    {
        "name":
            "5GS policy and QoS",

        "signals": {
            "policy control",
            "pcf",
            "qos policy",
            "5qi",
        },

        "specs": [
            "23.503",
        ],
    },

    {
        "name":
            "5G security",

        "signals": {
            "authentication",
            "ausf",
            "udm",
            "security",
            "5g aka",
            "aka",
        },

        "specs": [
            "33.501",
        ],
    },

    {
        "name":
            "5GS NAS",

        "signals": {
            "nas",
            "5gmm",
            "5gsm",
        },

        "specs": [
            "24.501",
        ],
    },

    {
        "name":
            "PFCP and N4",

        "signals": {
            "pfcp",
            "n4",
            "n4 interface",
        },

        "specs": [
            "29.244",
        ],
    },

    {
        "name":
            "NR RRC",

        "signals": {
            "rrc",
            "radio link failure",
            "rlf",
            "rrc re-establishment",
        },

        "specs": [
            "38.331",
        ],
    },

    {
        "name":
            "NR architecture",

        "signals": {
            "nr architecture",
            "gnb",
            "ng-ran",
            "handover",
            "inter-gnb handover",
        },

        "specs": [
            "38.300",
        ],
    },

    {
        "name":
            "NGAP",

        "signals": {
            "ngap",
            "n2",
            "n2 interface",
        },

        "specs": [
            "38.413",
        ],
    },

    {
        "name":
            "XnAP",

        "signals": {
            "xnap",
            "xn",
            "xn interface",
            "inter-gnb handover",
        },

        "specs": [
            "38.423",
        ],
    },

]

def build_hf_resolve_url(
    repo_id,
    file_path,
):

    return (
        "https://huggingface.co/datasets/"
        f"{repo_id}/resolve/main/"
        f"{file_path}"
    )

def is_tcc_runtime_file(
    file_path,
):

    file_path = str(
        file_path
    )


    return (

        file_path.startswith(
            "data/"
        )

        and

        file_path.endswith(
            ".parquet"
        )

        and

        file_path.count("/")
        ==
        1
    )

def is_3gpp_runtime_file(
    file_path,
):

    file_path = str(
        file_path
    )


    return (

        file_path.startswith(
            "marked/"
        )

        and

        file_path.endswith(
            "/raw.md"
        )
    )


# ============================================================
# REMOTE SOURCE DISCOVERY — CELL 2A DEPLOYMENT ADAPTATION
# ============================================================

REMOTE_SOURCE_REGISTRY: dict[str, dict[str, Any]] = {}
TCC_REMOTE_SHARDS: list[dict[str, Any]] = []
GPP_SPEC_PATH_INDEX = defaultdict(list)
SHARD_TERM_STATS = defaultdict(lambda: defaultdict(float))


def initialize_remote_sources(force_refresh: bool = False) -> dict[str, dict[str, Any]]:
    global REMOTE_SOURCE_REGISTRY, TCC_REMOTE_SHARDS, GPP_SPEC_PATH_INDEX

    if REMOTE_SOURCE_REGISTRY and not force_refresh:
        return REMOTE_SOURCE_REGISTRY

    hf_api = HfApi(token=HF_TOKEN or None)

    tcc_repo_files = hf_api.list_repo_files(
        repo_id=REMOTE_CORPORA["tcc"]["repo_id"],
        repo_type=REMOTE_CORPORA["tcc"]["repo_type"],
    )
    tcc_files = sorted([
        file_path for file_path in tcc_repo_files
        if is_tcc_runtime_file(file_path)
    ])
    tcc_urls = [
        build_hf_resolve_url(REMOTE_CORPORA["tcc"]["repo_id"], file_path)
        for file_path in tcc_files
    ]

    gpp_repo_files = hf_api.list_repo_files(
        repo_id=REMOTE_CORPORA["3gpp"]["repo_id"],
        repo_type=REMOTE_CORPORA["3gpp"]["repo_type"],
    )
    gpp_files = sorted([
        file_path for file_path in gpp_repo_files
        if is_3gpp_runtime_file(file_path)
    ])
    gpp_urls = [
        build_hf_resolve_url(REMOTE_CORPORA["3gpp"]["repo_id"], file_path)
        for file_path in gpp_files
    ]

    REMOTE_SOURCE_REGISTRY = {
        "tcc": {
            "repo_id": REMOTE_CORPORA["tcc"]["repo_id"],
            "source_family": "TCC",
            "format": "parquet",
            "file_count": len(tcc_files),
            "files": tcc_files,
            "urls": tcc_urls,
        },
        "3gpp": {
            "repo_id": REMOTE_CORPORA["3gpp"]["repo_id"],
            "source_family": "3GPP",
            "format": "markdown",
            "file_count": len(gpp_files),
            "files": gpp_files,
            "urls": gpp_urls,
        },
    }

    if not tcc_files or not gpp_files:
        raise RuntimeError("MCP Version A remote source discovery failed.")

    TCC_REMOTE_SHARDS = [
        {
            "shard_idx": shard_idx,
            "file_path": file_path,
            "url": url,
        }
        for shard_idx, (file_path, url) in enumerate(zip(tcc_files, tcc_urls))
    ]

    # GPP_SPEC_PATH_INDEX is completed after parse_3gpp_path is defined.
    GPP_SPEC_PATH_INDEX = defaultdict(list)

    return REMOTE_SOURCE_REGISTRY


def normalize_query(query):

    if not query or not str(query).strip():

        raise ValueError(
            "Query must not be empty."
        )

    return re.sub(
        r"\s+",
        " ",
        str(query).strip().lower(),
    )

def tokenize_query(
    query,
    remove_stopwords=True,
):

    tokens = re.findall(
        r"[a-z0-9]+(?:[.-][a-z0-9]+)*",
        normalize_query(query),
    )

    if not remove_stopwords:
        return tokens

    return [
        token
        for token in tokens
        if token not in QUERY_STOPWORDS
    ]

def build_query_phrases(query):

    terms = tokenize_query(
        query
    )

    bigrams = [
        " ".join(
            terms[i:i + 2]
        )
        for i in range(
            len(terms) - 1
        )
    ]

    trigrams = [
        " ".join(
            terms[i:i + 3]
        )
        for i in range(
            len(terms) - 2
        )
    ]

    return (
        bigrams,
        trigrams,
    )

def build_proximity_pairs(query):

    terms = tokenize_query(
        query
    )

    return [
        (
            terms[i],
            terms[i + 1],
        )
        for i in range(
            len(terms) - 1
        )
        if terms[i] != terms[i + 1]
    ]

def term_weight(term):

    if term in LOW_INFORMATION_TERMS:
        return 1.0

    if len(term) <= 2:
        return 0.5

    return 2.0

def escape_sql_literal(value):

    return str(
        value
    ).replace(
        "'",
        "''",
    )

def build_query_features(query):

    normalized = normalize_query(
        query
    )

    terms = tokenize_query(
        query
    )

    (
        bigrams,
        trigrams,
    ) = build_query_phrases(
        query
    )

    proximity_pairs = (
        build_proximity_pairs(
            query
        )
    )

    return {
        "normalized_query":
            normalized,

        "terms":
            terms,

        "bigrams":
            bigrams,

        "trigrams":
            trigrams,

        "proximity_pairs":
            proximity_pairs,
    }

def contains_signal(
    normalized_query,
    signal,
):

    pattern = (
        r"(?<![a-z0-9])"
        +
        re.escape(
            signal.lower()
        )
        +
        r"(?![a-z0-9])"
    )

    return (
        re.search(
            pattern,
            normalized_query,
        )
        is not None
    )

def extract_explicit_3gpp_specs(query):

    normalized = normalize_query(
        query
    )

    matches = re.findall(
        r"\b"
        r"(?:3gpp\s*)?"
        r"(?:ts\s*|tr\s*)?"
        r"(\d{2}\.\d{3})"
        r"\b",
        normalized,
    )

    return list(
        dict.fromkeys(
            matches
        )
    )

def infer_3gpp_spec_candidates(
    query,
    max_specs=MAX_GPP_SPEC_CANDIDATES,
):

    normalized = normalize_query(
        query
    )

    candidates = []
    reasons = []


    # Explicit specification references first.

    for spec in extract_explicit_3gpp_specs(
        query
    ):

        if spec not in candidates:

            candidates.append(
                spec
            )

            reasons.append(
                f"explicit:{spec}"
            )


    # Telecom-domain inference.

    for rule in GPP_SPEC_RULES:

        matched_signals = [
            signal
            for signal in rule[
                "signals"
            ]
            if contains_signal(
                normalized,
                signal,
            )
        ]

        if not matched_signals:
            continue

        for spec in rule[
            "specs"
        ]:

            if spec not in candidates:

                candidates.append(
                    spec
                )

        reasons.append(
            f"{rule['name']}: "
            +
            ", ".join(
                matched_signals
            )
        )

        if len(
            candidates
        ) >= max_specs:

            break


    return {

        "specs":
            candidates[
                :max_specs
            ],

        "reasons":
            reasons,
    }

def select_tcc_collections(query):

    normalized = normalize_query(
        query
    )

    collections = []
    reasons = []


    ietf_hits = [
        signal
        for signal in TCC_IETF_SIGNALS
        if contains_signal(
            normalized,
            signal,
        )
    ]

    if ietf_hits:

        collections.extend(
            [
                "IETF-RFCs",
                "IETF-Drafts",
            ]
        )

        reasons.append(
            "IETF: "
            +
            ", ".join(
                ietf_hits
            )
        )


    research_hits = [
        signal
        for signal in TCC_RESEARCH_SIGNALS
        if contains_signal(
            normalized,
            signal,
        )
    ]

    if research_hits:

        collections.extend(
            [
                "IEEE-Access",
                "OpenAlex",
            ]
        )

        reasons.append(
            "Research: "
            +
            ", ".join(
                research_hits
            )
        )


    patent_hits = [
        signal
        for signal in TCC_PATENT_SIGNALS
        if contains_signal(
            normalized,
            signal,
        )
    ]

    if patent_hits:

        collections.extend(
            [
                "USPTO",
                "EPO",
            ]
        )

        reasons.append(
            "Patents: "
            +
            ", ".join(
                patent_hits
            )
        )


    knowledge_hits = [
        signal
        for signal in TCC_KNOWLEDGE_SIGNALS
        if contains_signal(
            normalized,
            signal,
        )
    ]

    if knowledge_hits:

        collections.extend(
            [
                "Wikipedia-Telecom",
                "Wikidata-Telecom",
            ]
        )

        reasons.append(
            "generic knowledge fallback: "
            +
            ", ".join(
                knowledge_hits
            )
        )


    gpp_hits = [
        signal
        for signal in GPP_SOURCE_SIGNALS
        if contains_signal(
            normalized,
            signal,
        )
    ]

    if gpp_hits:

        collections.append(
            "3GPP-TSG"
        )

        reasons.append(
            "TCC 3GPP contribution material"
        )


    if not collections:

        # Experiment 3:
        # No permanent fallback collection.
        # Dynamic evidence-based collection discovery happens
        # later inside retrieve_tcc_remote().
        collections = []

        reasons.append(
            "dynamic TCC collection discovery required"
        )


    collections = list(
        dict.fromkeys(
            collections
        )
    )


    return {

        "collections":
            collections,

        "reasons":
            reasons,
    }

def route_telecom_query(query):

    normalized = normalize_query(
        query
    )


    gpp_hits = [
        signal
        for signal in GPP_SOURCE_SIGNALS
        if contains_signal(
            normalized,
            signal,
        )
    ]


    tcc_signals = (
        TCC_IETF_SIGNALS
        |
        TCC_RESEARCH_SIGNALS
        |
        TCC_PATENT_SIGNALS
        |
        TCC_KNOWLEDGE_SIGNALS
    )


    tcc_hits = [
        signal
        for signal in tcc_signals
        if contains_signal(
            normalized,
            signal,
        )
    ]


    explicit_specs = (
        extract_explicit_3gpp_specs(
            query
        )
    )


    # --------------------------------------------------------
    # Build source selections BEFORE deciding fallback behavior.
    # This avoids unbound-local errors and keeps the router
    # deterministic.
    # --------------------------------------------------------

    gpp_selection = (
        infer_3gpp_spec_candidates(
            query
        )
    )


    tcc_selection = (
        select_tcc_collections(
            query
        )
    )


    # --------------------------------------------------------
    # Primary authoritative routing
    # --------------------------------------------------------

    if explicit_specs:

        route = "3gpp"

    elif gpp_hits and tcc_hits:

        route = "hybrid"

    elif gpp_hits:

        route = "3gpp"

    elif tcc_hits:

        route = "tcc"

    else:

        # Demo continuity fallback:
        # keep the MCP path populated when no authoritative
        # source family can be confidently selected.
        route = "tcc"

        if not tcc_selection[
            "collections"
        ]:

            tcc_selection[
                "collections"
            ] = [
                "Wikipedia-Telecom",
                "Wikidata-Telecom",
            ]

            tcc_selection[
                "reasons"
            ].append(
                "generic MCP fallback"
            )


    # --------------------------------------------------------
    # Prevent duplicate standards retrieval in HYBRID.
    # --------------------------------------------------------

    if route == "hybrid":

        tcc_selection[
            "collections"
        ] = [
            collection
            for collection
            in tcc_selection[
                "collections"
            ]
            if collection != "3GPP-TSG"
        ]


        tcc_selection[
            "reasons"
        ] = [
            reason
            for reason
            in tcc_selection[
                "reasons"
            ]
            if reason
            !=
            "TCC 3GPP contribution material"
        ]


        if not tcc_selection[
            "collections"
        ]:

            tcc_selection[
                "reasons"
            ].append(
                "dynamic TCC collection discovery required"
            )


    return {

        "route":
            route,

        "gpp_signal_hits":
            gpp_hits,

        "tcc_signal_hits":
            tcc_hits,

        "gpp_specs":
            gpp_selection[
                "specs"
            ],

        "gpp_reasons":
            gpp_selection[
                "reasons"
            ],

        "tcc_collections":
            tcc_selection[
                "collections"
            ],

        "tcc_reasons":
            tcc_selection[
                "reasons"
            ],
    }


def select_spread_shards(
    parquet_files,
    max_shards=MAX_REMOTE_SHARDS,
):

    total_files = len(
        parquet_files
    )

    if total_files <= max_shards:

        return list(
            parquet_files
        )


    if max_shards <= 1:

        return [
            parquet_files[0]
        ]


    selected = []
    selected_indices = set()


    step = (
        (total_files - 1)
        /
        (max_shards - 1)
    )


    for i in range(
        max_shards
    ):

        idx = round(
            i * step
        )

        if idx not in selected_indices:

            selected.append(
                parquet_files[
                    idx
                ]
            )

            selected_indices.add(
                idx
            )


    if len(
        selected
    ) < max_shards:

        for idx, item in enumerate(
            parquet_files
        ):

            if idx not in selected_indices:

                selected.append(
                    item
                )

                selected_indices.add(
                    idx
                )

            if len(
                selected
            ) >= max_shards:

                break


    return selected[
        :max_shards
    ]

def get_query_shard_scores(
    query,
    parquet_files,
):

    terms = tokenize_query(
        query
    )

    shard_scores = []


    for shard in parquet_files:

        shard_idx = shard[
            "shard_idx"
        ]

        score = 0.0


        for term in terms:

            score += (
                SHARD_TERM_STATS[
                    term
                ].get(
                    shard_idx,
                    0.0,
                )
            )


        shard_scores.append(
            {
                "shard_idx":
                    shard_idx,

                "score":
                    score,

                "parquet_file":
                    shard,
            }
        )


    return shard_scores

def select_adaptive_shards(
    query,
    parquet_files,
    max_shards=MAX_REMOTE_SHARDS,
):

    if not parquet_files:

        return {
            "mode":
                "EMPTY",

            "shards":
                [],
        }


    max_shards = min(
        max_shards,
        len(
            parquet_files
        ),
    )


    scores = (
        get_query_shard_scores(
            query,
            parquet_files,
        )
    )


    useful = [
        item
        for item in scores
        if item[
            "score"
        ] > 0
    ]


    if not useful:

        return {

            "mode":
                "SPREAD",

            "shards":
                select_spread_shards(
                    parquet_files,
                    max_shards,
                ),
        }


    ranked = sorted(
        scores,
        key=lambda item:
            item[
                "score"
            ],
        reverse=True,
    )


    return {

        "mode":
            "ADAPTIVE",

        "shards":
            [
                item[
                    "parquet_file"
                ]
                for item in ranked[
                    :max_shards
                ]
            ],
    }

def update_shard_term_stats(
    query,
    shard_idx,
    relevance_score,
):

    if relevance_score <= 0:
        return


    for term in tokenize_query(
        query
    ):

        SHARD_TERM_STATS[
            term
        ][
            shard_idx
        ] += float(
            relevance_score
        )

def load_httpfs(con):

    try:

        con.execute(
            "LOAD httpfs"
        )

    except Exception:

        con.execute(
            "INSTALL httpfs"
        )

        con.execute(
            "LOAD httpfs"
        )

def build_collection_filter_sql(
    collections
):

    if not collections:

        raise ValueError(
            "At least one TCC collection must be selected."
        )


    values = [

        "'"
        +
        escape_sql_literal(
            collection
        )
        +
        "'"

        for collection
        in collections
    ]


    return (
        "("
        +
        ", ".join(
            values
        )
        +
        ")"
    )

def build_tcc_search_sql(
    query,
    parquet_url,
    collections,
    limit=PER_SHARD_LIMIT,
):

    features = (
        build_query_features(
            query
        )
    )


    normalized_query = (
        escape_sql_literal(
            features[
                "normalized_query"
            ]
        )
    )


    terms = features[
        "terms"
    ]

    bigrams = features[
        "bigrams"
    ]

    trigrams = features[
        "trigrams"
    ]

    proximity_pairs = features[
        "proximity_pairs"
    ]


    if not terms:

        raise ValueError(
            "Query contains no searchable terms."
        )


    collection_sql = (
        build_collection_filter_sql(
            collections
        )
    )


    score_parts = []
    matched_term_parts = []
    matched_phrase_parts = []
    proximity_parts = []


    # --------------------------------------------------------
    # Exact query
    # --------------------------------------------------------

    score_parts.append(
        f"""
        CASE
            WHEN title_l LIKE '%{normalized_query}%'
                THEN 40
            WHEN early_text_l LIKE '%{normalized_query}%'
                THEN 30
            WHEN text_l LIKE '%{normalized_query}%'
                THEN 20
            ELSE 0
        END
        """
    )


    # --------------------------------------------------------
    # Individual terms
    # --------------------------------------------------------

    for term in terms:

        safe_term = (
            escape_sql_literal(
                term
            )
        )

        weight = (
            term_weight(
                term
            )
        )


        score_parts.append(
            f"""
            CASE
                WHEN title_l LIKE '%{safe_term}%'
                    THEN {8 * weight}
                WHEN early_text_l LIKE '%{safe_term}%'
                    THEN {5 * weight}
                WHEN text_l LIKE '%{safe_term}%'
                    THEN {2 * weight}
                ELSE 0
            END
            """
        )


        matched_term_parts.append(
            f"""
            CASE
                WHEN title_l LIKE '%{safe_term}%'
                  OR text_l LIKE '%{safe_term}%'
                THEN 1
                ELSE 0
            END
            """
        )


    # --------------------------------------------------------
    # Bigrams
    # --------------------------------------------------------

    for phrase in bigrams:

        safe_phrase = (
            escape_sql_literal(
                phrase
            )
        )


        score_parts.append(
            f"""
            CASE
                WHEN title_l LIKE '%{safe_phrase}%'
                    THEN 14
                WHEN early_text_l LIKE '%{safe_phrase}%'
                    THEN 10
                WHEN text_l LIKE '%{safe_phrase}%'
                    THEN 6
                ELSE 0
            END
            """
        )


        matched_phrase_parts.append(
            f"""
            CASE
                WHEN title_l LIKE '%{safe_phrase}%'
                  OR text_l LIKE '%{safe_phrase}%'
                THEN 1
                ELSE 0
            END
            """
        )


    # --------------------------------------------------------
    # Trigrams
    # --------------------------------------------------------

    for phrase in trigrams:

        safe_phrase = (
            escape_sql_literal(
                phrase
            )
        )


        score_parts.append(
            f"""
            CASE
                WHEN title_l LIKE '%{safe_phrase}%'
                    THEN 20
                WHEN early_text_l LIKE '%{safe_phrase}%'
                    THEN 14
                WHEN text_l LIKE '%{safe_phrase}%'
                    THEN 8
                ELSE 0
            END
            """
        )


        matched_phrase_parts.append(
            f"""
            CASE
                WHEN title_l LIKE '%{safe_phrase}%'
                  OR text_l LIKE '%{safe_phrase}%'
                THEN 1
                ELSE 0
            END
            """
        )


    # --------------------------------------------------------
    # Proximity
    # --------------------------------------------------------

    for (
        term_a,
        term_b,
    ) in proximity_pairs:

        safe_a = escape_sql_literal(
            re.escape(
                term_a
            )
        )

        safe_b = escape_sql_literal(
            re.escape(
                term_b
            )
        )


        proximity_parts.append(
            f"""
            CASE
                WHEN regexp_matches(
                    text_l,
                    '{safe_a}.{{0,{PROXIMITY_WINDOW}}}{safe_b}'
                )
                OR regexp_matches(
                    text_l,
                    '{safe_b}.{{0,{PROXIMITY_WINDOW}}}{safe_a}'
                )
                THEN 10
                ELSE 0
            END
            """
        )


    score_expr = (
        " + ".join(
            score_parts
        )
        if score_parts
        else "0"
    )


    matched_terms_expr = (
        " + ".join(
            matched_term_parts
        )
        if matched_term_parts
        else "0"
    )


    matched_phrases_expr = (
        " + ".join(
            matched_phrase_parts
        )
        if matched_phrase_parts
        else "0"
    )


    proximity_expr = (
        " + ".join(
            proximity_parts
        )
        if proximity_parts
        else "0"
    )


    safe_url = (
        escape_sql_literal(
            parquet_url
        )
    )


    return f"""
        WITH scoped AS (

            SELECT
                identifier,
                collection,
                date,
                title,
                creator,
                text,

                lower(
                    coalesce(
                        title,
                        ''
                    )
                ) AS title_l,

                lower(
                    coalesce(
                        text,
                        ''
                    )
                ) AS text_l,

                lower(
                    substr(
                        coalesce(
                            text,
                            ''
                        ),
                        1,
                        {EARLY_TEXT_CHARS}
                    )
                ) AS early_text_l

            FROM read_parquet(
                '{safe_url}'
            )

            WHERE collection IN
                {collection_sql}
        )

        SELECT
            identifier,
            collection,
            date,
            title,
            creator,
            text,

            'TCC'
                AS source_family,

            ({matched_terms_expr})
                AS matched_terms,

            ({matched_phrases_expr})
                AS matched_phrases,

            ({proximity_expr})
                AS proximity_score,

            (
                ({score_expr})
                +
                ({proximity_expr})
            )
                AS relevance_score

        FROM scoped

        WHERE
            (
                title_l
                    LIKE '%{normalized_query}%'

                OR

                text_l
                    LIKE '%{normalized_query}%'

                OR

                ({matched_terms_expr}) > 0
            )

        ORDER BY
            relevance_score DESC,
            date DESC NULLS LAST

        LIMIT {int(limit)}
    """

def search_single_tcc_shard(
    query,
    shard,
    collections,
    limit=PER_SHARD_LIMIT,
):

    start_time = (
        time.perf_counter()
    )


    con = duckdb.connect(
        database=":memory:"
    )


    try:

        con.execute(
            f"SET threads = {int(DUCKDB_THREADS)}"
        )

        load_httpfs(
            con
        )


        sql = (
            build_tcc_search_sql(
                query=query,
                parquet_url=
                    shard[
                        "url"
                    ],
                collections=
                    collections,
                limit=
                    limit,
            )
        )


        result_df = (
            con.execute(
                sql
            )
            .fetchdf()
        )


        records = (
            result_df.to_dict(
                orient="records"
            )
        )


        for item in records:

            item[
                "shard_idx"
            ] = (
                shard[
                    "shard_idx"
                ]
            )

            item[
                "source_path"
            ] = (
                shard[
                    "file_path"
                ]
            )

            item[
                "source_shard"
            ] = (
                shard[
                    "url"
                ]
            )


        return {

            "shard_idx":
                shard[
                    "shard_idx"
                ],

            "file_path":
                shard[
                    "file_path"
                ],

            "elapsed_s":
                (
                    time.perf_counter()
                    -
                    start_time
                ),

            "results":
                records,

            "error":
                None,
        }


    except Exception as exc:

        return {

            "shard_idx":
                shard[
                    "shard_idx"
                ],

            "file_path":
                shard[
                    "file_path"
                ],

            "elapsed_s":
                (
                    time.perf_counter()
                    -
                    start_time
                ),

            "results":
                [],

            "error":
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
        }


    finally:

        con.close()

async def search_tcc_shard_with_timeout(
    query,
    shard,
    collections,
    limit=PER_SHARD_LIMIT,
):

    try:

        return await asyncio.wait_for(

            asyncio.to_thread(
                search_single_tcc_shard,
                query,
                shard,
                collections,
                limit,
            ),

            timeout=
                TCC_SEARCH_TIMEOUT_SECONDS,
        )


    except asyncio.TimeoutError:

        return {

            "shard_idx":
                shard[
                    "shard_idx"
                ],

            "file_path":
                shard[
                    "file_path"
                ],

            "elapsed_s":
                float(
                    TCC_SEARCH_TIMEOUT_SECONDS
                ),

            "results":
                [],

            "error":
                (
                    "TimeoutError: TCC shard search "
                    f"exceeded "
                    f"{TCC_SEARCH_TIMEOUT_SECONDS}s"
                ),
        }

async def search_selected_tcc_shards(
    query,
    selected_shards,
    collections,
    limit=PER_SHARD_LIMIT,
):

    semaphore = asyncio.Semaphore(
        PARALLEL_SHARD_CONCURRENCY
    )


    async def run_one(shard):

        async with semaphore:

            return await (
                search_tcc_shard_with_timeout(
                    query=query,
                    shard=shard,
                    collections=collections,
                    limit=limit,
                )
            )


    tasks = [
        run_one(
            shard
        )
        for shard
        in selected_shards
    ]


    return await asyncio.gather(
        *tasks
    )

def merge_tcc_results(
    shard_results
):

    combined = []


    for shard_result in shard_results:

        for item in shard_result[
            "results"
        ]:

            combined.append(
                dict(
                    item
                )
            )


    combined.sort(
        key=lambda item:
            float(
                item.get(
                    "relevance_score",
                    0,
                )
                or 0
            ),
        reverse=True,
    )


    unique = []
    seen = set()


    for item in combined:

        key = (
            str(
                item.get(
                    "collection",
                    "",
                )
            ),
            str(
                item.get(
                    "identifier",
                    "",
                )
            ),
            str(
                item.get(
                    "title",
                    "",
                )
            ),
        )


        if key in seen:
            continue


        seen.add(
            key
        )

        unique.append(
            item
        )


    return unique

def learn_from_tcc_results(
    query,
    ranked_results,
):

    for item in ranked_results:

        shard_idx = (
            item.get(
                "shard_idx"
            )
        )

        relevance_score = float(
            item.get(
                "relevance_score",
                0,
            )
            or 0
        )


        if (
            shard_idx is None
            or
            relevance_score <= 0
        ):
            continue


        update_shard_term_stats(
            query=query,
            shard_idx=shard_idx,
            relevance_score=
                relevance_score,
        )


# ============================================================
# EXPERIMENT 3 — DYNAMIC TCC COLLECTION DISCOVERY
# ============================================================

DYNAMIC_COLLECTION_PROBE_SHARDS = 1
DYNAMIC_COLLECTION_TOP_N = 2
DYNAMIC_COLLECTION_PROBE_LIMIT = 40


def build_tcc_collection_probe_sql(
    query,
    parquet_url,
    limit=DYNAMIC_COLLECTION_PROBE_LIMIT,
):

    features = (
        build_query_features(
            query
        )
    )

    normalized_query = (
        escape_sql_literal(
            features[
                "normalized_query"
            ]
        )
    )

    terms = features[
        "terms"
    ]

    bigrams = features[
        "bigrams"
    ]

    trigrams = features[
        "trigrams"
    ]

    if not terms:

        raise ValueError(
            "Query contains no searchable terms."
        )

    score_parts = []
    matched_term_parts = []

    for term in terms:

        safe_term = (
            escape_sql_literal(
                term
            )
        )

        weight = (
            term_weight(
                term
            )
        )

        score_parts.append(
            f"""
            CASE
                WHEN title_l LIKE '%{safe_term}%'
                    THEN {8 * weight}
                WHEN early_text_l LIKE '%{safe_term}%'
                    THEN {5 * weight}
                WHEN text_l LIKE '%{safe_term}%'
                    THEN {2 * weight}
                ELSE 0
            END
            """
        )

        matched_term_parts.append(
            f"""
            CASE
                WHEN title_l LIKE '%{safe_term}%'
                  OR text_l LIKE '%{safe_term}%'
                THEN 1
                ELSE 0
            END
            """
        )

    for phrase in bigrams:

        safe_phrase = (
            escape_sql_literal(
                phrase
            )
        )

        score_parts.append(
            f"""
            CASE
                WHEN title_l LIKE '%{safe_phrase}%'
                    THEN 14
                WHEN early_text_l LIKE '%{safe_phrase}%'
                    THEN 10
                WHEN text_l LIKE '%{safe_phrase}%'
                    THEN 6
                ELSE 0
            END
            """
        )

    for phrase in trigrams:

        safe_phrase = (
            escape_sql_literal(
                phrase
            )
        )

        score_parts.append(
            f"""
            CASE
                WHEN title_l LIKE '%{safe_phrase}%'
                    THEN 20
                WHEN early_text_l LIKE '%{safe_phrase}%'
                    THEN 14
                WHEN text_l LIKE '%{safe_phrase}%'
                    THEN 8
                ELSE 0
            END
            """
        )

    score_expr = (
        " + ".join(
            score_parts
        )
    )

    matched_terms_expr = (
        " + ".join(
            matched_term_parts
        )
    )

    safe_url = (
        escape_sql_literal(
            parquet_url
        )
    )

    return f"""
        WITH scoped AS (

            SELECT
                collection,
                title,
                text,

                lower(
                    coalesce(
                        title,
                        ''
                    )
                ) AS title_l,

                lower(
                    coalesce(
                        text,
                        ''
                    )
                ) AS text_l,

                lower(
                    substr(
                        coalesce(
                            text,
                            ''
                        ),
                        1,
                        {EARLY_TEXT_CHARS}
                    )
                ) AS early_text_l

            FROM read_parquet(
                '{safe_url}'
            )
        ),

        scored AS (

            SELECT
                collection,

                (
                    CASE
                        WHEN title_l LIKE '%{normalized_query}%'
                            THEN 40
                        WHEN early_text_l LIKE '%{normalized_query}%'
                            THEN 30
                        WHEN text_l LIKE '%{normalized_query}%'
                            THEN 20
                        ELSE 0
                    END
                    +
                    ({score_expr})
                ) AS relevance_score,

                ({matched_terms_expr})
                    AS matched_terms

            FROM scoped

            WHERE
                (
                    title_l LIKE '%{normalized_query}%'
                    OR
                    text_l LIKE '%{normalized_query}%'
                    OR
                    ({matched_terms_expr}) > 0
                )
        )

        SELECT
            collection,
            max(relevance_score)
                AS best_score,
            avg(relevance_score)
                AS mean_score,
            count(*)
                AS hit_count

        FROM scored

        GROUP BY collection

        ORDER BY
            best_score DESC,
            mean_score DESC,
            hit_count DESC

        LIMIT {int(limit)}
    """


def probe_tcc_collections_on_shard(
    query,
    shard,
):

    start_time = (
        time.perf_counter()
    )

    con = duckdb.connect(
        database=":memory:"
    )

    try:

        con.execute(
            f"SET threads = {int(DUCKDB_THREADS)}"
        )

        load_httpfs(
            con
        )

        sql = (
            build_tcc_collection_probe_sql(
                query=query,
                parquet_url=
                    shard[
                        "url"
                    ],
                limit=
                    DYNAMIC_COLLECTION_PROBE_LIMIT,
            )
        )

        result_df = (
            con.execute(
                sql
            )
            .fetchdf()
        )

        return {

            "shard_idx":
                shard[
                    "shard_idx"
                ],

            "elapsed_s":
                (
                    time.perf_counter()
                    -
                    start_time
                ),

            "results":
                result_df.to_dict(
                    orient="records"
                ),

            "error":
                None,
        }

    except Exception as exc:

        return {

            "shard_idx":
                shard[
                    "shard_idx"
                ],

            "elapsed_s":
                (
                    time.perf_counter()
                    -
                    start_time
                ),

            "results":
                [],

            "error":
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
        }

    finally:

        con.close()


async def discover_tcc_collections(
    query,
    selected_shards,
    top_n=DYNAMIC_COLLECTION_TOP_N,
):

    probe_shards = (
        selected_shards[
            :DYNAMIC_COLLECTION_PROBE_SHARDS
        ]
    )

    if not probe_shards:

        return {
            "collections": [],
            "candidates": [],
            "probe_results": [],
            "retrieval_time_s": 0.0,
        }

    start_time = (
        time.perf_counter()
    )

    probe_results = await asyncio.gather(
        *[
            asyncio.to_thread(
                probe_tcc_collections_on_shard,
                query,
                shard,
            )
            for shard
            in probe_shards
        ]
    )

    combined = defaultdict(
        lambda: {
            "best_score": 0.0,
            "weighted_score": 0.0,
            "hit_count": 0,
        }
    )

    for probe in probe_results:

        for item in probe.get(
            "results",
            [],
        ):

            collection = str(
                item.get(
                    "collection",
                    "",
                )
                or ""
            ).strip()

            if not collection:
                continue

            best_score = float(
                item.get(
                    "best_score",
                    0,
                )
                or 0
            )

            mean_score = float(
                item.get(
                    "mean_score",
                    0,
                )
                or 0
            )

            hit_count = int(
                item.get(
                    "hit_count",
                    0,
                )
                or 0
            )

            stats = combined[
                collection
            ]

            stats[
                "best_score"
            ] = max(
                stats[
                    "best_score"
                ],
                best_score,
            )

            stats[
                "weighted_score"
            ] += (
                mean_score
                *
                max(
                    hit_count,
                    1,
                )
            )

            stats[
                "hit_count"
            ] += hit_count

    candidates = []

    for collection, stats in combined.items():

        mean_score = (
            stats[
                "weighted_score"
            ]
            /
            max(
                stats[
                    "hit_count"
                ],
                1,
            )
        )

        candidates.append(
            {
                "collection":
                    collection,

                "best_score":
                    stats[
                        "best_score"
                    ],

                "mean_score":
                    mean_score,

                "hit_count":
                    stats[
                        "hit_count"
                    ],
            }
        )

    candidates.sort(
        key=lambda item: (
            float(
                item.get(
                    "best_score",
                    0,
                )
            ),
            float(
                item.get(
                    "mean_score",
                    0,
                )
            ),
            int(
                item.get(
                    "hit_count",
                    0,
                )
            ),
        ),
        reverse=True,
    )

    collections = [
        item[
            "collection"
        ]
        for item
        in candidates[
            :max(
                int(
                    top_n
                ),
                0,
            )
        ]
    ]

    return {
        "collections":
            collections,

        "candidates":
            candidates,

        "probe_results":
            probe_results,

        "retrieval_time_s":
            (
                time.perf_counter()
                -
                start_time
            ),
    }


async def retrieve_tcc_remote(
    query,
    collections=None,
    top_k=TOP_K_RESULTS,
    max_shards=MAX_REMOTE_SHARDS,
):

    start_time = (
        time.perf_counter()
    )

    if collections is None:

        selection = (
            select_tcc_collections(
                query
            )
        )

        collections = (
            selection[
                "collections"
            ]
        )

    collections = list(
        dict.fromkeys(
            collections
            or []
        )
    )

    shard_selection = (
        select_adaptive_shards(
            query=query,
            parquet_files=
                TCC_REMOTE_SHARDS,
            max_shards=
                max_shards,
        )
    )

    selected_shards = (
        shard_selection[
            "shards"
        ]
    )

    discovery = None

    # --------------------------------------------------------
    # Dynamic collection discovery only when the original
    # selector has no confident collection match.
    # --------------------------------------------------------

    if not collections:

        discovery = (
            await discover_tcc_collections(
                query=query,
                selected_shards=
                    selected_shards,
                top_n=
                    DYNAMIC_COLLECTION_TOP_N,
            )
        )

        collections = (
            discovery[
                "collections"
            ]
        )

    # If the evidence probe still finds no usable collection,
    # return cleanly rather than forcing a fallback corpus.

    if not collections:

        return {

            "engine":
                "tcc_remote",

            "query":
                query,

            "collections":
                [],

            "collection_discovery":
                discovery,

            "shard_selection":
                shard_selection[
                    "mode"
                ],

            "selected_shards":
                selected_shards,

            "shard_results":
                [],

            "shard_errors":
                0,

            "results":
                [],

            "result_count":
                0,

            "retrieval_time_s":
                (
                    time.perf_counter()
                    -
                    start_time
                ),
        }

    shard_results = (
        await search_selected_tcc_shards(
            query=query,
            selected_shards=
                selected_shards,
            collections=
                collections,
            limit=
                PER_SHARD_LIMIT,
        )
    )

    ranked_results = (
        merge_tcc_results(
            shard_results
        )
    )

    final_results = (
        ranked_results[
            :top_k
        ]
    )

    learn_from_tcc_results(
        query=query,
        ranked_results=
            final_results,
    )

    errors = [
        item
        for item in shard_results
        if item.get(
            "error"
        )
        is not None
    ]

    return {

        "engine":
            "tcc_remote",

        "query":
            query,

        "collections":
            collections,

        "collection_discovery":
            discovery,

        "shard_selection":
            shard_selection[
                "mode"
            ],

        "selected_shards":
            selected_shards,

        "shard_results":
            shard_results,

        "shard_errors":
            len(
                errors
            ),

        "results":
            final_results,

        "result_count":
            len(
                final_results
            ),

        "retrieval_time_s":
            (
                time.perf_counter()
                -
                start_time
            ),
    }

def format_3gpp_spec_number(
    identifier
):

    value = str(
        identifier
    )


    match = re.match(
        r"^(\d{5})(-\d+)?$",
        value,
    )


    if not match:
        return None


    base = match.group(
        1
    )

    suffix = (
        match.group(
            2
        )
        or
        ""
    )


    return (
        f"{base[:2]}."
        f"{base[2:]}"
        f"{suffix}"
    )

def parse_3gpp_path(
    file_path,
    url,
):

    release_match = re.search(
        r"/Rel-(\d+)/",
        f"/{file_path}",
    )


    spec_match = re.search(
        r"/(\d{5}(?:-\d+)?)/raw\.md$",
        f"/{file_path}",
    )


    if not spec_match:
        return None


    spec_number = (
        format_3gpp_spec_number(
            spec_match.group(
                1
            )
        )
    )


    if not spec_number:
        return None


    release = (
        int(
            release_match.group(
                1
            )
        )
        if release_match
        else None
    )


    return {

        "spec_number":
            spec_number,

        "release":
            release,

        "file_path":
            file_path,

        "url":
            url,
    }

def get_latest_3gpp_spec(
    spec_number
):

    available = (
        GPP_SPEC_PATH_INDEX.get(
            spec_number,
            [],
        )
    )


    if not available:
        return None


    return available[
        0
    ]

def fetch_3gpp_document(
    candidate
):

    start_time = (
        time.perf_counter()
    )


    request = urllib.request.Request(

        candidate[
            "url"
        ],

        headers={
            "User-Agent":
                "Telecom-AI-MCP-Version-A-Demo"
        },
    )


    try:

        with urllib.request.urlopen(
            request,
            timeout=
                GPP_FETCH_TIMEOUT_SECONDS,
        ) as response:

            raw = (
                response.read()
            )

            status = getattr(
                response,
                "status",
                None,
            )


        text = raw.decode(
            "utf-8",
            errors="replace",
        )


        return {

            **candidate,

            "http_status":
                status,

            "bytes":
                len(
                    raw
                ),

            "text":
                text,

            "elapsed_s":
                (
                    time.perf_counter()
                    -
                    start_time
                ),

            "error":
                None,
        }


    except Exception as exc:

        return {

            **candidate,

            "http_status":
                None,

            "bytes":
                0,

            "text":
                "",

            "elapsed_s":
                (
                    time.perf_counter()
                    -
                    start_time
                ),

            "error":
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
        }

def extract_3gpp_title(
    text,
    spec_number,
):

    for line in str(
        text
    ).splitlines()[
        :120
    ]:

        clean = re.sub(
            r"^#+\s*",
            "",
            line,
        ).strip()


        clean = re.sub(
            r"[*_~]+",
            "",
            clean,
        ).strip()


        if (
            len(
                clean
            ) >= 20

            and

            not clean.startswith(
                "!["
            )

            and

            "3gpp logo"
            not in clean.lower()
        ):

            return clean[
                :500
            ]


    return (
        f"3GPP Specification "
        f"{spec_number}"
    )

def score_text_block(
    query,
    title,
    text,
):

    features = (
        build_query_features(
            query
        )
    )


    normalized_query = (
        features[
            "normalized_query"
        ]
    )

    terms = (
        features[
            "terms"
        ]
    )

    bigrams = (
        features[
            "bigrams"
        ]
    )

    trigrams = (
        features[
            "trigrams"
        ]
    )

    proximity_pairs = (
        features[
            "proximity_pairs"
        ]
    )


    title_l = str(
        title
        or
        ""
    ).lower()


    text_l = str(
        text
        or
        ""
    ).lower()


    early_text_l = (
        text_l[
            :EARLY_TEXT_CHARS
        ]
    )


    score = 0.0
    matched_terms = 0
    matched_phrases = 0
    proximity_score = 0.0


    # Exact query.

    if normalized_query:

        if normalized_query in title_l:

            score += 40

        elif normalized_query in early_text_l:

            score += 30

        elif normalized_query in text_l:

            score += 20


    # Individual terms.

    for term in terms:

        weight = (
            term_weight(
                term
            )
        )


        if (
            term in title_l
            or
            term in text_l
        ):

            matched_terms += 1


        if term in title_l:

            score += (
                8
                *
                weight
            )

        elif term in early_text_l:

            score += (
                5
                *
                weight
            )

        elif term in text_l:

            score += (
                2
                *
                weight
            )


    # Bigrams.

    for phrase in bigrams:

        if (
            phrase in title_l
            or
            phrase in text_l
        ):

            matched_phrases += 1


        if phrase in title_l:

            score += 14

        elif phrase in early_text_l:

            score += 10

        elif phrase in text_l:

            score += 6


    # Trigrams.

    for phrase in trigrams:

        if (
            phrase in title_l
            or
            phrase in text_l
        ):

            matched_phrases += 1


        if phrase in title_l:

            score += 20

        elif phrase in early_text_l:

            score += 14

        elif phrase in text_l:

            score += 8


    # Proximity.

    for (
        term_a,
        term_b,
    ) in proximity_pairs:

        pattern_ab = re.compile(
            re.escape(
                term_a
            )
            +
            rf".{{0,{PROXIMITY_WINDOW}}}"
            +
            re.escape(
                term_b
            ),
            flags=re.DOTALL,
        )


        pattern_ba = re.compile(
            re.escape(
                term_b
            )
            +
            rf".{{0,{PROXIMITY_WINDOW}}}"
            +
            re.escape(
                term_a
            ),
            flags=re.DOTALL,
        )


        if (
            pattern_ab.search(
                text_l
            )
            or
            pattern_ba.search(
                text_l
            )
        ):

            proximity_score += 10


    score += (
        proximity_score
    )


    return {

        "matched_terms":
            matched_terms,

        "matched_phrases":
            matched_phrases,

        "proximity_score":
            proximity_score,

        "relevance_score":
            score,
    }

def rank_3gpp_document(
    query,
    document,
    windows_per_spec=
        GPP_WINDOWS_PER_SPEC,
):

    text = document.get(
        "text",
        "",
    )


    if not text:
        return []


    spec_number = (
        document[
            "spec_number"
        ]
    )


    title = (
        extract_3gpp_title(
            text,
            spec_number,
        )
    )


    step = max(
        1,
        (
            GPP_WINDOW_CHARS
            -
            GPP_WINDOW_OVERLAP
        ),
    )


    ranked = []


    for start in range(
        0,
        len(
            text
        ),
        step,
    ):

        window = (
            text[
                start:
                start
                +
                GPP_WINDOW_CHARS
            ]
        )


        if not window.strip():
            continue


        scoring = (
            score_text_block(
                query=query,
                title=title,
                text=window,
            )
        )


        if scoring[
            "relevance_score"
        ] <= 0:

            continue


        ranked.append(
            {

                "identifier":
                    spec_number,

                "collection":
                    "3GPP-Specifications",

                "date":
                    None,

                "title":
                    title,

                "creator":
                    "3GPP",

                "text":
                    window,

                "source_family":
                    "3GPP",

                "source_path":
                    document[
                        "file_path"
                    ],

                "source_shard":
                    document[
                        "url"
                    ],

                "release":
                    document[
                        "release"
                    ],

                "window_start":
                    start,

                **scoring,
            }
        )


    ranked.sort(
        key=lambda item:
            float(
                item[
                    "relevance_score"
                ]
            ),
        reverse=True,
    )


    return ranked[
        :windows_per_spec
    ]

async def retrieve_3gpp_remote(
    query,
    specs=None,
    top_k=TOP_K_RESULTS,
):

    start_time = (
        time.perf_counter()
    )


    if specs is None:

        inference = (
            infer_3gpp_spec_candidates(
                query
            )
        )

        specs = (
            inference[
                "specs"
            ]
        )


    specs = list(
        dict.fromkeys(
            specs
        )
    )


    selected_specs = []
    missing_specs = []


    for spec in specs:

        candidate = (
            get_latest_3gpp_spec(
                spec
            )
        )


        if candidate is None:

            missing_specs.append(
                spec
            )

            continue


        selected_specs.append(
            candidate
        )


    fetched_documents = []


    if selected_specs:

        fetched_documents = (
            await asyncio.gather(
                *[
                    asyncio.to_thread(
                        fetch_3gpp_document,
                        candidate,
                    )
                    for candidate
                    in selected_specs
                ]
            )
        )


    document_errors = [
        item
        for item in fetched_documents
        if item.get(
            "error"
        )
        is not None
    ]


    ranked_results = []


    for document in fetched_documents:

        if document.get(
            "error"
        ) is not None:

            continue


        ranked_results.extend(
            rank_3gpp_document(
                query=query,
                document=document,
            )
        )


    ranked_results.sort(
        key=lambda item:
            float(
                item.get(
                    "relevance_score",
                    0,
                )
                or 0
            ),
        reverse=True,
    )


    final_results = (
        ranked_results[
            :top_k
        ]
    )


    return {

        "engine":
            "3gpp_remote",

        "query":
            query,

        "requested_specs":
            specs,

        "selected_specs":
            selected_specs,

        "missing_specs":
            missing_specs,

        "fetched_documents":
            fetched_documents,

        "document_errors":
            len(
                document_errors
            ),

        "results":
            final_results,

        "result_count":
            len(
                final_results
            ),

        "retrieval_time_s":
            (
                time.perf_counter()
                -
                start_time
            ),
    }

def merge_cross_source_results(
    result_groups,
    top_k=TOP_K_RESULTS,
):

    combined = []


    for group in result_groups:

        combined.extend(
            group
        )


    combined.sort(
        key=lambda item:
            float(
                item.get(
                    "relevance_score",
                    0,
                )
                or 0
            ),
        reverse=True,
    )


    unique = []
    seen = set()


    for item in combined:

        key = (
            str(
                item.get(
                    "source_family",
                    "",
                )
            ),
            str(
                item.get(
                    "source_path",
                    "",
                )
            ),
            str(
                item.get(
                    "identifier",
                    "",
                )
            ),
            int(
                item.get(
                    "window_start",
                    0,
                )
                or 0
            ),
        )


        if key in seen:
            continue


        seen.add(
            key
        )


        unique.append(
            item
        )


        if len(
            unique
        ) >= top_k:

            break


    return unique

async def retrieve_version_a(
    query,
    top_k=TOP_K_RESULTS,
):
    """
    Conservative demo retrieval policy.

    MCP is intentionally used as a controlled authoritative
    knowledge-access tool, not as a universal broad retriever.

    Policy:
      - explicit / clearly inferred 3GPP -> dedicated 3GPP
      - clearly selected TCC collection -> that TCC collection only
      - clear mixed intent -> controlled 3GPP + selected TCC
      - no confident target -> return no MCP evidence

    Generic Wikipedia/Wikidata fallback is excluded.
    """

    overall_start = time.perf_counter()

    routing = route_telecom_query(
        query
    )

    route = str(
        routing.get(
            "route",
            "none",
        )
        or
        "none"
    ).strip().lower()

    gpp_specs = list(
        routing.get(
            "gpp_specs",
            [],
        )
        or
        []
    )

    tcc_collections = list(
        routing.get(
            "tcc_collections",
            [],
        )
        or
        []
    )


    # --------------------------------------------------------
    # No confident MCP source target
    # --------------------------------------------------------

    if route == "none":

        return {
            "query": query,
            "route": "none",
            "original_route": "none",
            "experiment": "CONTROLLED_MCP_WITH_GENERIC_FALLBACK",
            "routing": routing,
            "sources_searched": [],
            "tcc": None,
            "3gpp": None,
            "results": [],
            "result_count": 0,
            "stop_reason": "no_confident_mcp_source_target",
            "retrieval_time_s": (
                time.perf_counter()
                -
                overall_start
            ),
        }


    # --------------------------------------------------------
    # Dedicated 3GPP
    # --------------------------------------------------------

    if route == "3gpp":

        gpp_result = await retrieve_3gpp_remote(
            query=query,
            specs=gpp_specs,
            top_k=top_k,
        )

        final_results = list(
            gpp_result.get(
                "results",
                [],
            )
            or
            []
        )[:top_k]

        return {
            "query": query,
            "route": "3gpp",
            "original_route": "3gpp",
            "experiment": "CONTROLLED_MCP_WITH_GENERIC_FALLBACK",
            "routing": routing,
            "sources_searched": ["3GPP"],
            "tcc": None,
            "3gpp": gpp_result,
            "results": final_results,
            "result_count": len(final_results),
            "retrieval_time_s": (
                time.perf_counter()
                -
                overall_start
            ),
        }


    # --------------------------------------------------------
    # Selected TCC collection(s) only
    # --------------------------------------------------------

    if route == "tcc":

        if not tcc_collections:

            # Defensive generic fallback for demo continuity.
            tcc_collections = [
                "Wikipedia-Telecom",
                "Wikidata-Telecom",
            ]

            routing = dict(
                routing
            )

            routing["tcc_collections"] = list(
                tcc_collections
            )

            routing.setdefault(
                "tcc_reasons",
                [],
            ).append(
                "defensive generic MCP fallback"
            )

        tcc_result = await retrieve_tcc_remote(
            query=query,
            collections=tcc_collections,
            top_k=top_k,
            max_shards=MAX_REMOTE_SHARDS,
        )

        final_results = list(
            tcc_result.get(
                "results",
                [],
            )
            or
            []
        )[:top_k]

        return {
            "query": query,
            "route": "tcc",
            "original_route": "tcc",
            "experiment": "CONTROLLED_MCP_WITH_GENERIC_FALLBACK",
            "routing": routing,
            "sources_searched": ["TCC"],
            "tcc": tcc_result,
            "3gpp": None,
            "results": final_results,
            "result_count": len(final_results),
            "retrieval_time_s": (
                time.perf_counter()
                -
                overall_start
            ),
        }


    # --------------------------------------------------------
    # Controlled hybrid
    # --------------------------------------------------------

    if route == "hybrid":

        # If one side has no actual target, degrade cleanly to
        # the side that does instead of triggering broad search.
        if not tcc_collections:

            gpp_result = await retrieve_3gpp_remote(
                query=query,
                specs=gpp_specs,
                top_k=top_k,
            )

            final_results = list(
                gpp_result.get(
                    "results",
                    [],
                )
                or
                []
            )[:top_k]

            return {
                "query": query,
                "route": "3gpp",
                "original_route": "hybrid",
                "experiment": "CONTROLLED_MCP_WITH_GENERIC_FALLBACK",
                "routing": routing,
                "sources_searched": ["3GPP"],
                "tcc": None,
                "3gpp": gpp_result,
                "results": final_results,
                "result_count": len(final_results),
                "retrieval_time_s": (
                    time.perf_counter()
                    -
                    overall_start
                ),
            }

        (
            gpp_result,
            tcc_result,
        ) = await asyncio.gather(
            retrieve_3gpp_remote(
                query=query,
                specs=gpp_specs,
                top_k=top_k,
            ),
            retrieve_tcc_remote(
                query=query,
                collections=tcc_collections,
                top_k=top_k,
                max_shards=MAX_REMOTE_SHARDS,
            ),
        )

        final_results = merge_cross_source_results(
            [
                list(
                    gpp_result.get(
                        "results",
                        [],
                    )
                    or
                    []
                ),
                list(
                    tcc_result.get(
                        "results",
                        [],
                    )
                    or
                    []
                ),
            ],
            top_k=top_k,
        )

        return {
            "query": query,
            "route": "hybrid",
            "original_route": "hybrid",
            "experiment": "CONTROLLED_MCP_WITH_GENERIC_FALLBACK",
            "routing": routing,
            "sources_searched": [
                "3GPP",
                "TCC",
            ],
            "tcc": tcc_result,
            "3gpp": gpp_result,
            "results": final_results,
            "result_count": len(final_results),
            "retrieval_time_s": (
                time.perf_counter()
                -
                overall_start
            ),
        }


    # Defensive fallback
    return {
        "query": query,
        "route": "none",
        "original_route": route,
        "experiment": "CONTROLLED_MCP_WITH_GENERIC_FALLBACK",
        "routing": routing,
        "sources_searched": [],
        "tcc": None,
        "3gpp": None,
        "results": [],
        "result_count": 0,
        "stop_reason": "unsupported_route",
        "retrieval_time_s": (
            time.perf_counter()
            -
            overall_start
        ),
    }


# ============================================================
# COMPLETE 3GPP REGISTRY AFTER CELL 2C FUNCTIONS ARE DEFINED
# ============================================================

def finalize_3gpp_spec_registry() -> None:
    global GPP_SPEC_PATH_INDEX

    initialize_remote_sources()

    if GPP_SPEC_PATH_INDEX:
        return

    for file_path, url in zip(
        REMOTE_SOURCE_REGISTRY["3gpp"]["files"],
        REMOTE_SOURCE_REGISTRY["3gpp"]["urls"],
    ):
        metadata = parse_3gpp_path(file_path, url)
        if metadata:
            GPP_SPEC_PATH_INDEX[metadata["spec_number"]].append(metadata)

    for spec_number in GPP_SPEC_PATH_INDEX:
        GPP_SPEC_PATH_INDEX[spec_number].sort(
            key=lambda item: (
                item["release"] if item["release"] is not None else -1
            ),
            reverse=True,
        )


def ensure_mcp_runtime_ready() -> None:
    initialize_remote_sources()
    finalize_3gpp_spec_registry()


# Preserve the validated Cell 2C bootstrap of DuckDB HTTPFS.
_bootstrap_conn = duckdb.connect(database=":memory:")
try:
    load_httpfs(_bootstrap_conn)
finally:
    _bootstrap_conn.close()

# Initialize source metadata only; no corpus is downloaded.
ensure_mcp_runtime_ready()

# ============================================================
# FASTMCP KNOWLEDGE TOOL — CELL 2D
# ============================================================

mcp = FastMCP("Telecom Knowledge Service — Version A")


def clean_mcp_text(
    value,
    max_chars=MCP_EXCERPT_CHARS,
):

    if value is None:
        return ""


    text = str(
        value
    )


    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


    if len(
        text
    ) > max_chars:

        text = (
            text[
                :max_chars
            ].rstrip()
            +
            " ..."
        )


    return text

def safe_float(
    value,
    default=0.0,
):

    try:

        if value is None:

            return float(
                default
            )


        return float(
            value
        )


    except Exception:

        return float(
            default
        )

def safe_int(
    value,
    default=0,
):

    try:

        if value is None:

            return int(
                default
            )


        return int(
            value
        )


    except Exception:

        return int(
            default
        )

def format_mcp_evidence(
    item,
    rank,
):

    source_family = str(
        item.get(
            "source_family",
            "",
        )
        or
        ""
    )


    identifier = str(
        item.get(
            "identifier",
            "",
        )
        or
        ""
    )


    collection = str(
        item.get(
            "collection",
            "",
        )
        or
        ""
    )


    title = clean_mcp_text(
        item.get(
            "title",
            "",
        ),
        max_chars=500,
    )


    source_path = str(
        item.get(
            "source_path",
            "",
        )
        or
        ""
    )


    release = item.get(
        "release"
    )


    if release is not None:

        release = str(
            release
        )


    evidence_text = clean_mcp_text(
        item.get(
            "text",
            "",
        ),
        max_chars=
            MCP_EXCERPT_CHARS,
    )


    return {

        "rank":
            safe_int(
                rank
            ),

        "source_family":
            source_family,

        "collection":
            collection,

        "identifier":
            identifier,

        "title":
            title,

        "release":
            release,

        "source_path":
            source_path,

        # Retain provenance for later demo inspection.
        "source_shard":
            str(
                item.get(
                    "source_shard",
                    "",
                )
                or
                ""
            ),

        "section_heading":
            clean_mcp_text(
                item.get(
                    "section_heading",
                    "",
                ),
                max_chars=500,
            ),

        # Retrieval diagnostics.
        "relevance_score":
            safe_float(
                item.get(
                    "relevance_score",
                    0,
                )
            ),

        "matched_terms":
            safe_int(
                item.get(
                    "matched_terms",
                    0,
                )
            ),

        "matched_phrases":
            safe_int(
                item.get(
                    "matched_phrases",
                    0,
                )
            ),

        "proximity_score":
            safe_float(
                item.get(
                    "proximity_score",
                    0,
                )
            ),

        "primary_coverage":
            safe_float(
                item.get(
                    "primary_coverage",
                    0,
                )
            ),

        # Bounded content exposed to downstream generation.
        "evidence":
            evidence_text,

        "evidence_chars":
            len(
                evidence_text
            ),
    }

def build_mcp_trace(
    retrieval_result
):

    routing = (
        retrieval_result.get(
            "routing",
            {},
        )
    )


    trace = {

        "architecture":
            ARCHITECTURE_VERSION,

        "retrieval_architecture":
            RETRIEVAL_ARCHITECTURE,

        "route":
            retrieval_result.get(
                "route",
                "",
            ),

        "sources_searched":
            list(
                retrieval_result.get(
                    "sources_searched",
                    [],
                )
            ),

        "retrieval_time_s":
            safe_float(
                retrieval_result.get(
                    "retrieval_time_s",
                    0,
                )
            ),

        "gpp_specs":
            list(
                routing.get(
                    "gpp_specs",
                    [],
                )
            ),

        "tcc_collections":
            list(
                routing.get(
                    "tcc_collections",
                    [],
                )
            ),
    }


    # --------------------------------------------------------
    # 3GPP trace
    # --------------------------------------------------------

    gpp_result = (
        retrieval_result.get(
            "3gpp"
        )
    )


    if gpp_result is not None:

        trace[
            "3gpp"
        ] = {

            "requested_specs":
                list(
                    gpp_result.get(
                        "requested_specs",
                        [],
                    )
                ),

            "selected_specs":
                [

                    {
                        "spec_number":
                            str(
                                item.get(
                                    "spec_number",
                                    "",
                                )
                            ),

                        "release":
                            item.get(
                                "release"
                            ),
                    }

                    for item
                    in gpp_result.get(
                        "selected_specs",
                        [],
                    )
                ],

            "missing_specs":
                list(
                    gpp_result.get(
                        "missing_specs",
                        [],
                    )
                ),

            "document_errors":
                safe_int(
                    gpp_result.get(
                        "document_errors",
                        0,
                    )
                ),

            "retrieval_time_s":
                safe_float(
                    gpp_result.get(
                        "retrieval_time_s",
                        0,
                    )
                ),
        }


    else:

        trace[
            "3gpp"
        ] = None


    # --------------------------------------------------------
    # TCC trace
    # --------------------------------------------------------

    tcc_result = (
        retrieval_result.get(
            "tcc"
        )
    )


    if tcc_result is not None:

        trace[
            "tcc"
        ] = {

            "collections":
                list(
                    tcc_result.get(
                        "collections",
                        [],
                    )
                ),

            "shard_selection":
                str(
                    tcc_result.get(
                        "shard_selection",
                        "",
                    )
                ),

            "shards_searched":
                len(
                    tcc_result.get(
                        "selected_shards",
                        [],
                    )
                ),

            "shard_errors":
                safe_int(
                    tcc_result.get(
                        "shard_errors",
                        0,
                    )
                ),

            "retrieval_time_s":
                safe_float(
                    tcc_result.get(
                        "retrieval_time_s",
                        0,
                    )
                ),
        }


    else:

        trace[
            "tcc"
        ] = None


    return trace

@mcp.tool()
async def search_telecom_knowledge(
    query: str,
    top_k: int = TOP_K_RESULTS,
) -> dict:

    """
    Search authoritative telecommunications documentation.

    The knowledge service conservatively selects an authoritative
    retrieval route when the query provides a confident source target:

        - dedicated 3GPP
        - selected GSMA Telco Common Corpus collection
        - controlled Hybrid 3GPP + selected TCC

    If no confident authoritative target is identified, the tool
    uses a clearly identified generic telecom fallback collection so
    the MCP path remains populated in the demo.

    Returns ranked bounded evidence together with lightweight
    retrieval provenance and timing metadata.
    """


    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if (
        not query
        or
        not query.strip()
    ):

        raise ValueError(
            "query must not be empty."
        )


    requested_top_k = safe_int(
        top_k,
        default=
            TOP_K_RESULTS,
    )


    effective_top_k = max(
        1,
        min(
            requested_top_k,
            TOP_K_RESULTS,
            MAX_RETRIEVED_SOURCES,
        ),
    )


    tool_start = (
        time.perf_counter()
    )


    # --------------------------------------------------------
    # Dynamic Version A retrieval
    # --------------------------------------------------------

    retrieval_result = (
        await retrieve_version_a(
            query=
                query,

            top_k=
                effective_top_k,
        )
    )


    # --------------------------------------------------------
    # Format bounded MCP evidence
    # --------------------------------------------------------

    raw_results = (
        retrieval_result.get(
            "results",
            [],
        )
    )


    evidence = [

        format_mcp_evidence(
            item=
                item,

            rank=
                rank,
        )

        for rank, item
        in enumerate(
            raw_results[
                :effective_top_k
            ],
            start=1,
        )
    ]


    # --------------------------------------------------------
    # Evidence-source distribution
    # --------------------------------------------------------

    source_distribution = {}


    for item in evidence:

        source_family = (
            item[
                "source_family"
            ]
            or
            "UNKNOWN"
        )


        source_distribution[
            source_family
        ] = (

            source_distribution.get(
                source_family,
                0,
            )
            +
            1
        )


    # --------------------------------------------------------
    # Lightweight trace
    # --------------------------------------------------------

    trace = (
        build_mcp_trace(
            retrieval_result
        )
    )


    tool_elapsed_s = (
        time.perf_counter()
        -
        tool_start
    )


    trace[
        "tool_elapsed_s"
    ] = safe_float(
        tool_elapsed_s
    )


    # --------------------------------------------------------
    # Final MCP payload
    # --------------------------------------------------------

    return {

        "query":
            query,

        "route":
            retrieval_result.get(
                "route",
                "",
            ),

        "sources_searched":
            list(
                retrieval_result.get(
                    "sources_searched",
                    [],
                )
            ),

        "source_distribution":
            source_distribution,

        "result_count":
            len(
                evidence
            ),

        "top_k":
            effective_top_k,

        "evidence":
            evidence,

        "trace":
            trace,
    }



def get_mcp_runtime_status() -> dict[str, Any]:
    ensure_mcp_runtime_ready()
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "retrieval_architecture": RETRIEVAL_ARCHITECTURE,
        "tcc_repo": REMOTE_CORPORA["tcc"]["repo_id"],
        "tcc_files": REMOTE_SOURCE_REGISTRY["tcc"]["file_count"],
        "3gpp_repo": REMOTE_CORPORA["3gpp"]["repo_id"],
        "3gpp_files": REMOTE_SOURCE_REGISTRY["3gpp"]["file_count"],
        "persistent_local_kb": False,
        "top_k": TOP_K_RESULTS,
        "max_remote_tcc_shards": MAX_REMOTE_SHARDS,
        "max_retrieved_sources": MAX_RETRIEVED_SOURCES,
        "routes": ["3gpp", "tcc", "hybrid"],
        "tcc_access": "DuckDB HTTPFS",
        "3gpp_access": "Direct HTTP raw.md",
    }

