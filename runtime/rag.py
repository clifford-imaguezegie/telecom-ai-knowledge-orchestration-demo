from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any


# ============================================================
# TELECOM AI KNOWLEDGE ORCHESTRATION — RAG V1
# ============================================================

RAG_EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "BAAI/bge-m3",
).strip()

RAG_FAISS_INDEX_PATH = os.getenv(
    "RAG_FAISS_INDEX_PATH",
    "",
).strip()

RAG_METADATA_DIR = os.getenv(
    "RAG_METADATA_DIR",
    "",
).strip()


# Kaggle deployment assets
RAG_FAISS_KAGGLE_DATASET = os.getenv(
    "RAG_FAISS_KAGGLE_DATASET",
    "cliffordimaguezegie/telecom-bge-m3-faiss",
).strip()

RAG_METADATA_KAGGLE_DATASET = os.getenv(
    "RAG_METADATA_KAGGLE_DATASET",
    "cliffordimaguezegie/telecom-reconciled-chunks",
).strip()


# Validated Module 4 retrieval settings
RAG_TOP_K = 5

RAG_MAX_EVIDENCE_ITEMS = 5

RAG_MAX_EVIDENCE_CHARS = 2500

RAG_MAX_CONTEXT_CHARS = 12500

RAG_MODEL_MAX_SEQ_LENGTH = 1280


# ============================================================
# VALIDATED MODULE 4 RAG V1 IDENTITY
# ============================================================

EXPECTED_VECTOR_COUNT = 1_506_367

EXPECTED_VECTOR_DIMENSION = 1024

EXPECTED_METADATA_SHARDS = 151

STANDARD_SHARD_ROWS = 10_000


# ============================================================
# LAZY RUNTIME STATE
# ============================================================

_embedding_model = None

_faiss_index = None

_metadata_shards: list[Path] | None = None

_resolved_index_path: Path | None = None

_resolved_metadata_dir: Path | None = None

# Deployment hardening:
# serialize one-time heavy initialization inside a single
# Cloud Run / Streamlit Python process. These locks do not
# alter retrieval behaviour; they prevent concurrent sessions
# from racing through asset download, model load, or FAISS load.
_RAG_ASSET_LOCK = threading.RLock()
_RAG_EMBEDDING_LOCK = threading.Lock()
_RAG_FAISS_LOCK = threading.Lock()


# ============================================================
# OPTIONAL DEPENDENCY HELPERS
# ============================================================

def _import_faiss():
    """
    Import FAISS only when RAG retrieval is actually used.
    """

    try:
        import faiss

    except ImportError as exc:
        raise RuntimeError(
            "faiss is not installed. "
            "Install faiss-cpu before using RAG retrieval."
        ) from exc

    return faiss


def _import_sentence_transformer():
    """
    Import SentenceTransformer only when retrieval is used.
    """

    try:
        from sentence_transformers import SentenceTransformer

    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Install sentence-transformers before using RAG retrieval."
        ) from exc

    return SentenceTransformer


def _import_kagglehub():
    """
    Import KaggleHub only when automatic asset resolution
    is required.
    """

    try:
        import kagglehub

    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed. "
            "Install kagglehub before using automatic "
            "RAG asset download."
        ) from exc

    return kagglehub


# ============================================================
# SORTING / DISCOVERY HELPERS
# ============================================================

def _natural_sort_key(
    path: Path,
) -> list[Any]:
    """
    Natural filename sorting.

    Example:
        chunks_2.jsonl
        chunks_10.jsonl

    is sorted numerically rather than lexicographically.
    """

    parts = re.split(
        r"(\d+)",
        path.name.lower(),
    )

    return [
        int(part)
        if part.isdigit()
        else part
        for part in parts
    ]


def _discover_faiss_index(
    root: Path,
) -> Path:
    """
    Recursively discover the FAISS index inside a
    KaggleHub dataset directory.
    """

    candidates = []

    for pattern in (
        "*.faiss",
        "*.index",
    ):
        candidates.extend(
            root.rglob(
                pattern
            )
        )

    candidates = [
        path
        for path in candidates
        if path.is_file()
    ]

    if not candidates:
        raise RuntimeError(
            "No FAISS index file was found under "
            f"{root}"
        )

    # Prefer the largest candidate because the validated
    # production index is the large vector index.
    candidates.sort(
        key=lambda path: path.stat().st_size,
        reverse=True,
    )

    return candidates[0]


def _discover_metadata_shards(
    root: Path,
) -> list[Path]:
    """
    Recursively discover the 151 JSONL metadata shards.
    """

    shards = [
        path
        for path in root.rglob(
            "*.jsonl"
        )
        if path.is_file()
    ]

    shards.sort(
        key=_natural_sort_key
    )

    if not shards:
        raise RuntimeError(
            "No JSONL metadata shards were found under "
            f"{root}"
        )

    if len(shards) != EXPECTED_METADATA_SHARDS:
        raise RuntimeError(
            "Unexpected RAG metadata shard count. "
            f"Expected {EXPECTED_METADATA_SHARDS}, "
            f"found {len(shards)} under {root}."
        )

    return shards


# ============================================================
# KAGGLE ASSET DOWNLOAD
# ============================================================

def _download_kaggle_dataset(
    dataset_handle: str,
) -> Path:
    """
    Download or resolve a cached Kaggle dataset.

    Deployment hardening:
    - explicitly read KAGGLE_API_TOKEN from the runtime environment
    - validate presence / expected token prefix
    - normalize the value back into os.environ before importing
      and calling KaggleHub

    The token value is never logged or returned.
    """

    kaggle_token = os.getenv(
        "KAGGLE_API_TOKEN",
        "",
    ).strip()

    if not kaggle_token:
        raise RuntimeError(
            "KAGGLE_API_TOKEN is not available in the "
            "runtime environment."
        )

    if not kaggle_token.startswith("KGAT_"):
        raise RuntimeError(
            "KAGGLE_API_TOKEN is present but has an "
            "unexpected format."
        )

    # Ensure KaggleHub sees the exact normalized token value
    # in the current process environment.
    os.environ["KAGGLE_API_TOKEN"] = kaggle_token

    kagglehub = _import_kagglehub()

    try:
        resolved_path = kagglehub.dataset_download(
            dataset_handle
        )

    except Exception as exc:
        raise RuntimeError(
            "Unable to access Kaggle dataset "
            f"'{dataset_handle}'. "
            "KAGGLE_API_TOKEN is present and structurally valid; "
            "verify runtime authentication and dataset access."
        ) from exc

    path = Path(
        resolved_path
    )

    if not path.exists():
        raise RuntimeError(
            "KaggleHub returned a dataset path that "
            f"does not exist: {path}"
        )

    return path


# ============================================================
# ASSET RESOLUTION
# ============================================================

def _resolve_rag_assets_unlocked(
    force_refresh: bool = False,
) -> dict[str, Any]:
    """
    Resolve the deployment RAG assets.

    Resolution order:

    FAISS:
        1. RAG_FAISS_INDEX_PATH
        2. Kaggle dataset download/cache

    Metadata:
        1. RAG_METADATA_DIR
        2. Kaggle dataset download/cache

    Heavy assets are not downloaded until this function or
    retrieval is actually invoked.
    """

    global _resolved_index_path
    global _resolved_metadata_dir
    global _metadata_shards

    if force_refresh:
        _resolved_index_path = None
        _resolved_metadata_dir = None
        _metadata_shards = None

    # --------------------------------------------------------
    # FAISS INDEX
    # --------------------------------------------------------

    index_source = None

    if _resolved_index_path is None:

        if RAG_FAISS_INDEX_PATH:

            explicit_index = Path(
                RAG_FAISS_INDEX_PATH
            ).expanduser()

            if not explicit_index.is_file():
                raise RuntimeError(
                    "Configured RAG_FAISS_INDEX_PATH "
                    f"does not exist: {explicit_index}"
                )

            _resolved_index_path = (
                explicit_index.resolve()
            )

            index_source = (
                "EXPLICIT_LOCAL_PATH"
            )

        else:

            faiss_dataset_root = (
                _download_kaggle_dataset(
                    RAG_FAISS_KAGGLE_DATASET
                )
            )

            _resolved_index_path = (
                _discover_faiss_index(
                    faiss_dataset_root
                )
            )

            index_source = (
                "KAGGLE_DATASET"
            )

    else:
        index_source = (
            "RESOLVED_CACHE"
        )

    # --------------------------------------------------------
    # METADATA SHARDS
    # --------------------------------------------------------

    metadata_source = None

    if _resolved_metadata_dir is None:

        if RAG_METADATA_DIR:

            explicit_metadata = Path(
                RAG_METADATA_DIR
            ).expanduser()

            if not explicit_metadata.is_dir():
                raise RuntimeError(
                    "Configured RAG_METADATA_DIR "
                    f"does not exist: {explicit_metadata}"
                )

            metadata_shards = (
                _discover_metadata_shards(
                    explicit_metadata
                )
            )

            _resolved_metadata_dir = (
                explicit_metadata.resolve()
            )

            _metadata_shards = (
                metadata_shards
            )

            metadata_source = (
                "EXPLICIT_LOCAL_PATH"
            )

        else:

            metadata_dataset_root = (
                _download_kaggle_dataset(
                    RAG_METADATA_KAGGLE_DATASET
                )
            )

            metadata_shards = (
                _discover_metadata_shards(
                    metadata_dataset_root
                )
            )

            # The JSONL files may live below the Kaggle
            # root. Store the common parent for reporting.
            metadata_parents = {
                shard.parent.resolve()
                for shard in metadata_shards
            }

            if len(metadata_parents) == 1:
                _resolved_metadata_dir = (
                    next(
                        iter(
                            metadata_parents
                        )
                    )
                )
            else:
                _resolved_metadata_dir = (
                    metadata_dataset_root.resolve()
                )

            _metadata_shards = (
                metadata_shards
            )

            metadata_source = (
                "KAGGLE_DATASET"
            )

    else:
        metadata_source = (
            "RESOLVED_CACHE"
        )

    return {
        "index_path": str(
            _resolved_index_path
        ),
        "metadata_dir": str(
            _resolved_metadata_dir
        ),
        "metadata_shard_count": len(
            _metadata_shards
            or
            []
        ),
        "index_source": index_source,
        "metadata_source": metadata_source,
        "faiss_dataset":
            RAG_FAISS_KAGGLE_DATASET,
        "metadata_dataset":
            RAG_METADATA_KAGGLE_DATASET,
    }


def resolve_rag_assets(
    force_refresh: bool = False,
) -> dict[str, Any]:
    """
    Thread-safe wrapper around deployment asset resolution.

    Only one Streamlit session may resolve/download the heavy
    Kaggle-backed RAG assets at a time. Other sessions wait for
    the same local cache to become ready.
    """
    with _RAG_ASSET_LOCK:
        return _resolve_rag_assets_unlocked(
            force_refresh=force_refresh,
        )


# ============================================================
# METADATA SHARD REGISTRY
# ============================================================

def get_metadata_shards() -> list[Path]:
    """
    Return the validated metadata shards in deterministic order.
    """

    global _metadata_shards

    if _metadata_shards is None:
        resolve_rag_assets()

    if not _metadata_shards:
        raise RuntimeError(
            "RAG metadata shards could not be resolved."
        )

    return _metadata_shards


# ============================================================
# EMBEDDING MODEL
# ============================================================

def _get_embedding_model_unlocked():
    """
    Lazy-load the validated BGE-M3 embedding model.
    """

    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    SentenceTransformer = (
        _import_sentence_transformer()
    )

    model = SentenceTransformer(
        RAG_EMBEDDING_MODEL
    )

    model.max_seq_length = (
        RAG_MODEL_MAX_SEQ_LENGTH
    )

    _embedding_model = model

    return _embedding_model


def get_embedding_model():
    """
    Thread-safe BGE-M3 lazy loader.

    Prevents two concurrent Streamlit sessions from loading a
    second copy of the embedding model into memory.
    """
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    with _RAG_EMBEDDING_LOCK:
        if _embedding_model is not None:
            return _embedding_model

        return _get_embedding_model_unlocked()


# ============================================================
# FAISS INDEX
# ============================================================

def _get_faiss_index_unlocked():
    """
    Lazy-load the validated FAISS IndexFlatIP index.
    """

    global _faiss_index

    if _faiss_index is not None:
        return _faiss_index

    assets = resolve_rag_assets()

    index_path = Path(
        assets["index_path"]
    )

    faiss = _import_faiss()

    index = faiss.read_index(
        str(
            index_path
        )
    )

    if index.ntotal != EXPECTED_VECTOR_COUNT:
        raise RuntimeError(
            "Unexpected FAISS vector count. "
            f"Expected {EXPECTED_VECTOR_COUNT:,}, "
            f"found {index.ntotal:,}."
        )

    if index.d != EXPECTED_VECTOR_DIMENSION:
        raise RuntimeError(
            "Unexpected FAISS vector dimension. "
            f"Expected {EXPECTED_VECTOR_DIMENSION}, "
            f"found {index.d}."
        )

    _faiss_index = index

    return _faiss_index


def get_faiss_index():
    """
    Thread-safe FAISS lazy loader.

    Prevents duplicate ~multi-GB index loads when more than one
    Streamlit session reaches RAG during cold initialization.
    """
    global _faiss_index

    if _faiss_index is not None:
        return _faiss_index

    with _RAG_FAISS_LOCK:
        if _faiss_index is not None:
            return _faiss_index

        return _get_faiss_index_unlocked()


# ============================================================
# QUERY EMBEDDING
# ============================================================

def embed_query(
    query: str,
):
    """
    Encode one query using BGE-M3.

    Embeddings are normalized so inner product behaves as
    cosine similarity, matching the validated Module 4
    FAISS IndexFlatIP design.
    """

    query = str(
        query or ""
    ).strip()

    if not query:
        raise ValueError(
            "RAG query cannot be empty."
        )

    model = get_embedding_model()

    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embedding.astype(
        "float32"
    )


# ============================================================
# VECTOR ID → SHARD / LOCAL ROW
# ============================================================

def vector_id_to_metadata_position(
    vector_id: int,
) -> tuple[int, int]:
    """
    Resolve a global FAISS vector ID to:

        metadata shard index
        local row index

    Validated corpus construction:
        shards 0..149 -> 10,000 rows each
        shard 150     -> 6,367 rows
    """

    if vector_id < 0:
        raise ValueError(
            f"Invalid vector_id: {vector_id}"
        )

    if vector_id >= EXPECTED_VECTOR_COUNT:
        raise ValueError(
            "vector_id exceeds validated RAG corpus size: "
            f"{vector_id}"
        )

    shard_index = (
        vector_id
        //
        STANDARD_SHARD_ROWS
    )

    local_row = (
        vector_id
        %
        STANDARD_SHARD_ROWS
    )

    return (
        shard_index,
        local_row,
    )


# ============================================================
# METADATA ROW LOOKUP
# ============================================================

def load_metadata_row(
    vector_id: int,
) -> dict[str, Any]:
    """
    Retrieve one metadata row for a FAISS vector ID without
    loading the complete metadata corpus into memory.
    """

    shards = get_metadata_shards()

    shard_index, local_row = (
        vector_id_to_metadata_position(
            vector_id
        )
    )

    if shard_index >= len(shards):
        raise RuntimeError(
            "Resolved metadata shard index exceeds "
            "available shard count."
        )

    shard_path = shards[
        shard_index
    ]

    with shard_path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for row_index, line in enumerate(
            handle
        ):

            if row_index != local_row:
                continue

            line = line.strip()

            if not line:
                raise RuntimeError(
                    "Resolved metadata row is empty: "
                    f"vector_id={vector_id}, "
                    f"shard={shard_path.name}, "
                    f"row={local_row}"
                )

            try:
                metadata = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Invalid JSON metadata row: "
                    f"vector_id={vector_id}, "
                    f"shard={shard_path.name}, "
                    f"row={local_row}"
                ) from exc

            return metadata

    raise RuntimeError(
        "Metadata row could not be resolved: "
        f"vector_id={vector_id}, "
        f"shard={shard_path.name}, "
        f"row={local_row}"
    )


# ============================================================
# METADATA NORMALIZATION
# ============================================================

def _first_nonempty(
    metadata: dict[str, Any],
    *keys: str,
) -> str:
    """
    Return the first non-empty metadata value.
    """

    for key in keys:

        value = metadata.get(
            key
        )

        if value is None:
            continue

        text = str(
            value
        ).strip()

        if text:
            return text

    return ""


def normalize_metadata_result(
    vector_id: int,
    score: float,
    rank: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize one RAG retrieval result into the
    deployment evidence contract.
    """

    text = _first_nonempty(
        metadata,
        "text",
        "content",
        "page_content",
        "chunk_text",
    )

    source = _first_nonempty(
        metadata,
        "source",
        "source_name",
        "dataset",
        "corpus",
    )

    title = _first_nonempty(
        metadata,
        "title",
        "document_title",
        "doc_title",
        "name",
    )

    chunk_id = _first_nonempty(
        metadata,
        "chunk_id",
        "id",
    )

    document_id = _first_nonempty(
        metadata,
        "document_id",
        "doc_id",
    )

    path = _first_nonempty(
        metadata,
        "path",
        "file_path",
        "filepath",
        "url",
    )

    return {
        "rank": rank,
        "vector_id": vector_id,
        "score": float(
            score
        ),
        "chunk_id": chunk_id,
        "document_id": document_id,
        "source": source,
        "title": title,
        "path": path,
        "text": text,
        "text_chars": len(
            text
        ),
        "metadata": metadata,
    }


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_rag(
    query: str,
    top_k: int = RAG_TOP_K,
) -> dict[str, Any]:
    """
    Retrieve top-K evidence from the validated Module 4
    RAG V1 corpus.
    """

    start = time.perf_counter()

    query = str(
        query or ""
    ).strip()

    if not query:
        raise ValueError(
            "RAG query cannot be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    index = get_faiss_index()

    embedding_start = (
        time.perf_counter()
    )

    query_embedding = embed_query(
        query
    )

    embedding_elapsed_s = (
        time.perf_counter()
        -
        embedding_start
    )

    search_start = (
        time.perf_counter()
    )

    scores, vector_ids = (
        index.search(
            query_embedding,
            top_k,
        )
    )

    search_elapsed_s = (
        time.perf_counter()
        -
        search_start
    )

    metadata_start = (
        time.perf_counter()
    )

    results = []

    for rank, (
        vector_id,
        score,
    ) in enumerate(
        zip(
            vector_ids[0],
            scores[0],
        ),
        start=1,
    ):

        vector_id = int(
            vector_id
        )

        if vector_id < 0:
            continue

        metadata = load_metadata_row(
            vector_id
        )

        normalized = (
            normalize_metadata_result(
                vector_id=vector_id,
                score=float(score),
                rank=rank,
                metadata=metadata,
            )
        )

        results.append(
            normalized
        )

    metadata_elapsed_s = (
        time.perf_counter()
        -
        metadata_start
    )

    total_elapsed_s = (
        time.perf_counter()
        -
        start
    )

    return {
        "query": query,
        "top_k": top_k,
        "retrieved": len(
            results
        ),
        "results": results,
        "timing": {
            "embedding_s":
                embedding_elapsed_s,
            "faiss_search_s":
                search_elapsed_s,
            "metadata_lookup_s":
                metadata_elapsed_s,
            "total_s":
                total_elapsed_s,
        },
    }


# ============================================================
# RAG EVIDENCE PACKAGING
# ============================================================

def build_rag_evidence(
    retrieval_result: dict[str, Any],
    max_items: int = RAG_MAX_EVIDENCE_ITEMS,
    max_chars_per_item: int = RAG_MAX_EVIDENCE_CHARS,
    max_context_chars: int = RAG_MAX_CONTEXT_CHARS,
) -> list[dict[str, Any]]:
    """
    Convert RAG retrieval results into bounded evidence items.
    """

    evidence = []

    total_chars = 0

    for result in retrieval_result.get(
        "results",
        [],
    ):

        if len(evidence) >= max_items:
            break

        text = str(
            result.get(
                "text",
                "",
            )
        ).strip()

        if not text:
            continue

        remaining_chars = (
            max_context_chars
            -
            total_chars
        )

        if remaining_chars <= 0:
            break

        allowed_chars = min(
            max_chars_per_item,
            remaining_chars,
        )

        bounded_text = text[
            :allowed_chars
        ]

        evidence_id = (
            f"R{len(evidence) + 1}"
        )

        evidence.append(
            {
                "evidence_id":
                    evidence_id,
                "rank":
                    result.get(
                        "rank"
                    ),
                "vector_id":
                    result.get(
                        "vector_id"
                    ),
                "score":
                    result.get(
                        "score"
                    ),
                "source":
                    result.get(
                        "source",
                        "",
                    ),
                "title":
                    result.get(
                        "title",
                        "",
                    ),
                "chunk_id":
                    result.get(
                        "chunk_id",
                        "",
                    ),
                "document_id":
                    result.get(
                        "document_id",
                        "",
                    ),
                "path":
                    result.get(
                        "path",
                        "",
                    ),
                "content":
                    bounded_text,
                "retrieval_method":
                    "RAG_V1_FAISS",
            }
        )

        total_chars += len(
            bounded_text
        )

    return evidence


# ============================================================
# CONTEXT FORMATTING
# ============================================================

def format_rag_context(
    evidence: list[dict[str, Any]],
) -> str:
    """
    Format bounded RAG evidence for grounded Gemma generation.
    """

    blocks = []

    for item in evidence:

        evidence_id = item.get(
            "evidence_id",
            "",
        )

        source = item.get(
            "source",
            "",
        )

        title = item.get(
            "title",
            "",
        )

        chunk_id = item.get(
            "chunk_id",
            "",
        )

        content = item.get(
            "content",
            "",
        )

        blocks.append(
            f"[{evidence_id}]\n"
            f"Source: {source}\n"
            f"Title: {title}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Content:\n{content}"
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# COMPLETE RAG PIPELINE
# ============================================================

def retrieve_rag_evidence(
    query: str,
    top_k: int = RAG_TOP_K,
) -> dict[str, Any]:
    """
    Execute RAG retrieval and return the bounded deployment
    evidence package.
    """

    retrieval = retrieve_rag(
        query=query,
        top_k=top_k,
    )

    evidence = build_rag_evidence(
        retrieval
    )

    context = format_rag_context(
        evidence
    )

    return {
        "query": query,
        "retrieval_mode": "RAG_ONLY",
        "retrieved":
            retrieval["retrieved"],
        "evidence_count": len(
            evidence
        ),
        "evidence": evidence,
        "context": context,
        "timing":
            retrieval["timing"],
    }


# ============================================================
# RUNTIME STATUS
# ============================================================

def get_rag_runtime_status() -> dict[str, Any]:
    """
    Return lightweight RAG deployment status without loading
    the FAISS index or embedding model into memory.
    """

    assets = resolve_rag_assets()

    return {
        "embedding_model":
            RAG_EMBEDDING_MODEL,
        "index_path":
            assets["index_path"],
        "metadata_dir":
            assets["metadata_dir"],
        "metadata_shard_count":
            assets[
                "metadata_shard_count"
            ],
        "index_source":
            assets["index_source"],
        "metadata_source":
            assets["metadata_source"],
        "expected_vectors":
            EXPECTED_VECTOR_COUNT,
        "expected_dimension":
            EXPECTED_VECTOR_DIMENSION,
        "index_loaded":
            _faiss_index
            is not None,
        "embedding_model_loaded":
            _embedding_model
            is not None,
    }