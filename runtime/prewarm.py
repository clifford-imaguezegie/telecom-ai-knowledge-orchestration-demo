from __future__ import annotations

import threading
import time
from typing import Any

from runtime.rag import (
    get_embedding_model,
    get_faiss_index,
    resolve_rag_assets,
)

# ============================================================
# MODULE 4 DEPLOYMENT PREWARM
# ============================================================
#
# Cloud Run may serve multiple Streamlit sessions from one
# Python process. This lock ensures only one session performs
# the expensive cold-start initialization. Other sessions wait
# for the same initialized runtime rather than triggering a
# second Kaggle download / FAISS load / BGE-M3 load.
#
# This module does not change retrieval behaviour, embeddings,
# FAISS contents, scoring, or evaluation logic.
# ============================================================

_PREWARM_LOCK = threading.Lock()
_PREWARM_READY = False
_PREWARM_STATE: dict[str, Any] = {}


def prewarm_rag_runtime() -> dict[str, Any]:
    """Resolve and load the complete RAG runtime once per process."""

    global _PREWARM_READY
    global _PREWARM_STATE

    if _PREWARM_READY:
        return dict(_PREWARM_STATE)

    with _PREWARM_LOCK:
        if _PREWARM_READY:
            return dict(_PREWARM_STATE)

        started = time.perf_counter()

        assets_started = time.perf_counter()
        assets = resolve_rag_assets()
        assets_seconds = time.perf_counter() - assets_started

        faiss_started = time.perf_counter()
        index = get_faiss_index()
        faiss_seconds = time.perf_counter() - faiss_started

        embedding_started = time.perf_counter()
        get_embedding_model()
        embedding_seconds = time.perf_counter() - embedding_started

        _PREWARM_STATE = {
            "status": "ready",
            "index_path": assets.get("index_path"),
            "metadata_dir": assets.get("metadata_dir"),
            "metadata_shard_count": assets.get("metadata_shard_count"),
            "vector_count": int(index.ntotal),
            "vector_dimension": int(index.d),
            "embedding_model": "BAAI/bge-m3",
            "assets_seconds": round(assets_seconds, 3),
            "faiss_seconds": round(faiss_seconds, 3),
            "embedding_seconds": round(embedding_seconds, 3),
            "total_seconds": round(time.perf_counter() - started, 3),
        }

        _PREWARM_READY = True
        return dict(_PREWARM_STATE)


def is_rag_runtime_ready() -> bool:
    """Return whether this process has completed RAG prewarm."""

    return bool(_PREWARM_READY)
