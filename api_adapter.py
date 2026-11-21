"""Bounded public projection of the ORIGINAL Gemma/Granite/FAISS 3-path experiment.
Loaded in the original source repository, alongside api_server.py, for a tagged API revision.
Private retrieved passages, prompts, corpus paths and secrets are never returned.
"""
from __future__ import annotations
import re
from typing import Any
SCHEMA_VERSION="tebunor-rag-1"
PATHS=("llm_only","strict","augmented")

def bounded(value:Any,limit:int)->str:
    return "" if value is None else str(value).strip()[:limit]

def number(value:Any):
    try:
        v=float(value)
        return round(v,3) if v==v and abs(v)<1e12 else None
    except (TypeError,ValueError,OverflowError):return None

def validate_question(question:Any)->str:
    if not isinstance(question,str) or len(question)>500:
        raise ValueError("Question must contain 3–500 characters.")
    q=re.sub(r"\s+"," ",question).strip()
    if len(q)<3:raise ValueError("Question must contain 3–500 characters.")
    if any(ord(c)<32 for c in q):raise ValueError("Invalid control characters.")
    return q

def safe_sources(result:dict[str,Any]):
    rows=result.get("final_evidence") if isinstance(result,dict) else []
    if not isinstance(rows,list):return []
    output=[]
    for v in rows[:8]:
        if not isinstance(v,dict):continue
        output.append({
            "evidence_id":bounded(v.get("evidence_id"),85),
            "title":bounded(v.get("title") or v.get("source_family") or "Technical evidence",150),
            "family":bounded(v.get("source_family") or v.get("retrieval_system"),65),
            "retrieval_system":bounded(v.get("retrieval_system"),24),
            "reference":bounded(v.get("chunk_id"),70),
        })
    return output

def safe_trace(stages:Any):
    if not isinstance(stages,list):return []
    return [{"stage":bounded(x.get("stage"),110),
             "status":bounded(x.get("status"),32),
             "elapsed_s":number(x.get("elapsed_s"))}
            for x in stages[:16] if isinstance(x,dict)]

def safe_evaluation(raw:dict[str,Any]|None):
    if not isinstance(raw,dict):return None
    sources=raw.get("systems") or {}
    systems={}
    for key in PATHS:
        v=sources.get(key) or {}
        systems[key]={"relevance_score":number(v.get("relevance_score")),
                      "completeness_score":number(v.get("completeness_score")),
                      "hallucination_risk_pct":number(v.get("hallucination_risk_pct")),
                      "hallucination_risk_label":bounded(v.get("hallucination_risk_label"),30)}
    return {"applicable":bool(raw.get("applicable",False)),
            "systems":systems,
            "comparison_summary":bounded(raw.get("comparison_summary"),700),
            "question_intent":bounded(raw.get("question_intent"),300)}

def public_path(path:dict[str,Any],key:str):
    """The same bounded projection is used by both streaming and JSON routes."""
    if key not in PATHS or not isinstance(path,dict):
        raise ValueError("Invalid original system.")
    m=path.get("metadata") if isinstance(path.get("metadata"),dict) else {}
    value=path.get("result") if isinstance(path.get("result"),dict) else {}
    answer=bounded(value.get("answer"),12000)
    if not answer:raise ValueError("Original "+key+" path returned no answer.")
    timing=path.get("timing") if isinstance(path.get("timing"),dict) else {}
    return {
        "answer":answer,
        "title":bounded(m.get("name"),65),
        "mode":bounded(m.get("mode"),60),
        "objective":bounded(m.get("objective"),120),
        "policy":bounded(m.get("policy"),170),
        "sources":safe_sources(value),
        "orchestration_path":safe_trace(path.get("orchestration_path") or value.get("orchestration_path")),
        "elapsed_s":number(timing.get("total_wall_s")),
        "source_families":[bounded(x,75) for x in (m.get("sources") or [])[:7] if isinstance(x,str)],
    }

def public_result(result:dict[str,Any],evaluation:dict[str,Any]|None=None):
    paths=result.get("paths") or {}
    if not all(isinstance(paths.get(k),dict) for k in PATHS):
        raise ValueError("Three-path inference was incomplete.")
    safe={key:public_path(paths[key],key) for key in PATHS}
    timing=result.get("timing") if isinstance(result.get("timing"),dict) else {}
    routing=result.get("routing") if isinstance(result.get("routing"),dict) else {}
    return {"ok":True,"schema_version":SCHEMA_VERSION,
            "question":bounded(result.get("sanitized_question") or result.get("question"),500),
            "query_type":bounded(result.get("query_type"),65),
            "routing_label":bounded(routing.get("response_mode"),65),
            "paths":safe,"evaluation":safe_evaluation(evaluation),
            "timing":{"total_wall_s":number(timing.get("total_wall_s"))}}
