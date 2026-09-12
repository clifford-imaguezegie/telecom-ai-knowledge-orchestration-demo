from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import re
import time

from fastmcp import Client

from runtime.rag import retrieve_rag
from runtime.mcp import mcp

MODULE4_RAG_RETRIEVAL_K = 5
MODULE4_MCP_RETRIEVAL_K = 5
COMMON_MAX_EVIDENCE_ITEMS = 5
NORMALIZED_EVIDENCE_MAX_CHARS = 2500
COMMON_CONTEXT_MAX_CHARS = 12500
HYBRID_FUSION_METHOD = "RECIPROCAL_RANK_FUSION"
HYBRID_RRF_K = 60
HYBRID_NEAR_DUPLICATE_THRESHOLD = 0.88
HYBRID_MIN_ITEMS_PER_RETRIEVER = 2
HYBRID_SELECTION_POLICY = "RRF_WITH_POST_DEDUP_MINIMUM_RETRIEVER_REPRESENTATION"

def normalize_evidence_text(text):
    """
    Normalize whitespace while preserving technical content.
    """
    if text is None:
        return ''
    return re.sub('\\s+', ' ', str(text)).strip()

def canonical_evidence_text(text):
    """
    Canonical representation used later for duplicate detection.
    """
    text = normalize_evidence_text(text).lower()
    text = re.sub('[^a-z0-9]+', ' ', text)
    return re.sub('\\s+', ' ', text).strip()

def bound_evidence_text(text, max_chars=NORMALIZED_EVIDENCE_MAX_CHARS):
    """
    Bound one evidence item before it enters the common contract.
    """
    text = normalize_evidence_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()

def build_evidence_id(retrieval_system, document_id, chunk_id, text):
    """
    Produce a stable evidence identity while keeping RAG and MCP
    provenance distinct.
    """
    payload = f'{retrieval_system}|{document_id}|{chunk_id}|{canonical_evidence_text(text)}'
    digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]
    return f'{retrieval_system}:{digest}'

def infer_rag_source_family(item):
    """
    Infer a human-readable source family from RAG metadata.

    This affects provenance/display only.
    It does NOT alter RAG ranking.
    """
    metadata = item.get('metadata', {}) or {}
    fields = [item.get('source', ''), item.get('title', ''), item.get('file_path', ''), item.get('document_id', ''), metadata.get('source', ''), metadata.get('title', ''), metadata.get('file_path', ''), metadata.get('document_id', '')]
    blob = ' '.join((str(value).lower() for value in fields if value is not None))
    source_rules = [(['3gpp', '23.', '24.', '28.', '29.', '38.'], '3GPP'), (['etsi'], 'ETSI'), (['itu-t', 'itu '], 'ITU-T'), (['gsma'], 'GSMA'), (['o-ran', 'o ran', 'oran'], 'O-RAN'), (['tm forum', 'tmforum'], 'TM Forum'), (['camara'], 'CAMARA'), (['kubernetes'], 'Kubernetes'), (['ietf', 'rfc'], 'IETF')]
    for signals, family in source_rules:
        if any((signal in blob for signal in signals)):
            return family
    return str(item.get('source', '') or metadata.get('source', '') or 'RAG_CORPUS').strip()

def normalize_rag_results(rag_result):
    """
    Convert native Semantic RAG V1 results into the common
    Module 4 evidence representation.
    """
    normalized = []
    raw_results = rag_result.get('results', []) or []
    for fallback_rank, item in enumerate(raw_results, start=1):
        metadata = item.get('metadata', {}) or {}
        text = bound_evidence_text(item.get('text', ''))
        document_id = str(item.get('document_id', '') or metadata.get('document_id', '') or item.get('file_path', '') or metadata.get('file_path', '') or item.get('chunk_id', '') or item.get('vector_id', ''))
        chunk_id = str(item.get('chunk_id', '') or metadata.get('chunk_id', '') or '')
        rank = int(item.get('rank', fallback_rank) or fallback_rank)
        score = float(item.get('score', 0.0) or 0.0)
        family = infer_rag_source_family(item)
        title = str(item.get('title', '') or metadata.get('title', '') or '')
        source = item.get('source') or metadata.get('source')
        file_path = item.get('file_path') or metadata.get('file_path')
        normalized.append({'evidence_id': build_evidence_id(retrieval_system='RAG', document_id=document_id, chunk_id=chunk_id, text=text), 'retrieval_system': 'RAG', 'retrieval_systems': ['RAG'], 'source_family': family, 'source_families': [family], 'document_id': document_id, 'title': title, 'chunk_id': chunk_id, 'score': score, 'native_ranks': {'RAG': rank}, 'native_scores': {'RAG': score}, 'fusion_score': None, 'text': text, 'text_chars': len(text), 'provenance': [{'retrieval_system': 'RAG', 'vector_id': item.get('vector_id'), 'chunk_id': chunk_id, 'document_id': document_id, 'source': source, 'file_path': file_path, 'page': item.get('page') or metadata.get('page'), 'native_rank': rank, 'native_score': score}]})
    return normalized

def normalize_mcp_results(mcp_payload):
    """
    Convert Version A MCP evidence into the same Module 4
    evidence representation used by Semantic RAG.
    """
    normalized = []
    raw_evidence = mcp_payload.get('evidence', []) or []
    for fallback_rank, item in enumerate(raw_evidence, start=1):
        text = bound_evidence_text(item.get('evidence', ''))
        source_path = str(item.get('source_path', '') or '')
        identifier = str(item.get('identifier', '') or '')
        title = str(item.get('title', '') or '')
        document_id = identifier or source_path or title
        rank = int(item.get('rank', fallback_rank) or fallback_rank)
        score = float(item.get('relevance_score', item.get('bm25_score', 0.0)) or 0.0)
        family = str(item.get('source_family', '') or 'MCP')
        normalized.append({'evidence_id': build_evidence_id(retrieval_system='MCP', document_id=document_id, chunk_id='', text=text), 'retrieval_system': 'MCP', 'retrieval_systems': ['MCP'], 'source_family': family, 'source_families': [family], 'document_id': document_id, 'title': title, 'chunk_id': '', 'score': score, 'native_ranks': {'MCP': rank}, 'native_scores': {'MCP': score}, 'fusion_score': None, 'text': text, 'text_chars': len(text), 'provenance': [{'retrieval_system': 'MCP', 'source_family': family, 'collection': item.get('collection'), 'identifier': identifier, 'release': item.get('release'), 'section_heading': item.get('section_heading'), 'source_path': source_path, 'source_shard': item.get('source_shard'), 'native_rank': rank, 'native_score': score}]})
    return normalized

def apply_context_budget(evidence, max_items=COMMON_MAX_EVIDENCE_ITEMS, max_chars=COMMON_CONTEXT_MAX_CHARS):
    """
    Enforce the same generator-facing evidence budget
    regardless of retrieval architecture.
    """
    selected = []
    used_chars = 0
    for source_item in evidence:
        if len(selected) >= max_items:
            break
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        item = copy.deepcopy(source_item)
        text = str(item.get('text', ''))
        if len(text) > remaining:
            if remaining < 300:
                break
            text = text[:remaining].rstrip()
        item['text'] = text
        item['text_chars'] = len(text)
        selected.append(item)
        used_chars += len(text)
    for rank, item in enumerate(selected, start=1):
        item['presentation_rank'] = rank
    return (selected, used_chars)

def select_single_mode_evidence(candidates):
    """
    Preserve native ranking for RAG_ONLY or MCP_ONLY while
    applying the common generator-facing evidence budget.
    """
    ordered = sorted(candidates, key=lambda item: min(item.get('native_ranks', {'': 999999}).values(), default=999999))
    return apply_context_budget(ordered)

def build_evidence_context(evidence):
    """
    Convert normalized evidence into the text context that
    will later be supplied to the hosted LLM.
    """
    blocks = []
    for fallback_rank, item in enumerate(evidence, start=1):
        presentation_rank = item.get('presentation_rank', fallback_rank)
        systems = '/'.join(item.get('retrieval_systems', []))
        header = f"[E{presentation_rank}] Retrieval={systems} | Source={item.get('source_family', '')} | Title={item.get('title', '')} | Document={item.get('document_id', '')}"
        blocks.append(header + '\n' + item.get('text', ''))
    return '\n\n'.join(blocks)

def evidence_token_set(text):
    return set(re.findall('[a-z0-9]+', canonical_evidence_text(text)))

def text_jaccard(text_a, text_b):
    tokens_a = evidence_token_set(text_a)
    tokens_b = evidence_token_set(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

def add_rrf_score(item):
    item = copy.deepcopy(item)
    components = {}
    for system, rank in item.get('native_ranks', {}).items():
        rank = int(rank)
        if rank > 0:
            components[system] = 1.0 / (HYBRID_RRF_K + rank)
    item['rrf_components'] = components
    item['fusion_score'] = float(sum(components.values()))
    return item

def merge_duplicate_evidence(primary, secondary):
    merged = copy.deepcopy(primary)
    systems = list(dict.fromkeys(primary.get('retrieval_systems', []) + secondary.get('retrieval_systems', [])))
    families = list(dict.fromkeys(primary.get('source_families', []) + secondary.get('source_families', [])))
    merged['retrieval_systems'] = systems
    merged['retrieval_system'] = 'BOTH' if 'RAG' in systems and 'MCP' in systems else systems[0]
    merged['source_families'] = families
    merged['source_family'] = families[0] if len(families) == 1 else 'MULTI'
    merged['provenance'] = primary.get('provenance', []) + secondary.get('provenance', [])
    native_ranks = dict(primary.get('native_ranks', {}))
    for system, rank in secondary.get('native_ranks', {}).items():
        if system not in native_ranks or rank < native_ranks[system]:
            native_ranks[system] = rank
    merged['native_ranks'] = native_ranks
    native_scores = dict(primary.get('native_scores', {}))
    native_scores.update(secondary.get('native_scores', {}))
    merged['native_scores'] = native_scores
    rrf_components = dict(primary.get('rrf_components', {}))
    for system, score in secondary.get('rrf_components', {}).items():
        rrf_components[system] = max(float(score), float(rrf_components.get(system, 0.0)))
    merged['rrf_components'] = rrf_components
    merged['fusion_score'] = float(sum(rrf_components.values()))
    return merged

def deduplicate_hybrid_evidence(evidence):
    stats = {'input_items': len(evidence), 'exact_duplicate_merges': 0, 'near_duplicate_merges': 0}
    exact_map = {}
    for item in evidence:
        fingerprint = hashlib.sha256(canonical_evidence_text(item['text']).encode('utf-8')).hexdigest()
        if fingerprint in exact_map:
            exact_map[fingerprint] = merge_duplicate_evidence(exact_map[fingerprint], item)
            stats['exact_duplicate_merges'] += 1
        else:
            exact_map[fingerprint] = copy.deepcopy(item)
    exact_unique = list(exact_map.values())
    exact_unique.sort(key=lambda item: float(item.get('fusion_score', 0.0)), reverse=True)
    near_unique = []
    for candidate in exact_unique:
        duplicate_index = None
        for index, existing in enumerate(near_unique):
            similarity = text_jaccard(candidate['text'], existing['text'])
            if similarity >= HYBRID_NEAR_DUPLICATE_THRESHOLD:
                duplicate_index = index
                break
        if duplicate_index is None:
            near_unique.append(copy.deepcopy(candidate))
        else:
            near_unique[duplicate_index] = merge_duplicate_evidence(near_unique[duplicate_index], candidate)
            stats['near_duplicate_merges'] += 1
    near_unique.sort(key=lambda item: float(item.get('fusion_score', 0.0)), reverse=True)
    stats['output_items'] = len(near_unique)
    return (near_unique, stats)

def select_hybrid_evidence(fused_evidence):
    ranked = sorted(fused_evidence, key=lambda item: float(item.get('fusion_score', 0.0)), reverse=True)
    available_rag = sum(('RAG' in item.get('retrieval_systems', []) for item in ranked))
    available_mcp = sum(('MCP' in item.get('retrieval_systems', []) for item in ranked))
    required_rag = min(HYBRID_MIN_ITEMS_PER_RETRIEVER, available_rag)
    required_mcp = min(HYBRID_MIN_ITEMS_PER_RETRIEVER, available_mcp)
    selected_ids = []

    def get_selected_items():
        selected_set = set(selected_ids)
        return [item for item in ranked if item['evidence_id'] in selected_set]

    def representation_count(system):
        return sum((system in item.get('retrieval_systems', []) for item in get_selected_items()))
    for item in ranked:
        if representation_count('RAG') >= required_rag:
            break
        if 'RAG' in item.get('retrieval_systems', []) and item['evidence_id'] not in selected_ids:
            selected_ids.append(item['evidence_id'])
    for item in ranked:
        if representation_count('MCP') >= required_mcp:
            break
        if 'MCP' in item.get('retrieval_systems', []) and item['evidence_id'] not in selected_ids:
            selected_ids.append(item['evidence_id'])
    for item in ranked:
        if len(selected_ids) >= COMMON_MAX_EVIDENCE_ITEMS:
            break
        if item['evidence_id'] not in selected_ids:
            selected_ids.append(item['evidence_id'])
    selected_set = set(selected_ids)
    selected = [item for item in ranked if item['evidence_id'] in selected_set]
    selected, context_chars = apply_context_budget(selected)
    return {'evidence': selected, 'context_chars': context_chars, 'available_rag_after_dedup': int(available_rag), 'available_mcp_after_dedup': int(available_mcp), 'required_rag': int(required_rag), 'required_mcp': int(required_mcp)}

def fuse_hybrid_candidates(rag_candidates, mcp_candidates):
    start = time.perf_counter()
    if len(rag_candidates) > MODULE4_RAG_RETRIEVAL_K:
        raise RuntimeError('RAG candidate count exceeds Module 4 Top-K.')
    if len(mcp_candidates) > MODULE4_MCP_RETRIEVAL_K:
        raise RuntimeError('MCP candidate count exceeds Module 4 Top-K.')
    scored = [add_rrf_score(item) for item in list(rag_candidates) + list(mcp_candidates)]
    deduplicated, dedup_stats = deduplicate_hybrid_evidence(scored)
    hybrid_selection = select_hybrid_evidence(deduplicated)
    selected = hybrid_selection['evidence']
    context_chars = hybrid_selection['context_chars']
    elapsed = time.perf_counter() - start
    rag_supported = sum(('RAG' in item.get('retrieval_systems', []) for item in selected))
    mcp_supported = sum(('MCP' in item.get('retrieval_systems', []) for item in selected))
    both_supported = sum(('RAG' in item.get('retrieval_systems', []) and 'MCP' in item.get('retrieval_systems', []) for item in selected))
    return {'evidence': selected, 'context': build_evidence_context(selected), 'context_chars': context_chars, 'timing': {'fusion_only_s': elapsed}, 'diagnostics': {**dedup_stats, 'rag_input_candidates': len(rag_candidates), 'mcp_input_candidates': len(mcp_candidates), 'available_rag_after_dedup': hybrid_selection['available_rag_after_dedup'], 'available_mcp_after_dedup': hybrid_selection['available_mcp_after_dedup'], 'required_rag': hybrid_selection['required_rag'], 'required_mcp': hybrid_selection['required_mcp'], 'selected_items': len(selected), 'rag_supported_items': int(rag_supported), 'mcp_supported_items': int(mcp_supported), 'both_supported_items': int(both_supported), 'selection_policy': HYBRID_SELECTION_POLICY}}

def execute_rag_v1(query):
    """
    Call the already validated Cell 1D retrieve_rag() function
    without assuming whether its Top-K parameter is named
    'top_k' or 'k'.
    """
    if 'retrieve_rag' not in globals() or not callable(retrieve_rag):
        raise RuntimeError('Cell 1D retrieve_rag() is unavailable.')
    parameters = inspect.signature(retrieve_rag).parameters
    if 'top_k' in parameters:
        return retrieve_rag(query, top_k=MODULE4_RAG_RETRIEVAL_K)
    if 'k' in parameters:
        return retrieve_rag(query, k=MODULE4_RAG_RETRIEVAL_K)
    return retrieve_rag(query)

def retrieve_rag_candidates(query):
    start = time.perf_counter()
    raw = execute_rag_v1(query)
    elapsed = time.perf_counter() - start
    if not isinstance(raw, dict):
        raise RuntimeError('RAG retriever did not return a dictionary.')
    normalized = normalize_rag_results(raw)
    return {'query': query, 'retrieval_system': 'RAG', 'retrieval_top_k': MODULE4_RAG_RETRIEVAL_K, 'candidate_count': len(normalized), 'evidence': normalized, 'timing': {'retrieval_time_s': float(elapsed)}, 'raw': raw}

async def retrieve_mcp_candidates(query):
    start = time.perf_counter()
    client = Client(mcp)
    async with client:
        result = await client.call_tool('search_telecom_knowledge', {'query': query, 'top_k': MODULE4_MCP_RETRIEVAL_K})
    elapsed = time.perf_counter() - start
    payload = getattr(result, 'data', None)
    if payload is None:
        payload = getattr(result, 'structured_content', None)
    if not isinstance(payload, dict):
        raise RuntimeError('MCP returned no structured payload.')
    normalized = normalize_mcp_results(payload)
    return {'query': query, 'retrieval_system': 'MCP', 'retrieval_top_k': MODULE4_MCP_RETRIEVAL_K, 'candidate_count': len(normalized), 'evidence': normalized, 'timing': {'mcp_roundtrip_s': float(elapsed), 'retrieval_time_s': float(payload.get('trace', {}).get('retrieval_time_s', 0.0) or 0.0)}, 'trace': payload.get('trace', {}), 'payload': payload}

async def retrieve_module4_hybrid(query):
    """
    Execute RAG and MCP concurrently, then construct:

        RAG_ONLY
        MCP_ONLY
        HYBRID

    from the SAME retrieval run.
    """
    pair_start = time.perf_counter()
    rag_task = asyncio.to_thread(retrieve_rag_candidates, query)
    mcp_task = retrieve_mcp_candidates(query)
    rag, mcp_result = await asyncio.gather(rag_task, mcp_task)
    parallel_retrieval_wall_s = time.perf_counter() - pair_start
    rag_selected, rag_context_chars = select_single_mode_evidence(rag['evidence'])
    rag_mode = {'mode': 'RAG_ONLY', 'candidate_count': rag['candidate_count'], 'evidence': rag_selected, 'context': build_evidence_context(rag_selected), 'context_chars': rag_context_chars, 'timing': rag['timing']}
    mcp_selected, mcp_context_chars = select_single_mode_evidence(mcp_result['evidence'])
    mcp_mode = {'mode': 'MCP_ONLY', 'candidate_count': mcp_result['candidate_count'], 'evidence': mcp_selected, 'context': build_evidence_context(mcp_selected), 'context_chars': mcp_context_chars, 'timing': mcp_result['timing'], 'trace': mcp_result['trace']}
    hybrid = fuse_hybrid_candidates(rag_candidates=rag['evidence'], mcp_candidates=mcp_result['evidence'])
    hybrid_total_wall_s = parallel_retrieval_wall_s + hybrid['timing']['fusion_only_s']
    hybrid_mode = {'mode': 'HYBRID', 'candidate_count': {'RAG': rag['candidate_count'], 'MCP': mcp_result['candidate_count'], 'total_before_dedup': rag['candidate_count'] + mcp_result['candidate_count']}, 'evidence': hybrid['evidence'], 'context': hybrid['context'], 'context_chars': hybrid['context_chars'], 'timing': {'rag_retrieval_s': rag['timing']['retrieval_time_s'], 'mcp_retrieval_s': mcp_result['timing']['retrieval_time_s'], 'mcp_roundtrip_s': mcp_result['timing']['mcp_roundtrip_s'], 'parallel_retrieval_wall_s': float(parallel_retrieval_wall_s), 'fusion_only_s': hybrid['timing']['fusion_only_s'], 'hybrid_total_wall_s': float(hybrid_total_wall_s)}, 'diagnostics': hybrid['diagnostics'], 'mcp_trace': mcp_result['trace']}
    return {'query': query, 'RAG_ONLY': rag_mode, 'MCP_ONLY': mcp_mode, 'HYBRID': hybrid_mode}
