"""Weighted Reciprocal Rank Fusion (RRF) preserving both dense and sparse provenance scores."""
from __future__ import annotations

from typing import Any, Dict, List

from config.settings import settings


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    top_k: int = 10,
    rrf_k: int = settings.rrf_k,
    dense_weight: float = settings.dense_weight,
    sparse_weight: float = settings.sparse_weight,
) -> List[Dict[str, Any]]:
    """Combine dense and sparse rankings via Weighted Reciprocal Rank Fusion without overwriting metadata."""
    total_w = dense_weight + sparse_weight
    if total_w <= 0:
        w_d, w_s = 0.5, 0.5
    else:
        w_d, w_s = dense_weight / total_w, sparse_weight / total_w

    fused_map: Dict[str, Dict[str, Any]] = {}
    rrf_scores: Dict[str, float] = {}

    for rank, item in enumerate(dense_results, start=1):
        cid = str(item.get("chunk_id") or item.get("id"))
        if not cid:
            continue
        entry = dict(item)
        entry["dense_rank"] = rank
        entry["dense_score"] = float(item.get("dense_score", item.get("retrieval_score", 0.0)))
        entry.setdefault("sparse_rank", None)
        entry.setdefault("sparse_score", 0.0)
        fused_map[cid] = entry
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (w_d / (rrf_k + rank))

    for rank, item in enumerate(sparse_results, start=1):
        cid = str(item.get("chunk_id") or item.get("id"))
        if not cid:
            continue
        if cid not in fused_map:
            entry = dict(item)
            entry["dense_rank"] = None
            entry["dense_score"] = 0.0
            fused_map[cid] = entry
        fused_map[cid]["sparse_rank"] = rank
        fused_map[cid]["sparse_score"] = float(item.get("sparse_score", item.get("retrieval_score", 0.0)))
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (w_s / (rrf_k + rank))

    ranked_ids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:top_k]
    fused_list: List[Dict[str, Any]] = []
    for final_rank, cid in enumerate(ranked_ids, start=1):
        doc = fused_map[cid]
        doc["rrf_score"] = float(rrf_scores[cid])
        doc["rrf_rank"] = final_rank
        doc["retrieval_score"] = float(rrf_scores[cid])
        fused_list.append(doc)
    return fused_list
