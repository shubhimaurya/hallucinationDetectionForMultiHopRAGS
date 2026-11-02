"""Independent RAG Pipeline for RAG 2: PDF & Organizational Documents (pdf)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from config.model_config import RAG_SOURCE_CONFIGS
from config.settings import settings
from ingestion.indexing import SourceCollectionIndex
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import CrossEncoderReranker


class PDFRAGPipeline:
    """Dedicated retrieval pipeline for RAG 2: PDF & Organizational Documents with isolated index and metadata."""

    source_id: str = "pdf"

    def __init__(
        self,
        index: SourceCollectionIndex,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        if index.source_id != self.source_id:
            raise ValueError(
                f"PDFRAGPipeline requires index with source_id='{self.source_id}', got '{index.source_id}'"
            )
        self.index = index
        self.config = RAG_SOURCE_CONFIGS[self.source_id]
        self.reranker = reranker or CrossEncoderReranker()
        self.enabled: bool = True

    def retrieve(
        self,
        query: str,
        top_k: int = settings.rerank_top_k,
        dense_top_k: int = settings.dense_top_k,
        sparse_top_k: int = settings.sparse_top_k,
    ) -> List[Dict[str, Any]]:
        """Run dense + BM25 retrieval, RRF fusion, and cross-encoder reranking on pdf collection."""
        if not self.enabled:
            return []
        dense_hits = self.index.search_dense(query, top_k=dense_top_k)
        sparse_hits = self.index.search_sparse(query, top_k=sparse_top_k)
        fused = reciprocal_rank_fusion(
            dense_results=dense_hits,
            sparse_results=sparse_hits,
            top_k=max(top_k * 2, 8),
        )
        reranked = self.reranker.rerank(query=query, candidates=fused, top_k=top_k)
        for item in reranked:
            item["source_id"] = self.source_id
            item["source_display_name"] = self.config.display_name
            item["source_reliability"] = float(
                item.get("metadata", {}).get("source_reliability", self.config.default_reliability)
            )
        return reranked
