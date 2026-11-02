"""Multi-RAG Orchestrator coordinating the four independent retrieval pipelines."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from config.settings import settings
from ingestion.document_processor import DocumentIngestionProcessor
from retrieval.academic_rag import AcademicRAGPipeline
from retrieval.pdf_rag import PDFRAGPipeline
from retrieval.reranker import CrossEncoderReranker
from retrieval.structured_rag import StructuredRAGPipeline
from retrieval.web_rag import WebRAGPipeline


class MultiRAGHybridRetriever:
    """Orchestrates independent retrieval across Academic, PDF, Web, and Structured RAG pipelines."""

    def __init__(self, ingestion_processor: DocumentIngestionProcessor) -> None:
        self.processor = ingestion_processor
        self.shared_reranker = CrossEncoderReranker()
        self.pipelines: Dict[str, Any] = {
            "academic": AcademicRAGPipeline(
                self.processor.collections["academic"], self.shared_reranker
            ),
            "pdf": PDFRAGPipeline(
                self.processor.collections["pdf"], self.shared_reranker
            ),
            "web": WebRAGPipeline(
                self.processor.collections["web"], self.shared_reranker
            ),
            "structured": StructuredRAGPipeline(
                self.processor.collections["structured"], self.shared_reranker
            ),
        }

    def set_pipeline_enabled(self, source_id: str, enabled: bool) -> None:
        """Enable or disable an individual RAG pipeline."""
        if source_id not in self.pipelines:
            raise KeyError(f"Unknown RAG pipeline: {source_id}")
        self.pipelines[source_id].enabled = enabled

    def get_active_pipelines(self) -> List[str]:
        return [src for src, pipe in self.pipelines.items() if pipe.enabled]

    def retrieve_all(
        self,
        query: str,
        enabled_sources: Optional[List[str]] = None,
        top_k_per_source: int = 3,
    ) -> Dict[str, Any]:
        """Retrieve evidence independently from each enabled RAG pipeline and assemble pooled + source-wise views."""
        active = (
            [s for s in enabled_sources if s in self.pipelines]
            if enabled_sources is not None
            else self.get_active_pipelines()
        )
        source_results: Dict[str, List[Dict[str, Any]]] = {}
        pooled_chunks: List[Dict[str, Any]] = []

        for src_id in active:
            hits = self.pipelines[src_id].retrieve(query=query, top_k=top_k_per_source)
            source_results[src_id] = hits
            pooled_chunks.extend(hits)

        pooled_chunks.sort(key=lambda x: float(x.get("retrieval_score", 0.0)), reverse=True)
        return {
            "query": query,
            "active_sources": active,
            "source_wise_results": source_results,
            "pooled_results": pooled_chunks,
            "total_retrieved": len(pooled_chunks),
        }
