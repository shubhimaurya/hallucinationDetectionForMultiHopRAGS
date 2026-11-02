"""Unit tests for the 4 independent Multi-RAG pipelines, RRF fusion, and Cross-Encoder reranking."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app import bootstrap_sample_documents
from ingestion.document_processor import DocumentIngestionProcessor
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.hybrid_retriever import MultiRAGHybridRetriever
from retrieval.reranker import CrossEncoderReranker


class TestMultiRAGRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.processor = DocumentIngestionProcessor(storage_root=Path(self.tmp_dir.name))
        bootstrap_sample_documents(self.processor)
        self.retriever = MultiRAGHybridRetriever(self.processor)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_four_independent_collections_isolation(self) -> None:
        stats = self.processor.get_collection_stats()
        self.assertEqual(set(stats.keys()), {"academic", "pdf", "web", "structured"})
        for src_id, info in stats.items():
            self.assertGreaterEqual(info["document_count"], 1, f"{src_id} should have indexed docs")

    def test_enable_disable_individual_pipelines(self) -> None:
        self.retriever.set_pipeline_enabled("web", False)
        self.assertNotIn("web", self.retriever.get_active_pipelines())
        out = self.retriever.retrieve_all("Chandrayaan-3 lunar south pole")
        self.assertNotIn("web", out["source_wise_results"])

        self.retriever.set_pipeline_enabled("web", True)
        out2 = self.retriever.retrieve_all("Chandrayaan-3 lunar south pole")
        self.assertIn("web", out2["source_wise_results"])

    def test_rrf_preserves_dense_and_sparse_metadata(self) -> None:
        dense = [{"chunk_id": "C1", "doc_id": "D1", "text": "alpha beta", "dense_score": 0.9}]
        sparse = [{"chunk_id": "C1", "doc_id": "D1", "text": "alpha beta", "sparse_score": 4.2}]
        fused = reciprocal_rank_fusion(dense, sparse, top_k=5)
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0]["dense_score"], 0.9)
        self.assertAlmostEqual(fused[0]["sparse_score"], 4.2)
        self.assertGreater(fused[0]["rrf_score"], 0.0)

    def test_shingle_deduplication_fixes_prefix_bug(self) -> None:
        shared_header = "OFFICIAL ENTERPRISE REPORT HEADER " * 8  # > 200 chars identical prefix
        c1 = {"chunk_id": "1", "text": shared_header + "Section A discusses Chandrayaan-3 landing in 2023."}
        c2 = {"chunk_id": "2", "text": shared_header + "Section B discusses solid-state battery specific energy of 380 Wh/kg."}
        deduped = CrossEncoderReranker.deduplicate_chunks([c1, c2])
        # Unlike the reference repo's [:200] prefix check, shingle Jaccard keeps both distinct chunks
        self.assertEqual(len(deduped), 2)


if __name__ == "__main__":
    unittest.main()
