"""Unit tests for Cross-Source Conflict Detection (Contribution 4)."""
from __future__ import annotations

import unittest

from verification.claim_extractor import AtomicClaim
from verification.conflict_detector import CrossSourceConflictDetector
from verification.source_verifier import SourceWiseVerifier


class TestCrossSourceConflictDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.source_verifier = SourceWiseVerifier()
        self.detector = CrossSourceConflictDetector()

    def test_cross_source_disagreement_matrix(self) -> None:
        claim = AtomicClaim(
            claim_id="CLM-001",
            claim_text="LK-99 is a confirmed room-temperature superconductor at ambient pressure.",
            original_sentence="LK-99 is a confirmed room-temperature superconductor at ambient pressure.",
            sentence_index=0,
            char_start=0,
            char_end=72,
        )
        source_chunks = {
            "academic": [{"doc_id": "ACAD-1", "text": "LK-99 is not a room-temperature superconductor at ambient pressure.", "retrieval_score": 0.9}],
            "pdf": [{"doc_id": "PDF-1", "text": "Replication audits confirmed LK-99 is not a room-temperature superconductor at ambient pressure.", "retrieval_score": 0.88}],
            "web": [{"doc_id": "WEB-1", "text": "LK-99 is a confirmed room-temperature superconductor at ambient pressure.", "retrieval_score": 0.92}],
            "structured": [],
        }
        entries = self.source_verifier.verify_claim_by_source(claim, source_chunks)
        conflict = self.detector.analyze(claim.claim_id, entries)

        self.assertTrue(conflict.has_conflict)
        self.assertEqual(conflict.conflict_type, "CROSS_SOURCE_CONTRADICTION")
        self.assertIn("web", conflict.supporting_sources)
        self.assertIn("academic", conflict.contradicting_sources)
        self.assertIn("pdf", conflict.contradicting_sources)
        self.assertIn("structured", conflict.insufficient_sources)
        self.assertGreater(conflict.conflict_severity, 0.40)


if __name__ == "__main__":
    unittest.main()
