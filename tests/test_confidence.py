"""Unit tests for Adaptive Verification, Confidence Aggregation, Calibration, and Abstention (Contributions 5 & 6)."""
from __future__ import annotations

import unittest

from verification.adaptive_verifier import AdaptiveLLMVerifier
from verification.calibration import TemperatureCalibrator
from verification.claim_extractor import AtomicClaim
from verification.confidence_aggregator import AdaptiveConfidenceAggregator
from verification.conflict_detector import CrossSourceConflictDetector
from verification.source_verifier import SourceWiseVerifier


class TestConfidenceAndCalibration(unittest.TestCase):
    def setUp(self) -> None:
        self.sv = SourceWiseVerifier()
        self.cd = CrossSourceConflictDetector()
        self.av = AdaptiveLLMVerifier()
        self.agg = AdaptiveConfidenceAggregator()

    def test_high_confidence_consensus_bypasses_llm_judge(self) -> None:
        claim = AtomicClaim("CLM-001", "India launched Chandrayaan-3 on July 14, 2023.", "India launched Chandrayaan-3 on July 14, 2023.", 0, 0, 46)
        chunks = {
            "academic": [{"doc_id": "A1", "text": "India launched Chandrayaan-3 on July 14, 2023.", "retrieval_score": 0.9}],
            "pdf": [{"doc_id": "P1", "text": "India launched Chandrayaan-3 on July 14, 2023.", "retrieval_score": 0.9}],
        }
        entries = self.sv.verify_claim_by_source(claim, chunks)
        conflict = self.cd.analyze(claim.claim_id, entries)
        judge = self.av.verify_adaptively(claim.claim_id, claim.claim_text, entries, conflict)
        self.assertFalse(judge.invoked, "High-confidence consensus should bypass expensive LLM Judge")

        conf = self.agg.aggregate(claim.claim_id, entries, conflict, judge)
        self.assertEqual(conf.abstention_decision.final_classification, "SUPPORTED")
        self.assertGreater(conf.calibrated_confidence, 0.70)

    def test_insufficient_evidence_triggers_abstention(self) -> None:
        claim = AtomicClaim("CLM-002", "The rover chassis cost 42.7 million Euros.", "The rover chassis cost 42.7 million Euros.", 0, 0, 42)
        chunks = {
            "academic": [{"doc_id": "A1", "text": "Pragyan deployed its LIBS spectroscopy instrument.", "retrieval_score": 0.4}],
        }
        entries = self.sv.verify_claim_by_source(claim, chunks)
        conflict = self.cd.analyze(claim.claim_id, entries)
        judge = self.av.verify_adaptively(claim.claim_id, claim.claim_text, entries, conflict)
        conf = self.agg.aggregate(claim.claim_id, entries, conflict, judge)
        self.assertEqual(conf.abstention_decision.final_classification, "UNCERTAIN")
        self.assertTrue(conf.abstention_decision.abstained)

    def test_temperature_scaling_reduces_or_preserves_nll(self) -> None:
        cal = TemperatureCalibrator(1.0)
        logits = [2.5, 2.1, -2.2, -1.8, 1.9, -2.5]
        labels = [1, 1, 0, 0, 1, 0]
        t_opt = cal.fit(logits, labels)
        self.assertGreater(t_opt, 0.05)
        ece, brier, bins = cal.compute_ece_and_brier([cal.calibrate_logit(x) for x in logits], labels)
        self.assertLess(brier, 0.15)
        self.assertEqual(len(bins), 10)


if __name__ == "__main__":
    unittest.main()
