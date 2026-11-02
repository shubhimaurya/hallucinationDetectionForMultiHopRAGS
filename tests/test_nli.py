"""Unit tests for NLI verification and semantic-similarity trap detection (Contribution 3)."""
from __future__ import annotations

import unittest

from verification.nli_verifier import NLIVerifier


class TestNLIVerifier(unittest.TestCase):
    def setUp(self) -> None:
        self.nli = NLIVerifier()

    def test_entailment_detection(self) -> None:
        chunks = [{"doc_id": "D1", "text": "India launched the Chandrayaan-3 mission on July 14, 2023."}]
        res = self.nli.verify_claim_against_chunks("India launched Chandrayaan-3 in 2023.", chunks)
        self.assertEqual(res.label, "SUPPORTED")
        self.assertGreater(res.entailment_prob, 0.65)

    def test_contradiction_on_negation_and_numbers(self) -> None:
        chunks = [{"doc_id": "D1", "text": "Peptide CPX-402 did not reverse established myocardial fibrosis in Stage D heart failure."}]
        res = self.nli.verify_claim_against_chunks(
            "Peptide CPX-402 reversed established myocardial fibrosis in Stage D heart failure.",
            chunks,
        )
        self.assertEqual(res.label, "CONTRADICTED")
        self.assertGreater(res.contradiction_prob, 0.60)

        num_chunks = [{"doc_id": "D2", "text": "Peptide CPX-402 reduced cardiovascular events by 28% in 4,200 patients."}]
        res_num = self.nli.verify_claim_against_chunks(
            "Peptide CPX-402 reduced cardiovascular events by 85% in 4,200 patients.",
            num_chunks,
        )
        self.assertEqual(res_num.label, "CONTRADICTED")

    def test_not_enough_evidence_on_unrelated_premise(self) -> None:
        chunks = [{"doc_id": "D1", "text": "The Pragyan rover deployed its LIBS payload near the lunar south pole."}]
        res = self.nli.verify_claim_against_chunks(
            "The Pragyan rover chassis cost 42.7 million Euros to manufacture in Toulouse.",
            chunks,
        )
        self.assertEqual(res.label, "NOT ENOUGH EVIDENCE")
        self.assertGreater(res.neutral_prob, 0.50)


if __name__ == "__main__":
    unittest.main()
