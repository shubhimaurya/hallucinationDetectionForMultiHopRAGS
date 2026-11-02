"""Unit tests for Atomic Claim Decomposition (Contribution 1)."""
from __future__ import annotations

import unittest

from verification.claim_extractor import AtomicClaimExtractor


class TestAtomicClaimExtractor(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = AtomicClaimExtractor()

    def test_chandrayaan_compound_sentence_decomposition(self) -> None:
        ans = "India launched Chandrayaan-3 in 2023, and it successfully landed near the lunar south pole."
        claims = self.extractor.extract_claims(ans, decompose=True)
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0].claim_id, "CLM-001")
        self.assertEqual(claims[0].claim_text, "India launched Chandrayaan-3 in 2023.")
        self.assertEqual(claims[1].claim_id, "CLM-002")
        self.assertEqual(
            claims[1].claim_text,
            "Chandrayaan-3 successfully landed near the lunar south pole.",
        )

    def test_conversational_and_opinion_filtering(self) -> None:
        ans = "Hello there! I think space exploration is wonderful. India launched Chandrayaan-3 in 2023 [DOC-1]. Feel free to ask more!"
        claims = self.extractor.extract_claims(ans, decompose=True)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].claim_text, "India launched Chandrayaan-3 in 2023.")
        self.assertEqual(claims[0].cited_doc_ids, ["DOC-1"])


if __name__ == "__main__":
    unittest.main()
