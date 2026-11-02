"""End-to-end integration tests verifying query execution, citation grounding, E1-E4 experiments, and ablations."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app import initialize_system_state
from dashboard.claim_analysis import format_claim_cards
from dashboard.home import compute_home_summary
from dashboard.query import execute_multirag_query
from evaluation.ablation import AblationStudyRunner
from evaluation.experiments import ExperimentRunner


class TestEndToEndPipeline(unittest.TestCase):
    def test_full_pipeline_and_citation_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = initialize_system_state(storage_root=Path(tmp))
            home_summary = compute_home_summary(state)
            self.assertEqual(home_summary["total_indexed_documents"], 6)
            self.assertEqual(home_summary["active_rag_pipelines"], 4)

            res = execute_multirag_query(
                state,
                query="When did India launch Chandrayaan-3 and where did it land?",
                selected_sources=["academic", "pdf", "web", "structured"],
            )
            ver = res["verification"]
            self.assertGreaterEqual(ver["total_claims"], 1)
            self.assertGreaterEqual(ver["supported_claims"], 1)

            cards = format_claim_cards(ver)
            self.assertEqual(len(cards), ver["total_claims"])

            # Verify fabricated citation detection
            fake_Cite_ans = "India launched Chandrayaan-3 on July 14, 2023 [NON_EXISTENT_DOC_999]."
            ver_fake = state["verifier"].verify_answer(
                fake_Cite_ans, res["retrieval"]["source_wise_results"]
            )
            cite_audit = ver_fake["claims"][0]["citation_audit"]
            self.assertIn("NON_EXISTENT_DOC_999", cite_audit["hallucinated_cited_doc_ids"])

    def test_experiments_and_ablations_produce_real_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exp_runner = ExperimentRunner(output_dir=Path(tmp))
            e1 = exp_runner.evaluate_configuration("E1_ORIGINAL_BASELINE", "ms_conflict_bench")
            e4 = exp_runner.evaluate_configuration("E4_PROPOSED_ACVE", "ms_conflict_bench")

            self.assertGreater(e4["metrics"]["f1_score"], e1["metrics"]["f1_score"])
            self.assertGreater(e4["metrics"]["conflict_detection_accuracy"], e1["metrics"]["conflict_detection_accuracy"])

            abl_runner = AblationStudyRunner(output_dir=Path(tmp))
            abl = abl_runner.run_ablation_suite("ms_conflict_bench")
            self.assertEqual(len(abl), 7)


if __name__ == "__main__":
    unittest.main()
