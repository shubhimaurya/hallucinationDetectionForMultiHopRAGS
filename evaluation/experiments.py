"""Research Experiment Runner for E1 (Original Baseline), E2 (Claim-Level NLI), E3 (Hybrid Verification), and E4 (Proposed ACVE Framework)."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.model_config import VerificationConfig
from config.settings import settings
from evaluation.baseline import OriginalBaselineEvaluator
from evaluation.datasets import DatasetRegistry, EvaluationSample
from evaluation.metrics import compute_all_metrics
from verification.citation_validator import ACVEPipelineVerifier


class ExperimentRunner:
    """Executes E1, E2, E3, and E4 on identical held-out test splits after validation tuning."""

    EXPERIMENT_CONFIGS: Dict[str, str] = {
        "E1_ORIGINAL_BASELINE": "Collective answer-context similarity baseline (reproducing reference repo verification)",
        "E2_CLAIM_LEVEL_NLI": "Atomic claim extraction + pooled NLI verification (no conflict matrix, 0 LLM judge calls)",
        "E3_HYBRID_VERIFICATION": "Atomic claim extraction + pooled NLI + unconditional LLM Judge on 100% of claims",
        "E4_PROPOSED_ACVE": "Full ACVE: Source-wise Multi-RAG NLI + Cross-Source Conflict Detection + Adaptive LLM Judge + Calibrated Aggregation",
    }

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or settings.experiments_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_eval = OriginalBaselineEvaluator()

    @staticmethod
    def _pool_chunks_into_single_source(
        source_wise_chunks: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Pool all chunks into a single undifferentiated collection for E2 and E3."""
        pooled: List[Dict[str, Any]] = []
        for chunks in source_wise_chunks.values():
            for c in chunks:
                copy_c = dict(c)
                copy_c["source_id"] = "pooled_rag"
                copy_c["source_reliability"] = 0.80
                pooled.append(copy_c)
        return {"pooled_rag": pooled}

    def _build_verifier_for_mode(
        self,
        mode: str,
        val_samples: List[EvaluationSample],
    ) -> ACVEPipelineVerifier:
        if mode == "E2_CLAIM_LEVEL_NLI":
            cfg = VerificationConfig(
                use_claim_decomposition=True,
                use_nli_verification=True,
                use_conflict_detection=False,
                use_adaptive_llm_judge=False,
                disable_llm_judge_all=True,
                use_confidence_aggregation=False,
                use_source_reliability=False,
            )
            return ACVEPipelineVerifier(config=cfg)

        if mode == "E3_HYBRID_VERIFICATION":
            cfg = VerificationConfig(
                use_claim_decomposition=True,
                use_nli_verification=True,
                use_conflict_detection=False,
                use_adaptive_llm_judge=True,
                force_llm_judge_all=True,
                use_confidence_aggregation=True,
                use_source_reliability=False,
            )
            return ACVEPipelineVerifier(config=cfg)

        # E4_PROPOSED_ACVE: tune aggregation weights & temperature on the validation split (zero test leakage)
        cfg = VerificationConfig(
            use_claim_decomposition=True,
            use_nli_verification=True,
            use_conflict_detection=True,
            use_adaptive_llm_judge=True,
            use_confidence_aggregation=True,
            use_source_reliability=True,
        )
        verifier = ACVEPipelineVerifier(config=cfg)
        if val_samples:
            val_feats: List[Dict[str, float]] = []
            val_labels: List[int] = []
            for sample in val_samples:
                claims = verifier.claim_extractor.extract_claims(sample.generated_answer, decompose=True)
                for clm in claims:
                    src_entries = verifier.source_verifier.verify_claim_by_source(clm, sample.source_wise_chunks)
                    conflict_res = verifier.conflict_detector.analyze(clm.claim_id, src_entries)
                    judge_res = verifier.adaptive_verifier.verify_adaptively(
                        clm.claim_id, clm.claim_text, src_entries, conflict_res
                    )
                    feats = verifier.aggregator.extract_features(src_entries, conflict_res, judge_res)
                    val_feats.append(feats)
                    val_labels.append(sample.ground_truth_label)
            verifier.aggregator.fit_weights_on_validation(val_feats, val_labels)
        return verifier

    def evaluate_configuration(
        self,
        mode: str,
        dataset_name: str = "ms_conflict_bench",
    ) -> Dict[str, Any]:
        """Run a single experimental configuration (E1..E4) on the held-out test split."""
        val_samples = DatasetRegistry.load_dataset(dataset_name, split="val")
        test_samples = DatasetRegistry.load_dataset(dataset_name, split="test")
        if not test_samples:
            test_samples = DatasetRegistry.load_dataset(dataset_name)

        y_true: List[int] = []
        y_pred: List[int] = []
        y_prob: List[float] = []
        latencies: List[float] = []
        llm_calls: List[int] = []
        tokens_used: List[int] = []
        coverages: List[float] = []
        cite_precs: List[float] = []
        agreements: List[float] = []
        conflict_true: List[bool] = []
        conflict_pred: List[bool] = []
        raw_predictions: List[Dict[str, Any]] = []

        if mode == "E1_ORIGINAL_BASELINE":
            for sample in test_samples:
                res = self.baseline_eval.evaluate_sample(sample)
                y_true.append(sample.ground_truth_label)
                y_pred.append(int(res["pred_label"]))
                y_prob.append(float(res["pred_prob_supported"]))
                latencies.append(float(res["latency_ms"]))
                llm_calls.append(int(res["llm_calls"]))
                tokens_used.append(int(res["tokens_used"]))
                coverages.append(float(res["evidence_coverage"]))
                cite_precs.append(float(res["citation_precision"]))
                agreements.append(float(res["source_agreement"]))
                conflict_true.append(sample.has_source_conflict)
                conflict_pred.append(bool(res["detected_conflict"]))
                raw_predictions.append(
                    {
                        "sample_id": sample.sample_id,
                        "category": sample.category,
                        "ground_truth_label": sample.ground_truth_label,
                        **res,
                    }
                )
        else:
            verifier = self._build_verifier_for_mode(mode, val_samples)
            for sample in test_samples:
                t0 = time.perf_counter()
                chunks_input = (
                    sample.source_wise_chunks
                    if mode == "E4_PROPOSED_ACVE"
                    else self._pool_chunks_into_single_source(sample.source_wise_chunks)
                )
                out = verifier.verify_answer(sample.generated_answer, chunks_input)
                elapsed = (time.perf_counter() - t0) * 1000.0

                claims_list = out["claims"]
                # Answer is faithful (1) iff all extracted factual claims are SUPPORTED and none are HALLUCINATED/UNCERTAIN
                if claims_list:
                    min_prob = min(float(c["calibrated_confidence"]) for c in claims_list)
                    all_supported = all(
                        c["final_classification"] == "SUPPORTED" for c in claims_list
                    )
                    pred_lbl = 1 if all_supported else 0
                    avg_cov = sum(
                        float(c["confidence_result"]["evidence_coverage"]) for c in claims_list
                    ) / len(claims_list)
                    avg_cite = sum(
                        float(c["citation_audit"]["citation_precision"]) for c in claims_list
                    ) / len(claims_list)
                    avg_agree = sum(
                        float(c["conflict_analysis"]["cross_source_agreement"]) for c in claims_list
                    ) / len(claims_list)
                    det_conflict = any(
                        bool(c["conflict_analysis"]["has_conflict"]) for c in claims_list
                    )
                else:
                    min_prob = 0.25
                    pred_lbl = 0
                    avg_cov = 0.0
                    avg_cite = 0.0
                    avg_agree = 0.0
                    det_conflict = False

                y_true.append(sample.ground_truth_label)
                y_pred.append(pred_lbl)
                y_prob.append(min_prob)
                latencies.append(elapsed)
                llm_calls.append(int(out["llm_judge_calls"]))
                tokens_used.append(int(out["llm_judge_tokens"]))
                coverages.append(avg_cov)
                cite_precs.append(avg_cite)
                agreements.append(avg_agree)
                conflict_true.append(sample.has_source_conflict)
                conflict_pred.append(det_conflict)

                raw_predictions.append(
                    {
                        "sample_id": sample.sample_id,
                        "category": sample.category,
                        "ground_truth_label": sample.ground_truth_label,
                        "pred_label": pred_lbl,
                        "pred_prob_supported": round(min_prob, 4),
                        "detected_conflict": det_conflict,
                        "llm_judge_calls": out["llm_judge_calls"],
                        "claims_count": out["total_claims"],
                    }
                )

        metrics = compute_all_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            latencies_ms=latencies,
            llm_calls=llm_calls,
            tokens_used=tokens_used,
            evidence_coverages=coverages,
            citation_precisions=cite_precs,
            source_agreements=agreements,
            conflict_true=conflict_true,
            conflict_pred=conflict_pred,
        )

        return {
            "experiment_id": mode,
            "description": self.EXPERIMENT_CONFIGS.get(mode, mode),
            "dataset_name": dataset_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "predictions": raw_predictions,
        }

    def run_all_experiments(self, dataset_name: str = "ms_conflict_bench") -> Dict[str, Any]:
        """Execute E1, E2, E3, and E4 sequentially and persist results to disk."""
        results: Dict[str, Any] = {}
        for mode in self.EXPERIMENT_CONFIGS:
            results[mode] = self.evaluate_configuration(mode=mode, dataset_name=dataset_name)

        out_file = self.output_dir / f"experiments_{dataset_name}.json"
        out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results
