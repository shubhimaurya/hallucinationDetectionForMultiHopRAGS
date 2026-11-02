"""Component-Wise Ablation Study Suite (7 Configurations)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.model_config import VerificationConfig
from config.settings import settings
from evaluation.datasets import DatasetRegistry
from evaluation.experiments import ExperimentRunner
from evaluation.metrics import compute_all_metrics
from verification.citation_validator import ACVEPipelineVerifier


class AblationStudyRunner:
    """Evaluates the 7 required ablation configurations on the held-out test set."""

    ABLATION_VARIANTS: Dict[str, VerificationConfig] = {
        "1_without_claim_decomposition": VerificationConfig(
            use_claim_decomposition=False,
            use_nli_verification=True,
            use_conflict_detection=True,
            use_adaptive_llm_judge=True,
            use_confidence_aggregation=True,
            use_source_reliability=True,
        ),
        "2_without_nli_verification": VerificationConfig(
            use_claim_decomposition=True,
            use_nli_verification=False,
            nli_entailment_threshold=0.99,  # Disables NLI classification, forcing reliance on judge/heuristics
            use_conflict_detection=False,
            use_adaptive_llm_judge=True,
            force_llm_judge_all=True,
            use_confidence_aggregation=True,
            use_source_reliability=True,
        ),
        "3_without_conflict_detection": VerificationConfig(
            use_claim_decomposition=True,
            use_nli_verification=True,
            use_conflict_detection=False,
            use_adaptive_llm_judge=True,
            use_confidence_aggregation=True,
            use_source_reliability=True,
        ),
        "4_without_adaptive_llm_judging": VerificationConfig(
            use_claim_decomposition=True,
            use_nli_verification=True,
            use_conflict_detection=True,
            use_adaptive_llm_judge=False,
            disable_llm_judge_all=True,
            use_confidence_aggregation=True,
            use_source_reliability=True,
        ),
        "5_without_confidence_aggregation": VerificationConfig(
            use_claim_decomposition=True,
            use_nli_verification=True,
            use_conflict_detection=True,
            use_adaptive_llm_judge=True,
            use_confidence_aggregation=False,
            use_source_reliability=True,
        ),
        "6_without_source_reliability": VerificationConfig(
            use_claim_decomposition=True,
            use_nli_verification=True,
            use_conflict_detection=True,
            use_adaptive_llm_judge=True,
            use_confidence_aggregation=True,
            use_source_reliability=False,
        ),
        "7_complete_proposed_acve": VerificationConfig(
            use_claim_decomposition=True,
            use_nli_verification=True,
            use_conflict_detection=True,
            use_adaptive_llm_judge=True,
            use_confidence_aggregation=True,
            use_source_reliability=True,
        ),
    }

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or settings.experiments_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_ablation_suite(self, dataset_name: str = "ms_conflict_bench") -> Dict[str, Any]:
        """Run all 7 ablation configurations and record accuracy, F1, ECE, latency, and LLM calls."""
        test_samples = DatasetRegistry.load_dataset(dataset_name, split="test")
        if not test_samples:
            test_samples = DatasetRegistry.load_dataset(dataset_name)

        results: Dict[str, Any] = {}
        for variant_name, cfg in self.ABLATION_VARIANTS.items():
            verifier = ACVEPipelineVerifier(config=cfg)
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

            for sample in test_samples:
                t0 = time.perf_counter()
                chunks_in = (
                    sample.source_wise_chunks
                    if cfg.use_conflict_detection
                    else ExperimentRunner._pool_chunks_into_single_source(sample.source_wise_chunks)
                )
                out = verifier.verify_answer(sample.generated_answer, chunks_in)
                elapsed = (time.perf_counter() - t0) * 1000.0

                claims_list = out["claims"]
                if claims_list:
                    min_prob = min(float(c["calibrated_confidence"]) for c in claims_list)
                    all_sup = all(c["final_classification"] == "SUPPORTED" for c in claims_list)
                    pred_lbl = 1 if all_sup else 0
                    avg_cov = sum(float(c["confidence_result"]["evidence_coverage"]) for c in claims_list) / len(claims_list)
                    avg_cite = sum(float(c["citation_audit"]["citation_precision"]) for c in claims_list) / len(claims_list)
                    avg_agree = sum(float(c["conflict_analysis"]["cross_source_agreement"]) for c in claims_list) / len(claims_list)
                    det_conf = any(bool(c["conflict_analysis"]["has_conflict"]) for c in claims_list)
                else:
                    min_prob = 0.25
                    pred_lbl = 0
                    avg_cov = 0.0
                    avg_cite = 0.0
                    avg_agree = 0.0
                    det_conf = False

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
                conflict_pred.append(det_conf)

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
            results[variant_name] = {
                "variant": variant_name,
                "config": cfg.to_dict(),
                "metrics": metrics,
            }

        out_path = self.output_dir / f"ablation_{dataset_name}.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results
