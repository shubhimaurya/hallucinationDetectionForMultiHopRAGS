"""Mathematical Confidence Aggregation module: C(claim) = f(E, K, A, J, R).

Supports validation-set parameter learning via L2-regularized logistic regression
without test-set leakage, alongside full diagnostic decomposition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import minimize

from config.model_config import DEFAULT_VERIFICATION_CONFIG, VerificationConfig
from verification.abstention import AbstentionDecision, CalibratedAbstentionClassifier
from verification.calibration import TemperatureCalibrator, sigmoid
from verification.conflict_detector import ConflictAnalysisResult
from verification.llm_judge import LLMJudgeResult
from verification.source_verifier import SourceVerificationEntry


@dataclass
class AggregatedClaimConfidence:
    """Complete mathematical breakdown of C(claim) = f(E, K, A, J, R)."""

    claim_id: str
    entailment_strength_E: float
    contradiction_strength_K: float
    cross_source_agreement_A: float
    judge_assessment_J: float
    source_reliability_R: float
    evidence_coverage: float
    conflict_severity_CSI: float
    raw_logit: float
    uncalibrated_confidence: float
    calibrated_confidence: float
    abstention_decision: AbstentionDecision

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "E_entailment": round(self.entailment_strength_E, 4),
            "K_contradiction": round(self.contradiction_strength_K, 4),
            "A_agreement": round(self.cross_source_agreement_A, 4),
            "J_judge": round(self.judge_assessment_J, 4),
            "R_reliability": round(self.source_reliability_R, 4),
            "evidence_coverage": round(self.evidence_coverage, 4),
            "CSI_conflict": round(self.conflict_severity_CSI, 4),
            "raw_logit": round(self.raw_logit, 4),
            "uncalibrated_confidence": round(self.uncalibrated_confidence, 4),
            "calibrated_confidence": round(self.calibrated_confidence, 4),
            "final_classification": self.abstention_decision.final_classification,
            "fine_grained_subtype": self.abstention_decision.fine_grained_subtype,
            "abstained": self.abstention_decision.abstained,
            "abstention_reason": self.abstention_decision.abstention_reason,
        }


class AdaptiveConfidenceAggregator:
    """Computes C(claim) = sigma(z(E, K, A, J, R) / T) and supports validation-split tuning."""

    def __init__(
        self,
        config: VerificationConfig = DEFAULT_VERIFICATION_CONFIG,
        calibrator: TemperatureCalibrator | None = None,
    ) -> None:
        self.config = config
        self.calibrator = calibrator or TemperatureCalibrator(config.temperature)
        self.abstention = CalibratedAbstentionClassifier(config)

    def extract_features(
        self,
        source_entries: Dict[str, SourceVerificationEntry],
        conflict_result: ConflictAnalysisResult,
        judge_result: LLMJudgeResult,
    ) -> Dict[str, float]:
        """Compute the 5 primary variables (E, K, A, J, R) plus coverage and CSI."""
        if not source_entries:
            return {
                "E": 0.0,
                "K": 0.0,
                "A": 0.0,
                "J": 0.5,
                "R": 0.5,
                "coverage": 0.0,
                "CSI": 0.0,
            }

        # 1. Reliability-aware and raw E, K across active sources
        e_vals: List[float] = []
        k_vals: List[float] = []
        rel_num = 0.0
        rel_den = 0.0
        covered_sources = 0

        for entry in source_entries.values():
            pe = entry.nli_result.entailment_prob
            pk = entry.nli_result.contradiction_prob
            rel = entry.source_reliability if self.config.use_source_reliability else 0.80

            e_vals.append(pe * (0.75 + 0.25 * rel))
            k_vals.append(pk * (0.75 + 0.25 * rel))
            if max(pe, pk) >= self.config.substantive_evidence_floor:
                covered_sources += 1
            rel_num += rel * pe
            rel_den += rel * (pe + pk)

        E = float(max(e_vals)) if e_vals else 0.0
        K = float(max(k_vals)) if k_vals else 0.0
        raw_max_e = max(e.nli_result.entailment_prob for e in source_entries.values())
        raw_max_k = max(e.nli_result.contradiction_prob for e in source_entries.values())

        # Keep E and K bounded in [0, 1]
        E = min(1.0, max(E, raw_max_e * 0.95))
        K = min(1.0, max(K, raw_max_k * 0.95))

        # 2. Cross-source agreement A
        A = conflict_result.cross_source_agreement if self.config.use_conflict_detection else 0.5

        # 3. Source reliability balance R in [0, 1]
        R = (rel_num / rel_den) if (rel_den > 1e-6 and self.config.use_source_reliability) else 0.5

        # 4. Judge assessment J in [0, 1]
        if judge_result.invoked:
            J = judge_result.support_score
            # If judge resolved a cross-source conflict in favor of a higher-reliability source,
            # dampen the raw contradiction K from the lower-reliability source proportionally
            if judge_result.verification_decision == "SUPPORTED" and conflict_result.has_conflict:
                K = K * (1.0 - 0.55 * judge_result.confidence)
            elif judge_result.verification_decision == "CONTRADICTED" and conflict_result.has_conflict:
                E = E * (1.0 - 0.55 * judge_result.confidence)
        else:
            # When bypassed due to clear NLI consensus, align J with NLI consensus posterior
            total_ek = E + K
            J = (E / total_ek) if total_ek > 0.25 else 0.5

        coverage = covered_sources / max(1, len(source_entries))
        CSI = conflict_result.conflict_severity if self.config.use_conflict_detection else 0.0

        return {
            "E": E,
            "K": K,
            "A": A,
            "J": J,
            "R": R,
            "coverage": coverage,
            "CSI": CSI,
        }

    def compute_logit(self, feats: Dict[str, float]) -> float:
        """Compute z(c) = beta_0 + w_E*E - w_K*K + w_A*(A-0.5) + w_J*(J-0.5) + w_R*(R-0.5) - w_CSI*CSI."""
        if not self.config.use_confidence_aggregation:
            # Ablation: simple difference E - K
            return 3.0 * (feats["E"] - feats["K"])

        # Effective conflict penalty is reduced if the independent judge resolved the conflict with high confidence
        judge_resolution_factor = 1.0 - 0.65 * abs(feats["J"] - 0.5) * 2.0
        effective_csi = feats["CSI"] * max(0.2, judge_resolution_factor)

        z = (
            self.config.beta_0
            + self.config.w_entailment * feats["E"]
            - self.config.w_contradiction * feats["K"]
            + self.config.w_agreement * (feats["A"] - 0.5)
            + self.config.w_judge * (feats["J"] - 0.5)
            + self.config.w_reliability * (feats["R"] - 0.5)
            - self.config.w_conflict_penalty * effective_csi
        )
        return float(z)

    def aggregate(
        self,
        claim_id: str,
        source_entries: Dict[str, SourceVerificationEntry],
        conflict_result: ConflictAnalysisResult,
        judge_result: LLMJudgeResult,
    ) -> AggregatedClaimConfidence:
        """Aggregate all verification signals into a calibrated confidence score and 3-way classification."""
        feats = self.extract_features(source_entries, conflict_result, judge_result)
        raw_logit = self.compute_logit(feats)
        uncal_prob = sigmoid(raw_logit)
        cal_prob = self.calibrator.calibrate_logit(raw_logit)

        decision = self.abstention.classify(
            calibrated_confidence=cal_prob,
            entailment_strength=feats["E"],
            contradiction_strength=feats["K"],
            evidence_coverage=feats["coverage"],
            conflict_severity=feats["CSI"],
            judge_decision=judge_result.verification_decision,
        )

        return AggregatedClaimConfidence(
            claim_id=claim_id,
            entailment_strength_E=feats["E"],
            contradiction_strength_K=feats["K"],
            cross_source_agreement_A=feats["A"],
            judge_assessment_J=feats["J"],
            source_reliability_R=feats["R"],
            evidence_coverage=feats["coverage"],
            conflict_severity_CSI=feats["CSI"],
            raw_logit=raw_logit,
            uncalibrated_confidence=uncal_prob,
            calibrated_confidence=cal_prob,
            abstention_decision=decision,
        )

    def fit_weights_on_validation(
        self,
        feature_dicts: List[Dict[str, float]],
        val_labels: List[int],
    ) -> Dict[str, float]:
        """Tune aggregation weights on a validation split via L2-regularized logistic regression (zero test leakage)."""
        if len(feature_dicts) < 6 or len(feature_dicts) != len(val_labels):
            return self.config.to_dict()

        X = np.array(
            [
                [
                    1.0,
                    f["E"],
                    -f["K"],
                    f["A"] - 0.5,
                    f["J"] - 0.5,
                    f["R"] - 0.5,
                    -f["CSI"],
                ]
                for f in feature_dicts
            ],
            dtype=np.float64,
        )
        y = np.asarray(val_labels, dtype=np.float64)
        prior_w = np.array(
            [
                self.config.beta_0,
                self.config.w_entailment,
                self.config.w_contradiction,
                self.config.w_agreement,
                self.config.w_judge,
                self.config.w_reliability,
                self.config.w_conflict_penalty,
            ],
            dtype=np.float64,
        )

        def loss_fn(w: np.ndarray) -> float:
            logits = np.clip(X @ w, -25.0, 25.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            eps = 1e-7
            nll = -np.mean(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))
            # L2 regularization toward positive theoretical priors
            reg = 0.08 * float(np.sum((w - prior_w) ** 2))
            return float(nll + reg)

        bounds = [
            (-2.0, 2.0),   # beta_0
            (1.0, 6.0),    # w_E > 0
            (1.0, 6.0),    # w_K > 0
            (0.1, 3.0),    # w_A > 0
            (0.2, 4.0),    # w_J > 0
            (0.1, 3.0),    # w_R > 0
            (0.1, 3.0),    # w_CSI > 0
        ]
        res = minimize(loss_fn, x0=prior_w, bounds=bounds, method="L-BFGS-B")
        if res.success:
            w_opt = res.x
            self.config.beta_0 = float(w_opt[0])
            self.config.w_entailment = float(w_opt[1])
            self.config.w_contradiction = float(w_opt[2])
            self.config.w_agreement = float(w_opt[3])
            self.config.w_judge = float(w_opt[4])
            self.config.w_reliability = float(w_opt[5])
            self.config.w_conflict_penalty = float(w_opt[6])

        # Fit temperature scaling on validation logits
        val_logits = [self.compute_logit(f) for f in feature_dicts]
        learned_temp = self.calibrator.fit(val_logits, val_labels)
        self.config.temperature = learned_temp
        return self.config.to_dict()
