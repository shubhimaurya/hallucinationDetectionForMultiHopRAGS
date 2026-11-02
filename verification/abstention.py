"""Three-Way Claim Classification and Calibrated Abstention Module.

Distinguishes between:
- SUPPORTED
- HALLUCINATED (subtyped into DIRECTLY_CONTRADICTED vs UNSUPPORTED_FABRICATION)
- UNCERTAIN (subtyped into INSUFFICIENT_EVIDENCE, CONFLICTING_EVIDENCE, LOW_CONFIDENCE)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from config.model_config import DEFAULT_VERIFICATION_CONFIG, VerificationConfig


@dataclass
class AbstentionDecision:
    """Final three-way classification and uncertainty diagnostic for an atomic claim."""

    final_classification: str  # "SUPPORTED", "HALLUCINATED", "UNCERTAIN"
    fine_grained_subtype: str
    abstained: bool
    abstention_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_classification": self.final_classification,
            "fine_grained_subtype": self.fine_grained_subtype,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
        }


class CalibratedAbstentionClassifier:
    """Applies calibrated thresholds and sufficiency checks to classify claims or abstain safely."""

    def __init__(self, config: VerificationConfig = DEFAULT_VERIFICATION_CONFIG) -> None:
        self.config = config

    def classify(
        self,
        calibrated_confidence: float,
        entailment_strength: float,
        contradiction_strength: float,
        evidence_coverage: float,
        conflict_severity: float,
        judge_decision: str,
    ) -> AbstentionDecision:
        """Map aggregated confidence and evidence diagnostics to SUPPORTED, HALLUCINATED, or UNCERTAIN."""
        # 1. Check if any source has substantive evidence
        max_signal = max(entailment_strength, contradiction_strength)
        if max_signal < self.config.evidence_sufficiency_floor and evidence_coverage < 0.25:
            return AbstentionDecision(
                final_classification="UNCERTAIN",
                fine_grained_subtype="INSUFFICIENT_EVIDENCE",
                abstained=True,
                abstention_reason=(
                    f"Retrieved evidence signal ({max_signal:.2f}) is below the sufficiency floor "
                    f"({self.config.evidence_sufficiency_floor:.2f})."
                ),
            )

        # 2. Check if LLM judge explicitly declared UNCERTAIN under severe cross-source conflict
        if (
            judge_decision == "UNCERTAIN"
            and conflict_severity >= 0.35
            and self.config.hallucinated_threshold < calibrated_confidence < self.config.supported_threshold
        ):
            return AbstentionDecision(
                final_classification="UNCERTAIN",
                fine_grained_subtype="CONFLICTING_EVIDENCE",
                abstained=True,
                abstention_reason=(
                    f"Unresolved cross-source conflict (CSI={conflict_severity:.2f}) with borderline confidence ({calibrated_confidence:.2f})."
                ),
            )

        # 3. Direct Contradiction -> HALLUCINATED
        if (
            contradiction_strength >= 0.52
            and (contradiction_strength > entailment_strength or judge_decision == "CONTRADICTED")
        ):
            return AbstentionDecision(
                final_classification="HALLUCINATED",
                fine_grained_subtype="DIRECTLY_CONTRADICTED",
                abstained=False,
                abstention_reason="",
            )

        # 4. High Calibrated Confidence -> SUPPORTED
        if calibrated_confidence >= self.config.supported_threshold and contradiction_strength < 0.48:
            return AbstentionDecision(
                final_classification="SUPPORTED",
                fine_grained_subtype="VERIFIED_SUPPORTED",
                abstained=False,
                abstention_reason="",
            )

        # 5. Low Calibrated Confidence -> HALLUCINATED
        if calibrated_confidence <= self.config.hallucinated_threshold:
            subtype = (
                "DIRECTLY_CONTRADICTED"
                if contradiction_strength >= 0.45
                else "UNSUPPORTED_FABRICATION"
            )
            return AbstentionDecision(
                final_classification="HALLUCINATED",
                fine_grained_subtype=subtype,
                abstained=False,
                abstention_reason="",
            )

        # 6. Intermediate zone -> UNCERTAIN (Abstain)
        return AbstentionDecision(
            final_classification="UNCERTAIN",
            fine_grained_subtype="LOW_CONFIDENCE_BORDERLINE",
            abstained=True,
            abstention_reason=(
                f"Calibrated confidence ({calibrated_confidence:.2f}) lies in the abstention band "
                f"[{self.config.hallucinated_threshold:.2f}, {self.config.supported_threshold:.2f}]."
            ),
        )
