"""Hyperparameters and mathematical priors for NLI, Conflict Detection, Adaptive Verification, and Confidence Aggregation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SourceCollectionConfig:
    """Configuration for an independent RAG knowledge source."""

    source_id: str
    display_name: str
    description: str
    default_reliability: float
    expected_doc_types: list[str]


RAG_SOURCE_CONFIGS: Dict[str, SourceCollectionConfig] = {
    "academic": SourceCollectionConfig(
        source_id="academic",
        display_name="RAG 1: Academic Research Papers",
        description="Peer-reviewed scientific articles, conference papers, and archival technical specifications.",
        default_reliability=0.92,
        expected_doc_types=["pdf", "md", "txt"],
    ),
    "pdf": SourceCollectionConfig(
        source_id="pdf",
        display_name="RAG 2: PDF & Organizational Documents",
        description="Official institutional reports, mission manuals, policy handbooks, and standard operating procedures.",
        default_reliability=0.88,
        expected_doc_types=["pdf", "docx", "txt"],
    ),
    "web": SourceCollectionConfig(
        source_id="web",
        display_name="RAG 3: Web Knowledge Documents",
        description="Encyclopedia snapshots, news wire reports, blog posts, and community articles (variable reliability).",
        default_reliability=0.68,
        expected_doc_types=["md", "txt", "pdf"],
    ),
    "structured": SourceCollectionConfig(
        source_id="structured",
        display_name="RAG 4: Structured Knowledge Sources",
        description="Curated tabular records, telemetry logs, financial/scientific CSV tables, and fact registries.",
        default_reliability=0.94,
        expected_doc_types=["csv", "txt", "md"],
    ),
}


@dataclass
class VerificationConfig:
    """Thresholds and aggregation parameters for ACVE-Hallucination Detector."""

    # Source-wise NLI classification thresholds
    nli_entailment_threshold: float = 0.52
    nli_contradiction_threshold: float = 0.48
    nli_margin_delta: float = 0.08
    nli_evidence_window_tokens: int = 180

    # Cross-source conflict thresholds
    conflict_csi_threshold: float = 0.24
    substantive_evidence_floor: float = 0.32

    # Adaptive LLM Judge gatekeeper thresholds
    adaptive_high_entailment_bypass: float = 0.74
    adaptive_low_evidence_bypass: float = 0.25
    adaptive_ambiguity_margin: float = 0.22
    adaptive_conflict_trigger: float = 0.24

    # Confidence Aggregation prior parameters (tunable on validation set)
    # z(c) = beta_0 + w_E * E - w_K * K + w_A * (A - 0.5) + w_J * (J - 0.5) + w_R * (R - 0.5) - w_CSI * CSI
    beta_0: float = -0.35
    w_entailment: float = 3.20
    w_contradiction: float = 3.60
    w_agreement: float = 1.15
    w_judge: float = 1.85
    w_reliability: float = 0.90
    w_conflict_penalty: float = 1.40

    # Calibration & 3-way Abstention thresholds
    temperature: float = 1.00
    supported_threshold: float = 0.58
    hallucinated_threshold: float = 0.42
    evidence_sufficiency_floor: float = 0.30
    unresolved_conflict_ceiling: float = 0.55

    # Ablation flags
    use_claim_decomposition: bool = True
    use_nli_verification: bool = True
    use_conflict_detection: bool = True
    use_adaptive_llm_judge: bool = True
    use_confidence_aggregation: bool = True
    use_source_reliability: bool = True
    force_llm_judge_all: bool = False
    disable_llm_judge_all: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "nli_entailment_threshold": self.nli_entailment_threshold,
            "nli_contradiction_threshold": self.nli_contradiction_threshold,
            "conflict_csi_threshold": self.conflict_csi_threshold,
            "adaptive_high_entailment_bypass": self.adaptive_high_entailment_bypass,
            "adaptive_low_evidence_bypass": self.adaptive_low_evidence_bypass,
            "beta_0": self.beta_0,
            "w_entailment": self.w_entailment,
            "w_contradiction": self.w_contradiction,
            "w_agreement": self.w_agreement,
            "w_judge": self.w_judge,
            "w_reliability": self.w_reliability,
            "w_conflict_penalty": self.w_conflict_penalty,
            "temperature": self.temperature,
            "supported_threshold": self.supported_threshold,
            "hallucinated_threshold": self.hallucinated_threshold,
            "evidence_sufficiency_floor": self.evidence_sufficiency_floor,
            "use_claim_decomposition": self.use_claim_decomposition,
            "use_nli_verification": self.use_nli_verification,
            "use_conflict_detection": self.use_conflict_detection,
            "use_adaptive_llm_judge": self.use_adaptive_llm_judge,
            "use_confidence_aggregation": self.use_confidence_aggregation,
            "use_source_reliability": self.use_source_reliability,
        }


DEFAULT_VERIFICATION_CONFIG = VerificationConfig()
