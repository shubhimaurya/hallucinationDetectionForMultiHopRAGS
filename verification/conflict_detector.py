"""Cross-Source Conflict Detection module.

Constructs pairwise agreement and polarity-contradiction matrices across RAG 1..4,
computes Jensen-Shannon Divergence and the Conflict Severity Index (CSI), and
identifies inter-source disagreements without naive majority-vote suppression.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

from config.model_config import DEFAULT_VERIFICATION_CONFIG, VerificationConfig
from verification.source_verifier import SourceVerificationEntry


def _kl_div(p: List[float], q: List[float]) -> float:
    eps = 1e-9
    return sum(pi * math.log((pi + eps) / (qi + eps), 2) for pi, qi in zip(p, q) if pi > 0)


def jensen_shannon_divergence(p: List[float], q: List[float]) -> float:
    """Compute normalized base-2 Jensen-Shannon Divergence in [0, 1]."""
    m = [0.5 * (pi + qi) for pi, qi in zip(p, q)]
    jsd = 0.5 * _kl_div(p, m) + 0.5 * _kl_div(q, m)
    return max(0.0, min(1.0, jsd))


@dataclass
class ConflictAnalysisResult:
    """Detailed cross-source conflict report for a single atomic claim."""

    claim_id: str
    has_conflict: bool
    conflict_type: str
    conflict_severity: float  # CSI in [0, 1]
    cross_source_agreement: float  # A(c) in [0, 1]
    supporting_sources: List[str]
    contradicting_sources: List[str]
    insufficient_sources: List[str]
    source_verification_table: List[Dict[str, Any]]
    agreement_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    contradiction_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "has_conflict": self.has_conflict,
            "conflict_type": self.conflict_type,
            "conflict_severity": round(self.conflict_severity, 4),
            "cross_source_agreement": round(self.cross_source_agreement, 4),
            "supporting_sources": self.supporting_sources,
            "contradicting_sources": self.contradicting_sources,
            "insufficient_sources": self.insufficient_sources,
            "source_verification_table": self.source_verification_table,
            "agreement_matrix": self.agreement_matrix,
            "contradiction_matrix": self.contradiction_matrix,
            "summary": self.summary,
        }


class CrossSourceConflictDetector:
    """Detects disagreements across independent knowledge collections while keeping reliability, relevance, and agreement distinct."""

    def __init__(self, config: VerificationConfig = DEFAULT_VERIFICATION_CONFIG) -> None:
        self.config = config

    def analyze(
        self,
        claim_id: str,
        source_entries: Dict[str, SourceVerificationEntry],
    ) -> ConflictAnalysisResult:
        """Compute cross-source conflict matrices, CSI, and agreement score for a claim."""
        supporting: List[str] = []
        contradicting: List[str] = []
        insufficient: List[str] = []
        table_rows: List[Dict[str, Any]] = []

        sources = list(source_entries.keys())
        for src_id in sources:
            entry = source_entries[src_id]
            lbl = entry.nli_result.label
            if lbl == "SUPPORTED":
                supporting.append(src_id)
            elif lbl == "CONTRADICTED":
                contradicting.append(src_id)
            else:
                insufficient.append(src_id)
            table_rows.append(entry.to_dict())

        agreement_matrix: Dict[str, Dict[str, float]] = {s: {} for s in sources}
        contradiction_matrix: Dict[str, Dict[str, float]] = {s: {} for s in sources}

        max_csi = 0.0
        substantive_sources = [
            s
            for s in sources
            if max(
                source_entries[s].nli_result.entailment_prob,
                source_entries[s].nli_result.contradiction_prob,
            )
            >= self.config.substantive_evidence_floor
        ]

        pair_agreements: List[float] = []
        for i, src_a in enumerate(sources):
            ea = source_entries[src_a].nli_result
            dist_a = [ea.entailment_prob, ea.contradiction_prob, ea.neutral_prob]
            for j, src_b in enumerate(sources):
                if i == j:
                    agreement_matrix[src_a][src_b] = 1.0
                    contradiction_matrix[src_a][src_b] = 0.0
                    continue
                eb = source_entries[src_b].nli_result
                dist_b = [eb.entailment_prob, eb.contradiction_prob, eb.neutral_prob]

                jsd = jensen_shannon_divergence(dist_a, dist_b)
                polarity_clash = (
                    ea.entailment_prob * eb.contradiction_prob
                    + ea.contradiction_prob * eb.entailment_prob
                )
                pair_agree = max(0.0, min(1.0, (1.0 - jsd) * (1.0 - polarity_clash)))
                agreement_matrix[src_a][src_b] = round(pair_agree, 4)
                contradiction_matrix[src_a][src_b] = round(polarity_clash, 4)

                if i < j:
                    if polarity_clash > max_csi:
                        max_csi = polarity_clash
                    if src_a in substantive_sources and src_b in substantive_sources:
                        pair_agreements.append(pair_agree)

        # Boost CSI when discrete SUPPORTED and CONTRADICTED labels co-occur
        if supporting and contradicting:
            max_csi = max(max_csi, 0.55)

        if len(substantive_sources) >= 2 and pair_agreements:
            agreement_score = sum(pair_agreements) / len(pair_agreements)
        elif len(substantive_sources) == 1:
            agreement_score = 0.50  # Single-source evidence (uncorroborated)
        else:
            agreement_score = 0.0

        has_conflict = (
            (len(supporting) > 0 and len(contradicting) > 0)
            or (max_csi >= self.config.conflict_csi_threshold)
        )

        if supporting and contradicting:
            conflict_type = "CROSS_SOURCE_CONTRADICTION"
            summary = (
                f"Disagreement detected: supported by {supporting} but contradicted by {contradicting} "
                f"(CSI={max_csi:.2f})."
            )
        elif has_conflict:
            conflict_type = "PARTIAL_POLARITY_CONFLICT"
            summary = f"Probabilistic polarity tension across sources (CSI={max_csi:.2f})."
        elif len(supporting) >= 2 and not contradicting:
            conflict_type = "CONSENSUS_SUPPORT"
            summary = f"Multi-source consensus support across {supporting} (agreement={agreement_score:.2f})."
        elif len(supporting) == 1 and not contradicting:
            conflict_type = "SINGLE_SOURCE_SUPPORT"
            summary = f"Supported only by {supporting[0]}; other sources lack sufficient evidence."
        elif contradicting and not supporting:
            conflict_type = "CONSENSUS_CONTRADICTION"
            summary = f"Refuted by {contradicting} with no supporting sources."
        else:
            conflict_type = "INSUFFICIENT_EVIDENCE"
            summary = "No active RAG pipeline provides sufficient evidence to confirm or refute the claim."

        return ConflictAnalysisResult(
            claim_id=claim_id,
            has_conflict=has_conflict,
            conflict_type=conflict_type,
            conflict_severity=float(max_csi),
            cross_source_agreement=float(agreement_score),
            supporting_sources=supporting,
            contradicting_sources=contradicting,
            insufficient_sources=insufficient,
            source_verification_table=table_rows,
            agreement_matrix=agreement_matrix,
            contradiction_matrix=contradiction_matrix,
            summary=summary,
        )
