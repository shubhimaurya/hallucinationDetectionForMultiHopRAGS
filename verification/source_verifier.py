"""Source-Wise NLI Verification module preserving separate verification distributions per RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from config.model_config import (
    DEFAULT_VERIFICATION_CONFIG,
    RAG_SOURCE_CONFIGS,
    VerificationConfig,
)
from verification.claim_extractor import AtomicClaim
from verification.nli_verifier import NLIScoreResult, NLIVerifier


@dataclass
class SourceVerificationEntry:
    """Verification outcome for a single claim against a single RAG collection."""

    source_id: str
    source_display_name: str
    source_reliability: float
    retrieval_relevance: float
    nli_result: NLIScoreResult
    evidence_chunks_evaluated: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_display_name": self.source_display_name,
            "source_reliability": round(self.source_reliability, 4),
            "retrieval_relevance": round(self.retrieval_relevance, 4),
            "classification": self.nli_result.label,
            "entailment_prob": round(self.nli_result.entailment_prob, 4),
            "contradiction_prob": round(self.nli_result.contradiction_prob, 4),
            "neutral_prob": round(self.nli_result.neutral_prob, 4),
            "best_evidence_window": self.nli_result.best_evidence_window,
            "best_doc_id": self.nli_result.best_doc_id,
            "diagnostic_reason": self.nli_result.diagnostic_reason,
            "evidence_chunks_evaluated": self.evidence_chunks_evaluated,
        }


class SourceWiseVerifier:
    """Evaluates each atomic claim independently against each active RAG pipeline before any aggregation."""

    def __init__(
        self,
        nli_verifier: NLIVerifier | None = None,
        config: VerificationConfig = DEFAULT_VERIFICATION_CONFIG,
    ) -> None:
        self.config = config
        self.nli = nli_verifier or NLIVerifier(config=config)

    def verify_claim_by_source(
        self,
        claim: AtomicClaim,
        source_wise_chunks: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, SourceVerificationEntry]:
        """Run NLI verification independently for every RAG source in source_wise_chunks."""
        outcomes: Dict[str, SourceVerificationEntry] = {}

        for src_id, chunks in source_wise_chunks.items():
            src_cfg = RAG_SOURCE_CONFIGS.get(src_id)
            display_name = src_cfg.display_name if src_cfg else src_id
            default_rel = src_cfg.default_reliability if src_cfg else 0.75

            if chunks:
                max_rel_score = max(float(c.get("retrieval_score", 0.0)) for c in chunks)
                src_reliability = float(
                    chunks[0].get("source_reliability", default_rel)
                )
            else:
                max_rel_score = 0.0
                src_reliability = default_rel

            if not self.config.use_source_reliability:
                src_reliability = 0.80  # Uniform reliability when ablated

            nli_res = self.nli.verify_claim_against_chunks(claim.claim_text, chunks)
            outcomes[src_id] = SourceVerificationEntry(
                source_id=src_id,
                source_display_name=display_name,
                source_reliability=src_reliability,
                retrieval_relevance=max_rel_score,
                nli_result=nli_res,
                evidence_chunks_evaluated=len(chunks),
            )
        return outcomes
