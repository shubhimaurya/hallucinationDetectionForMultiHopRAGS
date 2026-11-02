"""Citation Grounding Validator and Claim-Level Explanation Generator.

Ensures that:
1. Every generated citation points to a real document retrieved during the query.
2. Any fabricated citation to an unretrieved document is flagged and stripped.
3. Every verified claim receives a complete, traceable explanation with source references.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from config.model_config import DEFAULT_VERIFICATION_CONFIG, VerificationConfig
from verification.adaptive_verifier import AdaptiveLLMVerifier
from verification.claim_extractor import AtomicClaim, AtomicClaimExtractor
from verification.confidence_aggregator import (
    AdaptiveConfidenceAggregator,
    AggregatedClaimConfidence,
)
from verification.conflict_detector import (
    ConflictAnalysisResult,
    CrossSourceConflictDetector,
)
from verification.llm_judge import LLMJudgeResult
from verification.source_verifier import SourceVerificationEntry, SourceWiseVerifier


@dataclass
class CitationAuditReport:
    """Validation report checking cited document IDs against actually retrieved documents."""

    cited_doc_ids: List[str]
    valid_cited_doc_ids: List[str]
    hallucinated_cited_doc_ids: List[str]
    citation_precision: float
    verified_supporting_doc_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cited_doc_ids": self.cited_doc_ids,
            "valid_cited_doc_ids": self.valid_cited_doc_ids,
            "hallucinated_cited_doc_ids": self.hallucinated_cited_doc_ids,
            "citation_precision": round(self.citation_precision, 4),
            "verified_supporting_doc_ids": self.verified_supporting_doc_ids,
        }


@dataclass
class VerifiedClaimRecord:
    """Complete end-to-end verification dossier for a single atomic claim."""

    claim: AtomicClaim
    source_verifications: Dict[str, SourceVerificationEntry]
    conflict_analysis: ConflictAnalysisResult
    judge_result: LLMJudgeResult
    confidence_result: AggregatedClaimConfidence
    citation_audit: CitationAuditReport
    explanation_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim.claim_id,
            "claim_text": self.claim.claim_text,
            "original_sentence": self.claim.original_sentence,
            "cited_doc_ids": self.claim.cited_doc_ids,
            "source_verifications": {
                k: v.to_dict() for k, v in self.source_verifications.items()
            },
            "conflict_analysis": self.conflict_analysis.to_dict(),
            "judge_result": self.judge_result.to_dict(),
            "confidence_result": self.confidence_result.to_dict(),
            "citation_audit": self.citation_audit.to_dict(),
            "final_classification": self.confidence_result.abstention_decision.final_classification,
            "fine_grained_subtype": self.confidence_result.abstention_decision.fine_grained_subtype,
            "calibrated_confidence": round(self.confidence_result.calibrated_confidence, 4),
            "explanation_text": self.explanation_text,
        }


class CitationValidator:
    """Validates inline citations against retrieved documents and builds traceable explanations."""

    @staticmethod
    def validate_claim_citations(
        claim: AtomicClaim,
        retrieved_doc_ids: set[str],
        source_entries: Dict[str, SourceVerificationEntry],
    ) -> CitationAuditReport:
        """Verify that cited document IDs exist in retrieved_doc_ids and entail the claim."""
        valid_cites = [cid for cid in claim.cited_doc_ids if cid in retrieved_doc_ids]
        hallucinated_cites = [cid for cid in claim.cited_doc_ids if cid not in retrieved_doc_ids]

        supporting_docs: List[str] = []
        for entry in source_entries.values():
            if (
                entry.nli_result.label == "SUPPORTED"
                and entry.nli_result.best_doc_id
                and entry.nli_result.best_doc_id in retrieved_doc_ids
            ):
                if entry.nli_result.best_doc_id not in supporting_docs:
                    supporting_docs.append(entry.nli_result.best_doc_id)

        if claim.cited_doc_ids:
            precision = len(valid_cites) / len(claim.cited_doc_ids)
        else:
            precision = 1.0 if supporting_docs else 0.0

        return CitationAuditReport(
            cited_doc_ids=claim.cited_doc_ids,
            valid_cited_doc_ids=valid_cites,
            hallucinated_cited_doc_ids=hallucinated_cites,
            citation_precision=precision,
            verified_supporting_doc_ids=supporting_docs,
        )

    @staticmethod
    def build_claim_explanation(
        claim: AtomicClaim,
        source_entries: Dict[str, SourceVerificationEntry],
        conflict: ConflictAnalysisResult,
        judge: LLMJudgeResult,
        conf: AggregatedClaimConfidence,
        cite_audit: CitationAuditReport,
    ) -> str:
        """Generate a human-readable, citation-grounded explanation for the claim verdict."""
        verdict = conf.abstention_decision.final_classification
        subtype = conf.abstention_decision.fine_grained_subtype
        sup_snippets: List[str] = []
        con_snippets: List[str] = []

        for src_id, entry in source_entries.items():
            r = entry.nli_result
            if r.label == "SUPPORTED" and r.best_doc_id:
                sup_snippets.append(f"[{r.best_doc_id}] ({src_id}, E={r.entailment_prob:.2f}): '{r.best_evidence_window}'")
            elif r.label == "CONTRADICTED" and r.best_doc_id:
                con_snippets.append(
                    f"[{r.best_doc_id}] ({src_id}, K={r.contradiction_prob:.2f}, {r.diagnostic_reason}): '{r.best_evidence_window}'"
                )

        parts = [
            f"Claim [{claim.claim_id}] classified as {verdict} ({subtype}) with calibrated confidence {conf.calibrated_confidence:.2f}.",
            f"Cross-source analysis: {conflict.summary}",
        ]
        if sup_snippets:
            parts.append("Supporting evidence: " + " | ".join(sup_snippets))
        if con_snippets:
            parts.append("Contradictory evidence: " + " | ".join(con_snippets))
        if judge.invoked:
            parts.append(f"Adaptive LLM Judge ({judge.verification_decision}, conf={judge.confidence:.2f}): {judge.explanation}")
        else:
            parts.append(f"Adaptive LLM Judge bypassed ({judge.trigger_reason}).")
        if cite_audit.hallucinated_cited_doc_ids:
            parts.append(
                f"WARNING: Non-existent citation(s) detected and rejected: {cite_audit.hallucinated_cited_doc_ids}."
            )
        return " ".join(parts)


class ACVEPipelineVerifier:
    """Complete ACVE verification orchestrator executing Contributions 1 through 7."""

    def __init__(self, config: VerificationConfig = DEFAULT_VERIFICATION_CONFIG) -> None:
        self.config = config
        self.claim_extractor = AtomicClaimExtractor()
        self.source_verifier = SourceWiseVerifier(config=config)
        self.conflict_detector = CrossSourceConflictDetector(config=config)
        self.adaptive_verifier = AdaptiveLLMVerifier(config=config)
        self.aggregator = AdaptiveConfidenceAggregator(config=config)
        self.citation_validator = CitationValidator()

    def verify_answer(
        self,
        answer_text: str,
        source_wise_chunks: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Decompose an answer into atomic claims and run full ACVE verification."""
        claims = self.claim_extractor.extract_claims(
            answer_text=answer_text,
            decompose=self.config.use_claim_decomposition,
        )

        retrieved_doc_ids: set[str] = set()
        for chunks in source_wise_chunks.values():
            for c in chunks:
                if c.get("doc_id"):
                    retrieved_doc_ids.add(str(c["doc_id"]))

        verified_records: List[VerifiedClaimRecord] = []
        Grounded_sentences: List[str] = []
        llm_judge_calls = 0
        total_judge_tokens = 0
        total_judge_latency_ms = 0.0

        for claim in claims:
            src_entries = self.source_verifier.verify_claim_by_source(claim, source_wise_chunks)
            conflict_res = self.conflict_detector.analyze(claim.claim_id, src_entries)
            judge_res = self.adaptive_verifier.verify_adaptively(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                source_entries=src_entries,
                conflict_result=conflict_res,
            )
            if judge_res.invoked:
                llm_judge_calls += 1
                total_judge_tokens += judge_res.tokens_used
                total_judge_latency_ms += judge_res.latency_ms

            conf_res = self.aggregator.aggregate(
                claim_id=claim.claim_id,
                source_entries=src_entries,
                conflict_result=conflict_res,
                judge_result=judge_res,
            )
            cite_audit = self.citation_validator.validate_claim_citations(
                claim=claim,
                retrieved_doc_ids=retrieved_doc_ids,
                source_entries=src_entries,
            )
            explanation = self.citation_validator.build_claim_explanation(
                claim=claim,
                source_entries=src_entries,
                conflict=conflict_res,
                judge=judge_res,
                conf=conf_res,
                cite_audit=cite_audit,
            )
            record = VerifiedClaimRecord(
                claim=claim,
                source_verifications=src_entries,
                conflict_analysis=conflict_res,
                judge_result=judge_res,
                confidence_result=conf_res,
                citation_audit=cite_audit,
                explanation_text=explanation,
            )
            verified_records.append(record)

            # Build citation-grounded sentence using ONLY verified retrieved doc IDs
            valid_cites = cite_audit.verified_supporting_doc_ids or cite_audit.valid_cited_doc_ids
            cite_tag = " " + " ".join(f"[{d}]" for d in valid_cites) if valid_cites else ""
            status_tag = conf_res.abstention_decision.final_classification
            Grounded_sentences.append(f"[{status_tag}] {claim.claim_text}{cite_tag}")

        supported_count = sum(
            1 for r in verified_records if r.confidence_result.abstention_decision.final_classification == "SUPPORTED"
        )
        hallucinated_count = sum(
            1 for r in verified_records if r.confidence_result.abstention_decision.final_classification == "HALLUCINATED"
        )
        uncertain_count = sum(
            1 for r in verified_records if r.confidence_result.abstention_decision.final_classification == "UNCERTAIN"
        )
        avg_conf = (
            sum(r.confidence_result.calibrated_confidence for r in verified_records) / len(verified_records)
            if verified_records
            else 0.0
        )

        return {
            "original_answer": answer_text,
            "citation_grounded_response": " ".join(Grounded_sentences),
            "total_claims": len(verified_records),
            "supported_claims": supported_count,
            "hallucinated_claims": hallucinated_count,
            "uncertain_claims": uncertain_count,
            "average_confidence": round(avg_conf, 4),
            "llm_judge_calls": llm_judge_calls,
            "llm_judge_tokens": total_judge_tokens,
            "llm_judge_latency_ms": round(total_judge_latency_ms, 2),
            "claims": [r.to_dict() for r in verified_records],
        }
