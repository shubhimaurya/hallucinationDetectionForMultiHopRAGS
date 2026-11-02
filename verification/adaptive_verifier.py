"""Adaptive Verification Gatekeeper deciding whether an expensive LLM Judge call is warranted."""
from __future__ import annotations

from typing import Dict, Tuple

from config.model_config import DEFAULT_VERIFICATION_CONFIG, VerificationConfig
from verification.conflict_detector import ConflictAnalysisResult
from verification.llm_judge import IndependentLLMJudge, LLMJudgeResult
from verification.source_verifier import SourceVerificationEntry


class AdaptiveLLMVerifier:
    """Conditionally routes claims to the Independent LLM Judge based on NLI confidence and cross-source conflict."""

    def __init__(
        self,
        judge: IndependentLLMJudge | None = None,
        config: VerificationConfig = DEFAULT_VERIFICATION_CONFIG,
    ) -> None:
        self.config = config
        self.judge = judge or IndependentLLMJudge()

    def should_invoke_judge(
        self,
        source_entries: Dict[str, SourceVerificationEntry],
        conflict_result: ConflictAnalysisResult,
    ) -> Tuple[bool, str]:
        """Return (invoke_flag, trigger_reason) according to adaptive gating rules."""
        if self.config.disable_llm_judge_all or not self.config.use_adaptive_llm_judge:
            return False, "LLM Judge disabled by configuration"

        if self.config.force_llm_judge_all:
            return True, "Unconditional hybrid judge mode (E3)"

        max_ent = max(
            (e.nli_result.entailment_prob for e in source_entries.values()), default=0.0
        )
        max_con = max(
            (e.nli_result.contradiction_prob for e in source_entries.values()), default=0.0
        )

        # Rule 1: Total evidence absence -> abstain directly without wasting an LLM Judge call
        if (
            max_ent < self.config.adaptive_low_evidence_bypass
            and max_con < self.config.adaptive_low_evidence_bypass
        ):
            return False, "Bypass: insufficient retrieved evidence across all sources"

        # Rule 2: Cross-source conflict -> MUST invoke independent LLM Judge
        if (
            conflict_result.has_conflict
            or conflict_result.conflict_severity >= self.config.adaptive_conflict_trigger
        ):
            return (
                True,
                f"Triggered: cross-source conflict ({conflict_result.conflict_type}, CSI={conflict_result.conflict_severity:.2f})",
            )

        # Rule 3: High-confidence NLI agreement without conflict -> bypass LLM Judge
        if (
            max_ent >= self.config.adaptive_high_entailment_bypass
            and max_con < 0.30
            and not conflict_result.contradicting_sources
        ):
            return False, f"Bypass: high-confidence NLI entailment ({max_ent:.2f}) with zero conflict"

        # Rule 4: High-confidence unanimous NLI contradiction -> bypass LLM Judge
        if (
            max_con >= 0.78
            and max_ent < 0.28
            and not conflict_result.supporting_sources
        ):
            return False, f"Bypass: clear NLI contradiction ({max_con:.2f}) with zero supporting sources"

        # Rule 5: Ambiguous NLI margin or moderate uncorroborated evidence -> invoke Judge
        if abs(max_ent - max_con) < self.config.adaptive_ambiguity_margin:
            return True, f"Triggered: narrow NLI margin (|E - K| = {abs(max_ent - max_con):.2f})"

        return True, f"Triggered: moderate NLI confidence (E={max_ent:.2f}, K={max_con:.2f}) requiring verification"

    def verify_adaptively(
        self,
        claim_id: str,
        claim_text: str,
        source_entries: Dict[str, SourceVerificationEntry],
        conflict_result: ConflictAnalysisResult,
    ) -> LLMJudgeResult:
        """Evaluate gatekeeper policy and either invoke the LLM Judge or return a non-invoked trace."""
        invoke, reason = self.should_invoke_judge(source_entries, conflict_result)
        if not invoke:
            return LLMJudgeResult(
                claim_id=claim_id,
                invoked=False,
                trigger_reason=reason,
                verification_decision="NOT_INVOKED",
                supporting_evidence=[],
                contradictory_evidence=[],
                explanation=f"Adaptive gatekeeper bypassed LLM Judge ({reason}).",
                confidence=0.5,
                uncertainty_reason="",
                latency_ms=0.0,
                tokens_used=0,
            )
        return self.judge.evaluate(
            claim_id=claim_id,
            claim_text=claim_text,
            source_entries=source_entries,
            conflict_result=conflict_result,
            trigger_reason=reason,
        )
