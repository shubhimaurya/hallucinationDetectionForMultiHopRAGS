"""Independent Evidence-Grounded LLM Judge returning structured JSON assessments."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import requests

from config.settings import settings
from verification.conflict_detector import ConflictAnalysisResult
from verification.source_verifier import SourceVerificationEntry

logger = logging.getLogger(__name__)

LLM_JUDGE_SYSTEM_PROMPT = """You are an impartial, evidence-bound factual verification judge.
CRITICAL RULES:
1. Evaluate the atomic claim strictly using the provided source-wise evidence snippets.
2. NEVER invent evidence or use external unretrieved knowledge as proof.
3. Weigh source reliability metadata and direct specificity when sources conflict.
4. Return ONLY valid JSON with keys:
   "claim_id", "verification_decision" ("SUPPORTED" | "CONTRADICTED" | "UNCERTAIN"),
   "supporting_evidence" (list of strings), "contradictory_evidence" (list of strings),
   "explanation" (string), "confidence" (float in [0, 1]), "uncertainty_reason" (string).
"""


@dataclass
class LLMJudgeResult:
    """Structured JSON output from the independent LLM Judge."""

    claim_id: str
    invoked: bool
    trigger_reason: str
    verification_decision: str  # "SUPPORTED", "CONTRADICTED", "UNCERTAIN"
    supporting_evidence: List[str] = field(default_factory=list)
    contradictory_evidence: List[str] = field(default_factory=list)
    explanation: str = ""
    confidence: float = 0.5
    uncertainty_reason: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0

    @property
    def support_score(self) -> float:
        """Convert directional decision and confidence into a continuous support score J in [0, 1]."""
        if not self.invoked:
            return 0.5
        if self.verification_decision == "SUPPORTED":
            return 0.5 + 0.5 * self.confidence
        if self.verification_decision == "CONTRADICTED":
            return 0.5 - 0.5 * self.confidence
        return 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "invoked": self.invoked,
            "trigger_reason": self.trigger_reason,
            "verification_decision": self.verification_decision,
            "supporting_evidence": self.supporting_evidence,
            "contradictory_evidence": self.contradictory_evidence,
            "explanation": self.explanation,
            "confidence": round(self.confidence, 4),
            "support_score": round(self.support_score, 4),
            "uncertainty_reason": self.uncertainty_reason,
            "latency_ms": round(self.latency_ms, 2),
            "tokens_used": self.tokens_used,
        }


class IndependentLLMJudge:
    """Executes evidence-grounded verification for ambiguous or conflicting claims."""

    def __init__(
        self,
        provider: str = settings.llm_provider,
        openai_api_key: str = settings.openai_api_key,
        openai_base_url: str = settings.openai_base_url,
        openai_model: str = settings.openai_model,
        ollama_base_url: str = settings.ollama_base_url,
        ollama_model: str = settings.ollama_model,
    ) -> None:
        self.provider = provider
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url.rstrip("/")
        self.openai_model = openai_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model

    def evaluate(
        self,
        claim_id: str,
        claim_text: str,
        source_entries: Dict[str, SourceVerificationEntry],
        conflict_result: ConflictAnalysisResult,
        trigger_reason: str,
    ) -> LLMJudgeResult:
        """Judge a claim against separated supporting and contradictory source evidence."""
        start_t = time.perf_counter()

        if self.provider == "openai" and self.openai_api_key:
            try:
                res = self._call_external_llm(claim_id, claim_text, source_entries, conflict_result, trigger_reason)
                res.latency_ms = (time.perf_counter() - start_t) * 1000.0
                return res
            except Exception as exc:
                logger.warning("OpenAI Judge call failed (%s); using local analytical judge.", exc)

        res = self._analytical_evidence_judge(
            claim_id=claim_id,
            claim_text=claim_text,
            source_entries=source_entries,
            conflict_result=conflict_result,
            trigger_reason=trigger_reason,
        )
        res.latency_ms = (time.perf_counter() - start_t) * 1000.0
        return res

    def _call_external_llm(
        self,
        claim_id: str,
        claim_text: str,
        source_entries: Dict[str, SourceVerificationEntry],
        conflict_result: ConflictAnalysisResult,
        trigger_reason: str,
    ) -> LLMJudgeResult:
        evidence_lines = []
        for src_id, entry in source_entries.items():
            evidence_lines.append(
                f"- Source [{src_id}] (Reliability={entry.source_reliability:.2f}, "
                f"NLI={entry.nli_result.label}, E={entry.nli_result.entailment_prob:.2f}, "
                f"K={entry.nli_result.contradiction_prob:.2f}, Doc={entry.nli_result.best_doc_id}): "
                f"{entry.nli_result.best_evidence_window}"
            )
        user_payload = (
            f"Claim ID: {claim_id}\nClaim: {claim_text}\n"
            f"Conflict Summary: {conflict_result.summary}\n"
            f"Evidence by Source:\n" + "\n".join(evidence_lines)
        )
        resp = requests.post(
            f"{self.openai_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.openai_model,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_payload},
                ],
            },
            timeout=20,
        )
        resp.raise_for_status()
        body = resp.json()
        parsed = json.loads(body["choices"][0]["message"]["content"])
        tokens = int(body.get("usage", {}).get("total_tokens", len(user_payload.split()) * 2))
        return LLMJudgeResult(
            claim_id=claim_id,
            invoked=True,
            trigger_reason=trigger_reason,
            verification_decision=str(parsed.get("verification_decision", "UNCERTAIN")),
            supporting_evidence=list(parsed.get("supporting_evidence", [])),
            contradictory_evidence=list(parsed.get("contradictory_evidence", [])),
            explanation=str(parsed.get("explanation", "")),
            confidence=float(parsed.get("confidence", 0.5)),
            uncertainty_reason=str(parsed.get("uncertainty_reason", "")),
            tokens_used=tokens,
        )

    def _analytical_evidence_judge(
        self,
        claim_id: str,
        claim_text: str,
        source_entries: Dict[str, SourceVerificationEntry],
        conflict_result: ConflictAnalysisResult,
        trigger_reason: str,
    ) -> LLMJudgeResult:
        """Evidence-strict analytical judge resolving cross-source conflicts using reliability-weighted specificity."""
        sup_ev: List[str] = []
        con_ev: List[str] = []
        weighted_sup = 0.0
        weighted_con = 0.0
        max_sup_rel = 0.0
        max_con_rel = 0.0

        for src_id, entry in source_entries.items():
            res = entry.nli_result
            rel = entry.source_reliability
            if res.entailment_prob >= 0.35 and res.best_evidence_window:
                sup_ev.append(f"[{src_id}:{res.best_doc_id}] {res.best_evidence_window}")
                weighted_sup += rel * res.entailment_prob
                max_sup_rel = max(max_sup_rel, rel * res.entailment_prob)
            if res.contradiction_prob >= 0.35 and res.best_evidence_window:
                con_ev.append(
                    f"[{src_id}:{res.best_doc_id}] {res.best_evidence_window} ({res.diagnostic_reason})"
                )
                weighted_con += rel * res.contradiction_prob
                max_con_rel = max(max_con_rel, rel * res.contradiction_prob)

        tokens_est = int((len(claim_text.split()) + sum(len(s.split()) for s in sup_ev + con_ev)) * 1.4) + 85

        if not sup_ev and not con_ev:
            return LLMJudgeResult(
                claim_id=claim_id,
                invoked=True,
                trigger_reason=trigger_reason,
                verification_decision="UNCERTAIN",
                supporting_evidence=[],
                contradictory_evidence=[],
                explanation="Neither supporting nor contradictory factual passages were found in the retrieved sources.",
                confidence=0.25,
                uncertainty_reason="Absence of corroborating or refuting retrieved evidence.",
                tokens_used=tokens_est,
            )

        diff = weighted_sup - weighted_con
        if weighted_con > weighted_sup + 0.08 or max_con_rel > max_sup_rel + 0.06:
            conf = min(0.95, 0.58 + 0.38 * min(1.0, abs(diff) + max_con_rel))
            return LLMJudgeResult(
                claim_id=claim_id,
                invoked=True,
                trigger_reason=trigger_reason,
                verification_decision="CONTRADICTED",
                supporting_evidence=sup_ev,
                contradictory_evidence=con_ev,
                explanation=(
                    f"High-reliability evidence directly contradicts the claim "
                    f"(weighted contradiction={weighted_con:.2f} vs support={weighted_sup:.2f})."
                ),
                confidence=conf,
                uncertainty_reason="Cross-source contradiction present" if sup_ev else "None",
                tokens_used=tokens_est,
            )

        if weighted_sup > weighted_con + 0.12 and max_sup_rel >= 0.45:
            conf = min(0.94, 0.55 + 0.38 * min(1.0, diff + max_sup_rel * 0.5))
            return LLMJudgeResult(
                claim_id=claim_id,
                invoked=True,
                trigger_reason=trigger_reason,
                verification_decision="SUPPORTED",
                supporting_evidence=sup_ev,
                contradictory_evidence=con_ev,
                explanation=(
                    f"Authoritative retrieved evidence supports the claim "
                    f"(weighted support={weighted_sup:.2f} vs contradiction={weighted_con:.2f})."
                ),
                confidence=conf,
                uncertainty_reason="Minor secondary source disagreement" if con_ev else "None",
                tokens_used=tokens_est,
            )

        return LLMJudgeResult(
            claim_id=claim_id,
            invoked=True,
            trigger_reason=trigger_reason,
            verification_decision="UNCERTAIN",
            supporting_evidence=sup_ev,
            contradictory_evidence=con_ev,
            explanation=(
                f"Evidence across sources is closely split or inconclusive "
                f"(support={weighted_sup:.2f}, contradiction={weighted_con:.2f})."
            ),
            confidence=0.45,
            uncertainty_reason="Unresolved conflict between comparable sources.",
            tokens_used=tokens_est,
        )
