"""E1: Original Repository Baseline Reproduction.

Faithfully reproduces the collective whole-answer verification behavior of the
reference repository (`LakkuAmulya-2/Enterprise-Advanced-RAG-Knowledge-Intelligence-System`),
which pools all retrieved context chunks together and evaluates overall lexical/semantic
overlap without atomic claim decomposition, without source-wise NLI polarity checking,
and without cross-source conflict detection.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from evaluation.datasets import EvaluationSample
from ingestion.embedding import STOPWORDS, EmbeddingGenerator


class OriginalBaselineEvaluator:
    """Reproduces the collective context-overlap faithfulness baseline (E1)."""

    def __init__(self) -> None:
        self.embedder = EmbeddingGenerator()

    def evaluate_sample(self, sample: EvaluationSample) -> Dict[str, Any]:
        """Evaluate entire generated answer collectively against pooled retrieved context."""
        start_t = time.perf_counter()

        pooled_texts: List[str] = []
        for chunks in sample.source_wise_chunks.values():
            for c in chunks:
                if c.get("text"):
                    pooled_texts.append(str(c["text"]))

        if not pooled_texts:
            elapsed = (time.perf_counter() - start_t) * 1000.0
            return {
                "sample_id": sample.sample_id,
                "pred_label": 0,
                "pred_prob_supported": 0.15,
                "latency_ms": elapsed,
                "llm_calls": 0,
                "tokens_used": 0,
                "evidence_coverage": 0.0,
                "citation_precision": 0.0,
                "source_agreement": 0.5,
                "detected_conflict": False,
            }

        combined_context = " ".join(pooled_texts)
        ans_clean = re.sub(r"\[[^\]]+\]", "", sample.generated_answer)

        # Collective cosine similarity + lexical token overlap (as used in standard similarity baselines)
        vecs = self.embedder.embed_documents([ans_clean, combined_context])
        cos_sim = float(vecs[0] @ vecs[1])

        ans_tokens = {
            w.lower().strip(".,?!:;")
            for w in ans_clean.split()
            if len(w) > 2 and w.lower() not in STOPWORDS
        }
        ctx_tokens = {
            w.lower().strip(".,?!:;")
            for w in combined_context.split()
            if len(w) > 2 and w.lower() not in STOPWORDS
        }
        lex_overlap = len(ans_tokens.intersection(ctx_tokens)) / max(1, len(ans_tokens))

        # Collective uncalibrated score (blind to negation, antonym clash, or cross-source conflict)
        collective_score = min(0.96, max(0.05, 0.45 * max(0.0, cos_sim) + 0.55 * lex_overlap))
        pred_label = 1 if collective_score >= 0.52 else 0
        elapsed = (time.perf_counter() - start_t) * 1000.0

        return {
            "sample_id": sample.sample_id,
            "pred_label": pred_label,
            "pred_prob_supported": round(collective_score, 4),
            "latency_ms": elapsed,
            "llm_calls": 0,
            "tokens_used": 0,
            "evidence_coverage": round(lex_overlap, 4),
            "citation_precision": 1.0 if "[" in sample.generated_answer else 0.5,
            "source_agreement": 0.5,
            "detected_conflict": False,
        }
