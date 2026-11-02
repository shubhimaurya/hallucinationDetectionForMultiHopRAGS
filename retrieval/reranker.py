"""Cross-Encoder reranking and shingle-based context deduplication."""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List

from config.settings import settings
from ingestion.embedding import STOPWORDS, SYNONYM_CANONICAL_MAP

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks hybrid retrieval candidates using a Cross-Encoder or calibrated lexical-semantic interaction scorer."""

    def __init__(
        self,
        model_name: str = settings.reranker_model_name,
        top_k: int = settings.rerank_top_k,
    ) -> None:
        self.model_name = model_name
        self.top_k = top_k
        self._ce_model = None
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._ce_model = CrossEncoder(model_name, local_files_only=True)
            logger.info("Loaded CrossEncoder model %s", model_name)
        except Exception:
            self._ce_model = None

    @staticmethod
    def _tokenize_content(text: str) -> List[str]:
        cleaned = re.sub(r"[^a-zA-Z0-9%.-]+", " ", text.lower())
        raw = [t.strip(".-") for t in cleaned.split() if len(t.strip(".-")) > 1]
        out: List[str] = []
        for t in raw:
            if t not in STOPWORDS:
                out.append(t)
                if t in SYNONYM_CANONICAL_MAP:
                    out.append(SYNONYM_CANONICAL_MAP[t])
        return out or raw

    def _score_pair_interaction(self, query: str, passage: str, title: str, rrf_score: float) -> float:
        """Compute fine-grained query-passage token coverage, exact phrase match, and numeric overlap."""
        q_tokens = self._tokenize_content(query)
        p_tokens = self._tokenize_content(f"{title} {passage}")
        if not q_tokens or not p_tokens:
            return float(rrf_score)

        q_set = set(q_tokens)
        p_set = set(p_tokens)
        overlap = q_set.intersection(p_set)
        coverage = len(overlap) / max(1, len(q_set))

        # Exact bigram overlap
        q_bigrams = {f"{q_tokens[i]}_{q_tokens[i+1]}" for i in range(len(q_tokens) - 1)}
        p_bigrams = {f"{p_tokens[i]}_{p_tokens[i+1]}" for i in range(len(p_tokens) - 1)}
        bigram_cov = len(q_bigrams.intersection(p_bigrams)) / max(1, len(q_bigrams)) if q_bigrams else 0.0

        # Numeric/year alignment bonus
        q_nums = {t for t in q_tokens if any(c.isdigit() for c in t)}
        p_nums = {t for t in p_tokens if any(c.isdigit() for c in t)}
        num_bonus = 0.15 if (q_nums and q_nums.issubset(p_nums)) else 0.0

        raw_logit = -1.5 + 3.2 * coverage + 1.8 * bigram_cov + num_bonus + (12.0 * rrf_score)
        return float(1.0 / (1.0 + math.exp(-raw_logit)))

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Score and sort candidate chunks by cross-encoder relevance."""
        if not candidates:
            return []
        limit = top_k if top_k is not None else self.top_k
        deduped = self.deduplicate_chunks(candidates)

        if self._ce_model is not None:
            pairs = [(query, f"{c.get('doc_title', '')}: {c.get('text', '')}") for c in deduped]
            raw_scores = self._ce_model.predict(pairs)
            for cand, sc in zip(deduped, raw_scores):
                prob = 1.0 / (1.0 + math.exp(-float(sc)))
                cand["rerank_score"] = float(prob)
                cand["retrieval_score"] = float(prob)
        else:
            for cand in deduped:
                sc = self._score_pair_interaction(
                    query=query,
                    passage=str(cand.get("text", "")),
                    title=str(cand.get("doc_title", "")),
                    rrf_score=float(cand.get("rrf_score", 0.01)),
                )
                cand["rerank_score"] = sc
                cand["retrieval_score"] = sc

        deduped.sort(key=lambda c: float(c.get("rerank_score", 0.0)), reverse=True)
        for rank, cand in enumerate(deduped[:limit], start=1):
            cand["rerank_rank"] = rank
        return deduped[:limit]

    @staticmethod
    def deduplicate_chunks(
        chunks: List[Dict[str, Any]],
        jaccard_threshold: float = 0.88,
    ) -> List[Dict[str, Any]]:
        """Deduplicate chunks using full-text 3-word shingle Jaccard similarity (fixes reference repo 200-char prefix bug)."""
        kept: List[Dict[str, Any]] = []
        kept_shingles: List[set[str]] = []

        for chunk in chunks:
            words = str(chunk.get("text", "")).lower().split()
            if len(words) < 3:
                shingles = set(words)
            else:
                shingles = {" ".join(words[i : i + 3]) for i in range(len(words) - 2)}

            is_dup = False
            for existing in kept_shingles:
                if not shingles or not existing:
                    continue
                inter = len(shingles.intersection(existing))
                union = len(shingles.union(existing))
                if union > 0 and (inter / union) >= jaccard_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(dict(chunk))
                kept_shingles.append(shingles)
        return kept
