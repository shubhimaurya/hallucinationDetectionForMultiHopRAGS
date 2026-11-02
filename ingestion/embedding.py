"""Dense embedding generator supporting Sentence-Transformers and deterministic semantic hashing."""
from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Dict, List

import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "so", "than", "too", "very",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "shall", "should", "can", "could", "may", "might",
}

SYNONYM_CANONICAL_MAP: Dict[str, str] = {
    "lunar": "moon",
    "spacecraft": "probe",
    "orbiter": "probe",
    "lander": "probe",
    "launched": "launch",
    "liftoff": "launch",
    "landed": "landing",
    "touchdown": "landing",
    "contradicts": "refutes",
    "disagrees": "conflicts",
    "hallucination": "fabrication",
    "entailment": "support",
    "verified": "supported",
    "cardiovascular": "heart",
    "myocardial": "heart",
    "efficacy": "effectiveness",
    "mortality": "death",
    "revenue": "income",
    "quarterly": "quarter",
    "photovoltaic": "solar",
    "efficiency": "yield",
    "superconductor": "superconducting",
    "ambient": "room_temperature",
}


class EmbeddingGenerator:
    """Generates normalized dense embeddings for documents and queries."""

    def __init__(
        self,
        model_name: str = settings.embedding_model_name,
        dim: int = settings.embedding_dim,
        backend: str = settings.embedding_backend,
    ) -> None:
        self.model_name = model_name
        self.dim = dim
        self._st_model = None
        self.active_backend = "deterministic_semantic_projection"

        if backend in ("auto", "sentence_transformers"):
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._st_model = SentenceTransformer(model_name, local_files_only=True)
                self.dim = int(self._st_model.get_sentence_embedding_dimension())
                self.active_backend = f"sentence_transformers:{model_name}"
                logger.info("Loaded SentenceTransformer model %s (dim=%d)", model_name, self.dim)
            except Exception:
                self._st_model = None
                self.active_backend = "deterministic_semantic_projection"

    @staticmethod
    def _extract_features(text: str) -> List[tuple[str, float]]:
        """Extract weighted unigram, canonical synonym, bigram, and character 4-gram features."""
        cleaned = re.sub(r"[^a-zA-Z0-9%.$-]+", " ", text.lower())
        raw_tokens = [t.strip(".") for t in cleaned.split() if len(t.strip(".")) > 1]
        tokens = [t for t in raw_tokens if t not in STOPWORDS]
        if not tokens:
            tokens = raw_tokens

        features: List[tuple[str, float]] = []
        for idx, tok in enumerate(tokens):
            # Boost numbers, years, and proper-like entities
            weight = 2.2 if any(ch.isdigit() for ch in tok) else 1.5
            features.append((f"uni:{tok}", weight))
            canon = SYNONYM_CANONICAL_MAP.get(tok)
            if canon:
                features.append((f"syn:{canon}", 1.4))
            # Stem-like 5-char prefix for morphological alignment
            if len(tok) >= 5:
                features.append((f"pre:{tok[:5]}", 0.9))
            # Bigrams
            if idx + 1 < len(tokens):
                features.append((f"bi:{tok}_{tokens[idx + 1]}", 1.2))
            # Character 4-grams for subword robustness
            if len(tok) >= 4:
                for c_idx in range(len(tok) - 3):
                    features.append((f"c4:{tok[c_idx : c_idx + 4]}", 0.35))
        return features

    def _encode_deterministic(self, text: str) -> np.ndarray:
        """Compute L2-normalized signed random-projection embedding via SHA-256 feature hashing."""
        vec = np.zeros(self.dim, dtype=np.float32)
        features = self._extract_features(text)
        if not features:
            return vec

        tf_counts: Dict[str, float] = {}
        for feat, w in features:
            tf_counts[feat] = tf_counts.get(feat, 0.0) + w

        for feat, total_w in tf_counts.items():
            scaled_w = (1.0 + math.log1p(total_w))
            digest = hashlib.sha256(feat.encode("utf-8")).digest()
            # Project each feature into 4 coordinates for low-collision Johnson-Lindenstrauss geometry
            for slot in range(4):
                offset = slot * 4
                idx = int.from_bytes(digest[offset : offset + 2], "little") % self.dim
                sign = 1.0 if (digest[offset + 2] & 1) == 0 else -1.0
                vec[idx] += sign * scaled_w * 0.5

        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec /= norm
        return vec

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of document chunks into an (N, dim) float32 array."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._st_model is not None:
            arr = self._st_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            return np.asarray(arr, dtype=np.float32)
        rows = [self._encode_deterministic(t) for t in texts]
        return np.vstack(rows).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string into a (dim,) float32 vector."""
        if self._st_model is not None:
            arr = self._st_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
            return np.asarray(arr[0], dtype=np.float32)
        return self._encode_deterministic(query)
