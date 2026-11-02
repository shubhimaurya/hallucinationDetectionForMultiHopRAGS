"""Configurable LLM Answer Generator supporting OpenAI, Ollama, and deterministic extractive synthesis."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from config.settings import settings
from generation.prompt_templates import (
    ANSWER_GENERATION_SYSTEM_PROMPT,
    build_answer_user_prompt,
)
from ingestion.chunking import split_sentences_safe
from ingestion.embedding import STOPWORDS

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    """Structured output from the answer generation module."""

    query: str
    answer_text: str
    cited_doc_ids: List[str]
    insufficient_evidence: bool
    provider_used: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer_text": self.answer_text,
            "cited_doc_ids": self.cited_doc_ids,
            "insufficient_evidence": self.insufficient_evidence,
            "provider_used": self.provider_used,
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class AnswerGenerator:
    """Generates citation-grounded answers using OpenAI-compatible APIs, Ollama, or local extractive synthesis."""

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

    @staticmethod
    def extract_citations(text: str) -> List[str]:
        """Extract all bracketed document citations such as [ACADEMIC-12345] or [DOC-01]."""
        matches = re.findall(r"\[([A-Za-z0-9_-]{3,40})\]", text)
        seen: List[str] = []
        for m in matches:
            if m not in seen:
                seen.append(m)
        return seen

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        min_relevance_threshold: float = 0.22,
    ) -> GeneratedAnswer:
        """Generate a grounded, cited response from retrieved evidence chunks."""
        start_t = time.perf_counter()

        # Filter chunks with substantive relevance
        valid_chunks = [
            c for c in retrieved_chunks if float(c.get("retrieval_score", 0.0)) >= min_relevance_threshold
        ]
        if not valid_chunks:
            elapsed = (time.perf_counter() - start_t) * 1000.0
            return GeneratedAnswer(
                query=query,
                answer_text="The retrieved evidence is insufficient to answer this query based on the active knowledge collections.",
                cited_doc_ids=[],
                insufficient_evidence=True,
                provider_used="abstention_guard",
                latency_ms=elapsed,
                prompt_tokens=len(query.split()),
                completion_tokens=16,
                retrieved_chunks=retrieved_chunks,
            )

        user_prompt = build_answer_user_prompt(query, valid_chunks[:8])
        prompt_tokens = int(len(user_prompt.split()) * 1.3)

        if self.provider == "openai" and self.openai_api_key:
            try:
                ans_text = self._call_openai(user_prompt)
                provider_used = f"openai:{self.openai_model}"
            except Exception as exc:
                logger.warning("OpenAI generation failed (%s); falling back to local extractive synthesizer.", exc)
                ans_text = self._synthesize_extractive(query, valid_chunks)
                provider_used = "local_extractive_fallback"
        elif self.provider == "ollama":
            try:
                ans_text = self._call_ollama(user_prompt)
                provider_used = f"ollama:{self.ollama_model}"
            except Exception as exc:
                logger.warning("Ollama generation failed (%s); falling back to local extractive synthesizer.", exc)
                ans_text = self._synthesize_extractive(query, valid_chunks)
                provider_used = "local_extractive_fallback"
        else:
            ans_text = self._synthesize_extractive(query, valid_chunks)
            provider_used = "local_extractive_synthesizer"

        elapsed = (time.perf_counter() - start_t) * 1000.0
        citations = self.extract_citations(ans_text)
        insufficient = "insufficient to answer" in ans_text.lower()
        comp_tokens = int(len(ans_text.split()) * 1.3)

        return GeneratedAnswer(
            query=query,
            answer_text=ans_text,
            cited_doc_ids=citations,
            insufficient_evidence=insufficient,
            provider_used=provider_used,
            latency_ms=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=comp_tokens,
            retrieved_chunks=valid_chunks,
        )

    def _call_openai(self, user_prompt: str) -> str:
        resp = requests.post(
            f"{self.openai_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.openai_model,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": ANSWER_GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data["choices"][0]["message"]["content"]).strip()

    def _call_ollama(self, user_prompt: str) -> str:
        resp = requests.post(
            f"{self.ollama_base_url}/api/chat",
            json={
                "model": self.ollama_model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": ANSWER_GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data["message"]["content"]).strip()

    def _synthesize_extractive(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Synthesize a concise, multi-source cited answer directly from top evidence sentences."""
        q_words = {
            w.strip(".,?!:;").lower()
            for w in query.split()
            if len(w.strip(".,?!:;")) > 2 and w.lower() not in STOPWORDS
        }

        scored_sentences: List[tuple[float, str, str, str]] = []
        seen_norm: set[str] = set()

        for chunk in chunks[:6]:
            doc_id = str(chunk.get("doc_id", "DOC-UNKNOWN"))
            src_id = str(chunk.get("source_id", "unknown"))
            chunk_score = float(chunk.get("retrieval_score", 0.5))
            for sent in split_sentences_safe(str(chunk.get("text", ""))):
                norm_s = re.sub(r"\s+", " ", sent.lower().strip())
                if len(norm_s) < 18 or norm_s in seen_norm:
                    continue
                s_words = {w.strip(".,?!:;").lower() for w in sent.split()}
                overlap = len(q_words.intersection(s_words)) / max(1, len(q_words))
                if overlap == 0.0 and len(scored_sentences) >= 2:
                    continue
                total_sc = 0.65 * overlap + 0.35 * chunk_score
                scored_sentences.append((total_sc, sent.rstrip("."), doc_id, src_id))
                seen_norm.add(norm_s)

        if not scored_sentences:
            return "The retrieved evidence is insufficient to answer this query."

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        selected = scored_sentences[:3]
        formatted_clauses = [f"{sent}. [{doc_id}]" for _, sent, doc_id, _ in selected]
        return " ".join(formatted_clauses)
