"""Abbreviation-safe overlapping text chunker preserving page numbers and document metadata."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.settings import settings
from ingestion.document_loader import LoadedDocument

ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "fig", "eq", "tab",
    "no", "vol", "pp", "et", "al", "inc", "ltd", "corp", "vs", "etc",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "u.s", "u.k", "e.g", "i.e", "cf",
}


def split_sentences_safe(text: str) -> List[str]:
    """Split text into sentences without breaking on abbreviations or decimal numbers."""
    if not text or not text.strip():
        return []
    normalized = re.sub(r"\s+", " ", text.strip())
    # Protect common abbreviations and initials
    tokens = normalized.split(" ")
    rebuilt: List[str] = []
    sentences: List[str] = []

    for idx, tok in enumerate(tokens):
        rebuilt.append(tok)
        if tok.endswith((".", "!", "?")):
            stripped = tok.rstrip(".!?").lower()
            # Check if abbreviation or single initial (e.g., "A.") or number ("Fig. 1.")
            is_abbrev = stripped in ABBREVIATIONS or (len(stripped) == 1 and stripped.isalpha())
            has_internal_dots = "." in stripped and len(stripped) <= 5
            next_is_lower = (
                idx + 1 < len(tokens)
                and len(tokens[idx + 1]) > 0
                and tokens[idx + 1][0].islower()
            )
            if not is_abbrev and not has_internal_dots and not next_is_lower:
                sentences.append(" ".join(rebuilt).strip())
                rebuilt = []
    if rebuilt:
        sentences.append(" ".join(rebuilt).strip())
    return [s for s in sentences if s]


@dataclass
class DocumentChunk:
    """Represents an indexed text chunk with full provenance metadata."""

    chunk_id: str
    source_id: str
    doc_id: str
    doc_title: str
    doc_type: str
    text: str
    chunk_index: int
    page_number: Optional[int]
    original_reference: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "doc_type": self.doc_type,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "original_reference": self.original_reference,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(
            chunk_id=str(data["chunk_id"]),
            source_id=str(data["source_id"]),
            doc_id=str(data["doc_id"]),
            doc_title=str(data.get("doc_title", data.get("title", "Untitled"))),
            doc_type=str(data.get("doc_type", "txt")),
            text=str(data.get("text", data.get("content", ""))),
            chunk_index=int(data.get("chunk_index", 0)),
            page_number=data.get("page_number"),
            original_reference=str(data.get("original_reference", data.get("doc_id", ""))),
            token_count=int(data.get("token_count", len(str(data.get("text", "")).split()))),
            metadata=dict(data.get("metadata", {})),
        )


class OverlappingChunker:
    """Splits a LoadedDocument into configurable overlapping chunks while preserving page references."""

    def __init__(
        self,
        chunk_size_tokens: int = settings.chunk_size_tokens,
        chunk_overlap_tokens: int = settings.chunk_overlap_tokens,
    ) -> None:
        if chunk_size_tokens <= 10:
            raise ValueError("chunk_size_tokens must be > 10")
        if chunk_overlap_tokens < 0 or chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be in [0, chunk_size_tokens)")
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate subword token count accurately (~1.3 tokens per whitespace word plus punctuation)."""
        words = text.split()
        return max(1, int(round(len(words) * 1.3))) if words else 0

    def chunk_document(self, doc: LoadedDocument) -> List[DocumentChunk]:
        """Split a LoadedDocument into DocumentChunk objects with sentence boundary preservation."""
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        for page in doc.pages:
            sentences = split_sentences_safe(page.text)
            if not sentences:
                continue

            current_sentences: List[str] = []
            current_tokens = 0

            for sentence in sentences:
                sent_tokens = self.estimate_tokens(sentence)
                if current_tokens + sent_tokens > self.chunk_size_tokens and current_sentences:
                    chunk_text = " ".join(current_sentences).strip()
                    chunks.append(
                        self._make_chunk(doc, chunk_text, chunk_idx, page.page_number, page.section_title)
                    )
                    chunk_idx += 1

                    # Carry over overlap sentences
                    overlap_sents: List[str] = []
                    overlap_toks = 0
                    for prev_sent in reversed(current_sentences):
                        pt = self.estimate_tokens(prev_sent)
                        if overlap_toks + pt <= self.chunk_overlap_tokens or not overlap_sents:
                            overlap_sents.insert(0, prev_sent)
                            overlap_toks += pt
                        else:
                            break
                    # Avoid infinite carryover if a single sentence exceeded chunk_size_tokens
                    if overlap_sents == current_sentences:
                        overlap_sents = []
                        overlap_toks = 0
                    current_sentences = overlap_sents + [sentence]
                    current_tokens = overlap_toks + sent_tokens
                else:
                    current_sentences.append(sentence)
                    current_tokens += sent_tokens

            if current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                chunks.append(
                    self._make_chunk(doc, chunk_text, chunk_idx, page.page_number, page.section_title)
                )
                chunk_idx += 1

        return chunks

    def _make_chunk(
        self,
        doc: LoadedDocument,
        text: str,
        chunk_idx: int,
        page_number: Optional[int],
        section_title: Optional[str],
    ) -> DocumentChunk:
        chunk_meta = dict(doc.metadata)
        if section_title:
            chunk_meta["section_title"] = section_title
        return DocumentChunk(
            chunk_id=f"{doc.doc_id}-CHK-{chunk_idx:03d}",
            source_id=doc.source_id,
            doc_id=doc.doc_id,
            doc_title=doc.title,
            doc_type=doc.doc_type,
            text=text,
            chunk_index=chunk_idx,
            page_number=page_number,
            original_reference=str(doc.metadata.get("original_reference", doc.filename)),
            token_count=self.estimate_tokens(text),
            metadata=chunk_meta,
        )
