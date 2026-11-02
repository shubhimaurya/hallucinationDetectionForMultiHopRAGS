"""Persistent vector + BM25 index store supporting isolated per-source collections, duplicate detection, deletion, and re-indexing."""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config.settings import settings
from ingestion.chunking import DocumentChunk
from ingestion.document_loader import LoadedDocument
from ingestion.embedding import STOPWORDS, EmbeddingGenerator

logger = logging.getLogger(__name__)


class SourceCollectionIndex:
    """Isolated vector + Okapi BM25 index for a single RAG collection (fixes reference repo BM25 overwrite bug)."""

    def __init__(
        self,
        source_id: str,
        embedder: Optional[EmbeddingGenerator] = None,
        storage_dir: Optional[Path] = None,
        k1: float = settings.bm25_k1,
        b: float = settings.bm25_b,
    ) -> None:
        self.source_id = source_id
        self.embedder = embedder or EmbeddingGenerator()
        self.storage_dir = storage_dir or (settings.indexes_dir / source_id)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.k1 = k1
        self.b = b

        self.documents_meta: Dict[str, Dict[str, Any]] = {}
        self.content_hashes: Dict[str, str] = {}  # sha256 -> doc_id
        self.chunks: List[DocumentChunk] = []
        self.embeddings: np.ndarray = np.zeros((0, self.embedder.dim), dtype=np.float32)

        # BM25 structures
        self._tf: List[Counter[str]] = []
        self._doc_lens: List[int] = []
        self._df: Counter[str] = Counter()
        self._avg_doc_len: float = 0.0

        self.load()

    @staticmethod
    def tokenize_bm25(text: str) -> List[str]:
        """Punctuation-stripping tokenizer for BM25 (fixes reference repo punctuation bug)."""
        cleaned = re.sub(r"[^a-zA-Z0-9%.-]+", " ", text.lower())
        tokens = [t.strip(".-") for t in cleaned.split() if len(t.strip(".-")) > 1]
        filtered = [t for t in tokens if t not in STOPWORDS]
        return filtered if filtered else tokens

    def _rebuild_bm25_stats(self) -> None:
        """Recompute BM25 term frequencies and document frequencies across all chunks in this collection."""
        self._tf = []
        self._doc_lens = []
        self._df = Counter()
        for chunk in self.chunks:
            # Include title and text for lexical recall
            tokens = self.tokenize_bm25(f"{chunk.doc_title} {chunk.text}")
            counts = Counter(tokens)
            self._tf.append(counts)
            self._doc_lens.append(len(tokens))
            for term in counts.keys():
                self._df[term] += 1
        total_len = sum(self._doc_lens)
        self._avg_doc_len = (total_len / len(self._doc_lens)) if self._doc_lens else 0.0

    def has_duplicate(self, content_hash: str) -> Optional[str]:
        """Return existing doc_id if a document with the same SHA-256 hash is already indexed."""
        return self.content_hashes.get(content_hash)

    def add_document(
        self,
        doc: LoadedDocument,
        chunks: List[DocumentChunk],
        allow_overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Index a document and its chunks into this isolated collection."""
        existing_doc_id = self.has_duplicate(doc.content_hash)
        if existing_doc_id and not allow_overwrite:
            return {
                "status": "duplicate_skipped",
                "doc_id": existing_doc_id,
                "source_id": self.source_id,
                "chunks_indexed": 0,
            }
        if existing_doc_id and allow_overwrite:
            self.delete_document(existing_doc_id)
        elif doc.doc_id in self.documents_meta:
            self.delete_document(doc.doc_id)

        if not chunks:
            return {
                "status": "empty_chunks",
                "doc_id": doc.doc_id,
                "source_id": self.source_id,
                "chunks_indexed": 0,
            }

        new_vecs = self.embedder.embed_documents([f"{c.doc_title}: {c.text}" for c in chunks])
        if self.embeddings.shape[0] == 0:
            self.embeddings = new_vecs
        else:
            self.embeddings = np.vstack([self.embeddings, new_vecs]).astype(np.float32)

        self.chunks.extend(chunks)
        self.documents_meta[doc.doc_id] = {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "filename": doc.filename,
            "doc_type": doc.doc_type,
            "source_id": doc.source_id,
            "content_hash": doc.content_hash,
            "chunk_count": len(chunks),
            "page_count": len(doc.pages),
            "created_at": doc.created_at,
            "metadata": doc.metadata,
        }
        self.content_hashes[doc.content_hash] = doc.doc_id
        self._rebuild_bm25_stats()
        self.save()
        return {
            "status": "indexed",
            "doc_id": doc.doc_id,
            "source_id": self.source_id,
            "chunks_indexed": len(chunks),
        }

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks/embeddings from this collection."""
        if doc_id not in self.documents_meta:
            return False
        doc_meta = self.documents_meta.pop(doc_id)
        c_hash = doc_meta.get("content_hash")
        if c_hash and c_hash in self.content_hashes:
            del self.content_hashes[c_hash]

        keep_indices = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        self.chunks = [self.chunks[i] for i in keep_indices]
        if keep_indices and self.embeddings.shape[0] > 0:
            self.embeddings = self.embeddings[keep_indices]
        else:
            self.embeddings = np.zeros((0, self.embedder.dim), dtype=np.float32)

        self._rebuild_bm25_stats()
        self.save()
        return True

    def reindex_all(self) -> Dict[str, Any]:
        """Recompute all dense embeddings and BM25 inverted statistics for this collection."""
        if self.chunks:
            self.embeddings = self.embedder.embed_documents(
                [f"{c.doc_title}: {c.text}" for c in self.chunks]
            )
        else:
            self.embeddings = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._rebuild_bm25_stats()
        self.save()
        return {
            "source_id": self.source_id,
            "documents_count": len(self.documents_meta),
            "chunks_count": len(self.chunks),
            "status": "reindexed",
        }

    def search_dense(self, query: str, top_k: int = settings.dense_top_k) -> List[Dict[str, Any]]:
        """Perform cosine similarity search over normalized vectors in this collection."""
        if not self.chunks or self.embeddings.shape[0] == 0:
            return []
        q_vec = self.embedder.embed_query(query)
        scores = np.dot(self.embeddings, q_vec)
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(ranked_idx, start=1):
            chunk = self.chunks[int(idx)]
            item = chunk.to_dict()
            item["dense_score"] = float(scores[int(idx)])
            item["dense_rank"] = rank
            item["retrieval_score"] = float(scores[int(idx)])
            results.append(item)
        return results

    def search_sparse(self, query: str, top_k: int = settings.sparse_top_k) -> List[Dict[str, Any]]:
        """Perform Okapi BM25 lexical search over this collection."""
        if not self.chunks or self._avg_doc_len <= 0.0:
            return []
        q_tokens = self.tokenize_bm25(query)
        if not q_tokens:
            return []

        n_docs = len(self.chunks)
        scores: List[tuple[int, float]] = []
        for idx, tf_counter in enumerate(self._tf):
            doc_len = self._doc_lens[idx]
            score = 0.0
            for term in q_tokens:
                tf = tf_counter.get(term, 0)
                if tf == 0:
                    continue
                df = self._df.get(term, 0)
                idf = math.log(((n_docs - df + 0.5) / (df + 0.5)) + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self._avg_doc_len))
                score += idf * ((tf * (self.k1 + 1.0)) / denom)
            if score > 0.0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results: List[Dict[str, Any]] = []
        for rank, (idx, sc) in enumerate(scores[:top_k], start=1):
            chunk = self.chunks[idx]
            item = chunk.to_dict()
            item["sparse_score"] = float(sc)
            item["sparse_rank"] = rank
            item["retrieval_score"] = float(sc)
            results.append(item)
        return results

    def save(self) -> None:
        """Persist chunks, document metadata, and embeddings to disk."""
        meta_path = self.storage_dir / "index_meta.json"
        vec_path = self.storage_dir / "embeddings.npy"
        payload = {
            "source_id": self.source_id,
            "documents_meta": self.documents_meta,
            "content_hashes": self.content_hashes,
            "chunks": [c.to_dict() for c in self.chunks],
        }
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        np.save(vec_path, self.embeddings)

    def load(self) -> bool:
        """Load persisted index state from disk if available."""
        meta_path = self.storage_dir / "index_meta.json"
        vec_path = self.storage_dir / "embeddings.npy"
        if not meta_path.exists():
            return False
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            self.documents_meta = dict(payload.get("documents_meta", {}))
            self.content_hashes = dict(payload.get("content_hashes", {}))
            self.chunks = [DocumentChunk.from_dict(c) for c in payload.get("chunks", [])]
            if vec_path.exists():
                loaded_vecs = np.load(vec_path)
                if loaded_vecs.shape[0] == len(self.chunks):
                    self.embeddings = loaded_vecs.astype(np.float32)
                else:
                    self.embeddings = self.embedder.embed_documents(
                        [f"{c.doc_title}: {c.text}" for c in self.chunks]
                    )
            self._rebuild_bm25_stats()
            return True
        except Exception as exc:
            logger.warning("Failed to load index for %s: %s", self.source_id, exc)
            return False
