"""End-to-end document ingestion manager coordinating loading, chunking, and multi-collection indexing."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.model_config import RAG_SOURCE_CONFIGS
from config.settings import settings
from ingestion.chunking import OverlappingChunker
from ingestion.document_loader import DocumentIngestionError, DocumentLoader
from ingestion.embedding import EmbeddingGenerator
from ingestion.indexing import SourceCollectionIndex

logger = logging.getLogger(__name__)


class DocumentIngestionProcessor:
    """Manages ingestion, duplicate detection, deletion, and re-indexing across all 4 RAG collections."""

    def __init__(
        self,
        storage_root: Optional[Path] = None,
        chunk_size_tokens: int = settings.chunk_size_tokens,
        chunk_overlap_tokens: int = settings.chunk_overlap_tokens,
    ) -> None:
        self.storage_root = storage_root or settings.indexes_dir
        self.loader = DocumentLoader()
        self.chunker = OverlappingChunker(
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )
        self.embedder = EmbeddingGenerator()
        self.collections: Dict[str, SourceCollectionIndex] = {
            source_id: SourceCollectionIndex(
                source_id=source_id,
                embedder=self.embedder,
                storage_dir=self.storage_root / source_id,
            )
            for source_id in RAG_SOURCE_CONFIGS
        }
        self.ingestion_log: List[Dict[str, Any]] = []

    def validate_source_id(self, source_id: str) -> str:
        norm = source_id.strip().lower()
        if norm not in self.collections:
            raise DocumentIngestionError(
                f"Unknown RAG collection '{source_id}'. Valid collections: {list(self.collections.keys())}"
            )
        return norm

    def ingest_file(
        self,
        file_path: str | Path,
        source_id: str,
        title: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        allow_overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Ingest a file from disk into the specified RAG source collection."""
        src = self.validate_source_id(source_id)
        try:
            doc = self.loader.load_from_path(
                file_path=file_path,
                source_id=src,
                title=title,
                extra_metadata=extra_metadata,
            )
            chunks = self.chunker.chunk_document(doc)
            result = self.collections[src].add_document(
                doc=doc, chunks=chunks, allow_overwrite=allow_overwrite
            )
            result["title"] = doc.title
            result["filename"] = doc.filename
            result["page_count"] = len(doc.pages)
            self.ingestion_log.append(result)
            return result
        except Exception as exc:
            err_entry = {
                "status": "error",
                "source_id": src,
                "filename": str(file_path),
                "error": str(exc),
            }
            self.ingestion_log.append(err_entry)
            raise

    def ingest_bytes(
        self,
        raw_bytes: bytes,
        filename: str,
        source_id: str,
        title: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        allow_overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Ingest raw uploaded bytes into the specified RAG source collection."""
        src = self.validate_source_id(source_id)
        try:
            doc = self.loader.load_from_bytes(
                raw_bytes=raw_bytes,
                filename=filename,
                source_id=src,
                title=title,
                extra_metadata=extra_metadata,
            )
            chunks = self.chunker.chunk_document(doc)
            result = self.collections[src].add_document(
                doc=doc, chunks=chunks, allow_overwrite=allow_overwrite
            )
            result["title"] = doc.title
            result["filename"] = doc.filename
            result["page_count"] = len(doc.pages)
            self.ingestion_log.append(result)
            return result
        except Exception as exc:
            err_entry = {
                "status": "error",
                "source_id": src,
                "filename": filename,
                "error": str(exc),
            }
            self.ingestion_log.append(err_entry)
            raise

    def delete_document(self, source_id: str, doc_id: str) -> bool:
        src = self.validate_source_id(source_id)
        return self.collections[src].delete_document(doc_id)

    def reindex_collection(self, source_id: str) -> Dict[str, Any]:
        src = self.validate_source_id(source_id)
        return self.collections[src].reindex_all()

    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}
        for src_id, idx in self.collections.items():
            cfg = RAG_SOURCE_CONFIGS[src_id]
            stats[src_id] = {
                "source_id": src_id,
                "display_name": cfg.display_name,
                "default_reliability": cfg.default_reliability,
                "document_count": len(idx.documents_meta),
                "chunk_count": len(idx.chunks),
                "documents": list(idx.documents_meta.values()),
            }
        return stats
