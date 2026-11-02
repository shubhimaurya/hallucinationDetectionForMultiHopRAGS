"""Application configuration and environment management for ACVE-Hallucination Detector."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _mask_secret(secret: Optional[str]) -> str:
    """Mask API keys for safe logging."""
    if not secret or len(secret) < 8:
        return "***UNSET***"
    return f"{secret[:3]}...{secret[-3:]}"


@dataclass
class Settings:
    """Centralized system settings with environment variable overrides and safe defaults."""

    # Project paths
    project_root: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    indexes_dir: Path = BASE_DIR / "data" / "indexes"
    experiments_dir: Path = BASE_DIR / "data" / "experiments"
    sample_docs_dir: Path = BASE_DIR / "data" / "sample_docs"

    # LLM Provider Configuration ("local_mock", "openai", "ollama")
    llm_provider: str = field(default_factory=lambda: os.getenv("ACVE_LLM_PROVIDER", "local_mock"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1:8b"))

    # Embedding & Reranking Configuration
    embedding_backend: str = field(default_factory=lambda: os.getenv("ACVE_EMBEDDING_BACKEND", "auto"))
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("ACVE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("ACVE_EMBEDDING_DIM", "384")))
    reranker_model_name: str = field(
        default_factory=lambda: os.getenv("ACVE_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    nli_model_name: str = field(
        default_factory=lambda: os.getenv(
            "ACVE_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        )
    )

    # Chunking & Ingestion Limits
    chunk_size_tokens: int = field(default_factory=lambda: int(os.getenv("ACVE_CHUNK_SIZE", "256")))
    chunk_overlap_tokens: int = field(default_factory=lambda: int(os.getenv("ACVE_CHUNK_OVERLAP", "40")))
    max_upload_file_size_mb: int = field(default_factory=lambda: int(os.getenv("ACVE_MAX_FILE_SIZE_MB", "25")))
    allowed_extensions: List[str] = field(
        default_factory=lambda: [".pdf", ".txt", ".docx", ".csv", ".md"]
    )

    # Retrieval Hyperparameters
    dense_top_k: int = 10
    sparse_top_k: int = 10
    rrf_k: int = 60
    dense_weight: float = 0.65
    sparse_weight: float = 0.35
    rerank_top_k: int = 5
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Active RAG Collections
    default_rag_sources: List[str] = field(
        default_factory=lambda: ["academic", "pdf", "web", "structured"]
    )

    def ensure_directories(self) -> None:
        """Create required data directories if they do not exist."""
        for directory in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.indexes_dir,
            self.experiments_dir,
            self.sample_docs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def safe_summary(self) -> Dict[str, object]:
        """Return configuration summary with sensitive keys masked."""
        return {
            "llm_provider": self.llm_provider,
            "openai_model": self.openai_model,
            "openai_api_key": _mask_secret(self.openai_api_key),
            "ollama_model": self.ollama_model,
            "embedding_model_name": self.embedding_model_name,
            "embedding_dim": self.embedding_dim,
            "reranker_model_name": self.reranker_model_name,
            "nli_model_name": self.nli_model_name,
            "chunk_size_tokens": self.chunk_size_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
            "max_upload_file_size_mb": self.max_upload_file_size_mb,
            "default_rag_sources": self.default_rag_sources,
        }


settings = Settings()
settings.ensure_directories()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
