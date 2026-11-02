"""Script to ingest all sample documents and build isolated indexes for RAG 1..4."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import bootstrap_sample_documents
from ingestion.document_processor import DocumentIngestionProcessor


def main() -> None:
    processor = DocumentIngestionProcessor()
    summary = bootstrap_sample_documents(processor)
    stats = processor.get_collection_stats()
    print("Built Multi-RAG indexes successfully:")
    print(json.dumps({"ingested": summary, "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
