"""Multi-format document loader supporting PDF, TXT, DOCX, CSV, and Markdown with security validation."""
from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import docx
from pypdf import PdfReader

from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentIngestionError(Exception):
    """Raised when document validation or extraction fails."""


@dataclass
class PageContent:
    """Represents text extracted from a specific page or logical section."""

    page_number: int
    text: str
    section_title: Optional[str] = None


@dataclass
class LoadedDocument:
    """Structured representation of a loaded document with provenance metadata."""

    doc_id: str
    title: str
    filename: str
    doc_type: str
    source_id: str
    content_hash: str
    pages: List[PageContent]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


# Patterns that look like prompt injection inside untrusted documents
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"system\s*prompt\s*override", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
    re.compile(r"<\s*/?\s*system\s*>", re.IGNORECASE),
]


def sanitize_untrusted_text(text: str) -> tuple[str, bool]:
    """Neutralize prompt-injection directives in retrieved/uploaded document text."""
    flagged = False
    cleaned = text
    for pat in PROMPT_INJECTION_PATTERNS:
        if pat.search(cleaned):
            flagged = True
            cleaned = pat.sub("[REDACTED_INSTRUCTION_INJECTION]", cleaned)
    # Normalize null bytes and control chars
    cleaned = cleaned.replace("\x00", "")
    return cleaned, flagged


class DocumentLoader:
    """Loads PDF, TXT, DOCX, CSV, and Markdown files into LoadedDocument instances."""

    def __init__(self, max_size_mb: int = settings.max_upload_file_size_mb) -> None:
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.allowed_extensions = set(settings.allowed_extensions)

    def validate_file(self, filename: str, raw_bytes: bytes) -> str:
        """Validate file extension and size constraints."""
        if not filename or ".." in filename or "/" in filename or "\\" in filename:
            safe_name = Path(filename).name
        else:
            safe_name = filename
        ext = Path(safe_name).suffix.lower()
        if ext not in self.allowed_extensions:
            raise DocumentIngestionError(
                f"Unsupported file format '{ext}'. Allowed formats: {sorted(self.allowed_extensions)}"
            )
        if len(raw_bytes) == 0:
            raise DocumentIngestionError(f"File '{safe_name}' is empty (0 bytes).")
        if len(raw_bytes) > self.max_size_bytes:
            raise DocumentIngestionError(
                f"File '{safe_name}' exceeds maximum size of {self.max_size_bytes // (1024 * 1024)} MB."
            )
        return ext.lstrip(".")

    def load_from_path(
        self,
        file_path: str | Path,
        source_id: str,
        title: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> LoadedDocument:
        """Load a document from a local filesystem path."""
        path = Path(file_path)
        if not path.exists():
            raise DocumentIngestionError(f"File not found: {path}")
        raw_bytes = path.read_bytes()
        return self.load_from_bytes(
            raw_bytes=raw_bytes,
            filename=path.name,
            source_id=source_id,
            title=title or path.stem.replace("_", " ").title(),
            extra_metadata=extra_metadata,
        )

    def load_from_bytes(
        self,
        raw_bytes: bytes,
        filename: str,
        source_id: str,
        title: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> LoadedDocument:
        """Load a document from raw bytes with SHA-256 content hashing and metadata retention."""
        safe_filename = Path(filename).name
        doc_type = self.validate_file(safe_filename, raw_bytes)
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        doc_id = f"{source_id.upper()}-{content_hash[:10]}"

        if doc_type == "pdf":
            pages = self._parse_pdf(raw_bytes, safe_filename)
        elif doc_type == "docx":
            pages = self._parse_docx(raw_bytes, safe_filename)
        elif doc_type == "csv":
            pages = self._parse_csv(raw_bytes, safe_filename)
        elif doc_type == "md":
            pages = self._parse_markdown(raw_bytes)
        elif doc_type == "txt":
            pages = self._parse_txt(raw_bytes)
        else:
            raise DocumentIngestionError(f"Unhandled document type: {doc_type}")

        sanitized_pages: List[PageContent] = []
        injection_detected = False
        for page in pages:
            cleaned_text, flagged = sanitize_untrusted_text(page.text)
            if flagged:
                injection_detected = True
            if cleaned_text.strip():
                sanitized_pages.append(
                    PageContent(
                        page_number=page.page_number,
                        text=cleaned_text.strip(),
                        section_title=page.section_title,
                    )
                )

        if not sanitized_pages:
            raise DocumentIngestionError(
                f"No extractable text found in document '{safe_filename}'."
            )

        meta: Dict[str, Any] = {
            "original_reference": safe_filename,
            "file_size_bytes": len(raw_bytes),
            "page_count": len(sanitized_pages),
            "prompt_injection_sanitized": injection_detected,
        }
        if extra_metadata:
            meta.update(extra_metadata)

        return LoadedDocument(
            doc_id=doc_id,
            title=title or Path(safe_filename).stem.replace("_", " ").title(),
            filename=safe_filename,
            doc_type=doc_type,
            source_id=source_id,
            content_hash=content_hash,
            pages=sanitized_pages,
            metadata=meta,
        )

    def _parse_pdf(self, raw_bytes: bytes, filename: str) -> List[PageContent]:
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            pages: List[PageContent] = []
            for idx, page in enumerate(reader.pages, start=1):
                extracted = page.extract_text() or ""
                if extracted.strip():
                    pages.append(PageContent(page_number=idx, text=extracted.strip()))
            return pages
        except Exception as exc:
            raise DocumentIngestionError(f"Failed to parse PDF '{filename}': {exc}") from exc

    def _parse_docx(self, raw_bytes: bytes, filename: str) -> List[PageContent]:
        try:
            doc = docx.Document(io.BytesIO(raw_bytes))
            pages: List[PageContent] = []
            current_paras: List[str] = []
            current_heading: Optional[str] = None
            section_idx = 1

            for para in doc.paragraphs:
                txt = para.text.strip()
                if not txt:
                    continue
                style_name = (para.style.name or "").lower() if para.style else ""
                if "heading" in style_name and current_paras:
                    pages.append(
                        PageContent(
                            page_number=section_idx,
                            text="\n".join(current_paras),
                            section_title=current_heading,
                        )
                    )
                    section_idx += 1
                    current_paras = [txt]
                    current_heading = txt
                else:
                    if "heading" in style_name and not current_heading:
                        current_heading = txt
                    current_paras.append(txt)

            # Also extract any tables in DOCX
            for table in doc.tables:
                rows_text = []
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    rows_text.append(" | ".join(cells))
                if rows_text:
                    current_paras.append("\n".join(rows_text))

            if current_paras:
                pages.append(
                    PageContent(
                        page_number=section_idx,
                        text="\n".join(current_paras),
                        section_title=current_heading,
                    )
                )
            return pages
        except Exception as exc:
            raise DocumentIngestionError(f"Failed to parse DOCX '{filename}': {exc}") from exc

    def _parse_csv(self, raw_bytes: bytes, filename: str) -> List[PageContent]:
        try:
            decoded = raw_bytes.decode("utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(decoded))
            pages: List[PageContent] = []
            rows_per_page = 15
            buffer: List[str] = []
            page_num = 1

            if reader.fieldnames:
                for row_idx, row in enumerate(reader, start=1):
                    kv_pairs = [
                        f"{k.strip()}: {str(v).strip()}"
                        for k, v in row.items()
                        if k and v is not None and str(v).strip() != ""
                    ]
                    if kv_pairs:
                        buffer.append(f"Record {row_idx} -> " + "; ".join(kv_pairs) + ".")
                    if len(buffer) >= rows_per_page:
                        pages.append(
                            PageContent(
                                page_number=page_num,
                                text="\n".join(buffer),
                                section_title=f"Rows {row_idx - len(buffer) + 1}-{row_idx}",
                            )
                        )
                        page_num += 1
                        buffer = []
                if buffer:
                    pages.append(
                        PageContent(
                            page_number=page_num,
                            text="\n".join(buffer),
                            section_title=f"Final Rows (Page {page_num})",
                        )
                    )
            return pages
        except Exception as exc:
            raise DocumentIngestionError(f"Failed to parse CSV '{filename}': {exc}") from exc

    def _parse_markdown(self, raw_bytes: bytes) -> List[PageContent]:
        decoded = raw_bytes.decode("utf-8", errors="replace")
        sections = re.split(r"(?m)^(#{1,3}\s+.+)$", decoded)
        pages: List[PageContent] = []
        current_title: Optional[str] = None
        page_num = 1

        buffer = ""
        for part in sections:
            part_str = part.strip()
            if not part_str:
                continue
            if re.match(r"^#{1,3}\s+.+$", part_str):
                if buffer.strip():
                    pages.append(
                        PageContent(
                            page_number=page_num,
                            text=buffer.strip(),
                            section_title=current_title,
                        )
                    )
                    page_num += 1
                current_title = re.sub(r"^#{1,3}\s+", "", part_str).strip()
                buffer = f"{current_title}.\n"
            else:
                buffer += part_str + "\n"
        if buffer.strip():
            pages.append(
                PageContent(
                    page_number=page_num,
                    text=buffer.strip(),
                    section_title=current_title,
                )
            )
        return pages

    def _parse_txt(self, raw_bytes: bytes) -> List[PageContent]:
        decoded = raw_bytes.decode("utf-8", errors="replace")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", decoded) if p.strip()]
        if not paragraphs:
            return []
        pages: List[PageContent] = []
        current: List[str] = []
        word_count = 0
        page_num = 1
        for para in paragraphs:
            words = len(para.split())
            if word_count + words > 350 and current:
                pages.append(PageContent(page_number=page_num, text="\n\n".join(current)))
                page_num += 1
                current = [para]
                word_count = words
            else:
                current.append(para)
                word_count += words
        if current:
            pages.append(PageContent(page_number=page_num, text="\n\n".join(current)))
        return pages
