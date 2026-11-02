"""Page 2: Document Management allowing upload, target collection selection, deletion, and re-indexing."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore

from config.model_config import RAG_SOURCE_CONFIGS


def render_documents_page(state: Dict[str, Any]) -> None:
    """Render Page 2: Document Management in Streamlit."""
    if st is None:
        return
    st.title("Multi-RAG Document Ingestion & Index Management")
    processor = state["processor"]

    with st.form("upload_doc_form"):
        st.subheader("Upload & Index Document")
        target_source = st.selectbox(
            "Target RAG Collection",
            options=list(RAG_SOURCE_CONFIGS.keys()),
            format_func=lambda k: RAG_SOURCE_CONFIGS[k].display_name,
        )
        custom_title = st.text_input("Optional Document Title")
        allow_overwrite = st.checkbox("Overwrite if duplicate SHA-256 hash exists", value=False)
        uploaded_file = st.file_uploader(
            "Select PDF, TXT, DOCX, CSV, or Markdown (.md) file",
            type=["pdf", "txt", "docx", "csv", "md"],
        )
        submitted = st.form_submit_button("Ingest & Index Document")
        if submitted and uploaded_file is not None:
            try:
                res = processor.ingest_bytes(
                    raw_bytes=uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                    source_id=target_source,
                    title=custom_title or None,
                    allow_overwrite=allow_overwrite,
                )
                st.success(f"Ingestion status: {res['status']} | Doc ID: {res['doc_id']} | Chunks: {res['chunks_indexed']}")
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

    st.subheader("Indexed Documents Across RAG Pipelines")
    stats = processor.get_collection_stats()
    all_docs = []
    for src_id, info in stats.items():
        for doc in info["documents"]:
            all_docs.append(
                {
                    "Doc ID": doc["doc_id"],
                    "Title": doc["title"],
                    "Filename": doc["filename"],
                    "Type": doc["doc_type"],
                    "RAG Pipeline": src_id,
                    "Pages": doc["page_count"],
                    "Chunks": doc["chunk_count"],
                    "Content Hash": doc["content_hash"][:12],
                }
            )
    if all_docs:
        st.dataframe(pd.DataFrame(all_docs), use_container_width=True)
    else:
        st.info("No documents currently indexed.")

    col_del, col_reidx = st.columns(2)
    with col_del:
        st.markdown("#### Delete Document")
        del_src = st.selectbox("Collection", options=list(RAG_SOURCE_CONFIGS.keys()), key="del_src")
        doc_ids = [d["doc_id"] for d in stats[del_src]["documents"]]
        del_doc_id = st.selectbox("Document ID", options=doc_ids if doc_ids else ["(none)"])
        if st.button("Delete Selected Document") and del_doc_id != "(none)":
            if processor.delete_document(del_src, del_doc_id):
                st.success(f"Deleted {del_doc_id} from {del_src}.")
    with col_reidx:
        st.markdown("#### Re-Index Collection")
        re_src = st.selectbox("Collection to Re-Index", options=list(RAG_SOURCE_CONFIGS.keys()), key="re_src")
        if st.button("Re-Index Selected Pipeline"):
            out = processor.reindex_collection(re_src)
            st.success(f"Re-indexed {re_src}: {out['chunks_count']} chunks updated.")
