"""Page 3: Multi-RAG Query Interface supporting pipeline toggles, answer generation, and end-to-end verification."""
from __future__ import annotations

import time
from typing import Any, Dict, List

import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore

from config.model_config import RAG_SOURCE_CONFIGS


def execute_multirag_query(
    state: Dict[str, Any],
    query: str,
    selected_sources: List[str],
    top_k_per_source: int = 3,
) -> Dict[str, Any]:
    """Run Multi-RAG retrieval, LLM answer generation, and full ACVE claim verification."""
    t0 = time.perf_counter()
    retriever = state["retriever"]
    generator = state["generator"]
    verifier = state["verifier"]

    retrieval_out = retriever.retrieve_all(
        query=query,
        enabled_sources=selected_sources,
        top_k_per_source=top_k_per_source,
    )
    gen_out = generator.generate(
        query=query,
        retrieved_chunks=retrieval_out["pooled_results"],
    )
    ver_out = verifier.verify_answer(
        answer_text=gen_out.answer_text,
        source_wise_chunks=retrieval_out["source_wise_results"],
    )
    total_ms = (time.perf_counter() - t0) * 1000.0

    record = {
        "query": query,
        "selected_sources": selected_sources,
        "retrieval": retrieval_out,
        "generation": gen_out.to_dict(),
        "verification": ver_out,
        "total_latency_ms": round(total_ms, 2),
    }
    state.setdefault("query_history", []).append(record)
    state["last_query_result"] = record
    return record


def render_query_page(state: Dict[str, Any]) -> None:
    """Render Page 3: Multi-RAG Query Interface in Streamlit."""
    if st is None:
        return
    st.title("Multi-RAG Query & Evidence Inspection Interface")

    default_q = "When did India launch Chandrayaan-3, where did it land, and is LK-99 a room-temperature superconductor?"
    query = st.text_area("Enter Natural-Language Research Query", value=default_q, height=90)

    col1, col2 = st.columns([3, 1])
    with col1:
        enable_all = st.checkbox("Enable All 4 RAG Pipelines", value=True)
        if enable_all:
            selected_sources = list(RAG_SOURCE_CONFIGS.keys())
        else:
            selected_sources = st.multiselect(
                "Select Active RAG Sources",
                options=list(RAG_SOURCE_CONFIGS.keys()),
                default=list(RAG_SOURCE_CONFIGS.keys()),
            )
    with col2:
        top_k = st.slider("Top-K per Source", min_value=1, max_value=6, value=3)

    if st.button("Retrieve, Generate & Verify Answer", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
        elif not selected_sources:
            st.warning("Please select at least one RAG source pipeline.")
        else:
            execute_multirag_query(state, query.strip(), selected_sources, top_k_per_source=top_k)

    last_res = state.get("last_query_result")
    if last_res:
        st.subheader("Generated Answer & Citation-Grounded Verification Output")
        st.markdown(f"**Raw Generated Answer:**\n\n{last_res['generation']['answer_text']}")
        st.markdown(
            f"**ACVE Citation-Grounded Output:**\n\n{last_res['verification']['citation_grounded_response']}"
        )

        st.subheader("Source-Wise Retrieved Evidence Chunks")
        for src_id, chunks in last_res["retrieval"]["source_wise_results"].items():
            with st.expander(f"{RAG_SOURCE_CONFIGS[src_id].display_name} ({len(chunks)} chunks retrieved)", expanded=True):
                if chunks:
                    rows = [
                        {
                            "Doc ID": c["doc_id"],
                            "Title": c["doc_title"],
                            "Type": c["doc_type"],
                            "Page": c["page_number"],
                            "Rerank Score": round(float(c.get("retrieval_score", 0.0)), 4),
                            "Snippet": c["text"][:220] + ("..." if len(c["text"]) > 220 else ""),
                        }
                        for c in chunks
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.write("No matching chunks in this pipeline.")
