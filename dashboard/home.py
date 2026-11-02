"""Page 1: Home Dashboard displaying system-wide Multi-RAG and verification telemetry."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore


def compute_home_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compute top-level KPI metrics for the Home Dashboard."""
    processor = state["processor"]
    col_stats = processor.get_collection_stats()
    total_docs = sum(int(v["document_count"]) for v in col_stats.values())
    total_chunks = sum(int(v["chunk_count"]) for v in col_stats.values())
    active_pipelines = len(state["retriever"].get_active_pipelines())

    history = state.get("query_history", [])
    total_queries = len(history)
    claims_verified = sum(int(h.get("verification", {}).get("total_claims", 0)) for h in history)
    hallucinations_detected = sum(
        int(h.get("verification", {}).get("hallucinated_claims", 0)) for h in history
    )
    confidences = [
        float(h.get("verification", {}).get("average_confidence", 0.0))
        for h in history
        if h.get("verification", {}).get("total_claims", 0) > 0
    ]
    latencies = [float(h.get("total_latency_ms", 0.0)) for h in history]

    return {
        "total_indexed_documents": total_docs,
        "total_indexed_chunks": total_chunks,
        "active_rag_pipelines": active_pipelines,
        "total_queries_processed": total_queries,
        "claims_verified": claims_verified,
        "hallucinations_detected": hallucinations_detected,
        "average_detection_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "average_response_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "collection_breakdown": [
            {
                "Source ID": k,
                "Pipeline Name": v["display_name"],
                "Reliability Prior": v["default_reliability"],
                "Indexed Documents": v["document_count"],
                "Indexed Chunks": v["chunk_count"],
            }
            for k, v in col_stats.items()
        ],
    }


def render_home_page(state: Dict[str, Any]) -> None:
    """Render Page 1: Home Dashboard in Streamlit."""
    if st is None:
        return
    st.title("ACVE-Hallucination Detector — Research Command Center")
    st.caption(
        "Adaptive Cross-Source Evidence Verification for Claim-Level Hallucination Detection in Multi-RAG Systems"
    )

    summary = compute_home_summary(state)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Indexed Documents", summary["total_indexed_documents"])
    c2.metric("Active RAG Pipelines", f"{summary['active_rag_pipelines']} / 4")
    c3.metric("Total Queries Processed", summary["total_queries_processed"])
    c4.metric("Atomic Claims Verified", summary["claims_verified"])

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Hallucinations Detected", summary["hallucinations_detected"])
    c6.metric("Avg Detection Confidence", f"{summary['average_detection_confidence']:.2%}")
    c7.metric("Avg Response Latency", f"{summary['average_response_latency_ms']:.1f} ms")
    c8.metric("Total Indexed Chunks", summary["total_indexed_chunks"])

    st.subheader("Independent Multi-RAG Collection Status")
    st.dataframe(pd.DataFrame(summary["collection_breakdown"]), use_container_width=True)
