"""Main Streamlit Dashboard and Single-Command Entry Point for ACVE-Hallucination Detector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from config.settings import settings
from dashboard.claim_analysis import render_claim_analysis_page
from dashboard.conflict_explorer import render_conflict_explorer_page
from dashboard.documents import render_documents_page
from dashboard.evaluation_dashboard import render_evaluation_dashboard_page
from dashboard.experiment_manager import render_experiment_manager_page
from dashboard.home import compute_home_summary, render_home_page
from dashboard.query import execute_multirag_query, render_query_page
from evaluation.ablation import AblationStudyRunner
from evaluation.experiments import ExperimentRunner
from generation.answer_generator import AnswerGenerator
from ingestion.document_processor import DocumentIngestionProcessor
from retrieval.hybrid_retriever import MultiRAGHybridRetriever
from verification.citation_validator import ACVEPipelineVerifier


def bootstrap_sample_documents(processor: DocumentIngestionProcessor) -> Dict[str, Any]:
    """Ensure sample documents are indexed across all 4 independent RAG collections on startup."""
    sample_dir = settings.sample_docs_dir
    mapping = [
        ("academic_lunar_and_materials_2023.md", "academic", "Lunar & Superconductor Audit (2023)"),
        ("academic_clinical_and_rag_2024.txt", "academic", "CARDIO-SHIELD & Solid-State Battery Study (2024)"),
        ("official_mission_and_governance_report.pdf", "pdf", "ISRO & Enterprise Governance Official PDF"),
        ("enterprise_engineering_sop.docx", "pdf", "Hyperion-7 & Security Engineering SOP"),
        ("web_techpulse_community_blog.md", "web", "TechPulse Community Blog & Rumor Roundup"),
        ("structured_verified_telemetry_registry.csv", "structured", "Verified Telemetry & Fact Registry CSV"),
    ]
    ingested_summary: Dict[str, Any] = {}
    for fname, src_id, title in mapping:
        fpath = sample_dir / fname
        if fpath.exists():
            res = processor.ingest_file(fpath, source_id=src_id, title=title, allow_overwrite=False)
            ingested_summary[fname] = res
    return ingested_summary


def initialize_system_state(storage_root: Path | None = None) -> Dict[str, Any]:
    """Initialize all core modules and bootstrap sample Multi-RAG collections."""
    processor = DocumentIngestionProcessor(storage_root=storage_root)
    bootstrap_sample_documents(processor)
    retriever = MultiRAGHybridRetriever(processor)
    generator = AnswerGenerator()
    verifier = ACVEPipelineVerifier()
    exp_runner = ExperimentRunner()
    abl_runner = AblationStudyRunner()

    state: Dict[str, Any] = {
        "processor": processor,
        "retriever": retriever,
        "generator": generator,
        "verifier": verifier,
        "experiment_runner": exp_runner,
        "ablation_runner": abl_runner,
        "query_history": [],
        "last_query_result": None,
        "latest_experiments": None,
        "latest_ablation": None,
    }
    # Run an initial demonstration query so all dashboard pages have rich state immediately
    execute_multirag_query(
        state,
        query="When did India launch Chandrayaan-3, where did it land, and is LK-99 a room-temperature superconductor?",
        selected_sources=["academic", "pdf", "web", "structured"],
        top_k_per_source=3,
    )
    return state


def run_streamlit_app() -> None:
    """Launch the 7-page Streamlit Enterprise Research Dashboard."""
    import streamlit as st

    st.set_page_config(
        page_title="ACVE Multi-RAG Hallucination Detector",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "acve_state" not in st.session_state:
        st.session_state["acve_state"] = initialize_system_state()

    state = st.session_state["acve_state"]

    st.sidebar.title("ACVE-Hallucination Detector")
    st.sidebar.markdown("**Multi-RAG Evidence Verification**")
    page = st.sidebar.radio(
        "Navigation",
        options=[
            "1. Home Dashboard",
            "2. Document Management",
            "3. Multi-RAG Query Interface",
            "4. Claim-Level Hallucination Analysis",
            "5. Source Conflict Explorer",
            "6. Evaluation Dashboard",
            "7. Experiment Management",
        ],
    )

    if page.startswith("1."):
        render_home_page(state)
    elif page.startswith("2."):
        render_documents_page(state)
    elif page.startswith("3."):
        render_query_page(state)
    elif page.startswith("4."):
        render_claim_analysis_page(state)
    elif page.startswith("5."):
        render_conflict_explorer_page(state)
    elif page.startswith("6."):
        render_evaluation_dashboard_page(state)
    elif page.startswith("7."):
        render_experiment_manager_page(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACVE-Hallucination Detector")
    parser.add_argument(
        "--cli-demo",
        action="store_true",
        help="Run a non-interactive end-to-end demonstration in the terminal and exit.",
    )
    args, _ = parser.parse_known_args()
    if args.cli_demo:
        app_state = initialize_system_state()
        home_kpis = compute_home_summary(app_state)
        print(json.dumps(home_kpis, indent=2))
    else:
        run_streamlit_app()
