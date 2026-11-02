"""Page 6: Evaluation Dashboard comparing E1 Baseline, E2 Claim-NLI, E3 Hybrid, and E4 Proposed ACVE."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore

from evaluation.comparison import ablation_to_dataframe, experiments_to_dataframe


def render_evaluation_dashboard_page(state: Dict[str, Any]) -> None:
    """Render Page 6: Evaluation Dashboard in Streamlit."""
    if st is None:
        return
    st.title("Benchmark Evaluation & Calibration Dashboard")

    exp_results = state.get("latest_experiments")
    abl_results = state.get("latest_ablation")

    if not exp_results:
        st.info("Running default benchmark evaluation (E1–E4 on ms_conflict_bench)...")
        exp_results = state["experiment_runner"].run_all_experiments("ms_conflict_bench")
        state["latest_experiments"] = exp_results

    df_exp = experiments_to_dataframe(exp_results)
    st.subheader("Baseline vs. Proposed ACVE Performance Comparison (E1 – E4)")
    st.dataframe(df_exp, use_container_width=True)

    e4_metrics = exp_results.get("E4_PROPOSED_ACVE", {}).get("metrics", {})
    if e4_metrics:
        st.subheader("Proposed ACVE (E4) Detailed Telemetry")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{e4_metrics['accuracy']:.2%}")
        m2.metric("F1-Score", f"{e4_metrics['f1_score']:.4f}")
        m3.metric("AUROC", f"{e4_metrics['auroc']:.4f}")
        m4.metric("ECE (Calibration Error)", f"{e4_metrics['expected_calibration_error']:.4f}")
        m5.metric("Avg LLM Calls / Query", f"{e4_metrics['avg_llm_calls_per_query']:.2f}")

    if abl_results:
        st.subheader("Component-Wise Ablation Study Results")
        st.dataframe(ablation_to_dataframe(abl_results), use_container_width=True)
