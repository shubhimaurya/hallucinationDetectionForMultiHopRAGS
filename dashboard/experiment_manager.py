"""Page 7: Experiment Management allowing dataset selection, configuration execution, comparison, and CSV/JSON export."""
from __future__ import annotations

import json
from typing import Any, Dict

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore

from evaluation.comparison import ablation_to_dataframe, experiments_to_dataframe
from evaluation.datasets import DatasetRegistry


def render_experiment_manager_page(state: Dict[str, Any]) -> None:
    """Render Page 7: Experiment Management in Streamlit."""
    if st is None:
        return
    st.title("Academic Experiment & Ablation Manager")

    dataset_choice = st.selectbox(
        "Select Evaluation Dataset",
        options=DatasetRegistry.AVAILABLE_DATASETS + ["all"],
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run E1–E4 Comparative Experiments", type="primary"):
            res = state["experiment_runner"].run_all_experiments(dataset_choice)
            state["latest_experiments"] = res
            st.success(f"Completed E1–E4 evaluation on '{dataset_choice}'.")
    with col2:
        if st.button("Run 7-Variant Ablation Study"):
            abl = state["ablation_runner"].run_ablation_suite(dataset_choice)
            state["latest_ablation"] = abl
            st.success(f"Completed 7-variant ablation study on '{dataset_choice}'.")

    if state.get("latest_experiments"):
        df_exp = experiments_to_dataframe(state["latest_experiments"])
        st.subheader("Experimental Results Table")
        st.dataframe(df_exp, use_container_width=True)
        csv_bytes = df_exp.to_csv(index=False).encode("utf-8")
        json_bytes = json.dumps(state["latest_experiments"], indent=2).encode("utf-8")
        dl1, dl2 = st.columns(2)
        dl1.download_button("Export Experiments CSV", data=csv_bytes, file_name=f"experiments_{dataset_choice}.csv", mime="text/csv")
        dl2.download_button("Export Experiments JSON", data=json_bytes, file_name=f"experiments_{dataset_choice}.json", mime="application/json")

    if state.get("latest_ablation"):
        df_abl = ablation_to_dataframe(state["latest_ablation"])
        st.subheader("Ablation Study Results Table")
        st.dataframe(df_abl, use_container_width=True)
        abl_csv = df_abl.to_csv(index=False).encode("utf-8")
        st.download_button("Export Ablation CSV", data=abl_csv, file_name=f"ablation_{dataset_choice}.csv", mime="text/csv")
