"""Comparison and tabular export utilities for experimental runs and ablation studies."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def experiments_to_dataframe(experiment_results: Dict[str, Any]) -> pd.DataFrame:
    """Convert E1-E4 experiment dictionary into a clean comparison DataFrame."""
    rows: List[Dict[str, Any]] = []
    for exp_id, payload in experiment_results.items():
        m = payload.get("metrics", {})
        rows.append(
            {
                "Experiment": exp_id,
                "Dataset": payload.get("dataset_name", ""),
                "Accuracy": m.get("accuracy", 0.0),
                "Precision": m.get("precision", 0.0),
                "Recall": m.get("recall", 0.0),
                "F1_Score": m.get("f1_score", 0.0),
                "AUROC": m.get("auroc", 0.0),
                "FPR": m.get("false_positive_rate", 0.0),
                "ECE": m.get("expected_calibration_error", 0.0),
                "Brier_Score": m.get("brier_score", 0.0),
                "Conflict_Acc": m.get("conflict_detection_accuracy", 0.0),
                "Avg_LLM_Calls": m.get("avg_llm_calls_per_query", 0.0),
                "Avg_Latency_ms": m.get("avg_verification_latency_ms", 0.0),
                "Avg_Tokens": m.get("avg_tokens_per_query", 0.0),
            }
        )
    return pd.DataFrame(rows)


def ablation_to_dataframe(ablation_results: Dict[str, Any]) -> pd.DataFrame:
    """Convert 7-variant ablation dictionary into a clean DataFrame with deltas vs complete ACVE."""
    rows: List[Dict[str, Any]] = []
    full_f1 = (
        ablation_results.get("7_complete_proposed_acve", {})
        .get("metrics", {})
        .get("f1_score", 0.0)
    )
    for var_name, payload in ablation_results.items():
        m = payload.get("metrics", {})
        f1 = float(m.get("f1_score", 0.0))
        rows.append(
            {
                "Ablation_Variant": var_name,
                "Accuracy": m.get("accuracy", 0.0),
                "Precision": m.get("precision", 0.0),
                "Recall": m.get("recall", 0.0),
                "F1_Score": f1,
                "Delta_F1_vs_Full": round(f1 - full_f1, 4),
                "AUROC": m.get("auroc", 0.0),
                "ECE": m.get("expected_calibration_error", 0.0),
                "Conflict_Acc": m.get("conflict_detection_accuracy", 0.0),
                "Avg_LLM_Calls": m.get("avg_llm_calls_per_query", 0.0),
                "Avg_Latency_ms": m.get("avg_verification_latency_ms", 0.0),
            }
        )
    return pd.DataFrame(rows)


def export_comparison_csv(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
