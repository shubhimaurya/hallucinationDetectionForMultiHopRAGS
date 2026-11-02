"""Visualization utilities generating publication-grade charts for experiments and calibration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_experiment_comparison_chart(
    experiment_results: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Render a grouped bar chart comparing Accuracy, F1-Score, AUROC, and (1 - ECE) across E1..E4."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(experiment_results.keys())
    short_labels = [l.split("_")[0] for l in labels]

    acc = [experiment_results[k]["metrics"]["accuracy"] for k in labels]
    f1 = [experiment_results[k]["metrics"]["f1_score"] for k in labels]
    auroc = [experiment_results[k]["metrics"]["auroc"] for k in labels]
    cal = [1.0 - experiment_results[k]["metrics"]["expected_calibration_error"] for k in labels]

    x = np.arange(len(labels))
    width = 0.20

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    ax.bar(x - 1.5 * width, acc, width, label="Accuracy", color="#1b4965")
    ax.bar(x - 0.5 * width, f1, width, label="F1-Score", color="#2a9d8f")
    ax.bar(x + 0.5 * width, auroc, width, label="AUROC", color="#e9c46a")
    ax.bar(x + 1.5 * width, cal, width, label="Calibration (1 - ECE)", color="#e76f51")

    ax.set_ylabel("Score")
    ax.set_title("Experimental Comparison Across Verification Architectures (E1 - E4)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylim(0.0, 1.12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def save_reliability_diagram(
    reliability_bins: List[Dict[str, float]],
    output_path: Path,
    title: str = "ACVE Calibration Reliability Diagram",
) -> Path:
    """Plot predicted confidence vs empirical accuracy across decile bins."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    confs = [b["avg_confidence"] for b in reliability_bins if b["count"] > 0]
    accs = [b["empirical_accuracy"] for b in reliability_bins if b["count"] > 0]

    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=150)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6c757d", label="Ideal Calibration")
    if confs:
        ax.plot(confs, accs, marker="o", linewidth=2, color="#1b4965", label="ACVE Empirical")
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
