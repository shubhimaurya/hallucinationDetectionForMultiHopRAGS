"""Comprehensive Evaluation Metrics Suite.

Computes:
- Classification metrics: Accuracy, Precision, Recall, F1, AUROC, Confusion Matrix, FPR, FNR
- Calibration metrics: Expected Calibration Error (ECE), Brier Score, Reliability Diagram bins
- Efficiency metrics: Average query latency, Verification latency, Average LLM calls, Token consumption, Memory usage
- Evidence metrics: Claim-level evidence coverage, Citation correctness, Source agreement, Conflict detection accuracy
"""
from __future__ import annotations

import resource
from typing import Any, Dict, List

import numpy as np

from verification.calibration import TemperatureCalibrator


def compute_auroc(labels: List[int], probs: List[float]) -> float:
    """Compute Area Under the ROC Curve (AUROC) via Mann-Whitney U rank statistic."""
    pos_scores = [p for y, p in zip(labels, probs) if y == 1]
    neg_scores = [p for y, p in zip(labels, probs) if y == 0]
    if not pos_scores or not neg_scores:
        return 0.5
    wins = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                wins += 1.0
            elif abs(ps - ns) <= 1e-9:
                wins += 0.5
    return float(wins / (len(pos_scores) * len(neg_scores)))


def compute_all_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[float],
    latencies_ms: List[float],
    llm_calls: List[int],
    tokens_used: List[int],
    evidence_coverages: List[float],
    citation_precisions: List[float],
    source_agreements: List[float],
    conflict_true: List[bool],
    conflict_pred: List[bool],
) -> Dict[str, Any]:
    """Compute all classification, calibration, efficiency, and evidence metrics from real predictions."""
    n = len(y_true)
    if n == 0:
        raise ValueError("Cannot compute metrics on empty prediction list.")

    # Here positive class (1) = HALLUCINATED for hallucination detection metrics,
    # OR we report both overall truthfulness classification and hallucination detection.
    # To make interpretation crystal clear and standard across hallucination benchmarks:
    # Let y_true_halluc = 1 - y_true (where 1 = Hallucination present, 0 = Faithful)
    y_true_h = [1 - y for y in y_true]
    y_pred_h = [1 - y for y in y_pred]
    y_prob_h = [1.0 - p for p in y_prob]

    tp = sum(1 for yt, yp in zip(y_true_h, y_pred_h) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true_h, y_pred_h) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true_h, y_pred_h) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true_h, y_pred_h) if yt == 1 and yp == 0)

    accuracy = (tp + tn) / n
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-9, precision + recall)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    auroc = compute_auroc(y_true_h, y_prob_h)

    ece, brier, reliability_bins = TemperatureCalibrator.compute_ece_and_brier(
        probs=y_prob, labels=y_true, n_bins=10
    )

    conflict_matches = sum(
        1 for ct, cp in zip(conflict_true, conflict_pred) if bool(ct) == bool(cp)
    )
    conflict_acc = conflict_matches / max(1, len(conflict_true))

    mem_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    return {
        "num_samples": n,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "auroc": round(auroc, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "confusion_matrix": {
            "tp_hallucination": tp,
            "tn_supported": tn,
            "fp_false_alarm": fp,
            "fn_missed_hallucination": fn,
        },
        "expected_calibration_error": round(ece, 4),
        "brier_score": round(brier, 4),
        "reliability_bins": reliability_bins,
        "avg_verification_latency_ms": round(float(np.mean(latencies_ms)), 3) if latencies_ms else 0.0,
        "avg_llm_calls_per_query": round(float(np.mean(llm_calls)), 3) if llm_calls else 0.0,
        "total_llm_calls": int(sum(llm_calls)),
        "avg_tokens_per_query": round(float(np.mean(tokens_used)), 2) if tokens_used else 0.0,
        "peak_memory_mb": round(mem_rss_mb, 2),
        "avg_evidence_coverage": round(float(np.mean(evidence_coverages)), 4) if evidence_coverages else 0.0,
        "avg_citation_correctness": round(float(np.mean(citation_precisions)), 4) if citation_precisions else 0.0,
        "avg_source_agreement": round(float(np.mean(source_agreements)), 4) if source_agreements else 0.0,
        "conflict_detection_accuracy": round(conflict_acc, 4),
    }
