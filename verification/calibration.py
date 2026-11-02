"""Post-hoc Temperature Scaling and Platt Calibration without test-set leakage."""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class TemperatureCalibrator:
    """Fits temperature T > 0 on a validation set by minimizing Negative Log-Likelihood (NLL)."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        self.temperature = max(0.05, float(initial_temperature))
        self.is_fitted = False

    def calibrate_logit(self, logit: float) -> float:
        return sigmoid(logit / self.temperature)

    def fit(self, val_logits: List[float], val_labels: List[int]) -> float:
        """Optimize temperature T on validation logits and binary ground-truth labels (1=Supported, 0=Hallucinated)."""
        if not val_logits or len(val_logits) != len(val_labels):
            return self.temperature
        logits_arr = np.asarray(val_logits, dtype=np.float64)
        labels_arr = np.asarray(val_labels, dtype=np.float64)

        def nll_objective(params: np.ndarray) -> float:
            temp = max(0.05, float(params[0]))
            scaled = np.clip(logits_arr / temp, -30.0, 30.0)
            probs = 1.0 / (1.0 + np.exp(-scaled))
            eps = 1e-7
            loss = -np.mean(
                labels_arr * np.log(probs + eps) + (1.0 - labels_arr) * np.log(1.0 - probs + eps)
            )
            # Mild L2 prior around T=1.0
            return float(loss + 0.01 * ((temp - 1.0) ** 2))

        res = minimize(nll_objective, x0=np.array([1.0]), bounds=[(0.10, 5.0)], method="L-BFGS-B")
        if res.success:
            self.temperature = float(res.x[0])
            self.is_fitted = True
        return self.temperature

    @staticmethod
    def compute_ece_and_brier(
        probs: List[float],
        labels: List[int],
        n_bins: int = 10,
    ) -> Tuple[float, float, List[Dict[str, float]]]:
        """Compute Expected Calibration Error (ECE), Brier Score, and reliability diagram bins."""
        if not probs or len(probs) != len(labels):
            return 0.0, 0.0, []
        p_arr = np.asarray(probs, dtype=np.float64)
        y_arr = np.asarray(labels, dtype=np.float64)
        brier = float(np.mean((p_arr - y_arr) ** 2))

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        bins_data: List[Dict[str, float]] = []
        n_total = len(p_arr)

        for b_idx in range(n_bins):
            low, high = bin_edges[b_idx], bin_edges[b_idx + 1]
            if b_idx == n_bins - 1:
                mask = (p_arr >= low) & (p_arr <= high)
            else:
                mask = (p_arr >= low) & (p_arr < high)
            count = int(np.sum(mask))
            if count > 0:
                avg_conf = float(np.mean(p_arr[mask]))
                avg_acc = float(np.mean(y_arr[mask]))
                ece += (count / n_total) * abs(avg_acc - avg_conf)
            else:
                avg_conf = float(0.5 * (low + high))
                avg_acc = 0.0
            bins_data.append(
                {
                    "bin_index": float(b_idx),
                    "bin_lower": float(low),
                    "bin_upper": float(high),
                    "count": float(count),
                    "avg_confidence": round(avg_conf, 4),
                    "empirical_accuracy": round(avg_acc, 4),
                }
            )
        return float(ece), brier, bins_data
