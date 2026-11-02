"""Script to run E1 Original Baseline evaluation and save results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.experiments import ExperimentRunner


def main() -> None:
    runner = ExperimentRunner()
    res = runner.evaluate_configuration("E1_ORIGINAL_BASELINE", dataset_name="ms_conflict_bench")
    print("E1 Original Baseline Metrics:")
    print(json.dumps(res["metrics"], indent=2))


if __name__ == "__main__":
    main()
