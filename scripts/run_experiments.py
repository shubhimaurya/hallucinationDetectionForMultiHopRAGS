"""Script to execute all E1-E4 experiments across all datasets, run the 7-variant ablation suite, and generate plots/CSVs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from evaluation.ablation import AblationStudyRunner
from evaluation.comparison import (
    ablation_to_dataframe,
    experiments_to_dataframe,
    export_comparison_csv,
)
from evaluation.datasets import DatasetRegistry
from evaluation.experiments import ExperimentRunner
from evaluation.visualization import (
    save_experiment_comparison_chart,
    save_reliability_diagram,
)


def main() -> None:
    exp_runner = ExperimentRunner()
    abl_runner = AblationStudyRunner()

    all_datasets_summary = {}
    for ds_name in DatasetRegistry.AVAILABLE_DATASETS + ["all"]:
        exp_res = exp_runner.run_all_experiments(dataset_name=ds_name)
        df_exp = experiments_to_dataframe(exp_res)
        export_comparison_csv(df_exp, settings.experiments_dir / f"comparison_{ds_name}.csv")
        all_datasets_summary[ds_name] = {
            k: v["metrics"] for k, v in exp_res.items()
        }
        if ds_name == "ms_conflict_bench":
            save_experiment_comparison_chart(
                exp_res, settings.experiments_dir / "e1_e4_comparison_chart.png"
            )
            e4_bins = exp_res["E4_PROPOSED_ACVE"]["metrics"]["reliability_bins"]
            save_reliability_diagram(
                e4_bins, settings.experiments_dir / "e4_reliability_diagram.png"
            )

    abl_res = abl_runner.run_ablation_suite(dataset_name="ms_conflict_bench")
    df_abl = ablation_to_dataframe(abl_res)
    export_comparison_csv(df_abl, settings.experiments_dir / "ablation_ms_conflict_bench.csv")

    abl_all = abl_runner.run_ablation_suite(dataset_name="all")
    df_abl_all = ablation_to_dataframe(abl_all)
    export_comparison_csv(df_abl_all, settings.experiments_dir / "ablation_all.csv")

    master_summary = {
        "experiments_by_dataset": all_datasets_summary,
        "ablation_ms_conflict_bench": {k: v["metrics"] for k, v in abl_res.items()},
        "ablation_all": {k: v["metrics"] for k, v in abl_all.items()},
    }
    (settings.experiments_dir / "master_experimental_summary.json").write_text(
        json.dumps(master_summary, indent=2), encoding="utf-8"
    )
    print("All experiments and ablation studies completed successfully.")


if __name__ == "__main__":
    main()
