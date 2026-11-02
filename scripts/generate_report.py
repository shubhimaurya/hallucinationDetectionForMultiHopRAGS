"""Script to compile experimental outputs into a Markdown research report summary."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from evaluation.comparison import ablation_to_dataframe, experiments_to_dataframe


def main() -> None:
    exp_file = settings.experiments_dir / "experiments_ms_conflict_bench.json"
    abl_file = settings.experiments_dir / "ablation_ms_conflict_bench.json"
    if not exp_file.exists() or not abl_file.exists():
        print("Run scripts/run_experiments.py first.")
        return

    exp_data = json.loads(exp_file.read_text(encoding="utf-8"))
    abl_data = json.loads(abl_file.read_text(encoding="utf-8"))

    df_exp = experiments_to_dataframe(exp_data)
    df_abl = ablation_to_dataframe(abl_data)

    report_md = "# Automated Experimental & Ablation Report\n\n"
    report_md += "## 1. E1 - E4 Comparative Evaluation (ms_conflict_bench)\n\n"
    report_md += df_exp.to_markdown(index=False) + "\n\n"
    report_md += "## 2. 7-Variant Component Ablation Study\n\n"
    report_md += df_abl.to_markdown(index=False) + "\n"

    out_path = settings.experiments_dir / "experimental_report_summary.md"
    out_path.write_text(report_md, encoding="utf-8")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
