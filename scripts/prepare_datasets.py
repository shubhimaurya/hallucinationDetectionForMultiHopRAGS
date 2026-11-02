"""Script to validate dataset splits (val vs test) and verify zero overlap/leakage."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.datasets import DatasetRegistry


def main() -> None:
    report = {}
    for ds_name in DatasetRegistry.AVAILABLE_DATASETS:
        val_s = DatasetRegistry.load_dataset(ds_name, split="val")
        test_s = DatasetRegistry.load_dataset(ds_name, split="test")
        val_ids = {s.sample_id for s in val_s}
        test_ids = {s.sample_id for s in test_s}
        overlap = val_ids.intersection(test_ids)
        assert len(overlap) == 0, f"Leakage detected in {ds_name}: {overlap}"
        report[ds_name] = {
            "total": len(val_s) + len(test_s),
            "val_count": len(val_s),
            "test_count": len(test_s),
            "leakage_overlap": len(overlap),
        }
    print("Dataset preparation & anti-leakage verification passed:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
