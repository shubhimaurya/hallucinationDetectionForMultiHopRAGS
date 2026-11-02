"""Script to initialize and verify benchmark datasets in data/raw and data/processed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from evaluation.datasets import DatasetRegistry


def main() -> None:
    settings.ensure_directories()
    counts = DatasetRegistry.export_all_to_disk(settings.processed_data_dir)
    manifest = {
        "status": "ready",
        "datasets_exported": counts,
        "output_directory": str(settings.processed_data_dir),
    }
    (settings.raw_data_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Exported benchmark datasets:", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
