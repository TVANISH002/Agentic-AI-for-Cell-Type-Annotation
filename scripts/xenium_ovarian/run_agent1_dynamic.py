from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cosmx_brcr_annotation_pipeline.run_agent1_orchestrated import run_agent1


def main():
    parser = argparse.ArgumentParser(
        description="Dataset-independent Agent 1 runner"
    )

    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--panel-gene-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--tissue", required=True)
    parser.add_argument("--platform", required=True)

    args = parser.parse_args()

    panel = args.panel_gene_list.resolve()
    output_dir = args.output_dir.resolve()

    if not panel.is_file():
        raise FileNotFoundError(panel)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {output_dir}"
        )

    print(f"Running Agent 1 for dataset: {args.dataset_id}")

    result_path = run_agent1(
        panel_gene_list_csv=panel,
        output_dir=output_dir,
        species=args.species,
        tissue=args.tissue,
        platform=args.platform,
    )

    print("\nAGENT 1 COMPLETED SUCCESSFULLY")
    print(f"Dataset: {args.dataset_id}")
    print(f"Species: {args.species}")
    print(f"Tissue: {args.tissue}")
    print(f"Platform: {args.platform}")
    print(f"Output: {result_path}")


if __name__ == "__main__":
    main()