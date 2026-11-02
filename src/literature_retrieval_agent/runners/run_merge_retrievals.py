from __future__ import annotations

import argparse
from pathlib import Path

from literature_retrieval_agent.report_writer import write_pilot_reports
from literature_retrieval_agent.retrieval_merger import merge_retrieval_pilots
from literature_retrieval_agent.schemas import RetrievalPilotResult


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge baseline and context-expanded retrieval pilots before "
            "paper ranking. Inputs remain unchanged."
        )
    )
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument(
        "--input-pilot",
        type=Path,
        action="append",
        required=True,
        help="Repeat this argument for each retrieval pilot to merge.",
    )
    parser.add_argument("--output-directory", type=Path, required=True)

    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    cluster_id = str(args.cluster_id)

    if len(args.input_pilot) < 2:
        raise ValueError("Provide --input-pilot at least twice.")

    pilots: list[RetrievalPilotResult] = []

    for input_path in args.input_pilot:
        resolved = input_path.resolve()

        if not resolved.is_file():
            raise FileNotFoundError(f"Retrieval pilot not found: {resolved}")

        pilot = RetrievalPilotResult.model_validate_json(
            resolved.read_text(encoding="utf-8-sig")
        )

        if str(pilot.request.cluster_id) != cluster_id:
            raise ValueError(
                "Input pilot cluster mismatch: "
                f"{pilot.request.cluster_id!r} != {cluster_id!r}."
            )

        pilots.append(pilot)

    merged = merge_retrieval_pilots(pilots)
    paths = write_pilot_reports(merged, args.output_directory.resolve())

    print("Retrieval pilots merged successfully.")
    print(f"Cluster: {merged.request.cluster_id}")
    print(f"Inputs merged: {len(pilots)}")
    print(f"Queries retained: {len(merged.query_plan.queries)}")
    print(f"Unique papers retained: {len(merged.unique_papers)}")
    print(f"Warnings: {len(merged.warnings)}")

    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
