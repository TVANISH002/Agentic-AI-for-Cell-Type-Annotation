"""Build review-only candidate-family summaries for all Agent 2 V2 clusters.

This runner preserves all existing raw candidate files. It creates additional
family-level summaries by grouping related biological cell-type labels and
combining unique supporting genes without double counting.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from marker_interpretation_agent_v2.normalization.family_aggregator import (
    CandidateFamilyAggregator,
)


DEFAULT_CLUSTERS_ROOT = Path("data/agent2_v2/clusters")
DEFAULT_COMBINED_OUTPUT = Path(
    "data/agent2_v2/all_clusters_candidate_family_summary.csv"
)
DEFAULT_VALIDATION_OUTPUT = Path(
    "data/agent2_v2/candidate_family_summary_validation.json"
)

_CLUSTER_PATTERN = re.compile(r"^cluster_(\d+)$")


def _discover_cluster_directories(clusters_root: Path) -> list[tuple[int, Path]]:
    """Find numeric cluster directories in deterministic order."""

    if not clusters_root.exists():
        raise FileNotFoundError(
            f"Clusters directory was not found: {clusters_root}"
        )

    discovered: list[tuple[int, Path]] = []

    for path in clusters_root.iterdir():
        if not path.is_dir():
            continue

        match = _CLUSTER_PATTERN.fullmatch(path.name)
        if match is None:
            continue

        discovered.append((int(match.group(1)), path))

    discovered.sort(key=lambda item: item[0])

    if not discovered:
        raise ValueError(
            f"No cluster directories were found under: {clusters_root}"
        )

    return discovered


def _validate_family_summary(
    cluster_id: int,
    raw_candidates: pd.DataFrame,
    family_summary: pd.DataFrame,
) -> list[str]:
    """Validate one family summary and return warning messages."""

    warnings: list[str] = []

    if raw_candidates.empty:
        warnings.append("raw_candidate_table_is_empty")

    if family_summary.empty:
        warnings.append("family_summary_is_empty")
        return warnings

    observed_cluster_ids = {
        str(value).strip()
        for value in family_summary["cluster_id"].dropna().tolist()
    }

    if observed_cluster_ids != {str(cluster_id)}:
        warnings.append(
            "family_summary_cluster_id_mismatch:"
            f"expected={cluster_id},observed={sorted(observed_cluster_ids)}"
        )

    expected_ranks = list(range(1, len(family_summary) + 1))
    observed_ranks = family_summary["family_rank"].astype(int).tolist()

    if observed_ranks != expected_ranks:
        warnings.append("family_ranks_are_not_contiguous")

    if len(family_summary) > len(raw_candidates):
        warnings.append(
            "family_count_exceeds_raw_candidate_count"
        )

    if family_summary["cell_type_family"].duplicated().any():
        warnings.append("duplicate_cell_type_families")

    if (
        family_summary["unique_supporting_gene_count"]
        .astype(int)
        .lt(0)
        .any()
    ):
        warnings.append("negative_support_count_detected")

    return warnings


def build_candidate_family_summaries(
    clusters_root: Path,
    combined_output: Path,
    validation_output: Path,
) -> pd.DataFrame:
    """Build per-cluster and combined candidate-family summaries."""

    aggregator = CandidateFamilyAggregator()
    cluster_directories = _discover_cluster_directories(clusters_root)

    combined_frames: list[pd.DataFrame] = []
    validation_records: list[dict[str, object]] = []

    for cluster_id, cluster_directory in cluster_directories:
        input_csv = cluster_directory / (
            f"cluster_{cluster_id}_ranked_candidate_cell_types.csv"
        )

        output_csv = cluster_directory / (
            f"cluster_{cluster_id}_candidate_family_summary.csv"
        )

        if not input_csv.exists():
            raise FileNotFoundError(
                "Expected ranked candidate file was not found: "
                f"{input_csv}"
            )

        raw_candidates = pd.read_csv(input_csv)

        family_summary = aggregator.aggregate_csv(
            input_csv=input_csv,
            output_csv=output_csv,
        )

        warnings = _validate_family_summary(
            cluster_id=cluster_id,
            raw_candidates=raw_candidates,
            family_summary=family_summary,
        )

        combined_frames.append(family_summary)

        top_family = (
            str(family_summary.iloc[0]["cell_type_family"])
            if not family_summary.empty
            else ""
        )

        top_support = (
            int(
                family_summary.iloc[0][
                    "unique_supporting_gene_count"
                ]
            )
            if not family_summary.empty
            else 0
        )

        validation_records.append(
            {
                "cluster_id": cluster_id,
                "input_file": str(input_csv),
                "output_file": str(output_csv),
                "raw_candidate_count": len(raw_candidates),
                "family_count": len(family_summary),
                "top_family": top_family,
                "top_family_unique_supporting_gene_count": top_support,
                "warnings": warnings,
                "is_valid": not warnings,
            }
        )

        print(
            f"Cluster {cluster_id}: "
            f"{len(raw_candidates)} raw candidates -> "
            f"{len(family_summary)} families | "
            f"top family: {top_family} "
            f"({top_support} unique genes)"
        )

    combined_summary = pd.concat(
        combined_frames,
        ignore_index=True,
    )

    combined_output.parent.mkdir(parents=True, exist_ok=True)
    combined_summary.to_csv(combined_output, index=False)

    invalid_clusters = [
        record["cluster_id"]
        for record in validation_records
        if not bool(record["is_valid"])
    ]

    validation_payload = {
        "is_valid": not invalid_clusters,
        "clusters_root": str(clusters_root),
        "processed_cluster_count": len(validation_records),
        "processed_cluster_ids": [
            record["cluster_id"]
            for record in validation_records
        ],
        "combined_family_record_count": len(combined_summary),
        "combined_output_file": str(combined_output),
        "invalid_cluster_ids": invalid_clusters,
        "cluster_results": validation_records,
    }

    validation_output.parent.mkdir(parents=True, exist_ok=True)
    validation_output.write_text(
        json.dumps(
            validation_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Combined family summary written to: "
        f"{combined_output}"
    )
    print(
        "Validation report written to: "
        f"{validation_output}"
    )
    print(
        "Validation status: "
        f"{'PASSED' if not invalid_clusters else 'REVIEW REQUIRED'}"
    )

    return combined_summary


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Build candidate-family summaries from Agent 2 V2 "
            "ranked candidate outputs."
        )
    )

    parser.add_argument(
        "--clusters-root",
        type=Path,
        default=DEFAULT_CLUSTERS_ROOT,
        help=(
            "Directory containing cluster_0, cluster_1, and other "
            "cluster result folders."
        ),
    )

    parser.add_argument(
        "--combined-output",
        type=Path,
        default=DEFAULT_COMBINED_OUTPUT,
        help="Combined family-summary CSV output path.",
    )

    parser.add_argument(
        "--validation-output",
        type=Path,
        default=DEFAULT_VALIDATION_OUTPUT,
        help="Family-summary validation JSON output path.",
    )

    return parser.parse_args()


def main() -> None:
    """Run candidate-family aggregation for all discovered clusters."""

    args = _parse_args()

    build_candidate_family_summaries(
        clusters_root=args.clusters_root,
        combined_output=args.combined_output,
        validation_output=args.validation_output,
    )


if __name__ == "__main__":
    main()