"""Build raw-versus-family candidate review reports for all clusters."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from math import isnan
from pathlib import Path

import pandas as pd

from marker_interpretation_agent_v2.normalization.family_review_builder import (
    CandidateFamilyReviewBuilder,
)


DEFAULT_CLUSTERS_ROOT = Path("data/agent2_v2/clusters")

DEFAULT_COMBINED_OUTPUT = Path(
    "data/agent2_v2/all_clusters_candidate_family_review.csv"
)

DEFAULT_VALIDATION_OUTPUT = Path(
    "data/agent2_v2/candidate_family_review_validation.json"
)

_CLUSTER_PATTERN = re.compile(r"^cluster_(\d+)$")

_ALLOWED_REVIEW_STATUSES = {
    "stable_candidate_interpretation",
    "review_recommended",
    "manual_review_required",
}


@dataclass(frozen=True)
class _ClusterReviewValidation:
    """Typed validation record for one processed cluster."""

    cluster_id: int
    raw_candidates_file: str
    family_summary_file: str
    review_output_file: str
    review_status: str
    review_flags: str
    raw_top_cell_type: str
    family_top_cell_type: str
    warnings: list[str]
    is_valid: bool


def _discover_cluster_directories(
    clusters_root: Path,
) -> list[tuple[int, Path]]:
    """Find numeric cluster directories in ascending order."""

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

        discovered.append(
            (
                int(match.group(1)),
                path,
            )
        )

    discovered.sort(
        key=lambda item: item[0]
    )

    if not discovered:
        raise ValueError(
            f"No cluster directories were found under: {clusters_root}"
        )

    return discovered


def _as_text(value: object) -> str:
    """Convert a scalar CSV value into clean text."""

    if value is None or value is pd.NA:
        return ""

    if isinstance(value, float) and isnan(value):
        return ""

    return str(value).strip()


def _validate_review(
    cluster_id: int,
    review: pd.DataFrame,
) -> list[str]:
    """Validate one generated cluster review."""

    warnings: list[str] = []

    if len(review) != 1:
        warnings.append(
            f"expected_one_review_record_found_{len(review)}"
        )
        return warnings

    row = review.iloc[0]

    observed_cluster_id = _as_text(
        row["cluster_id"]
    )

    if observed_cluster_id.endswith(".0"):
        observed_cluster_id = observed_cluster_id[:-2]

    if observed_cluster_id != str(cluster_id):
        warnings.append(
            "cluster_id_mismatch:"
            f"expected={cluster_id},"
            f"observed={observed_cluster_id}"
        )

    review_status = _as_text(
        row["review_status"]
    )

    if review_status not in _ALLOWED_REVIEW_STATUSES:
        warnings.append(
            f"unexpected_review_status:{review_status}"
        )

    if not _as_text(row["raw_top_cell_type"]):
        warnings.append(
            "missing_raw_top_cell_type"
        )

    if not _as_text(row["family_top_cell_type"]):
        warnings.append(
            "missing_family_top_cell_type"
        )

    if not _as_text(row["review_note"]):
        warnings.append(
            "missing_review_note"
        )

    return warnings


def build_candidate_family_reviews(
    clusters_root: Path,
    combined_output: Path,
    validation_output: Path,
) -> pd.DataFrame:
    """Build per-cluster and combined family review reports."""

    builder = CandidateFamilyReviewBuilder()

    cluster_directories = _discover_cluster_directories(
        clusters_root
    )

    combined_frames: list[pd.DataFrame] = []

    validation_records: list[
        _ClusterReviewValidation
    ] = []

    for cluster_id, cluster_directory in cluster_directories:
        raw_candidates_csv = cluster_directory / (
            f"cluster_{cluster_id}_ranked_candidate_cell_types.csv"
        )

        family_summary_csv = cluster_directory / (
            f"cluster_{cluster_id}_candidate_family_summary.csv"
        )

        output_csv = cluster_directory / (
            f"cluster_{cluster_id}_candidate_family_review.csv"
        )

        if not raw_candidates_csv.exists():
            raise FileNotFoundError(
                "Raw ranked candidate file was not found: "
                f"{raw_candidates_csv}"
            )

        if not family_summary_csv.exists():
            raise FileNotFoundError(
                "Candidate family summary was not found: "
                f"{family_summary_csv}"
            )

        review = builder.build_csv(
            raw_candidates_csv=raw_candidates_csv,
            family_summary_csv=family_summary_csv,
            output_csv=output_csv,
        )

        warnings = _validate_review(
            cluster_id=cluster_id,
            review=review,
        )

        combined_frames.append(review)

        row = review.iloc[0]

        review_status = _as_text(
            row["review_status"]
        )

        raw_top_cell_type = _as_text(
            row["raw_top_cell_type"]
        )

        family_top_cell_type = _as_text(
            row["family_top_cell_type"]
        )

        review_flags = _as_text(
            row["review_flags"]
        )

        validation_records.append(
            _ClusterReviewValidation(
                cluster_id=cluster_id,
                raw_candidates_file=str(
                    raw_candidates_csv
                ),
                family_summary_file=str(
                    family_summary_csv
                ),
                review_output_file=str(
                    output_csv
                ),
                review_status=review_status,
                review_flags=review_flags,
                raw_top_cell_type=raw_top_cell_type,
                family_top_cell_type=family_top_cell_type,
                warnings=warnings,
                is_valid=not warnings,
            )
        )

        print(
            f"Cluster {cluster_id}: "
            f"raw='{raw_top_cell_type}' | "
            f"family='{family_top_cell_type}' | "
            f"status={review_status}"
        )

    combined_review = pd.concat(
        combined_frames,
        ignore_index=True,
    )

    combined_review["cluster_id"] = (
        combined_review["cluster_id"]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )

    combined_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_review.to_csv(
        combined_output,
        index=False,
    )

    invalid_cluster_ids = [
        record.cluster_id
        for record in validation_records
        if not record.is_valid
    ]

    manual_review_cluster_ids = [
        record.cluster_id
        for record in validation_records
        if record.review_status
        == "manual_review_required"
    ]

    review_recommended_cluster_ids = [
        record.cluster_id
        for record in validation_records
        if record.review_status
        == "review_recommended"
    ]

    stable_cluster_ids = [
        record.cluster_id
        for record in validation_records
        if record.review_status
        == "stable_candidate_interpretation"
    ]

    validation_payload = {
        "is_valid": not invalid_cluster_ids,
        "clusters_root": str(
            clusters_root
        ),
        "processed_cluster_count": len(
            validation_records
        ),
        "processed_cluster_ids": [
            record.cluster_id
            for record in validation_records
        ],
        "stable_cluster_ids": (
            stable_cluster_ids
        ),
        "review_recommended_cluster_ids": (
            review_recommended_cluster_ids
        ),
        "manual_review_required_cluster_ids": (
            manual_review_cluster_ids
        ),
        "invalid_cluster_ids": (
            invalid_cluster_ids
        ),
        "combined_output_file": str(
            combined_output
        ),
        "cluster_results": [
            asdict(record)
            for record in validation_records
        ],
    }

    validation_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        "Combined review written to: "
        f"{combined_output}"
    )

    print(
        "Validation report written to: "
        f"{validation_output}"
    )

    print(
        "Stable clusters: "
        f"{stable_cluster_ids}"
    )

    print(
        "Review recommended: "
        f"{review_recommended_cluster_ids}"
    )

    print(
        "Manual review required: "
        f"{manual_review_cluster_ids}"
    )

    print(
        "Validation status: "
        f"{'PASSED' if not invalid_cluster_ids else 'REVIEW REQUIRED'}"
    )

    return combined_review


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare raw candidate leaders with "
            "candidate-family summaries for all clusters."
        )
    )

    parser.add_argument(
        "--clusters-root",
        type=Path,
        default=DEFAULT_CLUSTERS_ROOT,
        help=(
            "Directory containing cluster result folders."
        ),
    )

    parser.add_argument(
        "--combined-output",
        type=Path,
        default=DEFAULT_COMBINED_OUTPUT,
        help=(
            "Combined raw-versus-family review CSV path."
        ),
    )

    parser.add_argument(
        "--validation-output",
        type=Path,
        default=DEFAULT_VALIDATION_OUTPUT,
        help=(
            "Review validation JSON output path."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run candidate-family reviews for all clusters."""

    args = _parse_args()

    build_candidate_family_reviews(
        clusters_root=args.clusters_root,
        combined_output=args.combined_output,
        validation_output=args.validation_output,
    )


if __name__ == "__main__":
    main()