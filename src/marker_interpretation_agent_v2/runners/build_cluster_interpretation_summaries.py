"""Build concise interpretation summaries for all Agent 2 V2 clusters."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from math import isfinite, isnan
from pathlib import Path

import pandas as pd

from marker_interpretation_agent_v2.reporting.cluster_interpretation_summary_builder import (
    ClusterInterpretationSummaryBuilder,
)


DEFAULT_CLUSTERS_ROOT = Path(
    "data/agent2_v2/clusters"
)

DEFAULT_COMBINED_CSV = Path(
    "data/agent2_v2/all_clusters_interpretation_summary.csv"
)

DEFAULT_COMBINED_JSON = Path(
    "data/agent2_v2/all_clusters_interpretation_summary.json"
)

DEFAULT_VALIDATION_JSON = Path(
    "data/agent2_v2/cluster_interpretation_summary_validation.json"
)

_CLUSTER_PATTERN = re.compile(
    r"^cluster_(\d+)$"
)

_ALLOWED_INTERPRETATION_STATES = {
    "stable",
    "consistent_review_recommended",
    "ambiguous_manual_review",
}


@dataclass(frozen=True)
class _ClusterSummaryValidation:
    """Validation record for one processed cluster."""

    cluster_id: int
    raw_candidates_file: str
    subtype_summary_file: str
    review_file: str
    summary_output_file: str
    interpretation_state: str
    professor_facing_interpretation: str
    warnings: list[str]
    is_valid: bool


def _is_missing(value: object) -> bool:
    """Return True for scalar missing values."""

    if value is None or value is pd.NA:
        return True

    if isinstance(value, float):
        return isnan(value)

    return str(value).strip().casefold() in {
        "",
        "nan",
        "none",
        "<na>",
        "nat",
    }


def _as_text(value: object) -> str:
    """Convert one scalar value into clean text."""

    if _is_missing(value):
        return ""

    return str(value).strip()


def _as_int(value: object) -> int:
    """Convert one scalar value into a validated integer."""

    if _is_missing(value):
        return 0

    if isinstance(value, bool):
        return int(value)

    text = str(value).strip()

    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Expected a numeric integer value, found: {value!r}."
        ) from exc

    if not isfinite(number):
        raise ValueError(
            f"Expected a finite integer value, found: {value!r}."
        )

    if not number.is_integer():
        raise ValueError(
            f"Expected an integer value, found: {value!r}."
        )

    return int(number)


def _as_bool(value: object) -> bool:
    """Convert one scalar value into a Boolean."""

    if isinstance(value, bool):
        return value

    if _is_missing(value):
        return False

    return str(value).strip().casefold() in {
        "true",
        "1",
        "yes",
        "y",
    }


def _discover_cluster_directories(
    clusters_root: Path,
) -> list[tuple[int, Path]]:
    """Find cluster directories in numeric order."""

    if not clusters_root.exists():
        raise FileNotFoundError(
            f"Clusters directory was not found: {clusters_root}"
        )

    discovered: list[tuple[int, Path]] = []

    for path in clusters_root.iterdir():
        if not path.is_dir():
            continue

        match = _CLUSTER_PATTERN.fullmatch(
            path.name
        )

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


def _validate_summary(
    cluster_id: int,
    summary: pd.DataFrame,
) -> list[str]:
    """Validate one generated cluster summary."""

    warnings: list[str] = []

    if len(summary) != 1:
        warnings.append(
            f"expected_one_summary_record_found_{len(summary)}"
        )
        return warnings

    row = summary.iloc[0]

    observed_cluster_id = _as_text(
        row["cluster_id"]
    )

    if observed_cluster_id.endswith(".0"):
        observed_cluster_id = (
            observed_cluster_id[:-2]
        )

    if observed_cluster_id != str(cluster_id):
        warnings.append(
            "cluster_id_mismatch:"
            f"expected={cluster_id},"
            f"observed={observed_cluster_id}"
        )

    interpretation_state = _as_text(
        row["interpretation_state"]
    )

    if (
        interpretation_state
        not in _ALLOWED_INTERPRETATION_STATES
    ):
        warnings.append(
            "unexpected_interpretation_state:"
            f"{interpretation_state}"
        )

    if not _as_text(
        row["professor_facing_interpretation"]
    ):
        warnings.append(
            "missing_professor_facing_interpretation"
        )

    if not _as_text(
        row["normalized_top_subtype"]
    ):
        warnings.append(
            "missing_normalized_top_subtype"
        )

    normalized_support = _as_int(
        row["normalized_top_support_count"]
    )

    if normalized_support <= 0:
        warnings.append(
            "non_positive_normalized_support_count"
        )

    top_subtype_tie = _as_bool(
        row["top_subtype_tie"]
    )

    competing_subtype = _as_text(
        row["competing_subtype"]
    )

    if (
        interpretation_state
        == "ambiguous_manual_review"
        and top_subtype_tie
        and not competing_subtype
    ):
        warnings.append(
            "ambiguous_tie_missing_competing_subtype"
        )

    return warnings


def _summary_row_to_record(
    row: pd.Series,
) -> dict[str, object]:
    """Convert one summary row into JSON-safe values."""

    return {
        "cluster_id": _as_int(
            row["cluster_id"]
        ),
        "interpretation_state": _as_text(
            row["interpretation_state"]
        ),
        "professor_facing_interpretation": _as_text(
            row["professor_facing_interpretation"]
        ),
        "raw_top_cell_type": _as_text(
            row["raw_top_cell_type"]
        ),
        "raw_top_support_count": _as_int(
            row["raw_top_support_count"]
        ),
        "raw_top_supporting_genes": _as_text(
            row["raw_top_supporting_genes"]
        ),
        "normalized_top_subtype": _as_text(
            row["normalized_top_subtype"]
        ),
        "normalized_top_support_count": _as_int(
            row["normalized_top_support_count"]
        ),
        "normalized_top_supporting_genes": _as_text(
            row["normalized_top_supporting_genes"]
        ),
        "normalized_top_broad_lineage": _as_text(
            row["normalized_top_broad_lineage"]
        ),
        "competing_subtype": _as_text(
            row["competing_subtype"]
        ),
        "competing_support_count": _as_int(
            row["competing_support_count"]
        ),
        "competing_supporting_genes": _as_text(
            row["competing_supporting_genes"]
        ),
        "competing_broad_lineage": _as_text(
            row["competing_broad_lineage"]
        ),
        "support_gap": _as_int(
            row["support_gap"]
        ),
        "top_subtype_tie": _as_bool(
            row["top_subtype_tie"]
        ),
        "raw_normalized_leader_match": _as_bool(
            row["raw_normalized_leader_match"]
        ),
        "review_status": _as_text(
            row["review_status"]
        ),
        "review_flags": _as_text(
            row["review_flags"]
        ),
        "review_note": _as_text(
            row["review_note"]
        ),
    }


def build_cluster_interpretation_summaries(
    clusters_root: Path,
    combined_csv: Path,
    combined_json: Path,
    validation_json: Path,
) -> pd.DataFrame:
    """Build per-cluster and combined interpretation summaries."""

    builder = (
        ClusterInterpretationSummaryBuilder()
    )

    cluster_directories = (
        _discover_cluster_directories(
            clusters_root
        )
    )

    combined_frames: list[pd.DataFrame] = []

    validation_records: list[
        _ClusterSummaryValidation
    ] = []

    for (
        cluster_id,
        cluster_directory,
    ) in cluster_directories:
        raw_candidates_file = (
            cluster_directory
            / f"cluster_{cluster_id}_ranked_candidate_cell_types.csv"
        )

        subtype_summary_file = (
            cluster_directory
            / f"cluster_{cluster_id}_candidate_family_summary.csv"
        )

        review_file = (
            cluster_directory
            / f"cluster_{cluster_id}_candidate_family_review.csv"
        )

        summary_output_file = (
            cluster_directory
            / f"cluster_{cluster_id}_interpretation_summary.csv"
        )

        summary = builder.build_csv(
            raw_candidates_csv=(
                raw_candidates_file
            ),
            subtype_summary_csv=(
                subtype_summary_file
            ),
            review_csv=review_file,
            output_csv=summary_output_file,
        )

        warnings = _validate_summary(
            cluster_id=cluster_id,
            summary=summary,
        )

        combined_frames.append(
            summary
        )

        row = summary.iloc[0]

        interpretation_state = _as_text(
            row["interpretation_state"]
        )

        professor_interpretation = _as_text(
            row[
                "professor_facing_interpretation"
            ]
        )

        validation_records.append(
            _ClusterSummaryValidation(
                cluster_id=cluster_id,
                raw_candidates_file=str(
                    raw_candidates_file
                ),
                subtype_summary_file=str(
                    subtype_summary_file
                ),
                review_file=str(
                    review_file
                ),
                summary_output_file=str(
                    summary_output_file
                ),
                interpretation_state=(
                    interpretation_state
                ),
                professor_facing_interpretation=(
                    professor_interpretation
                ),
                warnings=warnings,
                is_valid=not warnings,
            )
        )

        print(
            f"Cluster {cluster_id}: "
            f"{interpretation_state} | "
            f"{professor_interpretation}"
        )

    combined_summary = pd.concat(
        combined_frames,
        ignore_index=True,
    )

    combined_summary["cluster_id"] = (
        combined_summary["cluster_id"]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )

    combined_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_summary.to_csv(
        combined_csv,
        index=False,
    )

    json_records = [
        _summary_row_to_record(row)
        for _, row in combined_summary.iterrows()
    ]

    state_counts = {
        state: sum(
            1
            for record in json_records
            if record["interpretation_state"]
            == state
        )
        for state
        in sorted(
            _ALLOWED_INTERPRETATION_STATES
        )
    }

    combined_json_payload = {
        "schema_version": "1.0.0",
        "cluster_count": len(
            json_records
        ),
        "interpretation_state_counts": (
            state_counts
        ),
        "clusters": json_records,
    }

    combined_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_json.write_text(
        json.dumps(
            combined_json_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    invalid_cluster_ids = [
        record.cluster_id
        for record in validation_records
        if not record.is_valid
    ]

    stable_cluster_ids = [
        record.cluster_id
        for record in validation_records
        if record.interpretation_state
        == "stable"
    ]

    review_cluster_ids = [
        record.cluster_id
        for record in validation_records
        if record.interpretation_state
        == "consistent_review_recommended"
    ]

    ambiguous_cluster_ids = [
        record.cluster_id
        for record in validation_records
        if record.interpretation_state
        == "ambiguous_manual_review"
    ]

    validation_payload = {
        "is_valid": not invalid_cluster_ids,
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
            review_cluster_ids
        ),
        "ambiguous_cluster_ids": (
            ambiguous_cluster_ids
        ),
        "invalid_cluster_ids": (
            invalid_cluster_ids
        ),
        "combined_csv": str(
            combined_csv
        ),
        "combined_json": str(
            combined_json
        ),
        "cluster_results": [
            asdict(record)
            for record in validation_records
        ],
    }

    validation_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_json.write_text(
        json.dumps(
            validation_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()

    print(
        "Combined CSV written to: "
        f"{combined_csv}"
    )

    print(
        "Combined JSON written to: "
        f"{combined_json}"
    )

    print(
        "Validation report written to: "
        f"{validation_json}"
    )

    print(
        f"Stable clusters: {stable_cluster_ids}"
    )

    print(
        "Review recommended: "
        f"{review_cluster_ids}"
    )

    print(
        "Ambiguous/manual review: "
        f"{ambiguous_cluster_ids}"
    )

    print(
        "Validation status: "
        f"{'PASSED' if not invalid_cluster_ids else 'REVIEW REQUIRED'}"
    )

    return combined_summary


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Build concise interpretation summaries "
            "for all Agent 2 V2 clusters."
        )
    )

    parser.add_argument(
        "--clusters-root",
        type=Path,
        default=DEFAULT_CLUSTERS_ROOT,
    )

    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=DEFAULT_COMBINED_CSV,
    )

    parser.add_argument(
        "--combined-json",
        type=Path,
        default=DEFAULT_COMBINED_JSON,
    )

    parser.add_argument(
        "--validation-json",
        type=Path,
        default=DEFAULT_VALIDATION_JSON,
    )

    return parser.parse_args()


def main() -> None:
    """Run summary generation for all clusters."""

    args = _parse_args()

    build_cluster_interpretation_summaries(
        clusters_root=args.clusters_root,
        combined_csv=args.combined_csv,
        combined_json=args.combined_json,
        validation_json=args.validation_json,
    )


if __name__ == "__main__":
    main()