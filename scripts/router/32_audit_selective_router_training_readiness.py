from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROVISIONAL_FEATURE_COUNT = 14


def parse_target(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        number = int(float(text))
    except Exception:
        return None

    return number if number in {0, 1} else None


def parse_bool(value):
    if pd.isna(value):
        return False

    text = str(value).strip().casefold()

    if not text:
        return False

    if text in {"true", "yes", "y"}:
        return True

    if text in {"false", "no", "n"}:
        return False

    try:
        return float(text) == 1.0
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--xy-linkage",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/"
            "router_selective_training_readiness_v1"
        ),
    )

    args = parser.parse_args()

    if not args.xy_linkage.exists():
        raise FileNotFoundError(
            f"X-y linkage table not found: {args.xy_linkage}"
        )

    frame = pd.read_csv(
        args.xy_linkage
    )

    required = [
        "row_id",
        "dataset",
        "cluster_id",
        "primary_router_sufficiency_target",
        "target_eligible",
        "has_independent_primary_target",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "X-y linkage table missing required columns: "
            + ", ".join(missing)
        )

    frame = frame.copy()

    frame["_target"] = frame[
        "primary_router_sufficiency_target"
    ].map(
        parse_target
    )

    frame["_eligible"] = frame[
        "target_eligible"
    ].map(
        parse_bool
    )

    frame["_independent"] = frame[
        "has_independent_primary_target"
    ].map(
        parse_bool
    )

    usable = frame[
        frame["_eligible"]
        & frame["_independent"]
        & frame["_target"].isin([0, 1])
    ].copy()

    labeled_rows = int(
        len(
            usable
        )
    )

    positive_rows = int(
        (
            usable["_target"]
            == 1
        ).sum()
    )

    negative_rows = int(
        (
            usable["_target"]
            == 0
        ).sum()
    )

    both_classes_present = (
        positive_rows > 0
        and negative_rows > 0
    )

    independent_datasets = sorted(
        usable[
            "dataset"
        ]
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    independent_dataset_count = len(
        independent_datasets
    )

    dataset_counts = (
        usable.groupby(
            "dataset",
            dropna=False,
        )
        .agg(
            labeled_clusters=(
                "row_id",
                "count",
            ),
            sufficient=(
                "_target",
                lambda s: int(
                    (
                        s == 1
                    ).sum()
                ),
            ),
            insufficient=(
                "_target",
                lambda s: int(
                    (
                        s == 0
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )

    # Minimum structural requirement for:
    # development dataset(s) + validation dataset + held-out test dataset.
    preferred_dataset_split_possible = (
        independent_dataset_count >= 3
    )

    # With 14 provisional features and only 7 targets, the current sample is
    # smaller than the feature dimension. Regularization can technically fit
    # such a model, but it would not constitute a defensible final calibration.
    labels_exceed_feature_count = (
        labeled_rows
        > PROVISIONAL_FEATURE_COUNT
    )

    labels_per_feature = (
        labeled_rows
        / PROVISIONAL_FEATURE_COUNT
        if PROVISIONAL_FEATURE_COUNT > 0
        else None
    )

    final_training_ready = (
        preferred_dataset_split_possible
        and both_classes_present
        and labels_exceed_feature_count
    )

    # We intentionally distinguish:
    # - structurally impossible now
    # - statistically adequate later (must still be judged after more data).
    blockers = []

    if independent_dataset_count < 3:
        blockers.append(
            "Fewer than three independently labeled datasets are available, "
            "so separate development, validation, and held-out test datasets "
            "cannot currently be formed."
        )

    if not both_classes_present:
        blockers.append(
            "Both Router sufficiency classes are not represented."
        )

    if not labels_exceed_feature_count:
        blockers.append(
            "The number of independent labeled clusters does not exceed the "
            "14-feature provisional model dimension; final coefficient "
            "estimation would be poorly supported."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_summary_path = (
        args.output_dir
        / "selective_router_training_readiness_by_dataset_v1.csv"
    )

    audit_path = (
        args.output_dir
        / "selective_router_training_readiness_audit_v1.json"
    )

    dataset_counts.to_csv(
        dataset_summary_path,
        index=False,
    )

    audit = {
        "audit_role": (
            "SELECTIVE_ROUTER_FINAL_TRAINING_READINESS"
        ),
        "xy_linkage_source": str(
            args.xy_linkage
        ),
        "provisional_feature_count": (
            PROVISIONAL_FEATURE_COUNT
        ),
        "independent_labeled_clusters": (
            labeled_rows
        ),
        "sufficient_targets": (
            positive_rows
        ),
        "insufficient_targets": (
            negative_rows
        ),
        "both_classes_present": (
            both_classes_present
        ),
        "independently_labeled_dataset_count": (
            independent_dataset_count
        ),
        "independently_labeled_datasets": (
            independent_datasets
        ),
        "preferred_dataset_level_dev_validation_test_possible": (
            preferred_dataset_split_possible
        ),
        "independent_labels_exceed_feature_count": (
            labels_exceed_feature_count
        ),
        "independent_labels_per_provisional_feature": (
            labels_per_feature
        ),
        "final_router_training_ready": (
            final_training_ready
        ),
        "blocking_reasons": (
            blockers
        ),
        "interpretation": (
            "The current data may be used for engineering checks and case "
            "analysis, but not for final Router coefficient estimation, "
            "validation-threshold calibration, or held-out publication claims."
        ),
        "minimum_three_dataset_note": (
            "Three independently labeled datasets are only the minimum "
            "structural requirement for separate development, validation, "
            "and held-out test partitions. It is not a claim that three "
            "datasets alone guarantee adequate statistical power."
        ),
        "model_trained": False,
        "scaling_fitted": False,
        "threshold_selected": False,
        "router_v4_replaced": False,
        "existing_pipeline_outputs_modified": False,
        "status": (
            "READY"
            if final_training_ready
            else "NOT_READY"
        ),
    }

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Dataset readiness summary: "
        f"{dataset_summary_path}"
    )
    print(
        f"Training readiness audit: "
        f"{audit_path}"
    )

    print()
    print(
        f"Independent labeled clusters: {labeled_rows}"
    )
    print(
        f"  sufficient (1): {positive_rows}"
    )
    print(
        f"  insufficient (0): {negative_rows}"
    )
    print(
        "Both classes present: "
        + (
            "YES"
            if both_classes_present
            else "NO"
        )
    )

    print()
    print(
        "Independently labeled datasets: "
        f"{independent_dataset_count}"
    )

    if independent_datasets:
        for dataset in independent_datasets:
            print(
                f"  {dataset}"
            )

    print(
        "Separate dataset-level development / validation / "
        "held-out test possible: "
        + (
            "YES"
            if preferred_dataset_split_possible
            else "NO"
        )
    )

    print()
    print(
        f"Provisional predictive features: "
        f"{PROVISIONAL_FEATURE_COUNT}"
    )
    print(
        "Independent labels exceed feature count: "
        + (
            "YES"
            if labels_exceed_feature_count
            else "NO"
        )
    )
    print(
        "Independent labels per provisional feature: "
        f"{labels_per_feature:.3f}"
    )

    print()
    print(
        "FINAL ROUTER TRAINING READINESS: "
        + (
            "READY"
            if final_training_ready
            else "NOT READY"
        )
    )

    if blockers:
        print()
        print(
            "BLOCKING REASONS:"
        )

        for blocker in blockers:
            print(
                f"  - {blocker}"
            )

    print()
    print(
        "Model trained: NO"
    )
    print(
        "Scaling fitted: NO"
    )
    print(
        "Threshold selected: NO"
    )
    print(
        "Router V4 replaced: NO"
    )
    print(
        "No existing Agent/Router/orchestrator outputs were modified."
    )


if __name__ == "__main__":
    main()
