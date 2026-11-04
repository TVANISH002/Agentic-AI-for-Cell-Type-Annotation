from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROVISIONAL_MODEL_FEATURES = [
    "candidate_count",
    "top1_support_count",
    "top2_present",
    "top3_present",
    "top2_support_count",
    "top3_support_count",
    "top1_cross_database_fraction",
    "direct_evidence_coverage",
    "top1_support_fraction_of_input",
    "top2_support_fraction_of_input",
    "top1_top2_gap_fraction_of_input",
    "leading_non_direct_fraction",
    "first_direct_supported_rank_fraction",
    "top1_mean_supporting_gene_rank_fraction",
]

TARGET_COLUMN = "primary_router_sufficiency_target"
ELIGIBILITY_COLUMN = "target_eligible"
INDEPENDENT_COLUMN = "has_independent_primary_target"

TARGET_METADATA_COLUMNS = [
    "ontology_relation",
    TARGET_COLUMN,
    ELIGIBILITY_COLUMN,
    "nearest_shared_ancestor_id",
    "nearest_shared_ancestor_label",
    "prediction_to_shared_distance",
    "reference_to_shared_distance",
    INDEPENDENT_COLUMN,
]

REFERENCE_OR_TARGET_TERMS = [
    "target",
    "reference",
    "expert",
    "ground_truth",
    "ontology_relation",
    "shared_ancestor",
    "llm",
    "expected",
]


def norm_id(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
    except Exception:
        pass

    return text


def as_bool(value: Any) -> bool:
    return (
        str(value)
        .strip()
        .casefold()
        in {
            "true",
            "1",
            "yes",
            "y",
        }
    )


def as_target(value: Any) -> int | None:
    if pd.isna(value):
        return None

    text = str(value).strip()

    if text == "":
        return None

    try:
        num = int(float(text))
    except Exception:
        return None

    return num if num in {0, 1} else None


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-matrix",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--target-audit",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/"
            "router_selective_xy_linkage_v1"
        ),
    )

    args = parser.parse_args()

    if not args.model_matrix.exists():
        raise FileNotFoundError(
            f"Model-ready matrix not found: "
            f"{args.model_matrix}"
        )

    if not args.target_audit.exists():
        raise FileNotFoundError(
            f"Target audit not found: "
            f"{args.target_audit}"
        )

    x = pd.read_csv(
        args.model_matrix
    )

    targets = pd.read_csv(
        args.target_audit
    )

    required_x = [
        "row_id",
        "dataset",
        "cluster_id",
        *PROVISIONAL_MODEL_FEATURES,
    ]

    required_targets = [
        "row_id",
        "dataset",
        "cluster_id",
        TARGET_COLUMN,
        ELIGIBILITY_COLUMN,
        INDEPENDENT_COLUMN,
    ]

    missing_x = [
        column
        for column in required_x
        if column not in x.columns
    ]

    missing_targets = [
        column
        for column in required_targets
        if column not in targets.columns
    ]

    if missing_x:
        raise ValueError(
            "Model-ready matrix missing columns: "
            + ", ".join(missing_x)
        )

    if missing_targets:
        raise ValueError(
            "Target audit missing columns: "
            + ", ".join(missing_targets)
        )

    x = x.copy()
    targets = targets.copy()

    x["_row_key"] = x["row_id"].map(norm_id)
    targets["_row_key"] = targets["row_id"].map(norm_id)

    x_dupes = sorted(
        set(
            x.loc[
                x["_row_key"].duplicated(keep=False),
                "_row_key",
            ].tolist()
        )
    )

    target_dupes = sorted(
        set(
            targets.loc[
                targets["_row_key"].duplicated(keep=False),
                "_row_key",
            ].tolist()
        )
    )

    x_keys = set(x["_row_key"])
    target_keys = set(targets["_row_key"])

    x_only = sorted(x_keys - target_keys)
    target_only = sorted(target_keys - x_keys)

    row_id_sets_match = (
        not x_only
        and not target_only
        and not x_dupes
        and not target_dupes
    )

    # -------------------------------------------------------------------------
    # Verify biological metadata identity before attaching targets.
    # -------------------------------------------------------------------------
    x_indexed = x.set_index("_row_key", drop=False)
    t_indexed = targets.set_index("_row_key", drop=False)

    identity_mismatches = []

    if row_id_sets_match:
        for row_key in sorted(
            x_keys,
            key=lambda value: int(value)
            if value.isdigit()
            else value,
        ):
            x_dataset = str(
                x_indexed.loc[row_key, "dataset"]
            ).strip().casefold()

            t_dataset = str(
                t_indexed.loc[row_key, "dataset"]
            ).strip().casefold()

            x_cluster = norm_id(
                x_indexed.loc[row_key, "cluster_id"]
            )

            t_cluster = norm_id(
                t_indexed.loc[row_key, "cluster_id"]
            )

            if (
                x_dataset != t_dataset
                or x_cluster != t_cluster
            ):
                identity_mismatches.append(
                    {
                        "row_id": row_key,
                        "x_dataset": x_dataset,
                        "target_dataset": t_dataset,
                        "x_cluster_id": x_cluster,
                        "target_cluster_id": t_cluster,
                    }
                )

    stable_identity_pass = (
        row_id_sets_match
        and not identity_mismatches
    )

    # -------------------------------------------------------------------------
    # Duplicate biological key audit.
    # -------------------------------------------------------------------------
    x["_bio_key"] = (
        x["dataset"].astype(str).str.strip().str.casefold()
        + "::"
        + x["cluster_id"].map(norm_id)
    )

    target_bio_key = (
        targets["dataset"].astype(str).str.strip().str.casefold()
        + "::"
        + targets["cluster_id"].map(norm_id)
    )

    x_bio_dupes = sorted(
        set(
            x.loc[
                x["_bio_key"].duplicated(keep=False),
                "_bio_key",
            ].tolist()
        )
    )

    target_bio_dupes = sorted(
        set(
            target_bio_key[
                target_bio_key.duplicated(keep=False)
            ].tolist()
        )
    )

    # -------------------------------------------------------------------------
    # Join ONLY target metadata to X using row_id after identity validation.
    # -------------------------------------------------------------------------
    target_columns_to_join = [
        "_row_key",
        "dataset",
        "cluster_id",
    ]

    for column in TARGET_METADATA_COLUMNS:
        if column in targets.columns:
            target_columns_to_join.append(column)

    target_join = targets[
        target_columns_to_join
    ].copy()

    target_join = target_join.rename(
        columns={
            "dataset": "target_dataset",
            "cluster_id": "target_cluster_id",
        }
    )

    joined = x.merge(
        target_join,
        on="_row_key",
        how="left",
        validate="one_to_one",
    )

    # Target eligibility accounting.
    # Primary usable target requires BOTH:
    #   target_eligible == true
    #   has_independent_primary_target == true
    # and a valid binary target.

    joined["_target_eligible_bool"] = joined[
        ELIGIBILITY_COLUMN
    ].map(as_bool)

    joined["_independent_bool"] = joined[
        INDEPENDENT_COLUMN
    ].map(as_bool)

    joined["_target_binary"] = joined[
        TARGET_COLUMN
    ].map(as_target)

    joined["_primary_usable_target"] = (
        joined["_target_eligible_bool"]
        & joined["_independent_bool"]
        & joined["_target_binary"].isin([0, 1])
    )

    usable = joined[
        joined["_primary_usable_target"]
    ].copy()

    sufficient_count = int(
        (
            usable["_target_binary"]
            == 1
        ).sum()
    )

    insufficient_count = int(
        (
            usable["_target_binary"]
            == 0
        ).sum()
    )

    eligible_count = int(
        len(usable)
    )

    # Dataset-level accounting.
    dataset_rows = []

    for dataset_name, group in joined.groupby(
        "dataset",
        dropna=False,
        sort=True,
    ):
        usable_group = group[
            group["_primary_usable_target"]
        ]

        dataset_rows.append(
            {
                "dataset": str(dataset_name),
                "total_rows": int(len(group)),
                "independent_primary_targets": int(
                    len(usable_group)
                ),
                "sufficient_targets": int(
                    (
                        usable_group["_target_binary"]
                        == 1
                    ).sum()
                ),
                "insufficient_targets": int(
                    (
                        usable_group["_target_binary"]
                        == 0
                    ).sum()
                ),
            }
        )

    dataset_summary = pd.DataFrame(dataset_rows)

    lung_usable_count = 0

    if not dataset_summary.empty:
        lung_rows = dataset_summary[
            dataset_summary["dataset"]
            .astype(str)
            .str.strip()
            .str.casefold()
            == "lung"
        ]

        if not lung_rows.empty:
            lung_usable_count = int(
                lung_rows.iloc[0][
                    "independent_primary_targets"
                ]
            )

    # Label leakage checks.
    predictive_columns = list(
        PROVISIONAL_MODEL_FEATURES
    )

    prohibited_predictive_matches = [
        column
        for column in predictive_columns
        if any(
            term in column.casefold()
            for term in REFERENCE_OR_TARGET_TERMS
        )
    ]

    target_like_columns_in_x_table = [
        column
        for column in x.columns
        if any(
            term in column.casefold()
            for term in REFERENCE_OR_TARGET_TERMS
        )
    ]

    # Metadata columns are intentionally excluded from predictive feature list.
    metadata_in_predictive_features = [
        column
        for column in [
            "row_id",
            "dataset",
            "cluster_id",
        ]
        if column in predictive_columns
    ]

    # Validate target consistency:
    # independently eligible row must have target_eligible and valid y.
    independent_without_valid_target = []

    for _, row in joined.iterrows():
        if row["_independent_bool"]:
            if (
                not row["_target_eligible_bool"]
                or row["_target_binary"] not in {0, 1}
            ):
                independent_without_valid_target.append(
                    {
                        "row_id": norm_id(row["row_id"]),
                        "dataset": str(row["dataset"]),
                        "cluster_id": norm_id(
                            row["cluster_id"]
                        ),
                        "target_eligible": row[
                            ELIGIBILITY_COLUMN
                        ],
                        "target_value": row[
                            TARGET_COLUMN
                        ],
                    }
                )


    expected_current = {
        "total_rows": 20,
        "eligible_targets": 7,
        "sufficient_targets": 4,
        "insufficient_targets": 3,
        "lung_independent_targets": 0,
    }

    expected_accounting_pass = (
        len(joined)
        == expected_current["total_rows"]
        and eligible_count
        == expected_current["eligible_targets"]
        and sufficient_count
        == expected_current["sufficient_targets"]
        and insufficient_count
        == expected_current["insufficient_targets"]
        and lung_usable_count
        == expected_current["lung_independent_targets"]
    )

    status_pass = (
        stable_identity_pass
        and not x_bio_dupes
        and not target_bio_dupes
        and not prohibited_predictive_matches
        and not target_like_columns_in_x_table
        and not metadata_in_predictive_features
        and not independent_without_valid_target
        and expected_accounting_pass
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    linkage_path = (
        args.output_dir
        / "selective_router_xy_linkage_audit_table_v1.csv"
    )

    eligible_path = (
        args.output_dir
        / "selective_router_independent_targets_v1.csv"
    )

    dataset_summary_path = (
        args.output_dir
        / "selective_router_target_dataset_summary_v1.csv"
    )

    mismatch_path = (
        args.output_dir
        / "selective_router_xy_identity_mismatches_v1.csv"
    )

    audit_path = (
        args.output_dir
        / "selective_router_xy_linkage_audit_v1.json"
    )

    linkage_output_columns = [
        "row_id",
        "dataset",
        "cluster_id",
        "ontology_relation",
        TARGET_COLUMN,
        ELIGIBILITY_COLUMN,
        INDEPENDENT_COLUMN,
        "_primary_usable_target",
    ]

    available_linkage_columns = [
        column
        for column in linkage_output_columns
        if column in joined.columns
    ]

    joined[
        available_linkage_columns
    ].to_csv(
        linkage_path,
        index=False,
    )

    eligible_columns = [
        "row_id",
        "dataset",
        "cluster_id",
        "ontology_relation",
        TARGET_COLUMN,
    ]

    available_eligible_columns = [
        column
        for column in eligible_columns
        if column in usable.columns
    ]

    usable[
        available_eligible_columns
    ].to_csv(
        eligible_path,
        index=False,
    )

    dataset_summary.to_csv(
        dataset_summary_path,
        index=False,
    )

    pd.DataFrame(
        identity_mismatches,
        columns=[
            "row_id",
            "x_dataset",
            "target_dataset",
            "x_cluster_id",
            "target_cluster_id",
        ],
    ).to_csv(
        mismatch_path,
        index=False,
    )

    audit = {
        "audit_role": (
            "SELECTIVE_ROUTER_XY_LINKAGE_VALIDATION"
        ),
        "model_matrix": str(
            args.model_matrix
        ),
        "target_audit": str(
            args.target_audit
        ),
        "x_rows": int(
            len(x)
        ),
        "target_rows": int(
            len(targets)
        ),
        "row_id_unique_x": (
            not bool(x_dupes)
        ),
        "row_id_unique_targets": (
            not bool(target_dupes)
        ),
        "row_id_sets_match": (
            not x_only
            and not target_only
        ),
        "dataset_cluster_identity_mismatch_count": len(
            identity_mismatches
        ),
        "duplicate_biological_keys_x": (
            x_bio_dupes
        ),
        "duplicate_biological_keys_targets": (
            target_bio_dupes
        ),
        "stable_identity_linkage_pass": (
            stable_identity_pass
        ),
        "predictive_feature_count": len(
            predictive_columns
        ),
        "predictive_features": (
            predictive_columns
        ),
        "target_or_reference_like_predictive_features": (
            prohibited_predictive_matches
        ),
        "target_or_reference_like_columns_present_in_x_table": (
            target_like_columns_in_x_table
        ),
        "metadata_columns_used_predictively": (
            metadata_in_predictive_features
        ),
        "current_target_accounting": {
            "total_rows": int(
                len(joined)
            ),
            "eligible_independent_targets": (
                eligible_count
            ),
            "sufficient_targets": (
                sufficient_count
            ),
            "insufficient_targets": (
                insufficient_count
            ),
            "lung_independent_targets": (
                lung_usable_count
            ),
        },
        "expected_current_target_accounting": (
            expected_current
        ),
        "expected_accounting_pass": (
            expected_accounting_pass
        ),
        "independent_rows_without_valid_target": (
            independent_without_valid_target
        ),
        "llm_only_lung_labels_used_as_primary_targets": (
            lung_usable_count > 0
        ),
        "current_sample_sufficient_for_final_router_calibration": False,
        "current_sample_note": (
            "Seven independent targets are insufficient for defensible "
            "final coefficient estimation, threshold calibration, and "
            "held-out publication evaluation."
        ),
        "model_trained": False,
        "scaling_fitted": False,
        "threshold_selected": False,
        "router_v4_replaced": False,
        "existing_pipeline_outputs_modified": False,
        "status": (
            "PASS"
            if status_pass
            else "FAIL"
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
        f"X-y linkage table: {linkage_path}"
    )
    print(
        f"Independent target rows: {eligible_path}"
    )
    print(
        f"Dataset target summary: {dataset_summary_path}"
    )
    print(
        f"Identity mismatches: {mismatch_path}"
    )
    print(
        f"Audit: {audit_path}"
    )

    print()
    print(
        f"X rows: {len(x)}"
    )
    print(
        f"Target-audit rows: {len(targets)}"
    )
    print(
        "X row_id unique: "
        + (
            "YES"
            if not x_dupes
            else "NO"
        )
    )
    print(
        "Target row_id unique: "
        + (
            "YES"
            if not target_dupes
            else "NO"
        )
    )
    print(
        "row_id sets match exactly: "
        + (
            "YES"
            if not x_only
            and not target_only
            else "NO"
        )
    )
    print(
        "Dataset/cluster identity mismatches: "
        f"{len(identity_mismatches)}"
    )
    print(
        "Duplicate biological keys in X: "
        f"{len(x_bio_dupes)}"
    )
    print(
        "Duplicate biological keys in target audit: "
        f"{len(target_bio_dupes)}"
    )

    print()
    print(
        f"Predictive features: {len(predictive_columns)}"
    )
    print(
        "Target/reference-like predictive features: "
        f"{len(prohibited_predictive_matches)}"
    )
    print(
        "Target/reference-like columns present in X table: "
        f"{len(target_like_columns_in_x_table)}"
    )
    print(
        "Metadata columns used predictively: "
        f"{len(metadata_in_predictive_features)}"
    )

    print()
    print(
        f"Independent eligible targets: {eligible_count}"
    )
    print(
        f"  sufficient (1): {sufficient_count}"
    )
    print(
        f"  insufficient (0): {insufficient_count}"
    )
    print(
        f"Lung independent primary targets: {lung_usable_count}"
    )
    print(
        "LLM-only lung labels used as primary targets: "
        + (
            "YES"
            if lung_usable_count > 0
            else "NO"
        )
    )
    print(
        "Expected target accounting: "
        + (
            "PASS"
            if expected_accounting_pass
            else "FAIL"
        )
    )

    print()
    print(
        "X-y LINKAGE STATUS: "
        + (
            "PASS"
            if status_pass
            else "FAIL"
        )
    )
    print()
    print(
        "Current 7 targets sufficient for FINAL Router calibration: NO"
    )
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
