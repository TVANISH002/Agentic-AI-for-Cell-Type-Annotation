from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd



RAW_EVIDENCE_FEATURES = [
    "candidate_count",
    "top1_support_count",
    "top2_present",
    "top3_present",
    "top2_support_count",
    "top3_support_count",
    "top1_cross_database_support_count",
    "top1_cross_database_fraction",
    "direct_evidence_coverage",
    "top_marker_direct_supported",
    "top1_support_fraction_of_input",
    "top2_support_fraction_of_input",
    "top1_top2_gap_fraction_of_input",
    "leading_non_direct_fraction",
    "first_direct_supported_rank_fraction",
    "top1_mean_supporting_gene_rank_fraction",
]

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

METADATA_COLUMNS = [
    "row_id",
    "dataset",
    "cluster_id",
]

PROHIBITED_PREDICTIVE_COLUMNS = [
    "row_id",
    "dataset",
    "cluster_id",
    "ontology_relation",
    "primary_router_sufficiency_target",
    "target_eligible",
    "has_independent_primary_target",
    "reference_label",
    "reference_label_raw",
    "expert_label",
    "expert_annotation",
    "llm_label",
    "llm_annotation",
    "ground_truth",
    "router_decision",
    "router_confidence_score",
]


def to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        lowered = (
            series.astype(str)
            .str.strip()
            .str.casefold()
        )

        mapping = {
            "true": 1.0,
            "false": 0.0,
            "yes": 1.0,
            "no": 0.0,
        }

        mapped = lowered.map(
            mapping
        )

        parsed = pd.to_numeric(
            series,
            errors="coerce",
        )

        mask = (
            parsed.isna()
            & mapped.notna()
        )

        parsed.loc[
            mask
        ] = mapped.loc[
            mask
        ]

        return parsed.astype(
            float
        )

    return pd.to_numeric(
        series,
        errors="coerce",
    ).astype(
        float
    )


def normalized_key(value: Any) -> str:
    if pd.isna(
        value
    ):
        return ""

    text = str(
        value
    ).strip()

    try:
        number = float(
            text
        )

        if number.is_integer():
            return str(
                int(
                    number
                )
            )
    except Exception:
        pass

    return text


def values_equal(
    left: Any,
    right: Any,
    tol: float = 1e-8,
) -> bool:
    if pd.isna(
        left
    ) and pd.isna(
        right
    ):
        return True

    if pd.isna(
        left
    ) or pd.isna(
        right
    ):
        return False

    try:
        return math.isclose(
            float(
                left
            ),
            float(
                right
            ),
            rel_tol=tol,
            abs_tol=tol,
        )
    except Exception:
        return (
            str(
                left
            ).strip()
            == str(
                right
            ).strip()
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature-matrix",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--identity-audit",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/"
            "router_selective_model_matrix_v2"
        ),
    )

    args = parser.parse_args()

    if not args.feature_matrix.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: "
            f"{args.feature_matrix}"
        )

    if not args.identity_audit.exists():
        raise FileNotFoundError(
            f"Identity audit not found: "
            f"{args.identity_audit}"
        )

    features = pd.read_csv(
        args.feature_matrix
    )

    identity = pd.read_csv(
        args.identity_audit
    )

    required_feature_columns = [
        "row_id",
        *RAW_EVIDENCE_FEATURES,
    ]

    missing_feature_columns = [
        column
        for column in required_feature_columns
        if column not in features.columns
    ]

    required_identity_columns = [
        "row_id",
        "dataset",
        "cluster_id",
    ]

    missing_identity_columns = [
        column
        for column in required_identity_columns
        if column not in identity.columns
    ]

    if missing_feature_columns:
        raise ValueError(
            "Feature matrix missing required columns: "
            + ", ".join(
                missing_feature_columns
            )
        )

    if missing_identity_columns:
        raise ValueError(
            "Identity audit missing required columns: "
            + ", ".join(
                missing_identity_columns
            )
        )

    features = features.copy()
    identity = identity.copy()

    features[
        "_row_key"
    ] = features[
        "row_id"
    ].map(
        normalized_key
    )

    identity[
        "_row_key"
    ] = identity[
        "row_id"
    ].map(
        normalized_key
    )

    feature_duplicate_keys = (
        features.loc[
            features[
                "_row_key"
            ].duplicated(
                keep=False
            ),
            "_row_key",
        ]
        .tolist()
    )

    identity_duplicate_keys = (
        identity.loc[
            identity[
                "_row_key"
            ].duplicated(
                keep=False
            ),
            "_row_key",
        ]
        .tolist()
    )

    feature_keys = set(
        features[
            "_row_key"
        ]
    )

    identity_keys = set(
        identity[
            "_row_key"
        ]
    )

    feature_only_keys = sorted(
        feature_keys
        - identity_keys
    )

    identity_only_keys = sorted(
        identity_keys
        - feature_keys
    )

    one_to_one_key_match = (
        not feature_duplicate_keys
        and not identity_duplicate_keys
        and not feature_only_keys
        and not identity_only_keys
        and len(
            features
        )
        == len(
            identity
        )
    )

    # -------------------------------------------------------------------------
    # Verify that row_id truly refers to the same biological/evidence row in
    # both tables. Compare all overlapping frozen evidence features.
    # -------------------------------------------------------------------------
    overlapping_evidence_features = [
        feature
        for feature in RAW_EVIDENCE_FEATURES
        if feature in identity.columns
    ]

    feature_indexed = features.set_index(
        "_row_key",
        drop=False,
    )

    identity_indexed = identity.set_index(
        "_row_key",
        drop=False,
    )

    evidence_mismatches = []

    if one_to_one_key_match:
        for row_key in sorted(
            feature_keys,
            key=lambda value: int(
                value
            )
            if value.isdigit()
            else value,
        ):
            for column in overlapping_evidence_features:
                left = feature_indexed.loc[
                    row_key,
                    column,
                ]

                right = identity_indexed.loc[
                    row_key,
                    column,
                ]

                if not values_equal(
                    left,
                    right,
                ):
                    evidence_mismatches.append(
                        {
                            "row_id": row_key,
                            "feature": column,
                            "feature_matrix_value": (
                                None
                                if pd.isna(
                                    left
                                )
                                else left
                            ),
                            "identity_audit_value": (
                                None
                                if pd.isna(
                                    right
                                )
                                else right
                            ),
                        }
                    )

    evidence_identity_pass = (
        one_to_one_key_match
        and len(
            evidence_mismatches
        )
        == 0
    )

    # -------------------------------------------------------------------------
    # Join metadata using verified row_id only.
    # No target columns are carried into the model-ready matrix.
    # -------------------------------------------------------------------------
    identity_metadata = identity[
        [
            "_row_key",
            "dataset",
            "cluster_id",
        ]
    ].copy()

    joined = features.merge(
        identity_metadata,
        on="_row_key",
        how="left",
        validate="one_to_one",
    )

    dataset_missing = int(
        joined[
            "dataset"
        ].isna().sum()
    )

    cluster_missing = int(
        joined[
            "cluster_id"
        ].isna().sum()
    )

    # -------------------------------------------------------------------------
    # Numeric model inputs.
    # -------------------------------------------------------------------------
    numeric = pd.DataFrame(
        {
            feature: to_numeric(
                joined[
                    feature
                ]
            )
            for feature in PROVISIONAL_MODEL_FEATURES
        }
    )

    structural_top3_mask = (
        numeric[
            "top3_present"
        ]
        == 0
    ) & (
        numeric[
            "top3_support_count"
        ].isna()
    )

    invalid_missing_top3_mask = (
        numeric[
            "top3_present"
        ]
        == 1
    ) & (
        numeric[
            "top3_support_count"
        ].isna()
    )

    contradictory_top3_mask = (
        numeric[
            "top3_present"
        ]
        == 0
    ) & (
        numeric[
            "top3_support_count"
        ].notna()
    )

    numeric.loc[
        structural_top3_mask,
        "top3_support_count",
    ] = 0.0

    invalid_binary_rows = []

    for column in [
        "top2_present",
        "top3_present",
    ]:
        for index, value in numeric[
            column
        ].items():
            if pd.isna(
                value
            ) or value not in {
                0.0,
                1.0,
            }:
                invalid_binary_rows.append(
                    {
                        "row_index": int(
                            index
                        ),
                        "feature": column,
                        "value": (
                            None
                            if pd.isna(
                                value
                            )
                            else float(
                                value
                            )
                        ),
                    }
                )

    remaining_missing = []

    nonfinite = []

    for feature in PROVISIONAL_MODEL_FEATURES:
        for index, value in numeric[
            feature
        ].items():
            if pd.isna(
                value
            ):
                remaining_missing.append(
                    {
                        "row_index": int(
                            index
                        ),
                        "feature": feature,
                    }
                )
            elif not math.isfinite(
                float(
                    value
                )
            ):
                nonfinite.append(
                    {
                        "row_index": int(
                            index
                        ),
                        "feature": feature,
                        "value": str(
                            value
                        ),
                    }
                )

    zero_variance_features = [
        feature
        for feature in PROVISIONAL_MODEL_FEATURES
        if numeric[
            feature
        ].nunique(
            dropna=True
        )
        <= 1
    ]

    # -------------------------------------------------------------------------
    # Final output:
    # metadata columns are present for joining/audit, but predictive contract
    # remains exactly the 14 features below.
    # -------------------------------------------------------------------------
    output = pd.DataFrame(
        {
            "row_id": joined[
                "row_id"
            ],
            "dataset": joined[
                "dataset"
            ],
            "cluster_id": joined[
                "cluster_id"
            ],
        }
    )

    for feature in PROVISIONAL_MODEL_FEATURES:
        output[
            feature
        ] = numeric[
            feature
        ]

    prohibited_in_predictive_contract = [
        column
        for column in PROHIBITED_PREDICTIVE_COLUMNS
        if column in PROVISIONAL_MODEL_FEATURES
    ]

    leaked_target_columns_in_output = [
        column
        for column in output.columns
        if column in {
            "ontology_relation",
            "primary_router_sufficiency_target",
            "target_eligible",
            "has_independent_primary_target",
            "reference_label",
            "reference_label_raw",
            "expert_label",
            "llm_label",
            "ground_truth",
        }
    ]

    status_pass = (
        one_to_one_key_match
        and evidence_identity_pass
        and dataset_missing == 0
        and cluster_missing == 0
        and not invalid_binary_rows
        and not remaining_missing
        and not nonfinite
        and int(
            invalid_missing_top3_mask.sum()
        )
        == 0
        and int(
            contradictory_top3_mask.sum()
        )
        == 0
        and not prohibited_in_predictive_contract
        and not leaked_target_columns_in_output
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        args.output_dir
        / "selective_router_model_ready_features_v2.csv"
    )

    mismatch_path = (
        args.output_dir
        / "row_id_evidence_identity_mismatches_v2.csv"
    )

    audit_path = (
        args.output_dir
        / "selective_router_model_ready_audit_v2.json"
    )

    variance_path = (
        args.output_dir
        / "selective_router_model_ready_variance_v2.csv"
    )

    output.to_csv(
        output_path,
        index=False,
    )

    pd.DataFrame(
        evidence_mismatches,
        columns=[
            "row_id",
            "feature",
            "feature_matrix_value",
            "identity_audit_value",
        ],
    ).to_csv(
        mismatch_path,
        index=False,
    )

    pd.DataFrame(
        [
            {
                "feature": feature,
                "unique_nonmissing_values": int(
                    numeric[
                        feature
                    ].nunique(
                        dropna=True
                    )
                ),
                "variance": (
                    float(
                        numeric[
                            feature
                        ].var(
                            ddof=1
                        )
                    )
                    if numeric[
                        feature
                    ].notna().sum()
                    > 1
                    else np.nan
                ),
                "zero_variance_current_sample": (
                    feature
                    in zero_variance_features
                ),
            }
            for feature in PROVISIONAL_MODEL_FEATURES
        ]
    ).to_csv(
        variance_path,
        index=False,
    )

    audit = {
        "audit_role": (
            "SELECTIVE_ROUTER_MODEL_MATRIX_WITH_VERIFIED_IDENTIFIERS"
        ),
        "feature_matrix": str(
            args.feature_matrix
        ),
        "identity_audit": str(
            args.identity_audit
        ),
        "feature_rows": int(
            len(
                features
            )
        ),
        "identity_rows": int(
            len(
                identity
            )
        ),
        "row_id_unique_feature_matrix": (
            not bool(
                feature_duplicate_keys
            )
        ),
        "row_id_unique_identity_audit": (
            not bool(
                identity_duplicate_keys
            )
        ),
        "row_id_sets_match": (
            not feature_only_keys
            and not identity_only_keys
        ),
        "feature_only_row_ids": (
            feature_only_keys
        ),
        "identity_only_row_ids": (
            identity_only_keys
        ),
        "overlapping_evidence_features_checked": (
            overlapping_evidence_features
        ),
        "overlapping_evidence_feature_count": len(
            overlapping_evidence_features
        ),
        "evidence_identity_mismatch_count": len(
            evidence_mismatches
        ),
        "stable_identity_join_pass": (
            evidence_identity_pass
        ),
        "metadata_columns": (
            METADATA_COLUMNS
        ),
        "metadata_is_predictive": False,
        "provisional_predictive_features": (
            PROVISIONAL_MODEL_FEATURES
        ),
        "provisional_predictive_feature_count": len(
            PROVISIONAL_MODEL_FEATURES
        ),
        "prohibited_fields_in_predictive_contract": (
            prohibited_in_predictive_contract
        ),
        "target_or_reference_columns_in_output": (
            leaked_target_columns_in_output
        ),
        "structural_top3_rows_encoded_zero": int(
            structural_top3_mask.sum()
        ),
        "invalid_missing_top3_rows": int(
            invalid_missing_top3_mask.sum()
        ),
        "contradictory_top3_rows": int(
            contradictory_top3_mask.sum()
        ),
        "remaining_missing_model_values": len(
            remaining_missing
        ),
        "nonfinite_model_values": len(
            nonfinite
        ),
        "zero_variance_features_current_sample": (
            zero_variance_features
        ),
        "target_labels_used_for_feature_construction": False,
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
        f"Model-ready matrix V2: {output_path}"
    )
    print(
        f"Evidence identity mismatches: {mismatch_path}"
    )
    print(
        f"Variance profile: {variance_path}"
    )
    print(
        f"Audit: {audit_path}"
    )

    print()
    print(
        f"Feature rows: {len(features)}"
    )
    print(
        f"Identity rows: {len(identity)}"
    )
    print(
        "Feature row_id unique: "
        + (
            "YES"
            if not feature_duplicate_keys
            else "NO"
        )
    )
    print(
        "Identity row_id unique: "
        + (
            "YES"
            if not identity_duplicate_keys
            else "NO"
        )
    )
    print(
        "row_id sets match exactly: "
        + (
            "YES"
            if not feature_only_keys
            and not identity_only_keys
            else "NO"
        )
    )
    print(
        "Overlapping evidence features checked: "
        f"{len(overlapping_evidence_features)}"
    )
    print(
        "Evidence identity mismatches: "
        f"{len(evidence_mismatches)}"
    )
    print(
        "Stable row_id -> dataset/cluster identity: "
        + (
            "PASS"
            if evidence_identity_pass
            else "FAIL"
        )
    )

    print()
    print(
        f"Metadata dataset missing: {dataset_missing}"
    )
    print(
        f"Metadata cluster_id missing: {cluster_missing}"
    )
    print(
        "Provisional predictive features: "
        f"{len(PROVISIONAL_MODEL_FEATURES)}"
    )
    print(
        "Prohibited fields in predictive contract: "
        f"{len(prohibited_in_predictive_contract)}"
    )
    print(
        "Target/reference columns in model-ready output: "
        f"{len(leaked_target_columns_in_output)}"
    )

    print()
    print(
        "Structural top3 missing rows encoded as 0: "
        f"{int(structural_top3_mask.sum())}"
    )
    print(
        "Invalid missing top3 rows: "
        f"{int(invalid_missing_top3_mask.sum())}"
    )
    print(
        "Contradictory top3 rows: "
        f"{int(contradictory_top3_mask.sum())}"
    )
    print(
        "Remaining missing model values: "
        f"{len(remaining_missing)}"
    )
    print(
        "Non-finite model values: "
        f"{len(nonfinite)}"
    )

    print()
    print(
        "CURRENT-SAMPLE ZERO-VARIANCE FEATURES:"
    )
    if zero_variance_features:
        for feature in zero_variance_features:
            print(
                f"  {feature}"
            )
    else:
        print(
            "  none"
        )

    print()
    print(
        "MODEL-READY MATRIX V2 STATUS: "
        + (
            "PASS"
            if status_pass
            else "FAIL"
        )
    )
    print()
    print(
        "Dataset/cluster identifiers are metadata only: YES"
    )
    print(
        "Target labels used for feature construction: NO"
    )
    print(
        "Model trained: NO"
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
