from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# FREEZE PROVISIONAL SELECTIVE ROUTER MODEL-FEATURE CONTRACT




RAW_FEATURE_POOL = [
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


STRUCTURALLY_EXCLUDED_FROM_PROVISIONAL_MODEL = {
    "top1_cross_database_support_count": {
        "retained_in_raw_schema": True,
        "reason": (
            "Redundant with top1_cross_database_fraction when "
            "top1_support_count is retained. The fraction equals "
            "cross-database support count divided by top1 support count, "
            "up to the established four-decimal reporting precision."
        ),
        "preferred_representation": "top1_cross_database_fraction",
    },
    "top_marker_direct_supported": {
        "retained_in_raw_schema": True,
        "reason": (
            "In the audited table, top_marker_direct_supported is exactly "
            "determined by whether leading_non_direct_fraction equals zero "
            "(20/20 rows). leading_non_direct_fraction is retained because "
            "it provides richer rank-aware information than a binary flag."
        ),
        "preferred_representation": "leading_non_direct_fraction",
    },
}


PROVISIONAL_MODEL_FEATURES = [
    feature
    for feature in RAW_FEATURE_POOL
    if feature
    not in STRUCTURALLY_EXCLUDED_FROM_PROVISIONAL_MODEL
]


CONDITIONAL_FEATURE_POLICIES = {
    "top2_present": {
        "raw_schema_status": "retain",
        "provisional_model_status": "retain",
        "current_sample_note": (
            "Constant at 1.0 in the current 20-cluster engineering sample."
        ),
        "future_fit_rule": (
            "Exclude from a fitted development model if it remains "
            "zero-variance in the development data; do not delete it "
            "from the tissue-agnostic raw evidence schema."
        ),
    },
    "top3_present": {
        "raw_schema_status": "retain",
        "provisional_model_status": "retain",
        "current_sample_note": (
            "Near-constant (1 in 19/20 rows), but it exactly identifies "
            "the structural absence of top3_support_count."
        ),
        "future_fit_rule": (
            "Retain as a structural-presence indicator unless future "
            "development data show it has no usable variance."
        ),
    },
    "top3_support_count": {
        "raw_schema_status": "retain_missing_as_nan_when_top3_absent",
        "provisional_model_status": "retain",
        "future_model_encoding": (
            "Presence-gated encoding: when top3_present=0, encode the "
            "model-input support count as 0 while retaining top3_present. "
            "If top3_present=1 and top3_support_count is missing, raise "
            "a data-quality error rather than silently imputing."
        ),
    },
}


PENDING_DEVELOPMENT_ABLATIONS = [
    {
        "features": [
            "top2_support_count",
            "top3_support_count",
        ],
        "reason": (
            "Highly correlated by both Pearson and Spearman in the current "
            "20-cluster sample, but they represent different candidate ranks "
            "and are not structurally equivalent."
        ),
    },
    {
        "features": [
            "top1_support_count",
            "top1_support_fraction_of_input",
        ],
        "reason": (
            "Empirically correlated, but one represents absolute evidence "
            "magnitude and the other normalizes by input-marker count."
        ),
    },
    {
        "features": [
            "leading_non_direct_fraction",
            "first_direct_supported_rank_fraction",
        ],
        "reason": (
            "Empirically related rank-aware features. Retain provisionally "
            "because they encode different aspects of missing leading evidence."
        ),
    },
    {
        "features": [
            "top1_support_fraction_of_input",
            "top2_support_fraction_of_input",
            "top1_top2_gap_fraction_of_input",
        ],
        "reason": (
            "Normalized support quantities share underlying counts. Their "
            "joint value must be evaluated through development-data ablation."
        ),
    },
]


MODEL_PREPROCESSING_POLICY = {
    "continuous_feature_scaling": (
        "Fit scaling parameters on development/training data only; "
        "apply frozen parameters to validation/test data."
    ),
    "binary_presence_indicators": (
        "Preserve as explicit 0/1 structural indicators."
    ),
    "structural_missingness": (
        "Use presence-gated zero encoding only when candidate absence is "
        "the documented cause of missing support count. Do not use generic "
        "mean/median imputation for structural absence."
    ),
    "unexpected_missing_values": (
        "Treat as data-quality errors unless a preprocessing rule is "
        "predeclared using development data only."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature-matrix",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/analysis/"
            "router_selective_provisional_feature_contract_v1"
        ),
    )

    args = parser.parse_args()

    if not args.feature_matrix.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {args.feature_matrix}"
        )

    frame = pd.read_csv(
        args.feature_matrix
    )

    missing_raw = [
        feature
        for feature in RAW_FEATURE_POOL
        if feature not in frame.columns
    ]

    missing_provisional = [
        feature
        for feature in PROVISIONAL_MODEL_FEATURES
        if feature not in frame.columns
    ]

    overlap_error = sorted(
        set(
            PROVISIONAL_MODEL_FEATURES
        )
        & set(
            STRUCTURALLY_EXCLUDED_FROM_PROVISIONAL_MODEL
        )
    )

    contract_pass = (
        not missing_raw
        and not missing_provisional
        and not overlap_error
        and len(
            PROVISIONAL_MODEL_FEATURES
        )
        == 14
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    contract = {
        "contract_name": (
            "Selective Evidence-Sufficiency Router "
            "Provisional Model Feature Contract"
        ),
        "version": "v1",
        "status": (
            "PROVISIONAL_FEATURE_SET_FROZEN_PENDING_DEVELOPMENT_ABLATION"
        ),
        "feature_matrix": str(
            args.feature_matrix
        ),
        "current_engineering_rows": int(
            len(
                frame
            )
        ),
        "raw_feature_pool_count": len(
            RAW_FEATURE_POOL
        ),
        "raw_feature_pool": (
            RAW_FEATURE_POOL
        ),
        "provisional_model_feature_count": len(
            PROVISIONAL_MODEL_FEATURES
        ),
        "provisional_model_features": (
            PROVISIONAL_MODEL_FEATURES
        ),
        "structurally_excluded_from_provisional_model": (
            STRUCTURALLY_EXCLUDED_FROM_PROVISIONAL_MODEL
        ),
        "conditional_feature_policies": (
            CONDITIONAL_FEATURE_POLICIES
        ),
        "pending_development_ablations": (
            PENDING_DEVELOPMENT_ABLATIONS
        ),
        "model_preprocessing_policy": (
            MODEL_PREPROCESSING_POLICY
        ),
        "decision_basis": {
            "expert_or_reference_targets_used": False,
            "model_performance_used": False,
            "correlation_alone_used_for_removal": False,
            "structural_relationships_used": True,
        },
        "final_feature_set": False,
        "final_model_fitted": False,
        "final_threshold_selected": False,
        "router_v4_replaced": False,
        "existing_outputs_modified": False,
    }

    audit = {
        "raw_feature_pool_count": len(
            RAW_FEATURE_POOL
        ),
        "provisional_model_feature_count": len(
            PROVISIONAL_MODEL_FEATURES
        ),
        "structural_exclusion_count": len(
            STRUCTURALLY_EXCLUDED_FROM_PROVISIONAL_MODEL
        ),
        "missing_raw_features": missing_raw,
        "missing_provisional_features": missing_provisional,
        "excluded_feature_overlap_error": overlap_error,
        "target_labels_used": False,
        "model_trained": False,
        "threshold_selected": False,
        "status": (
            "PASS"
            if contract_pass
            else "FAIL"
        ),
    }

    contract_path = (
        args.output_dir
        / "selective_router_provisional_feature_contract_v1.json"
    )

    audit_path = (
        args.output_dir
        / "selective_router_provisional_feature_contract_audit_v1.json"
    )

    contract_path.write_text(
        json.dumps(
            contract,
            indent=2,
        ),
        encoding="utf-8",
    )

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Provisional feature contract: {contract_path}"
    )
    print(
        f"Contract audit: {audit_path}"
    )
    print()
    print(
        f"Raw evidence features preserved: {len(RAW_FEATURE_POOL)}"
    )
    print(
        "Structurally excluded from provisional fitted model: "
        f"{len(STRUCTURALLY_EXCLUDED_FROM_PROVISIONAL_MODEL)}"
    )
    print(
        "Provisional model features: "
        f"{len(PROVISIONAL_MODEL_FEATURES)}"
    )
    print()

    print(
        "STRUCTURAL EXCLUSIONS:"
    )
    for feature, information in (
        STRUCTURALLY_EXCLUDED_FROM_PROVISIONAL_MODEL.items()
    ):
        print(
            f"  {feature}"
        )
        print(
            f"    preferred: "
            f"{information['preferred_representation']}"
        )

    print()
    print(
        "PROVISIONAL MODEL FEATURE CONTRACT STATUS: "
        + (
            "PASS"
            if contract_pass
            else "FAIL"
        )
    )
    print()
    print(
        "Final feature set: NO"
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
        "Expert/reference targets used for this feature decision: NO"
    )
    print(
        "No existing Agent/Router/orchestrator outputs were modified."
    )


if __name__ == "__main__":
    main()
